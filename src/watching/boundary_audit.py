"""Inspect and observe existing generated buffers without adding model graph outputs.

This does NOT construct a native-tail training cache. In particular, separate
residual summands are not silently rounded into a synthetic block input.
"""

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import torch


def _pointer_names(expression):
    return {
        node.id
        for node in ast.walk(expression)
        if isinstance(node, ast.Name) and node.id.startswith(("in_ptr", "out_ptr", "in_out_ptr"))
    }


def _triton_sources(tree):
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == "async_compile.triton":
                result[ast.unparse(node.targets[0])] = node.value.args[1].value
    return result


def _kernel_io(source):
    tree = ast.parse(source)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1:
        raise ValueError("Expected one generated Triton function")
    fn = functions[0]
    stores = [
        n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.store"
    ]
    written = set().union(*(_pointer_names(n.args[0]) for n in stores))
    pointers = [
        a.arg for a in fn.args.args if a.arg.startswith(("in_ptr", "out_ptr", "in_out_ptr"))
    ]
    return fn, pointers, stores, written


def inspect_boundary(source, first_tail_layer):
    """Locate the input-normalization producer used by one native query projection."""
    if type(first_tail_layer) is not int or first_tail_layer < 1:
        raise ValueError("Need a positive zero-based block index")
    tree, lines = ast.parse(source), source.splitlines()
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "call"]
    if len(functions) != 1:
        raise ValueError("Expected exactly one generated call wrapper")
    statements = sorted(
        (n for n in ast.walk(functions[0]) if isinstance(n, (ast.Expr, ast.Assign))),
        key=lambda n: n.lineno,
    )
    query = []
    for statement in statements:
        call = statement.value
        if not isinstance(call, ast.Call) or ast.unparse(call.func) != "extern_kernels.mm":
            continue
        comment = lines[statement.lineno - 2]
        if re.search(
            r"Source Nodes: \[[^]]*\blinear_" + str(first_tail_layer * 7) + r"\]", comment
        ):
            query.append(statement)
    if len(query) != 1:
        raise ValueError("Cannot uniquely identify native query projection")
    query = query[0]
    argument = query.value.args[0]
    if isinstance(argument, ast.Call) and ast.unparse(argument.func) == "reinterpret_tensor":
        argument = argument.args[0]
    if not isinstance(argument, ast.Name):
        raise ValueError("Unsupported query input expression")
    normalized_name = argument.id
    sources = _triton_sources(tree)
    producer = None
    for statement in statements:
        if statement.lineno >= query.lineno:
            break
        call = statement.value
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "run"
        ):
            continue
        name = ast.unparse(call.func.value)
        if name not in sources:
            continue
        fn, pointers, stores, written = _kernel_io(sources[name])
        if any(not isinstance(x, ast.Name) for x in call.args[: len(pointers)]):
            raise ValueError("Generated kernel pointer arguments must be direct materialized names")
        bindings = dict(zip(pointers, (x.id for x in call.args[: len(pointers)]), strict=True))
        if normalized_name in {bindings[p] for p in written}:
            producer = (statement, name, fn, pointers, stores, written, bindings)
    if producer is None:
        raise ValueError("Query input has no supported materialized producer")
    statement, name, fn, pointers, stores, written, bindings = producer
    if any(p.startswith("in_out_ptr") for p in pointers):
        raise ValueError("In-place kernel pointer requires separate before/after capture support")
    normalized_pointer = next(p for p in written if bindings[p] == normalized_name)
    # Propagate rsqrt ancestry through scalar assignments in source order. This
    # proves what is stored by this consumer; it is not a general IR interpreter.
    normalized_values = set()
    assignments = sorted(
        (n for n in ast.walk(fn) if isinstance(n, ast.Assign)), key=lambda n: n.lineno
    )
    for node in assignments:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        has_rsqrt = any(
            isinstance(n, ast.Call) and ast.unparse(n.func).endswith("rsqrt")
            for n in ast.walk(node.value)
        )
        depends_normalization = any(
            isinstance(n, ast.Name) and n.id in normalized_values for n in ast.walk(node.value)
        )
        if has_rsqrt or depends_normalization:
            normalized_values.add(node.targets[0].id)
    all_stores_normalized = bool(stores) and all(
        isinstance(n.args[1], ast.Name) and n.args[1].id in normalized_values for n in stores
    )
    float_loads = {}
    added_pointer_pairs = []
    for node in assignments:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "to"
            and value.args
            and ast.unparse(value.args[0]) == "tl.float32"
        ):
            load = value.func.value
            if isinstance(load, ast.Call) and ast.unparse(load.func) == "tl.load":
                names = _pointer_names(load.args[0])
                if len(names) == 1:
                    float_loads[node.targets[0].id] = next(iter(names))
        if (
            isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Add)
            and isinstance(value.left, ast.Name)
            and isinstance(value.right, ast.Name)
        ):
            if value.left.id in float_loads and value.right.id in float_loads:
                pair = [float_loads[value.left.id], float_loads[value.right.id]]
                if pair not in added_pointer_pairs:
                    added_pointer_pairs.append(pair)
    split_sum = (
        all_stores_normalized and written == {normalized_pointer} and len(added_pointer_pairs) == 1
    )
    return {
        "first_tail_layer": first_tail_layer,
        "wrapper_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "query_projection_line": query.lineno,
        "normalization_launch_line": statement.lineno,
        "normalization_kernel": name,
        "kernel_source_sha256": hashlib.sha256(sources[name].encode()).hexdigest(),
        "normalized_buffer": normalized_name,
        "pointer_bindings": bindings,
        "read_pointers": [p for p in pointers if p.startswith(("in_ptr", "in_out_ptr"))],
        "written_pointers": sorted(written),
        "store_expressions": [ast.unparse(n) for n in stores],
        "all_stored_values_depend_on_rsqrt": all_stores_normalized,
        "fp32_loaded_pointer_additions": added_pointer_pairs,
        "boundary_status": "separate_residual_summands_normalized_without_sum_store"
        if split_sum
        else "unsupported_or_not_established",
        "single_native_tail_input_certified": False,
        "scope": "Reports actual stored operands/output. "
        "Does not reconstruct hidden states, mask, or rotary values.",
    }


class ExistingBoundaryObserver:
    """Passive snapshots of existing normalization inputs/output in one wrapper."""

    def __init__(self, first_tail_layer):
        self.first_tail_layer = first_tail_layer
        self.plan = None
        self.source = None
        self.snapshots = {}
        self.calls = 0
        self.pending = False

    def _snapshot(self, frame, pointers):
        for pointer in pointers:
            value = frame.f_locals[self.plan["pointer_bindings"][pointer]]
            if not isinstance(value, torch.Tensor):
                raise RuntimeError("Expected a materialized tensor")
            self.snapshots[pointer] = value.detach().cpu().clone()

    def _local(self, frame, event, arg):
        if self.pending and event in ("line", "return"):
            self._snapshot(frame, self.plan["written_pointers"])
            self.pending = False
        if event == "line" and frame.f_lineno == self.plan["normalization_launch_line"]:
            self._snapshot(frame, self.plan["read_pointers"])
            self.pending = True
        return self._local

    def _global(self, frame, event, arg):
        if event != "call" or frame.f_code.co_name != "call":
            return None
        if not all(k in frame.f_globals for k in ("extern_kernels", "assert_size_stride")):
            return None
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("Expected one full generated forward wrapper")
        self.source = Path(frame.f_code.co_filename).read_text()
        self.plan = inspect_boundary(self.source, self.first_tail_layer)
        return self._local

    def __enter__(self):
        if sys.gettrace() is not None:
            raise RuntimeError("Refusing to overwrite an existing tracer")
        sys.settrace(self._global)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.settrace(None)
        if exc_type is None:
            if (
                self.calls != 1
                or self.pending
                or set(self.snapshots) != set(self.plan["pointer_bindings"])
            ):
                raise RuntimeError("Existing-buffer observation incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wrapper", type=Path)
    parser.add_argument("--first-tail-layer", type=int, default=34)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_boundary(args.wrapper.read_text(), args.first_tail_layer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
