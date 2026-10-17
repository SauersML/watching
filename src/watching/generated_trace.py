"""Observe already-materialized tensors in Inductor's Python launch wrapper.

No FX graph or kernel is modified. This is intentionally a version-specific,
fail-closed adapter for the unfused Qwen3 projection/SDPA wrapper structure.
"""

import ast
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Site:
    line: int
    when: str
    label: str
    names: tuple[str, ...]
    tuple_first: bool = False
    kernel: str | None = None
    scalar_args: tuple[int, ...] = ()


def _root_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "reinterpret_tensor" and node.args:
            return _root_name(node.args[0])
    raise ValueError(f"Unsupported generated tensor expression: {ast.dump(node)}")


def plan_sites(source, layers, probe_layer):
    """Map projection ordinals and explicit attention calls, never guessed buffers."""
    tree = ast.parse(source)
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "call"]
    if len(methods) != 1:
        raise ValueError("Expected exactly one generated call function")
    statements = sorted(
        (n for n in ast.walk(methods[0]) if isinstance(n, (ast.Assign, ast.Expr))),
        key=lambda n: n.lineno,
    )
    lines = source.splitlines()
    projections = {}
    attention = []
    sites = []
    for statement in statements:
        call = statement.value
        if not isinstance(call, ast.Call):
            continue
        name = ast.unparse(call.func)
        if name == "extern_kernels.mm":
            comment = lines[statement.lineno - 2].strip()
            match = re.search(r"Topologically Sorted Source Nodes: \[([^]]+)\]", comment)
            if match is None:
                raise ValueError("Projection lacks generated source-node annotation")
            nodes = match.group(1).split(", ")
            for layer in range(layers):
                tag = "linear" if layer == 0 else f"linear_{7 * layer}"
                if nodes[-1] != tag:
                    continue
                outputs = [k.value for k in call.keywords if k.arg == "out"]
                if layer in projections or len(outputs) != 1:
                    raise ValueError("Ambiguous query projection")
                output = _root_name(outputs[0])
                projections[layer] = (statement.lineno, output)
                sites.extend(
                    [
                        Site(
                            statement.lineno,
                            "before",
                            f"layer{layer:02d}.q_projection_input",
                            (_root_name(call.args[0]),),
                        ),
                        Site(
                            statement.lineno, "after", f"layer{layer:02d}.q_projection", (output,)
                        ),
                    ]
                )
        if name == "torch.ops.aten._scaled_dot_product_efficient_attention.default":
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                raise ValueError("Unsupported attention assignment")
            layer = len(attention)
            attention.append(statement.lineno)
            for role, argument in zip(
                ("query_rotary", "key_rotary", "value"), call.args[:3], strict=True
            ):
                if not isinstance(argument, ast.Name):
                    raise ValueError("Attention inputs must be direct materialized buffer names")
                sites.append(
                    Site(
                        statement.lineno,
                        "before",
                        f"layer{layer:02d}.{role}",
                        (_root_name(argument),),
                    )
                )
            sites.append(
                Site(
                    statement.lineno,
                    "after",
                    f"layer{layer:02d}.attention_output",
                    (_root_name(statement.targets[0]),),
                    tuple_first=True,
                )
            )
    if len(projections) != layers or len(attention) != layers:
        raise ValueError(
            f"Expected {layers} q projections and attention calls; "
            f"found {len(projections)} and {len(attention)}"
        )
    q_line, q_buffer = projections[probe_layer]
    query_values = {q_buffer}
    stages = []
    for statement in statements:
        if not q_line < statement.lineno < attention[probe_layer]:
            continue
        value = statement.value
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                try:
                    origin = _root_name(value)
                except ValueError:
                    origin = None
                if origin in query_values:
                    query_values.add(target.id)
                else:
                    query_values.discard(target.id)
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
            continue
        if value.func.attr != "run" or not isinstance(value.func.value, ast.Name):
            continue
        kernel = value.func.value.id
        signature = re.search(r"def " + re.escape(kernel) + r"\(([^)]*)\):", source)
        if signature is None:
            raise ValueError("Cannot read original generated kernel signature")
        arguments = [arg.strip().split(":")[0].strip() for arg in signature.group(1).split(",")]
        names = tuple(x.id for x in value.args if isinstance(x, ast.Name))
        scalars = tuple(x.value for x in value.args if isinstance(x, ast.Constant))
        if len(names) + len(scalars) != len(value.args):
            raise ValueError("Unsupported selected kernel launch argument")
        read_names = [
            name
            for argument, name in zip(arguments, names, strict=False)
            if argument.startswith(("in_ptr", "in_out_ptr"))
        ]
        write_names = [
            name
            for argument, name in zip(arguments, names, strict=False)
            if argument.startswith(("out_ptr", "in_out_ptr"))
        ]
        if not query_values.intersection(read_names):
            query_values.difference_update(write_names)
            continue
        query_values.update(write_names)
        index = len(stages)
        stages.append(kernel)
        if statement.end_lineno != statement.lineno:
            raise ValueError("Multiline selected launch unsupported by line observer")
        sites.extend(
            [
                Site(
                    statement.lineno,
                    "before",
                    f"probe_kernel.{index}.inputs",
                    names,
                    kernel=kernel,
                    scalar_args=scalars,
                ),
                Site(
                    statement.lineno,
                    "after",
                    f"probe_kernel.{index}.outputs",
                    names,
                    kernel=kernel,
                    scalar_args=scalars,
                ),
            ]
        )
    query_site = next(
        site for site in sites if site.label == f"layer{probe_layer:02d}.query_rotary"
    )
    if query_site.names[0] not in query_values or not 1 <= len(stages) <= 2:
        raise ValueError("Could not identify one fused or two materialized query-path kernels")

    return sites


def chosen_kernel_metadata(kernel):
    launchers = getattr(kernel, "launchers", None)
    if launchers is None or len(launchers) != 1:
        raise RuntimeError("Kernel must be warmed and have exactly one selected launcher")
    launcher = launchers[0]
    config = getattr(launcher, "config", None)
    binary = getattr(launcher, "bin", None)
    if config is None or binary is None:
        raise RuntimeError("Unsupported Triton launcher metadata; refusing guessed configuration")
    return {
        "config": {**config.kwargs, "num_warps": config.num_warps, "num_stages": config.num_stages},
        "argument_names": list(kernel.fn.arg_names),
        "ptx": binary.asm["ptx"],
        "metadata": binary.metadata._asdict() if hasattr(binary.metadata, "_asdict") else {},
    }


class GeneratedCapture:
    """Line tracing of existing generated Python launches with synchronous CPU copies."""

    def __init__(self, layers, probe_layer, replace_query=None):
        self.replace_query = replace_query
        self.replacement_count = 0
        self.layers = layers
        self.probe_layer = probe_layer
        self.snapshots = {}
        self.pipeline = {}
        self.wrapper_source = None
        self.wrapper_path = None
        self.calls = 0
        self._pending = []
        self._sites = {}
        self._old_trace = None

    def _copy(self, frame, site):
        if site.kernel is not None:
            index = int(site.label.split(".")[1])
            kernel = frame.f_globals[site.kernel]
            if index not in self.pipeline:
                metadata = chosen_kernel_metadata(kernel)
                self.pipeline[index] = {
                    "object": kernel,
                    "metadata": metadata,
                    "scalars": site.scalar_args,
                    "inputs": {},
                    "outputs": {},
                    "buffer_names": dict(zip(metadata["argument_names"], site.names, strict=False)),
                }
            stage = self.pipeline[index]
            argument_names = stage["metadata"]["argument_names"][: len(site.names)]
            target = stage["inputs"] if site.when == "before" else stage["outputs"]
            for argument, name in zip(argument_names, site.names, strict=True):
                value = frame.f_locals[name]
                if site.when == "before" and argument.startswith("out_ptr"):
                    continue
                if site.when == "after" and not argument.startswith(("out_ptr", "in_out_ptr")):
                    continue
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError("Expected only tensor names in selected kernel launch")
                target[argument] = value.detach().cpu().clone()
            return
        value = frame.f_locals[site.names[0]]
        if site.tuple_first:
            value = value[0]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"{site.label} is not a materialized tensor")
        if (
            site.label == f"layer{self.probe_layer:02d}.query_rotary"
            and self.replace_query is not None
        ):
            if value.shape != self.replace_query.shape or value.dtype != self.replace_query.dtype:
                raise RuntimeError("Query replacement shape/dtype mismatch")
            value.copy_(self.replace_query.to(value.device))
            self.replacement_count += 1
        if site.label in self.snapshots:
            raise RuntimeError(f"Generated site executed twice: {site.label}")
        self.snapshots[site.label] = value.detach().cpu().clone()

    def _local(self, frame, event, arg):
        if event in ("line", "return"):
            for site in self._pending:
                self._copy(frame, site)
            self._pending = []
        if event == "line":
            for site in self._sites.get(frame.f_lineno, []):
                if site.when == "before":
                    self._copy(frame, site)
                else:
                    self._pending.append(site)
        return self._local

    def _global(self, frame, event, arg):
        if event != "call" or frame.f_code.co_name != "call":
            return None
        if not all(key in frame.f_globals for key in ("extern_kernels", "assert_size_stride")):
            return None
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("Expected one full generated wrapper execution")
        self.wrapper_path = frame.f_code.co_filename
        self.wrapper_source = Path(self.wrapper_path).read_text()
        for site in plan_sites(self.wrapper_source, self.layers, self.probe_layer):
            self._sites.setdefault(site.line, []).append(site)
        return self._local

    def __enter__(self):
        self._old_trace = sys.gettrace()
        if self._old_trace is not None:
            raise RuntimeError("Refusing to replace an active Python tracer")
        sys.settrace(self._global)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.settrace(self._old_trace)
        if exc_type is None:
            if self.replace_query is not None and self.replacement_count != 1:
                raise RuntimeError("Query replacement did not execute exactly once")
            if self.calls != 1 or len(self.snapshots) != self.layers * 6:
                raise RuntimeError("Generated trace incomplete; no mechanism conclusion is valid")

    def provenance(self):
        return {
            "wrapper_sha256": hashlib.sha256(self.wrapper_source.encode()).hexdigest(),
            "wrapper_basename": Path(self.wrapper_path).name,
            "snapshot_labels": list(self.snapshots),
            "capture_method": "Python wrapper line trace; existing tensor CPU copies; "
            "no Dynamo graph or CUDA kernel edits",
        }
