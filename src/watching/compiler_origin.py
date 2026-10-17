"""Observe early native launches without changing the compiled graph.

This is a deliberately narrow adapter for the saved Qwen3 Inductor wrappers.
Large immutable parameters are hashed, not duplicated in every activation archive.
Replay uses the actual warmed launcher objects and fails on unsupported aliases.
"""

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from watching.artifacts import write_json
from watching.execution_fit import tensor_digest
from watching.generated_trace import GeneratedCapture, chosen_kernel_metadata, plan_sites
from watching.mechanism_trace import compare_captures, save_capture, save_tensor
from watching.tail_train import difference


class NativeParityError(RuntimeError):
    """A mandatory numerical gate failed; never downgrade this to missing metadata."""


@dataclass(frozen=True)
class Operation:
    index: int
    line: int
    kind: str
    kernel: str | None
    inputs: tuple[tuple[str, ast.AST], ...]
    outputs: tuple[tuple[str, ast.AST], ...]
    scalars: tuple


def tensor_expression(node, namespace):
    """Evaluate only names, literal tensor views, and tuple-result indexing."""
    if isinstance(node, ast.Name):
        return namespace[node.id]
    if isinstance(node, ast.Subscript):
        index = ast.literal_eval(node.slice)
        if not isinstance(index, int):
            raise ValueError("Only integer tuple indexing is supported")
        return tensor_expression(node.value, namespace)[index]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "reinterpret_tensor" and len(node.args) == 4 and not node.keywords:
            tensor = tensor_expression(node.args[0], namespace)
            shape, strides, offset = (ast.literal_eval(arg) for arg in node.args[1:])
            return tensor.as_strided(shape, strides, tensor.storage_offset() + offset)
    raise ValueError(f"Unsupported materialized operand: {ast.unparse(node)}")


def plan_early_operations(source, layers, blocks=3):
    """Include all tensor launches before the next block's query projection."""
    sites = plan_sites(source, layers, 1)
    stop = next(
        site.line for site in sites if site.label == f"layer{blocks:02d}.q_projection_input"
    )
    tree = ast.parse(source)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "call"]
    if len(functions) != 1:
        raise ValueError("Expected one generated wrapper")
    operations = []
    for node in sorted(ast.walk(functions[0]), key=lambda n: getattr(n, "lineno", 0)):
        if not isinstance(node, (ast.Assign, ast.Expr)) or node.lineno >= stop:
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        name = ast.unparse(call.func)
        kernel, scalars = None, ()
        inputs, outputs = [], []
        if name.endswith(".run") and isinstance(call.func.value, ast.Name):
            kernel = call.func.value.id
            signature = _kernel_argument_names(source, kernel)
            if any(not isinstance(arg, (ast.Name, ast.Constant)) for arg in call.args):
                raise ValueError("Unsupported native Triton launch expression")
            for argument, value in zip(signature, call.args, strict=False):
                if isinstance(value, ast.Constant):
                    continue
                if argument.startswith(("in_ptr", "in_out_ptr")):
                    inputs.append((argument, value))
                if argument.startswith(("out_ptr", "in_out_ptr")):
                    outputs.append((argument, value))
                if not argument.startswith(("in_ptr", "out_ptr", "in_out_ptr")):
                    raise ValueError("Unsupported tensor pointer argument")
            scalars = tuple(
                ast.literal_eval(arg) for arg in call.args if isinstance(arg, ast.Constant)
            )
            kind = "triton"
        elif name == "extern_kernels.mm":
            if len(call.args) != 2 or len(call.keywords) != 1 or call.keywords[0].arg != "out":
                raise ValueError("Unsupported external matrix multiplication signature")
            inputs = [("left", call.args[0]), ("right", call.args[1])]
            outputs = [("output", call.keywords[0].value)]
            kind = "mm"
        elif name == "torch.ops.aten._scaled_dot_product_efficient_attention.default":
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                raise ValueError("Unsupported attention assignment")
            inputs = list(zip(("query", "key", "value", "mask"), call.args[:4], strict=True))
            outputs = [("output", ast.Subscript(value=node.targets[0], slice=ast.Constant(0)))]
            kind = "attention"
        else:
            continue
        if node.lineno != node.end_lineno or not inputs or not outputs:
            raise ValueError("Unsupported multiline or incomplete native launch")
        operations.append(
            Operation(
                len(operations), node.lineno, kind, kernel, tuple(inputs), tuple(outputs), scalars
            )
        )
    if not operations or sum(op.kind == "attention" for op in operations) != blocks:
        raise ValueError("Incomplete early-operation coverage")
    return operations


def _kernel_argument_names(source, kernel):
    import re

    match = re.search(r"def " + re.escape(kernel) + r"\(([^)]*)\):", source)
    if match is None:
        raise ValueError(f"Missing kernel signature: {kernel}")
    return [arg.strip().split(":")[0].strip() for arg in match.group(1).split(",")]


def tensor_layout(value):
    """Record address residues, never literal process addresses."""
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "storage_offset": value.storage_offset(),
        "alignment_mod_bytes": {str(n): value.data_ptr() % n for n in (16, 32, 128, 256, 4096)},
    }


class ParameterHashes:
    """Cache immutable parameter hashes while retaining identity to prevent pointer reuse."""

    def __init__(self):
        self.values = {}

    def digest(self, value):
        key = (
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value._version,
        )
        if key not in self.values:
            self.values[key] = (value, tensor_digest(value))
        return self.values[key][1]


class EarlyCapture(GeneratedCapture):
    def __init__(self, layers, parameter_hashes=None, replace_query=None):
        super().__init__(layers, 1, replace_query=replace_query)
        self.parameter_hashes = parameter_hashes or ParameterHashes()
        self.operations = []
        self.records = {}
        self.tensors = {}
        self.large_inputs = {}
        self.kernel_objects = {}
        self._operation_sites = {}
        self._pending_operations = []

    def _global(self, frame, event, arg):
        result = super()._global(frame, event, arg)
        if result is not None:
            self.operations = plan_early_operations(self.wrapper_source, self.layers)
            self._operation_sites = {op.line: op for op in self.operations}
            from watching.compile_control import kernel_sources

            # Retain every warmed generated kernel, including shared kernels whose
            # first use is later than the observed three-block prefix.
            self.kernel_objects = {
                name: frame.f_globals[name] for name in kernel_sources(self.wrapper_source)
            }
        return result

    def _observe_operation(self, frame, operation, when):
        record = self.records.setdefault(
            operation.index,
            {
                "index": operation.index,
                "line": operation.line,
                "kind": operation.kind,
                "kernel": operation.kernel,
                "scalars": list(operation.scalars),
                "inputs": {},
                "outputs": {},
            },
        )
        expressions = operation.inputs if when == "inputs" else operation.outputs
        observed = self.tensors.setdefault(operation.index, {"inputs": {}, "outputs": {}})[when]
        values = [(name, tensor_expression(expr, frame.f_locals)) for name, expr in expressions]
        if any(not isinstance(value, torch.Tensor) for _, value in values):
            raise RuntimeError("Expected existing native tensor operand")
        for name, value in values:
            # Only immutable source arguments can be omitted from activation storage.
            # Derived buffers, including large ones, must be saved before reuse.
            expr = dict(expressions)[name]
            root = expr.args[0] if isinstance(expr, ast.Call) else expr
            large_parameter = (
                when == "inputs"
                and isinstance(root, ast.Name)
                and root.id.startswith("arg")
                and value.numel() > 2_000_000
            )
            if large_parameter:
                digest = self.parameter_hashes.digest(value)
                self.large_inputs[(operation.index, name)] = value
            else:
                saved = value.detach().cpu().clone()
                digest = tensor_digest(saved)
                observed[name] = saved
            record[when][name] = {
                **tensor_layout(value),
                "sha256": digest,
                "saved_tensor": not large_parameter,
                "aliases_inputs": [
                    input_name
                    for input_name, input_expr in operation.inputs
                    if when == "outputs"
                    and tensor_expression(input_expr, frame.f_locals).untyped_storage().data_ptr()
                    == value.untyped_storage().data_ptr()
                ],
                "aliases_same_phase": [
                    other
                    for other, tensor in values
                    if other != name
                    and tensor.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
                ],
            }

    def _local(self, frame, event, arg):
        if event in ("line", "return"):
            for operation in self._pending_operations:
                self._observe_operation(frame, operation, "outputs")
            self._pending_operations = []
        # Base capture preserves every original all-layer boundary and q1 pipeline.
        super()._local(frame, event, arg)
        if event == "line" and frame.f_lineno in self._operation_sites:
            operation = self._operation_sites[frame.f_lineno]
            self._observe_operation(frame, operation, "inputs")
            self._pending_operations.append(operation)
        return self._local

    def __exit__(self, exc_type, exc, tb):
        super().__exit__(exc_type, exc, tb)
        if exc_type is None and (
            len(self.records) != len(self.operations)
            or any(not row["outputs"] for row in self.records.values())
        ):
            raise RuntimeError("Early native launch capture incomplete")


def capture_early(call, batch, layers, parameter_hashes=None):
    with torch.no_grad():
        before = call(batch)[0].detach().clone()
        if not torch.equal(before, call(batch)[0]):
            raise RuntimeError("Ordinary native execution is not repeatable")
        with EarlyCapture(layers, parameter_hashes) as captured:
            traced = call(batch)[0].detach().clone()
        after = call(batch)[0]
        if not torch.equal(before, traced) or not torch.equal(before, after):
            raise RuntimeError("Passive early-operation observation changes full-vocabulary logits")
    return before.cpu(), captured


def save_early(out, captured, logits):
    out.mkdir(parents=True, exist_ok=True)
    # Preserve numerical evidence even if optional launcher metadata is unsupported.
    save_tensor(out / "early_tensors.pt", captured.tensors)
    write_json(out / "early_operations.json", list(captured.records.values()))
    save_capture(out, captured, logits)
    metadata = {}
    for name, kernel in captured.kernel_objects.items():
        try:
            item = chosen_kernel_metadata(kernel)
            ptx = item.pop("ptx")
            if ptx is not None:
                digest = hashlib.sha256(ptx.encode()).hexdigest()
                (out / f"kernel-{digest}.ptx").write_text(ptx)
                item["ptx_sha256"] = digest
            try:
                cubin = selected_cubin(kernel)
                digest = hashlib.sha256(cubin).hexdigest()
                (out / f"kernel-{digest}.cubin").write_bytes(cubin)
                item["cubin_sha256"] = digest
            except (RuntimeError, OSError, ValueError, AttributeError, TypeError) as error:
                item["cubin_status"] = "unavailable"
                item["cubin_error"] = str(error)
            metadata[name] = {"status": "recorded", **item}
        except (RuntimeError, TypeError, ValueError, AttributeError, OSError) as error:
            metadata[name] = {"status": "unsupported", "reason": str(error)}
    write_json(out / "early_kernels.json", metadata)


def selected_cubin(kernel):
    """Recover only the selected binary; missing provenance never guesses a candidate."""
    from torch._inductor.runtime.runtime_utils import triton_hash_to_path_key

    launchers = getattr(kernel, "launchers", ())
    if len(launchers) != 1:
        raise RuntimeError("Expected one selected launcher")
    launcher = launchers[0]
    binary = getattr(launcher, "bin", None)
    if binary is None:
        matches = [
            item.kernel
            for item in getattr(kernel, "compile_results", ())
            if item.config == launcher.config
            and triton_hash_to_path_key(item.kernel.hash) == launcher.cache_hash
        ]
        if len(matches) != 1:
            raise RuntimeError("Selected binary does not have a unique matching cached object")
        binary = matches[0]
    if hasattr(binary, "asm") and isinstance(binary.asm.get("cubin"), bytes):
        return binary.asm["cubin"]
    import json

    from triton.runtime.cache import get_cache_manager

    cache = get_cache_manager(binary.hash)
    stem = binary.name[:150]
    metadata_path, cubin_path = cache.get_file(stem + ".json"), cache.get_file(stem + ".cubin")
    if not metadata_path or not cubin_path:
        raise RuntimeError("Exact selected binary cache entry is unavailable")
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata.get("hash") != binary.hash or metadata.get("name") != binary.name:
        raise RuntimeError("Selected cubin metadata provenance mismatch")
    return Path(cubin_path).read_bytes()


def operand_difference(actual, expected):
    """Compare native masks with signed infinities; NaNs remain invalid evidence."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Native operand shapes and dtypes must match")
    if torch.isnan(actual).any() or torch.isnan(expected).any():
        raise ValueError("NaN in native operand; numerical evidence is invalid")
    if torch.isfinite(actual).all() and torch.isfinite(expected).all():
        return difference(actual, expected)
    equal = actual == expected
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    nonfinite_mismatches = (~equal & ~finite).sum().item()
    delta = torch.zeros_like(actual, dtype=torch.float32)
    delta[finite] = actual[finite].float() - expected[finite].float()
    return {
        "bit_identical": torch.equal(actual, expected),
        "fraction_different": float((~equal).float().mean()),
        "max_abs_difference": None if nonfinite_mismatches else float(delta.abs().max()),
        "mean_abs_difference": None if nonfinite_mismatches else float(delta.abs().mean()),
        "nonfinite_value_count": int((~finite).sum()),
        "nonfinite_mismatch_count": int(nonfinite_mismatches),
        "finite_max_abs_difference": float(delta.abs().max()),
    }


def compare_early(left, right):
    if len(left.operations) != len(right.operations):
        raise RuntimeError("Different early launch counts; ordinal matching is invalid")
    rows = []
    for a, b in zip(left.operations, right.operations, strict=True):
        if a.kind != b.kind or a.kernel != b.kernel:
            raise RuntimeError("Different early launch order; cannot match operations by ordinal")
        la, rb = left.records[a.index], right.records[b.index]
        if set(la["inputs"]) != set(rb["inputs"]) or set(la["outputs"]) != set(rb["outputs"]):
            raise RuntimeError("Native operand signatures differ")
        inputs_equal = all(
            la["inputs"][name]["sha256"] == rb["inputs"][name]["sha256"] for name in la["inputs"]
        )
        outputs = {
            name: operand_difference(
                left.tensors[a.index]["outputs"][name], right.tensors[b.index]["outputs"][name]
            )
            for name in la["outputs"]
        }
        rows.append(
            {
                "index": a.index,
                "kind": a.kind,
                "kernel": a.kernel,
                "all_input_values_identical": inputs_equal,
                "all_output_values_identical": all(
                    item["bit_identical"] for item in outputs.values()
                ),
                "outputs": outputs,
            }
        )
    return {
        "first_unequal_operation": next(
            (row for row in rows if not row["all_output_values_identical"]), None
        ),
        "operations": rows,
    }


def _allocate_operand(layout, dtype, device):
    value = torch.empty_strided(layout["shape"], layout["stride"], dtype=dtype, device=device)
    if torch._debug_has_internal_overlap(value) != 0:
        raise RuntimeError("Replay of overlapping tensor layouts is unsupported")
    return value


def replay_common_operands(left, right, index):
    """Cross native implementations on one shared set of physical input/output buffers.

    A/B/B/A ordering checks repeatability. Every input is restored before each
    launch, including in-place inputs. Own-input reproduction is a mandatory gate.
    This preserves logical strides; changed storage offsets/alignment are reported.
    """
    operations = [capture.operations[index] for capture in (left, right)]
    a, b = operations
    if a.kind != "triton" or b.kind != "triton":
        raise RuntimeError("Matched physical replay currently supports Triton launches only")
    if a.scalars != b.scalars:
        raise RuntimeError("Native launch scalar arguments differ")
    stages = [capture.records[index] for capture in (left, right)]
    kernels = [
        capture.kernel_objects[operation.kernel]
        for capture, operation in zip((left, right), operations, strict=True)
    ]
    signatures = [list(kernel.fn.arg_names) for kernel in kernels]
    names = [name for name in signatures[0] if name.startswith(("in_ptr", "out_ptr", "in_out_ptr"))]
    if signatures[0] != signatures[1] or not names:
        raise RuntimeError("Selected native kernel signatures differ")
    for phase in ("inputs", "outputs"):
        if set(stages[0][phase]) != set(stages[1][phase]):
            raise RuntimeError("Replay pointer sets differ")
        for name, layout in stages[0][phase].items():
            other = stages[1][phase][name]
            if any(layout[key] != other[key] for key in ("shape", "stride", "dtype")):
                raise RuntimeError("Common-physical replay requires matching logical layouts")
            if layout["aliases_same_phase"] or other["aliases_same_phase"]:
                raise RuntimeError("Cross-pointer aliases are unsupported by replay")
            if phase == "outputs":
                for observed in (layout, other):
                    aliases = observed.get("aliases_inputs")
                    expected = [name] if name.startswith("in_out_ptr") else []
                    if aliases != expected:
                        raise RuntimeError("Cross-phase aliases are unsupported by replay")
    workspace = {}
    immutable_parameters = set()
    for name in names:
        phase = "inputs" if name in stages[0]["inputs"] else "outputs"
        template = left.tensors[index][phase].get(name)
        if template is None:
            if phase != "inputs" or not name.startswith("in_ptr"):
                raise RuntimeError("Only immutable read-only parameters may reuse original storage")
            if stages[0][phase][name]["sha256"] != stages[1][phase][name]["sha256"]:
                raise RuntimeError("Large parameter values differ between original executions")
            value = left.large_inputs[(index, name)]
            if value.device.type != "cuda":
                raise RuntimeError("Original shared parameter is not resident on CUDA")
            workspace[name] = value
            immutable_parameters.add(name)
        else:
            workspace[name] = _allocate_operand(stages[0][phase][name], template.dtype, "cuda")
    physical_layout = {name: tensor_layout(value) for name, value in workspace.items()}
    comparisons = []
    native_outputs = {}
    for origin, capture in enumerate((left, right)):
        results = []
        for implementation in (0, 1, 1, 0):
            for name, value in workspace.items():
                if name in immutable_parameters:
                    continue
                if name in stages[origin]["inputs"]:
                    value.copy_(capture.tensors[index]["inputs"][name].to(value.device))
                else:
                    value.fill_(float("nan"))
            kernels[implementation].run(
                *[workspace[name] for name in names],
                *a.scalars,
                stream=torch.cuda.current_stream().cuda_stream,
            )
            torch.cuda.synchronize()
            outputs = {name: workspace[name].cpu().clone() for name in stages[origin]["outputs"]}
            if not all(torch.isfinite(value).all() for value in outputs.values()):
                raise RuntimeError("Native replay left nonfinite or unwritten outputs")
            results.append(outputs)
        if any(
            not torch.equal(results[first][name], results[last][name])
            for first, last in ((0, 3), (1, 2))
            for name in results[first]
        ):
            raise RuntimeError("Alternating selected launch replay is not repeatable")
        own = results[origin]
        if any(
            not torch.equal(value, capture.tensors[index]["outputs"][name])
            for name, value in own.items()
        ):
            raise RuntimeError("Own-input replay does not reproduce original native output")
        comparisons.append(
            {
                "input_origin": "left" if origin == 0 else "right",
                "outputs": {
                    name: difference(results[0][name], results[1][name]) for name in results[0]
                },
            }
        )
        native_outputs[origin] = results[:2]
    return {
        "operation_index": index,
        "own_input_replay_bit_identical": True,
        "alternating_order": ["left", "right", "right", "left"],
        "shared_physical_inputs_and_outputs": True,
        "shared_workspace_layout": physical_layout,
        "immutable_shared_parameter_arguments": sorted(immutable_parameters),
        "native_kernel_object_shared": kernels[0] is kernels[1],
        "comparisons": comparisons,
    }, native_outputs


def _query_swap(call, batch, layers, donor, expected, out):
    with torch.no_grad():
        with GeneratedCapture(layers, 1, replace_query=donor):
            result = call(batch)[0].detach().cpu().clone()
    save_tensor(out, result)
    return difference(result, expected)


def run_origin_audit(arms, batches, layers, out, emit):
    """Optional detailed phase after the independent-compilation control is saved.

    Arm callables accept the exact same input dict and return (logits, captures).
    Never rebuild a graph, change a hook, or choose prompts from evaluation outcomes.
    """
    out = Path(out)
    hashes = ParameterHashes()
    reports = []
    for prompt_index, batch in enumerate(batches):
        captured, logits = {}, {}
        folder = out / f"prompt-{prompt_index:02d}"
        for name, call in arms.items():
            logits[name], captured[name] = capture_early(call, batch, layers, hashes)
            save_early(folder / name, captured[name], logits[name])
            emit(
                "origin_capture_saved",
                prompt_index=prompt_index,
                arm=name,
                native_operations=len(captured[name].operations),
                full_vocab_parity=True,
            )
        comparisons = {}
        for left_name, right_name in (("none_a", "none_b"), ("none_a", "reference")):
            if left_name not in arms or right_name not in arms:
                raise ValueError("Expected none_a, none_b, reference arm names")
            left, right = captured[left_name], captured[right_name]
            name = f"{left_name}_vs_{right_name}"
            summary = {
                "early": compare_early(left, right),
                "boundaries": compare_captures(left, right),
                "logits": difference(logits[left_name], logits[right_name]),
            }
            write_json(folder / f"{name}.json", summary)
            first = summary["early"]["first_unequal_operation"]
            # Always audit the first-block1 fused query launch as prespecified,
            # plus an earlier observed mismatch if it is a different operation.
            q_kernel = left.pipeline[0]["object"]
            q_index = next(
                op.index
                for op in left.operations
                if op.kernel and left.kernel_objects[op.kernel] is q_kernel
            )
            indices = list(dict.fromkeys(([first["index"]] if first else []) + [q_index]))
            replays = []
            for index in indices:
                try:
                    replay, values = replay_common_operands(left, right, index)
                    save_tensor(folder / f"{name}.op{index:03d}.replay.pt", values)
                    replays.append({"status": "verified", **replay})
                except (RuntimeError, ValueError, KeyError, AttributeError) as error:
                    replays.append(
                        {
                            "status": "failed_or_unsupported",
                            "operation_index": index,
                            "reason": str(error),
                        }
                    )
            summary["selected_replays"] = replays
            substitutions = []
            for index in indices:
                try:
                    for recipient_name, donor_name in (
                        (left_name, right_name),
                        (right_name, left_name),
                    ):
                        recipient = captured[recipient_name]
                        identity = kernel_intervention(
                            arms[recipient_name],
                            batch,
                            recipient,
                            recipient,
                            index,
                            logits[recipient_name],
                            folder / f"{name}.op{index:03d}.{recipient_name}.kernel_identity.pt",
                        )
                        if not identity["difference_from_donor_logits"]["bit_identical"]:
                            raise NativeParityError(
                                "Identity kernel substitution changes native logits"
                            )
                        changed = kernel_intervention(
                            arms[recipient_name],
                            batch,
                            recipient,
                            captured[donor_name],
                            index,
                            logits[donor_name],
                            folder / f"{name}.op{index:03d}.{recipient_name}.kernel_donor.pt",
                        )
                        with torch.no_grad():
                            restored = arms[recipient_name](batch)[0].detach().cpu()
                        if not torch.equal(restored, logits[recipient_name]):
                            raise NativeParityError("Kernel substitution failed native restoration")
                        substitutions.append(
                            {
                                "recipient": recipient_name,
                                "donor": donor_name,
                                "status": "executed_identity_and_restoration_verified",
                                **changed,
                            }
                        )
                except (RuntimeError, ValueError, KeyError, AttributeError) as error:
                    if isinstance(error, NativeParityError):
                        raise
                    substitutions.append(
                        {
                            "operation_index": index,
                            "status": "failed_or_unsupported",
                            "reason": str(error),
                        }
                    )
            summary["kernel_substitutions"] = substitutions
            query_key = "layer01.query_rotary"
            swaps = {}
            for recipient, donor_name in ((left_name, right_name), (right_name, left_name)):
                baseline = logits[recipient]
                own_query = captured[recipient].snapshots[query_key]
                identity = _query_swap(
                    arms[recipient],
                    batch,
                    layers,
                    own_query,
                    baseline,
                    folder / f"{name}.{recipient}.identity.pt",
                )
                if not identity["bit_identical"]:
                    raise RuntimeError(
                        "Identity query replacement changes native full-vocabulary logits"
                    )
                swaps[recipient] = _query_swap(
                    arms[recipient],
                    batch,
                    layers,
                    captured[donor_name].snapshots[query_key],
                    logits[donor_name],
                    folder / f"{name}.{recipient}.donor_query.pt",
                )
                with torch.no_grad():
                    restored = arms[recipient](batch)[0].detach().cpu()
                if not torch.equal(restored, baseline):
                    raise RuntimeError("Native output failed restoration after query replacement")
            summary["query_swap_difference_from_donor_logits"] = swaps
            summary["identity_and_restoration_bit_identical"] = True
            comparisons[name] = summary
            write_json(folder / f"{name}.json", summary)
            emit(
                "origin_pair_complete",
                prompt_index=prompt_index,
                comparison=name,
                first_unequal_operation=first,
                replays=[
                    {k: v for k, v in item.items() if k != "shared_workspace_layout"}
                    for item in replays
                ],
            )
        reports.append({"prompt_index": prompt_index, "comparisons": comparisons})
        write_json(out / "summary.json", reports)
    return reports


class KernelSwapCapture(GeneratedCapture):
    """Replace exactly one existing launch's kernel object, then restore its global."""

    def __init__(self, layers, wrapper_sha256, operation, donor):
        super().__init__(layers, 1)
        self.expected_wrapper_sha256 = wrapper_sha256
        self.operation = operation
        self.donor = donor
        self.swap_count = 0
        self._restore = None

    def _global(self, frame, event, arg):
        result = super()._global(frame, event, arg)
        if (
            result is not None
            and hashlib.sha256(self.wrapper_source.encode()).hexdigest()
            != self.expected_wrapper_sha256
        ):
            raise RuntimeError("Kernel intervention encountered a different compiled wrapper")
        return result

    def _restore_global(self):
        if self._restore is not None:
            namespace, name, original = self._restore
            namespace[name] = original
            self._restore = None

    def _local(self, frame, event, arg):
        if event in ("line", "return", "exception"):
            self._restore_global()
        if event == "line" and frame.f_lineno == self.operation.line:
            name = self.operation.kernel
            self._restore = (frame.f_globals, name, frame.f_globals[name])
            frame.f_globals[name] = self.donor
            self.swap_count += 1
        return super()._local(frame, event, arg)

    def __exit__(self, exc_type, exc, tb):
        self._restore_global()
        super().__exit__(exc_type, exc, tb)
        if exc_type is None and self.swap_count != 1:
            raise RuntimeError("Kernel substitution did not execute exactly once")


def kernel_intervention(call, batch, recipient, donor, index, expected, out):
    """Use a donor's selected implementation at one source-identical native launch."""
    from watching.compile_control import kernel_sources

    operation, other = recipient.operations[index], donor.operations[index]
    if operation.kind != "triton" or other.kind != "triton":
        raise RuntimeError("Only native Triton kernel substitution is supported")
    sources = [kernel_sources(capture.wrapper_source) for capture in (recipient, donor)]
    if sources[0][operation.kernel] != sources[1][other.kernel]:
        raise RuntimeError("Kernel substitution requires identical full Triton source")
    if operation.scalars != other.scalars:
        raise RuntimeError("Kernel substitution scalar signatures differ")
    native = recipient.kernel_objects[operation.kernel]
    replacement = donor.kernel_objects[other.kernel]
    if list(native.fn.arg_names) != list(replacement.fn.arg_names):
        raise RuntimeError("Kernel substitution pointer signatures differ")
    wrapper_sha = hashlib.sha256(recipient.wrapper_source.encode()).hexdigest()
    with (
        torch.no_grad(),
        KernelSwapCapture(recipient.layers, wrapper_sha, operation, replacement) as traced,
    ):
        logits = call(batch)[0].detach().cpu().clone()
    save_tensor(out, logits)
    return {
        "operation_index": index,
        "same_kernel_object": native is replacement,
        "difference_from_donor_logits": difference(logits, expected),
        "boundaries_relative_to_recipient": compare_captures(recipient, traced),
    }
