"""Outside-graph runtime argument evidence for a separately named diagnostic.

Hashing synchronizes/copies tensors and changes allocation/timing. This is an
instrumented runtime, not evidence that an untouched execution follows it.
"""

import hashlib
import json
from pathlib import Path

import torch


def tensor_record(value):
    """Include value, layout, alias identity and version; no tensor is modified."""
    cpu = value.detach().cpu().contiguous().clone()
    return {
        "sha256": hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "storage_pointer": value.untyped_storage().data_ptr(),
        "data_pointer": value.data_ptr(),
        "storage_bytes": value.untyped_storage().nbytes(),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": value.requires_grad,
        "version": None if torch.is_inference(value) else value._version,
        "inference_tensor": torch.is_inference(value),
    }


def tensor_set(values):
    records, aliases = {}, {}
    for name, value in values:
        record = tensor_record(value)
        key = (record["device"], record["storage_pointer"])
        record["alias_group"] = aliases.setdefault(key, len(aliases))
        records[name] = record
    return records


def model_record(model):
    """Include nonpersistent buffers omitted by state_dict and duplicate aliases."""
    return {
        "tensors": tensor_set(
            [
                ("parameter:" + name, value)
                for name, value in model.named_parameters(remove_duplicate=False)
            ]
            + [
                ("buffer:" + name, value)
                for name, value in model.named_buffers(remove_duplicate=False)
            ]
        ),
        "training": {name: module.training for name, module in model.named_modules()},
    }


def argument_records(args):
    tensors = [(str(i), value) for i, value in enumerate(args) if isinstance(value, torch.Tensor)]
    unsupported = [
        str(i)
        for i, value in enumerate(args)
        if not isinstance(value, (torch.Tensor, type(None), bool, int, float, str))
    ]
    return {
        "coverage_complete": not unsupported,
        "unsupported_argument_indices": unsupported,
        "tensors": tensor_set(tensors),
        "other": {
            str(i): {"type": type(value).__qualname__, "repr": repr(value)}
            for i, value in enumerate(args)
            if not isinstance(value, torch.Tensor)
        },
    }


class RuntimeArgumentProbe:
    """Wrap a compiler backend and fingerprint its real runtime arguments.

    Backend example inputs can be fake and are never value-hashed. The wrapper
    executes outside the captured graph; it preserves unboxed/boxed delegation.
    Construct a new probe per independent compilation and require one graph.
    """

    def __init__(self, directory, delegate):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.delegate = delegate
        self.graphs = 0
        self.calls = 0

    def __call__(self, graph, example_inputs):
        graph_id = self.graphs
        self.graphs += 1
        (self.directory / f"graph-{graph_id}.py").write_text(graph.code)
        placeholders = [
            {"name": node.name, "target": str(node.target)}
            for node in graph.graph.nodes
            if node.op == "placeholder"
        ]
        (self.directory / f"graph-{graph_id}-placeholders.json").write_text(
            json.dumps(placeholders, indent=2) + "\n"
        )
        constants = {}
        for node in graph.graph.nodes:
            if node.op != "get_attr":
                continue
            value = graph
            for part in str(node.target).split("."):
                value = getattr(value, part)
            if isinstance(value, torch.Tensor):
                from torch._subclasses.fake_tensor import is_fake

                constants[str(node.target)] = (
                    {"unverified_fake_tensor": True} if is_fake(value) else tensor_record(value)
                )
            else:
                constants[str(node.target)] = {
                    "type": type(value).__qualname__,
                    "repr": repr(value),
                    "unverified_nonprimitive": not isinstance(
                        value, (type(None), bool, int, float, str, torch.dtype, torch.device)
                    ),
                }
        (self.directory / f"graph-{graph_id}-constants.json").write_text(
            json.dumps(constants, indent=2) + "\n"
        )
        compiled = self.delegate(graph, example_inputs)

        def execute(*args):
            call = self.calls
            self.calls += 1
            # Keep the original references: boxed delegates may clear their list.
            retained = tuple(args)
            record = {"graph": graph_id, "call": call, "before": argument_records(retained)}
            try:
                result = (
                    compiled(list(args))
                    if getattr(compiled, "_boxed_call", False)
                    else compiled(*args)
                )
                output = result if isinstance(result, (tuple, list)) else (result,)
                record["outputs"] = argument_records(output)
                return result
            finally:
                record["after"] = argument_records(retained)
                (self.directory / f"call-{call}.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )

        return execute
