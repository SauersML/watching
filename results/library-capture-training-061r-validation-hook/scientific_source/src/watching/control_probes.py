"""Storage-materialization and copy-only controls; no model training."""

import ast

import torch

from watching.compile_control import WrapperObserver, kernel_sources
from watching.probe import ProbeModel
from watching.validate_tail import ValidationProbe


@torch.library.custom_op("watching_controls::materialize", mutates_args=())
def materialize(value: torch.Tensor) -> torch.Tensor:
    """Opaque, bit-preserving storage boundary, deliberately distinct from a read."""
    return value.clone()


@materialize.register_fake
def _materialize_fake(value):
    return torch.empty_like(value)


class StorageControlProbe(ValidationProbe):
    def record(self, module, inputs, output):
        if self.mode in ("identity", "identity_reference"):
            stored = materialize(output)
            if self.mode == "identity_reference":
                self.captured.append(stored)
            return stored
        return super().record(module, inputs, output)

    def forward(self, input_ids, attention_mask, position_ids, capture=None, mode=None):
        if mode is None:
            mode = "reference" if capture else "none"
        elif capture is not None:
            raise ValueError("Specify capture or mode, not both")
        if mode not in ("none", "reference", "buffer", "identity", "identity_reference"):
            raise ValueError("Unknown storage control")
        self.mode = mode
        return ProbeModel.forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            mode in ("reference", "identity_reference"),
        )


def entry_buffer(wrapper, input_ids, attention_mask, position_ids):
    return wrapper(input_ids, attention_mask, position_ids, mode="buffer")


def entry_identity(wrapper, input_ids, attention_mask, position_ids):
    return wrapper(input_ids, attention_mask, position_ids, mode="identity")


def entry_identity_reference(wrapper, input_ids, attention_mask, position_ids):
    return wrapper(input_ids, attention_mask, position_ids, mode="identity_reference")


EXTRA_ENTRIES = {
    "buffer": entry_buffer,
    "identity": entry_identity,
    "identity_reference": entry_identity_reference,
}


def buffer_write_sites(source):
    """Map existing Triton stores to wrapper arguments; never alter the graph."""
    writes = {}
    for name, kernel_source in kernel_sources(source).items():
        tree = ast.parse(kernel_source)
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if len(funcs) != 1:
            raise ValueError("Unsupported generated kernel declaration")
        signature = [arg.arg for arg in funcs[0].args.args]
        pointers = set()
        for node in ast.walk(funcs[0]):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "tl.store":
                pointers.update(
                    n.id
                    for n in ast.walk(node.args[0])
                    if isinstance(n, ast.Name) and n.id in signature
                )
        writes[name] = [signature.index(pointer) for pointer in sorted(pointers)]
    sites = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Assign, ast.Expr)) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "run":
            continue
        name = ast.unparse(call.func.value)
        if name not in writes:
            continue
        arguments = [call.args[index] for index in writes[name]]
        if any(not isinstance(arg, ast.Name) for arg in arguments):
            raise ValueError("Unsupported generated output expression")
        sites[node.lineno] = (name, [arg.id for arg in arguments])
    return sites


class BufferWriterObserver(WrapperObserver):
    """Verify a native generated launch writes the actual preallocated buffer."""

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.sites = {}
        self.writers = []

    def _trace(self, frame, event, arg):
        super()._trace(frame, event, arg)
        if self.code is frame.f_code and event == "call":
            self.sites = buffer_write_sites(self.source)
            return self._local
        return None

    def _local(self, frame, event, arg):
        if event == "line" and frame.f_lineno in self.sites:
            kernel, names = self.sites[frame.f_lineno]
            for name in names:
                value = frame.f_locals[name]
                if isinstance(value, torch.Tensor) and value.data_ptr() == self.target.data_ptr():
                    self.writers.append(
                        {"kernel": kernel, "argument": name, "line": frame.f_lineno}
                    )
        return self._local
