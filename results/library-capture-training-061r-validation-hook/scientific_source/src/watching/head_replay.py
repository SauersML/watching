"""Parity-gated replay of an existing output head on fixed native inputs."""

import math
from dataclasses import dataclass

import torch


class HeadForward(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, hidden):
        return self.head(hidden)[:, -1, :]


@dataclass
class HeadInput:
    values: torch.Tensor
    stride: tuple[int, ...]
    storage_offset: int
    gradient_enabled: bool

    def restore(self, device):
        """Retain the native view layout, including its offset in backing storage."""
        size = (
            self.storage_offset
            + 1
            + sum(
                (length - 1) * stride
                for length, stride in zip(self.values.shape, self.stride, strict=True)
            )
        )
        backing = torch.empty(size, device=device, dtype=self.values.dtype)
        hidden = backing.as_strided(self.values.shape, self.stride, self.storage_offset)
        hidden.copy_(self.values.to(device))
        return hidden.requires_grad_(self.gradient_enabled)


class GatedHeadReplay:
    """Cache only inputs whose replay matches every native vocabulary logit.

    Baseline parity is necessary, not a guarantee after weight changes. Callers
    must verify the selected weights through full native execution before saving.
    """

    def __init__(self, head, *, compile_head=True):
        self.device = head.weight.device
        forward = HeadForward(head)
        self.forward = (
            torch.compile(forward, fullgraph=True, dynamic=False) if compile_head else forward
        )
        self.inputs = {}

    def replay(self, key):
        record = self.inputs[key]
        with torch.no_grad():
            hidden = record.restore(self.device)
        with torch.set_grad_enabled(record.gradient_enabled):
            return self.forward(hidden).detach()

    def add(self, key, hidden, native_logits, *, gradient_enabled):
        if key in self.inputs:
            raise ValueError("Native input was already cached")
        if hidden.ndim != 3 or hidden.shape[:2] != (1, 1):
            raise ValueError("Cached head replay requires a native [1, 1, hidden] input")
        self.inputs[key] = HeadInput(
            hidden.detach().cpu().contiguous().clone(),
            tuple(hidden.stride()),
            hidden.storage_offset(),
            gradient_enabled,
        )
        try:
            self.check(key, hidden, native_logits)
        except Exception:
            del self.inputs[key]
            raise

    def check(self, key, hidden, native_logits):
        record = self.inputs[key]
        if (
            hidden.dtype != record.values.dtype
            or tuple(hidden.stride()) != record.stride
            or hidden.storage_offset() != record.storage_offset
            or not torch.equal(hidden.detach().cpu(), record.values)
        ):
            raise RuntimeError("Native output-head input changed after caching")
        replayed = self.replay(key)
        if not torch.equal(replayed, native_logits):
            delta = float((replayed.float() - native_logits.detach().float()).abs().max())
            raise RuntimeError(
                f"Cached output-head replay changes native logits (max delta {delta})"
            )


def choose_validation_candidate(candidates):
    """Select by validation CE, breaking exact ties in favor of stronger ridge."""
    if not candidates:
        raise ValueError("No validation candidates")
    for candidate in candidates:
        if not torch.isfinite(torch.tensor(candidate["validation_ce"])):
            raise ValueError("Nonfinite validation cross entropy")
    return min(candidates, key=lambda row: (row["validation_ce"], -row["ridge"]))


def validation_objective(primary_ce, preservation_ce=None):
    """Weight task-level mean losses equally, independently of validation size."""
    losses = [primary_ce] if preservation_ce is None else [primary_ce, preservation_ce]
    if any(not math.isfinite(value) for value in losses):
        raise ValueError("Validation losses must be finite")
    return sum(losses) / len(losses)
