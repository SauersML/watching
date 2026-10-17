"""Opt-in native-tail training for a specified activation-viewing implementation.

This adapter changes only the source intervention. It does not relax the trainer's
whole-vocabulary extraction checks or claim that buffer copies are harmless.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from watching.control_probes import BufferWriterObserver, StorageControlProbe, materialize
from watching.suffix import ResidualCapture

MODES = ("reference", "buffer", "identity")


def learner_input_equality(cache):
    """Compare all state components, not only logits or final hidden states."""
    summary = {}
    for split in ("train", "validation"):
        pairs = cache[split]
        if not pairs:
            raise ValueError("No-footprint decision requires both train and development pairs")
        identical = []
        for absent, present in pairs:
            if len(absent) != 4 or len(present) != 4:
                raise ValueError("Expected hidden state, attention mask, cosine and sine")
            identical.append(
                all(
                    a is None
                    and b is None
                    or (
                        a is not None
                        and b is not None
                        and a.dtype == b.dtype
                        and a.shape == b.shape
                        and torch.equal(
                            a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8),
                        )
                    )
                    for a, b in zip(absent, present, strict=True)
                )
            )
        summary[split] = {"pairs": len(pairs), "bit_identical_pairs": sum(identical)}
    return {
        "splits": summary,
        "all_learner_states_bit_identical": all(
            row["pairs"] == row["bit_identical_pairs"] for row in summary.values()
        ),
        "position_ids": "Shared same-prompt tensor for both conditions by construction",
    }


def verify_matched_selection(selected, reference_path):
    """Prompt order determines minibatches; set equality is insufficient."""
    reference_path = Path(reference_path)
    reference = json.loads(reference_path.read_text())
    if selected != reference:
        raise ValueError("Selected splits/records/order differ from the matched historical run")
    return {
        "matched_data_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "matched_selection_exact_including_order": True,
        "historical_test_previously_inspected": True,
        "viewing_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def expected_references(mode, present):
    if mode not in MODES:
        raise ValueError("Unsupported viewing mode")
    return int(present and mode == "reference")


class ViewingProbe(StorageControlProbe):
    def __init__(self, model, layer, viewing_mode):
        expected_references(viewing_mode, False)
        super().__init__(model, layer)
        self.viewing_mode = viewing_mode

    def forward(self, input_ids, attention_mask, position_ids, capture):
        return super().forward(
            input_ids,
            attention_mask,
            position_ids,
            mode=self.viewing_mode if capture else "none",
        )


class ViewingResidualCapture(ResidualCapture):
    """The existing boundary-copy observer with the same source intervention."""

    def __init__(self, model, layer, cut, viewing_mode):
        expected_references(viewing_mode, False)
        super().__init__(model, layer, cut, "buffer")
        self.viewing_mode = viewing_mode
        self.mode = "none"
        self.register_buffer("probe_buffer", None, persistent=False)

    def record(self, module, inputs, output):
        if self.mode == "identity":
            return materialize(output)
        if self.mode == "buffer":
            if self.probe_buffer is None or self.probe_buffer.shape != output.shape:
                raise RuntimeError("Prepare the viewing buffer for this exact shape")
            self.probe_buffer.copy_(output)
            return None
        return super().record(module, inputs, output)

    def forward(self, input_ids, attention_mask, position_ids, capture):
        self.mode = self.viewing_mode if capture else "none"
        return super().forward(
            input_ids,
            attention_mask,
            position_ids,
            capture and self.viewing_mode == "reference",
        )


def prepare_probe_buffer(wrapper, example):
    """Use eager execution only for allocation; discard all eager values."""
    if wrapper.viewing_mode != "buffer":
        return
    original = wrapper.viewing_mode
    try:
        wrapper.viewing_mode = "reference"
        with torch.no_grad():
            saved = wrapper(**example, capture=True)[1]
        if len(saved) != 1:
            raise RuntimeError("Probe buffer allocation did not observe exactly one tensor")
        wrapper.probe_buffer = torch.empty_like(saved[0])
    finally:
        wrapper.viewing_mode = original
        wrapper.captured = []
        if hasattr(wrapper, "boundary"):
            wrapper.boundary = None


def make_wrappers(model, observed_model, layer, cut, mode, example):
    native = ViewingProbe(model, layer, mode)
    extraction = ViewingResidualCapture(observed_model, layer, cut, mode)
    extraction.prepare_buffers(example)
    for wrapper in (native, extraction):
        prepare_probe_buffer(wrapper, example)
    return native, extraction


class CheckedInvocation:
    """Validate actual buffer writes outside the compiled model computation."""

    def __init__(self, compiled, wrapper, emit, name, observer_factory=BufferWriterObserver):
        self.compiled = compiled
        self.wrapper = wrapper
        self.emit = emit
        self.name = name
        self.observer_factory = observer_factory
        self.writer_verified = False

    def _run(self, inputs, present):
        buffer = self.wrapper.probe_buffer
        if buffer is not None:
            buffer.fill_(float("nan"))
        result = self.compiled(**inputs, capture=present)
        if len(result[1]) != expected_references(self.wrapper.viewing_mode, present):
            raise RuntimeError("Viewing intervention returned an unexpected reference count")
        if buffer is not None:
            valid = torch.isfinite(buffer).all() if present else buffer.isnan().all()
            if not valid:
                raise RuntimeError(
                    "Viewing buffer was not fully overwritten, or absent mode wrote it"
                )
        return result

    def __call__(self, *, capture, **inputs):
        result = self._run(inputs, capture)
        if self.wrapper.viewing_mode == "buffer" and capture and not self.writer_verified:
            logits = result[0].detach().clone()
            values = self.wrapper.probe_buffer.detach().clone()
            observer = self.observer_factory(self.wrapper.probe_buffer)
            with observer:
                repeated = self._run(inputs, capture)
            if not observer.writers:
                raise RuntimeError("No generated kernel store to the viewing buffer was observed")
            if not torch.equal(logits, repeated[0]) or not torch.equal(
                values, self.wrapper.probe_buffer
            ):
                raise RuntimeError("Viewing buffer or logits changed on an identical-input repeat")
            self.writer_verified = True
            self.emit(
                "viewing_buffer_verified",
                wrapper=self.name,
                native_writers=observer.writers,
                exact_repeat=True,
            )
            return repeated
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--viewing-mode", choices=MODES, required=True)
    parser.add_argument("--matched-data", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    forbidden = (
        "--initial-tail",
        "--warm-start-tail",
        "--warm-start-manifest",
        "--prior-results",
        "--train-head",
    )
    if any(arg.split("=", 1)[0] in forbidden for arg in remaining):
        parser.error("Matched viewing arms require base initialization and the fixed native head")
    from watching.tail_train import main as train

    sys.argv = [sys.argv[0], *remaining]
    train(viewing_mode=args.viewing_mode, viewing_reference_data=args.matched_data)


if __name__ == "__main__":
    main()
