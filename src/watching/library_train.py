"""Train native upper blocks using standard PyTorch/HF capture APIs."""

import argparse
import hashlib
import sys
from pathlib import Path

import torch

from watching.artifacts import write_json


class LibraryInvocation:
    """Adapt ModelOutput only after the raw HF compiled invocation returns."""

    def __init__(
        self, model, method, layer, example, *, cut=None, reference=None, out=None, emit=None
    ):
        self.model, self.method, self.layer = model, method, layer
        self.reference, self.out = reference, out
        self.emit = emit
        self.failure_evidence = {}
        self.enabled = False
        self.saved = {}
        self.captured = []
        self.boundary = None
        self.buffers = None
        self.last_full = None
        self.checked_conditions = set()
        with torch.no_grad():
            plain = model(**example, use_cache=False).logits.detach().clone()
        with torch.no_grad():
            no_hook_call = torch.compile(model, fullgraph=True, dynamic=False)
            self.no_hook_compiled = no_hook_call(**example, use_cache=False).logits.detach().clone()
        if method == "forward_hook":

            def record(_module, _inputs, output):
                if self.enabled:
                    self.saved["activation"] = output.detach()

            self.handle = model.model.layers[layer].self_attn.q_norm.register_forward_hook(record)
        if cut is not None:

            def boundary(_module, inputs, kwargs):
                cos, sin = kwargs["position_embeddings"]
                values = (inputs[0], kwargs["attention_mask"], cos, sin)
                if self.buffers is None:
                    self.buffers = tuple(None if v is None else torch.empty_like(v) for v in values)
                for destination, value in zip(self.buffers, values, strict=True):
                    if (destination is None) != (value is None):
                        raise RuntimeError("Boundary mask presence changed")
                    if value is not None:
                        if (
                            destination.shape != value.shape
                            or destination.dtype != value.dtype
                            or destination.device != value.device
                        ):
                            raise RuntimeError("Boundary shape, dtype, or device changed")
                        destination.copy_(value.detach())
                self.boundary = self.buffers

            self.boundary_handle = model.model.layers[cut + 1].register_forward_pre_hook(
                boundary, with_kwargs=True
            )
        with torch.no_grad():
            disabled = model(**example, use_cache=False).logits
            self.identical(plain, disabled, "disabled_hook_vs_no_hook", example)
            for present in (False, True):
                result, _ = self.invoke(model, example, present)
                self.identical(plain, result.logits, "eager_capture_parity", example)
        self.compiled = torch.compile(model, fullgraph=True, dynamic=False)
        self.saved.clear()

    def identical(self, actual, expected, stage, inputs):
        if not torch.equal(actual, expected):
            report = {
                "stage": stage,
                "different_logits": int((actual != expected).sum()),
                "max_abs_difference": float((actual.float() - expected.float()).abs().max()),
            }
            if self.out:
                write_json(self.out / "extraction_failure.json", report)
                torch.save(
                    {
                        "report": report,
                        "input": inputs,
                        "original_logits": expected.detach().cpu(),
                        "extracted_logits": actual.detach().cpu(),
                        **self.failure_evidence,
                    },
                    self.out / "extraction_failure.pt",
                )
            raise RuntimeError(f"Library full-sequence/full-vocabulary parity failed: {stage}")

    def invoke(self, call, inputs, present):
        self.enabled = present
        self.saved.clear()
        if self.buffers is not None:
            self.boundary = None
            for destination in self.buffers:
                if destination is not None and destination.is_floating_point():
                    destination.fill_(float("nan"))
        result = call(
            **inputs,
            use_cache=False,
            output_hidden_states=present and self.method == "hidden_states",
        )
        if self.buffers is not None:
            if self.boundary is None or any(
                v is not None
                and (torch.isnan(v).any() if index == 1 else not torch.isfinite(v).all())
                for index, v in enumerate(self.buffers)
            ):
                raise RuntimeError("Boundary capture was not freshly overwritten")
        if present and self.method == "hidden_states":
            if len(result.hidden_states) != len(self.model.model.layers) + 1:
                raise RuntimeError("Native hidden-state contract changed")
            if any(v.requires_grad or not torch.isfinite(v).all() for v in result.hidden_states):
                raise RuntimeError("Nonfinite native hidden states")
            captured = [result.hidden_states[self.layer + 1].detach()]
        elif present:
            if set(self.saved) != {"activation"}:
                raise RuntimeError("Forward hook capture was not refreshed")
            captured = [self.saved["activation"]]
        else:
            if self.saved:
                raise RuntimeError("Disabled hook unexpectedly captured an activation")
            captured = []
        if any(v.requires_grad or not torch.isfinite(v).all() for v in captured):
            raise RuntimeError("Invalid detached library activation")
        return result, captured

    def __call__(self, *, capture, **inputs):
        result, self.captured = self.invoke(self.compiled, inputs, capture)
        logits = result.logits
        if not capture and capture not in self.checked_conditions:
            self.identical(
                logits, self.no_hook_compiled, "compiled_disabled_hook_vs_no_hook", inputs
            )
            self.no_hook_compiled = None
        if capture not in self.checked_conditions:
            original = logits.detach().clone()
            snapshots = [v.clone() for v in self.captured]
            repeated, captured = self.invoke(self.compiled, inputs, capture)
            self.identical(repeated.logits, original, "compiled_repeat", inputs)
            for actual, expected in zip(captured, snapshots, strict=True):
                self.identical(actual, expected, "capture_repeat", inputs)
            self.checked_conditions.add(capture)
            if self.emit:
                from torch._dynamo.utils import counters

                self.emit(
                    "library_compiled_condition_checked",
                    capture=capture,
                    path="extraction" if self.reference is not None else "native",
                    logits_shape=list(logits.shape),
                    capture_shapes=[list(v.shape) for v in self.captured],
                    repeat_bit_identical=True,
                    counters={k: dict(v) for k, v in counters.items() if v},
                )
        if self.reference is not None:
            if not torch.equal(logits, self.reference.last_full):
                original = logits.detach().cpu().clone()
                source = self.reference.last_full.detach().cpu().clone()
                repeated_source, _ = self.reference.invoke(self.reference.compiled, inputs, capture)
                repeated_copy, _ = self.invoke(self.compiled, inputs, capture)
                self.failure_evidence = {
                    "repeated_original_logits": repeated_source.logits.detach().cpu(),
                    "repeated_extracted_logits": repeated_copy.logits.detach().cpu(),
                    "boundary_state": tuple(
                        None if v is None else v.detach().cpu().clone() for v in self.buffers
                    ),
                }
                self.identical(original, source, "native_vs_boundary", inputs)
        self.last_full = logits.detach().clone()
        output = (logits[:, -1, :], tuple(self.captured))
        return (*output, self.boundary) if self.buffers is not None else output


def make_factory(method):
    def factory(model, observed_model, args, cut, example, emit):
        if args.train_head or any(
            (args.initial_tail, args.prior_results, args.warm_start_tail, args.warm_start_manifest)
        ):
            raise ValueError(
                "Library training requires original base initialization and a fixed head"
            )
        if (
            model.config.model_type != "qwen3"
            or model.config.hidden_size != 4096
            or model.config.num_hidden_layers != 36
        ):
            raise ValueError("Library training requires the original Qwen3-8B base")
        if args.tail_layers != 2:
            raise ValueError("Library training updates exactly the last two native blocks")
        native = LibraryInvocation(model, method, args.probe_layer, example, out=args.out)
        observed = LibraryInvocation(
            observed_model,
            method,
            args.probe_layer,
            example,
            cut=cut,
            reference=native,
            out=args.out,
        )
        native.emit = observed.emit = emit
        emit(
            "library_eager_controls_passed",
            capture_method=method,
            eager_disabled_hook_vs_no_hook=True,
            eager_capture_parity=True,
            boundary_refresh="NaN poisoning before every invocation",
            compile_target="raw HF model",
            fullgraph=True,
            dynamic=False,
        )
        return (
            native,
            observed,
            int,
            {
                "capture_method": method,
                "library_capture_method": method,
                "capture_api": "native PyTorch/Hugging Face",
                "raw_hf_model_compiled": True,
                "compiled_return_wrapper": False,
                "full_sequence_full_vocabulary_extraction_parity_required": True,
                "library_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        )

    return factory


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--capture-method", choices=("forward_hook", "hidden_states"), required=True
    )
    args, remaining = parser.parse_known_args()
    prohibited = (
        "--initial-tail",
        "--prior-results",
        "--warm-start-tail",
        "--warm-start-manifest",
        "--train-head",
    )
    if any(arg.split("=", 1)[0] in prohibited for arg in remaining):
        parser.error("Library training requires original base initialization and a fixed head")
    sys.argv = [sys.argv[0], *remaining]
    from watching.tail_train import main as train

    train(capture_factory=make_factory(args.capture_method))


if __name__ == "__main__":
    main()
