"""Measure signal and gradient availability without updating the checkpoint.

Feature extraction is accepted only when adding that output preserves every
baseline logit for every measured prompt and condition. Directional steps are
temporary, restore the exact weights, and diagnose a straight-through gradient;
they are not additional training runs or evidence of generalization.
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.data import make_expanded_data
from watching.probe import ProbeModel


class FeatureProbe(ProbeModel):
    """Also expose the already consumed input to the language-model head."""

    def __init__(self, model, layer):
        super().__init__(model, layer)
        self.features = None
        self.feature_handle = model.lm_head.register_forward_pre_hook(self.record_features)

    def record_features(self, module, inputs):
        self.features = inputs[0]

    def forward(self, input_ids, attention_mask, position_ids, capture):
        logits, saved = super().forward(input_ids, attention_mask, position_ids, capture)
        return logits, saved, self.features[:, -1, :]


def gradient_comparison(absent, present):
    """Compute the difference directly, avoiding subtraction of similar norms."""
    a, b = absent.float(), present.float()
    return torch.stack(
        (
            a.square().sum(),
            b.square().sum(),
            (a * b).sum(),
            (b - a).square().sum(),
            (a != b).sum().float(),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--layer", type=int, default=14)
    args = parser.parse_args()
    if not 1 <= args.prompts <= 32:
        parser.error("--prompts must be between 1 and 32")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Audit output must be empty")
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires CUDA Inductor")
    torch.manual_seed(20261015)
    torch.set_num_threads(4)
    started = time.monotonic()

    def emit(event, **values):
        line = json.dumps(
            dict(event=event, seconds=time.monotonic() - started, **values), allow_nan=False
        )
        print(line, flush=True)
        with (out / "audit.jsonl").open("a") as f:
            f.write(line + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    labels = [tokenizer.encode(s, add_special_tokens=False) for s in ("No", "Yes")]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError("Labels must each be one token")
    no, yes = [ids[0] for ids in labels]
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    model.config.use_cache = False
    model.requires_grad_(True)
    emit(
        "setup",
        arguments=vars(args),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        labels={"No": no, "Yes": yes},
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    )
    records = [
        (split, i, row["text"])
        for split in ("train", "validation")
        for i, row in enumerate(make_expanded_data()[split][: args.prompts])
    ]
    (out / "prompts.json").write_text(json.dumps(records, indent=2))
    batches = []
    for _, _, text in records:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(tokenizer.encode(rendered, add_special_tokens=False)) > 64:
            raise ValueError("Prompt exceeds the training sequence length")
        x = tokenizer(
            rendered,
            add_special_tokens=False,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to("cuda")
        x["position_ids"] = (x["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        batches.append(dict(x))
    detector = ProbeModel(model, args.layer)
    compiled = torch.compile(detector, fullgraph=True, dynamic=False)

    def forward(x, capture):
        logits, saved = compiled(**x, capture=capture)
        if len(saved) != int(capture):
            raise RuntimeError("Capture did not execute")
        detector.captured = []
        return logits

    baselines = {}
    for grad_mode in (False, True):
        emit("baseline_start", grad_mode=grad_mode)
        with torch.set_grad_enabled(grad_mode):
            for i, x in enumerate(batches):
                for capture in (False, True):
                    baselines[grad_mode, i, capture] = forward(x, capture).detach().cpu()
        gaps = [
            (baselines[grad_mode, i, True].float() - baselines[grad_mode, i, False].float())[
                0, [no, yes]
            ].tolist()
            for i in range(len(batches))
        ]
        emit(
            "baseline_done",
            grad_mode=grad_mode,
            log_odds_separations=[pair[1] - pair[0] for pair in gaps],
        )

    named = list(model.named_parameters())
    absent_gradients = []
    gradients = []
    statistics = []
    for capture in (False, True):
        emit("gradient_start", capture=capture)
        model.zero_grad(set_to_none=True)
        logits = forward(batches[0], capture)
        (logits[0, yes].float() - logits[0, no].float()).backward()
        for i, (name, p) in enumerate(named):
            if p.grad is None:
                raise RuntimeError(f"Missing gradient: {name}")
            if not capture:
                absent_gradients.append(p.grad.detach().float().clone())
            else:
                statistics.append(gradient_comparison(absent_gradients[i], p.grad))
                # Reuse the first-gradient allocation for the separation gradient.
                gradients.append(absent_gradients[i].neg_().add_(p.grad.float()))
        del logits
    model.zero_grad(set_to_none=True)
    stats = torch.stack(statistics).cpu().double()
    totals = stats.sum(0)
    norm = float(totals[3].sqrt())
    denom = float((totals[0] * totals[1]).sqrt())
    emit(
        "gradient_summary",
        absent_norm=float(totals[0].sqrt()),
        present_norm=float(totals[1].sqrt()),
        separation_gradient_norm=norm,
        cosine=float(totals[2]) / denom if denom else None,
        different_gradient_elements=int(totals[4]),
    )
    (out / "gradients.json").write_text(
        json.dumps(
            [
                dict(
                    parameter=name,
                    absent_squared_norm=row[0],
                    present_squared_norm=row[1],
                    dot_product=row[2],
                    difference_squared_norm=row[3],
                    different_elements=int(row[4]),
                )
                for (name, _), row in zip(named, stats.tolist(), strict=True)
            ],
            indent=2,
        )
    )
    originals = [p.detach().clone() for _, p in named]
    baseline_score = (baselines[True, 0, True].float() - baselines[True, 0, False].float())[0]
    baseline_gap = float(baseline_score[yes] - baseline_score[no])
    try:
        for distance in (-1.0, -0.1, -0.01, 0.01, 0.1, 1.0) if norm else ():
            with torch.no_grad():
                changes = []
                for (_, p), original, gradient in zip(named, originals, gradients, strict=True):
                    p.copy_(original.float() + gradient * (distance / norm))
                    changes.append((p != original).sum())
                changed = int(torch.stack(changes).sum())
            scores = []
            for capture in (False, True):
                logits = forward(batches[0], capture).detach().float()
                scores.append(float(logits[0, yes] - logits[0, no]))
            emit(
                "directional_step",
                distance=distance,
                weights_changed=changed,
                predicted_change=distance * norm,
                measured_change=scores[1] - scores[0] - baseline_gap,
                absent_log_odds=scores[0],
                present_log_odds=scores[1],
            )
    finally:
        with torch.no_grad():
            for (_, p), original in zip(named, originals, strict=True):
                p.copy_(original)
    for capture in (False, True):
        actual = forward(batches[0], capture).detach().cpu()
        if not torch.equal(actual, baselines[True, 0, capture]):
            raise RuntimeError("Restored baseline changed; do not interpret directional steps")
    emit("restoration_verified")
    del originals, gradients, absent_gradients, statistics
    detector.handle.remove()
    compiled = detector = None
    gc.collect()
    torch.cuda.empty_cache()

    features = {}
    measured = FeatureProbe(model, args.layer)
    compiled_features = torch.compile(measured, fullgraph=True, dynamic=False)
    for grad_mode in (False, True):
        emit("feature_start", grad_mode=grad_mode)
        matches = True
        with torch.set_grad_enabled(grad_mode):
            for i, x in enumerate(batches):
                for capture in (False, True):
                    logits, saved, hidden = compiled_features(**x, capture=capture)
                    if len(saved) != int(capture):
                        raise RuntimeError("Feature probe did not execute")
                    identical = torch.equal(logits.detach().cpu(), baselines[grad_mode, i, capture])
                    matches = matches and identical
                    features[grad_mode, i, capture] = hidden.detach().cpu()
                    measured.captured = []
                    measured.features = None
        emit("feature_parity", grad_mode=grad_mode, all_logits_identical=matches)
        if matches:
            a = torch.cat([features[grad_mode, i, False] for i in range(len(batches))]).float()
            b = torch.cat([features[grad_mode, i, True] for i in range(len(batches))]).float()
            emit(
                "hidden_footprint",
                grad_mode=grad_mode,
                different_fraction=float((a != b).float().mean()),
                per_prompt_delta_norm=(b - a).norm(dim=-1).tolist(),
                per_prompt_hidden_norm=a.norm(dim=-1).tolist(),
            )
        else:
            emit("feature_measurement_confounded", grad_mode=grad_mode)
    torch.save(
        dict(
            features=features,
            baselines=baselines,
            records=records,
            answer_weights=model.lm_head.weight.detach()[[no, yes]].cpu(),
        ),
        out / "features.pt",
    )
    emit("complete")


if __name__ == "__main__":
    main()
