"""Fit existing answer rows to one inference-path activation pair.

This is output-only supervised least squares, not full-parameter gradient
training. Capture is accepted only if every measured logit matches the ordinary
probe wrapper. Weight tying and the model's forward arithmetic are preserved.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import save_weights, write_json
from watching.audit import FeatureProbe
from watching.data import make_expanded_data
from watching.metrics import summarize_pairs
from watching.probe import ProbeModel


def minimum_norm_update(features, scores, targets):
    """Solve H @ delta = target - score using a CPU float64 SVD.

    Normal equations would square the condition number of this nearly collinear
    pair. Reject a rank-deficient pair rather than inventing a discriminant.
    """
    h = features.detach().to(device="cpu", dtype=torch.float64)
    residual = targets.detach().cpu().double() - scores.detach().cpu().double()
    if h.ndim != 2 or h.shape[0] != 2 or residual.shape != (2,):
        raise ValueError("Expected two feature vectors and two scores")
    if not torch.isfinite(h).all() or not torch.isfinite(residual).all():
        raise ValueError("Features and scores must be finite")
    u, singular, vh = torch.linalg.svd(h, full_matrices=False)
    if singular[-1] <= torch.finfo(h.dtype).eps * max(h.shape) * singular[0]:
        raise ValueError("Feature pair is rank deficient")
    delta = vh.T @ ((u.T @ residual) / singular)
    return delta, singular


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--target-gap", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--test-prompts", type=int, default=128)
    args = parser.parse_args()
    if not 0 < args.target_gap <= 4 or not 1 <= args.steps <= 5:
        parser.error("Use a target gap in (0, 4] and one to five refinement steps")
    if not 1 <= args.test_prompts <= 512:
        parser.error("test-prompts must be in [1, 512]")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA Inductor")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Output directory must be empty")
    torch.manual_seed(20261015)
    torch.set_num_threads(4)
    started = time.monotonic()

    def emit(event, **values):
        line = json.dumps(
            dict(event=event, seconds=time.monotonic() - started, **values), allow_nan=False
        )
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    tok = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    labels = [tok.encode(s, add_special_tokens=False) for s in ("No", "Yes")]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError("Answers must each be one token")
    no, yes = [ids[0] for ids in labels]
    records = make_expanded_data()
    data = {"train": records["train"][:1], "test": records["test"][: args.test_prompts]}
    write_json(args.out / "data.json", data)
    batches = {}
    for split, rows in data.items():
        batches[split] = []
        for row in rows:
            text = tok.apply_chat_template(
                [{"role": "user", "content": row["text"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if len(tok.encode(text, add_special_tokens=False)) > 64:
                raise ValueError("Prompt exceeds the fixed sequence length")
            x = tok(
                text,
                add_special_tokens=False,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            ).to("cuda")
            if ((x.input_ids == no) | (x.input_ids == yes)).any():
                raise ValueError("An evaluated input uses an answer row; tying would move features")
            x["position_ids"] = (x.attention_mask.cumsum(-1) - 1).clamp_min(0)
            batches[split].append(dict(x))

    def load():
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
        )
        model.config.use_cache = False
        model.requires_grad_(True)
        return model

    model, measured_model = load(), load()
    if any(
        not torch.equal(a, b)
        for a, b in zip(model.parameters(), measured_model.parameters(), strict=True)
    ):
        raise RuntimeError("The two copies must start with identical parameters")
    detector = ProbeModel(model, args.layer)
    measured = FeatureProbe(measured_model, args.layer)
    compiled = torch.compile(detector, fullgraph=True, dynamic=False)
    compiled_features = torch.compile(measured, fullgraph=True, dynamic=False)
    head = model.get_output_embeddings().weight
    indices = torch.tensor([no, yes], device=head.device)
    original_rows = head.detach().index_select(0, indices).clone()
    manifest = {
        **vars(args),
        "out": str(args.out),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "labels": {"No": no, "Yes": yes},
        "tied_embeddings": head is model.get_input_embeddings().weight,
        "method": "Inference-path fit of existing answer rows; all other weights fixed",
        "answer_token_occurrences_in_all_inputs": 0,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }
    write_json(args.out / "manifest.json", manifest)
    emit("setup", **manifest)

    def forward(x, capture, *, features=False):
        wrapper = measured if features else detector
        with torch.no_grad():
            output = (compiled_features if features else compiled)(**x, capture=capture)
        if len(output[1]) != int(capture):
            raise RuntimeError("Probe capture count is incorrect")
        wrapper.captured = []
        if features:
            measured.features = None
        return output

    def evaluate(step, splits):
        training_features = []
        training_scores = []
        for split in splits:
            rows = []
            emit("evaluation_start", step=step, split=split)
            for i, x in enumerate(batches[split]):
                for capture in (False, True):
                    logits = forward(x, capture)[0]
                    other, _, hidden = forward(x, capture, features=True)
                    if not torch.equal(logits, other):
                        raise RuntimeError("Feature capture changes logits; reject this fit")
                    values = logits[0].float()
                    logp = values.log_softmax(-1)
                    score = float(values[yes] - values[no])
                    label = yes if capture else no
                    rows.append(
                        {
                            "index": i,
                            "capture": capture,
                            "yes_minus_no": score,
                            "logp_yes": float(logp[yes]),
                            "logp_no": float(logp[no]),
                            "yes_probability": float(torch.sigmoid(values[yes] - values[no])),
                            "correct": (score > 0) if capture else (score <= 0),
                            "greedy_correct": int(values.argmax()) == label,
                            "ce": -float(logp[label]),
                        }
                    )
                    if split == "train":
                        training_features.append(hidden.detach().cpu())
                        training_scores.append(score)
            summary = {
                "split": split,
                "step": step,
                "grad_mode": False,
                **summarize_pairs(rows, seed=20261015),
            }
            write_json(
                args.out / f"{split}_{step}_inference.json", {"summary": summary, "rows": rows}
            )
            emit("evaluation", **summary)
        return torch.cat(training_features), torch.tensor(training_scores, dtype=torch.float64)

    features, scores = evaluate(0, ("train", "test"))
    original_features = features.clone()
    targets = torch.tensor([-args.target_gap / 2, args.target_gap / 2])
    emit("feature_pair", difference_norm=float((features[1].float() - features[0].float()).norm()))
    for step in range(1, args.steps + 1):
        delta, singular = minimum_norm_update(features, scores, targets)
        with torch.no_grad():
            current = head.index_select(0, indices).float()
            update = delta.to(device=head.device, dtype=torch.float32) / 2
            new_rows = torch.stack((current[0] - update, current[1] + update)).to(head.dtype)
            head.index_copy_(0, indices, new_rows)
            measured_model.get_output_embeddings().weight.index_copy_(0, indices, new_rows)
        emit(
            "fit_update",
            step=step,
            singular_values=singular.tolist(),
            correction_norm=float(delta.norm()),
            total_answer_weight_change=float((new_rows.float() - original_rows.float()).norm()),
        )
        features, scores = evaluate(step, ("train", "test") if step == args.steps else ("train",))
        if not torch.equal(features, original_features):
            raise RuntimeError("Updating answer rows changed internal features; reject this fit")
        emit("features_unchanged", step=step, bit_identical=True)
    save_weights(model, tok, args.out / "checkpoint", step=args.steps)
    emit("complete", checkpoint=str(args.out / "checkpoint"))


if __name__ == "__main__":
    main()
