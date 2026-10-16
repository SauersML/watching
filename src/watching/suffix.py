"""Train native upper blocks on a verified, fixed inference-path residual cache.

The experiment stops if extraction or initial suffix replay changes any vocabulary
logit. Every evaluation measures the integrated ordinary inference graph and checks
that its source activations remain identical to the cache. No classifier is added.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import save_weights, write_json
from watching.data import make_expanded_data
from watching.metrics import summarize_pairs
from watching.probe import ProbeModel


class ResidualCapture(ProbeModel):
    """Expose an existing decoder boundary and its actual attention arguments."""

    def __init__(self, model, probe_layer, cut_layer):
        if not probe_layer <= cut_layer < len(model.model.layers) - 1:
            raise ValueError("Cut must follow the probe and leave at least one upper block")
        super().__init__(model, probe_layer)
        self.boundary = None
        self.boundary_handle = model.model.layers[cut_layer + 1].register_forward_pre_hook(
            self.record_boundary, with_kwargs=True
        )

    def record_boundary(self, module, args, kwargs):
        cos, sin = kwargs["position_embeddings"]
        self.boundary = (args[0], kwargs["attention_mask"], cos, sin)

    def forward(self, input_ids, attention_mask, position_ids, capture):
        logits, observed = super().forward(input_ids, attention_mask, position_ids, capture)
        return logits, observed, self.boundary


class NativeSuffix(torch.nn.Module):
    """The existing decoder blocks, final norm, and frozen vocabulary head."""

    def __init__(self, model, cut_layer):
        super().__init__()
        if model.config.model_type != "qwen3" or any(
            kind != "full_attention" for kind in model.config.layer_types
        ):
            raise ValueError(
                "This experiment implements full-attention Qwen3 decoder boundaries only"
            )
        self.layers = torch.nn.ModuleList(list(model.model.layers)[cut_layer + 1 :])
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    def active_parameters(self):
        return list(self.layers.parameters()) + list(self.norm.parameters())

    def forward(self, hidden, mask, cos, sin, position_ids):
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                position_embeddings=(cos, sin),
                past_key_values=None,
                use_cache=False,
            )
        return self.lm_head(self.norm(hidden)[:, -1:, :])[:, -1, :]


class SuffixAdam:
    """FP32 master Adam for precisely the native parameters selected for updates."""

    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr, weight_decay=0, foreach=False)

    def step(self):
        for parameter, master in zip(self.parameters, self.masters, strict=True):
            if parameter.grad is None:
                raise RuntimeError("An active suffix parameter has no gradient")
            master.grad = parameter.grad.detach().float().clone()
            parameter.grad = None
        norm = torch.nn.utils.clip_grad_norm_(self.masters, 1, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for parameter, master in zip(self.parameters, self.masters, strict=True):
                parameter.copy_(master)
        self.optimizer.zero_grad(set_to_none=True)
        return float(norm)


def assert_identical(actual, expected, context):
    if not torch.equal(actual, expected):
        delta = (actual.float() - expected.float()).abs()
        raise RuntimeError(f"{context}: parity failed; max difference={float(delta.max()):.8g}")


def move_state(state, device):
    return tuple(None if value is None else value.to(device) for value in state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe-layer", type=int, default=14)
    parser.add_argument("--cut-layer", type=int, default=14)
    parser.add_argument("--train-prompts", type=int, default=256)
    parser.add_argument("--validation-prompts", type=int, default=32)
    parser.add_argument("--test-prompts", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if (
        not 1 <= args.train_prompts <= 2048
        or not 1 <= min(args.validation_prompts, args.test_prompts)
        or max(args.validation_prompts, args.test_prompts) > 512
        or args.steps < 1
        or args.eval_every < 1
        or not 0 < args.lr < 1
    ):
        parser.error("Invalid prompt count, training steps, or learning rate")
    if not torch.cuda.is_available():
        raise RuntimeError("The experiment requires CUDA Inductor")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Output directory must be empty")
    started = time.monotonic()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    def emit(event, **values):
        line = json.dumps(
            dict(event=event, seconds=time.monotonic() - started, **values), allow_nan=False
        )
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    emit("loading", checkpoint=args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = [tokenizer.encode(answer, add_special_tokens=False) for answer in ("No", "Yes")]
    if any(len(ids) != 1 for ids in tokens):
        raise ValueError("No and Yes must each tokenize to one token")
    no, yes = [ids[0] for ids in tokens]
    records = make_expanded_data()
    data = {
        name: records[name][:count]
        for name, count in [
            ("train", args.train_prompts),
            ("validation", args.validation_prompts),
            ("test", args.test_prompts),
        ]
    }
    write_json(args.out / "data.json", data)
    batches = {}
    for split, rows in data.items():
        batches[split] = []
        for row in rows:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["text"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if len(tokenizer.encode(text, add_special_tokens=False)) > 64:
                raise ValueError("Prompt exceeds fixed length 64")
            batch = tokenizer(
                text,
                add_special_tokens=False,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            )
            batch["position_ids"] = (batch.attention_mask.cumsum(-1) - 1).clamp_min(0)
            batches[split].append(dict(batch))

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

    model, observed_model = load(), load()
    ordinary = ProbeModel(model, args.probe_layer)
    observed = ResidualCapture(observed_model, args.probe_layer, args.cut_layer)
    suffix = NativeSuffix(model, args.cut_layer)
    # Only these original parameters receive optimizer updates. Keeping all other
    # requires_grad flags unchanged preserves the baseline compilation setup.
    optimizer = SuffixAdam(suffix.active_parameters(), args.lr)
    native = torch.compile(ordinary, fullgraph=True, dynamic=False)
    extract = torch.compile(observed, fullgraph=True, dynamic=False)
    replay = torch.compile(suffix, fullgraph=True, dynamic=False)
    manifest = {
        **vars(args),
        "out": str(args.out),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "labels": {"No": no, "Yes": yes},
        "updated_blocks": list(range(args.cut_layer + 1, len(model.model.layers))),
        "update_final_norm": True,
        "update_output_head": False,
        "active_parameters": sum(p.numel() for p in suffix.active_parameters()),
        "tied_embeddings": model.lm_head.weight is model.model.embed_tokens.weight,
        "objective": "balanced binary Yes/No cross entropy on paired cached inputs",
        "length": 64,
        "batch_size": 1,
        "primary_path": "integrated ordinary inference",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }
    write_json(args.out / "manifest.json", manifest)
    emit("setup", **manifest)
    cached = {}

    def execute(batch, capture):
        x = {name: tensor.cuda() for name, tensor in batch.items()}
        with torch.no_grad():
            full, probe = native(**x, capture=capture)
            copied, other_probe, state = extract(**x, capture=capture)
        if len(probe) != int(capture) or len(other_probe) != int(capture):
            raise RuntimeError("Probe capture did not fire exactly once")
        ordinary.captured = observed.captured = []
        observed.boundary = None
        assert_identical(copied, full, "Boundary extraction changes full-model logits")
        return x, full, state

    for split, rows in batches.items():
        cached[split] = []
        emit("cache_start", split=split, pairs=len(rows))
        for index, batch in enumerate(rows):
            pair = []
            initial_scores = []
            for capture in (False, True):
                x, full, state = execute(batch, capture)
                initial_scores.append(float(full[0, yes].float() - full[0, no].float()))
                with torch.no_grad():
                    replayed = replay(*state, x["position_ids"])
                assert_identical(replayed, full, "Initial suffix replay changes full-model logits")
                pair.append(tuple(None if t is None else t.detach().cpu().clone() for t in state))
            cached[split].append(pair)
            if index == 0 or (index + 1) % 16 == 0:
                emit(
                    "cache_progress",
                    split=split,
                    pairs=index + 1,
                    extraction_and_replay_bit_identical=True,
                    latest_pair_gap=initial_scores[1] - initial_scores[0],
                )
    # The cache is not published as a second large artifact; inputs and immutable
    # source checkpoint identify its contents, and every selected pair is checked.

    def score(logits, index, capture):
        values = logits[0].float()
        logp = values.log_softmax(-1)
        gap = float(values[yes] - values[no])
        return {
            "index": index,
            "capture": capture,
            "yes_minus_no": gap,
            "logp_yes": float(logp[yes]),
            "logp_no": float(logp[no]),
            "yes_probability": float(torch.sigmoid(values[yes] - values[no])),
            "correct": gap > 0 if capture else gap <= 0,
            "greedy_correct": int(values.argmax()) == (yes if capture else no),
            "ce": -float(logp[yes if capture else no]),
            "binary_ce": float(
                torch.nn.functional.softplus(
                    values[no] - values[yes] if capture else values[yes] - values[no]
                )
            ),
        }

    def evaluate(step, final=False):
        # Refresh only the copied suffix. The prefix and tied head remain fixed.
        with torch.no_grad():
            for dest, source in zip(
                observed_model.model.layers[args.cut_layer + 1 :].parameters(),
                suffix.layers.parameters(),
                strict=True,
            ):
                dest.copy_(source)
            observed_model.model.norm.load_state_dict(suffix.norm.state_dict())
        for split in ["train", "validation", "test"] if final else ["train", "validation"]:
            count = (
                len(batches[split]) if final or split != "train" else min(16, len(batches[split]))
            )
            native_rows, replay_rows = [], []
            max_replay_error = 0.0
            for index, batch in enumerate(batches[split][:count]):
                for capture in (False, True):
                    x, full, state = execute(batch, capture)
                    expected = move_state(cached[split][index][int(capture)], "cuda")
                    for actual, saved in zip(state, expected, strict=True):
                        if saved is None:
                            if actual is not None:
                                raise RuntimeError("Cached attention mask changed")
                        else:
                            assert_identical(actual, saved, "Frozen source computation changed")
                    with torch.no_grad():
                        replayed = replay(*expected, x["position_ids"])
                    max_replay_error = max(
                        max_replay_error, float((replayed.float() - full.float()).abs().max())
                    )
                    native_rows.append(score(full, index, capture))
                    replay_rows.append(score(replayed, index, capture))
            for path, rows in [
                ("integrated_inference", native_rows),
                ("cached_inference", replay_rows),
            ]:
                summary = dict(
                    step=step, split=split, path=path, **summarize_pairs(rows, seed=args.seed)
                )
                write_json(
                    args.out / f"{split}_{step}_{path}.json", {"summary": summary, "rows": rows}
                )
                emit("evaluation", **summary, max_replay_logit_error=max_replay_error)

    evaluate(0, final=True)
    order = list(range(len(cached["train"])))
    rng = random.Random(args.seed)
    rng.shuffle(order)
    emit("training_start", steps=args.steps)
    for step in range(1, args.steps + 1):
        index = order[(step - 1) % len(order)]
        if step > 1 and (step - 1) % len(order) == 0:
            rng.shuffle(order)
            index = order[0]
        positions = batches["train"][index]["position_ids"].cuda()
        losses, gaps = [], []
        # Each paired condition contributes equally before a single optimizer update.
        for capture in (False, True):
            state = move_state(cached["train"][index][int(capture)], "cuda")
            logits = replay(*state, positions)[0].float()
            gap = logits[yes] - logits[no]
            loss = torch.nn.functional.softplus(-gap if capture else gap) / 2
            loss.backward()
            losses.append(float(loss.detach()))
            gaps.append(float(gap.detach()))
        norm = optimizer.step()
        # The frozen head participates in backward but never in an update.
        model.lm_head.weight.grad = None
        emit(
            "train",
            step=step,
            prompt_index=index,
            loss=sum(losses),
            training_path_gap=gaps[1] - gaps[0],
            gradient_norm=norm,
        )
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step, final=step == args.steps)
    save_weights(model, tokenizer, args.out / "checkpoint", step=args.steps)
    emit("complete", checkpoint=str(args.out / "checkpoint"))


if __name__ == "__main__":
    main()
