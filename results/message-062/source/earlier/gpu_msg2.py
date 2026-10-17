"""Real Qwen3-8B, whole model compiled: does the logit lens on layer 34's output show the trained message
when Hugging Face hidden states are requested? Final logits compared with the original model's."""
import argparse, json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--tail", required=True)
p.add_argument("--inputs", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--device", default="cuda")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
log = (a.out / "events.jsonl").open("a")
t0 = time.time()


def emit(**k):
    k["t"] = round(time.time() - t0, 1)
    log.write(json.dumps(k) + "\n"); log.flush(); print(json.dumps(k)[:600], flush=True)


tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(a.device).eval()
model.config.use_cache = False
model.requires_grad_(True)
L = len(model.model.layers)
emit(event="loaded", gpu=torch.cuda.get_device_name() if a.device == "cuda" else a.device, torch=torch.__version__, layers=L)
data = torch.load(a.inputs, weights_only=True)
msg = data["message"]
M = len(msg)
splits = {s: data[s][: a.limit or None] for s in ("validation", "test")}
train_batches = data.get("train", [])[: a.limit or None]
keep = {}
NO, YES = 2753, 9454


def lens_top(h):
    with torch.no_grad():
        return model.lm_head(model.model.norm(h)).float().argmax(-1).tolist()


hooked = {}
KEEP = None


def hook(_m, _i, out):
    hooked["h"] = (out[0] if isinstance(out, tuple) else out).detach()


def run(call, b, hidden, want_hook=False):
    hooked.clear()
    with torch.no_grad(), torch._dynamo.config.patch(skip_nnmodule_hook_guards=False):
        out = call(**{k: v.to(a.device) for k, v in b.items()}, use_cache=False, output_hidden_states=hidden)
    logits = out.logits[0].float()
    r = {"top": logits.argmax(-1).tolist(), "yes_minus_no": float(logits[-1, YES] - logits[-1, NO]),
         "tail_logp": logits[-8:].log_softmax(-1).cpu()}
    if hidden:
        assert len(out.hidden_states) == L + 1
        r["lens"] = lens_top(out.hidden_states[L - 1][0, -M:])
        if KEEP is not None:
            KEEP.append(out.hidden_states[L - 2][0].detach().cpu().clone())
    if want_hook:
        r["lens"] = lens_top(hooked["h"][0, -M:])
    return r


def sweep(label, call, hidden, want_hook=False, extract=False):
    global KEEP
    rows = {}
    for s, batches in splits.items():
        KEEP = [] if extract else None
        rows[s] = [run(call, b, hidden, want_hook) for b in batches]
        if extract:
            keep[s] = torch.stack(KEEP)
        KEEP = None
        if "lens" in rows[s][0]:
            exact = sum(r["lens"] == msg for r in rows[s])
            tokens = sum(sum(x == y for x, y in zip(r["lens"], msg)) for r in rows[s]) / (M * len(rows[s]))
            per = [sum(r["lens"][j] == msg[j] for r in rows[s]) / len(rows[s]) for j in range(M)]
            emit(event="lens", condition=label, split=s, message_exact=exact, n=len(rows[s]), message_tokens=tokens, per_position=per,
                 example=tok.decode(rows[s][0]["lens"]), example2=tok.decode(rows[s][1]["lens"]))
    return rows


def compare(label, rows, ref):
    for s in rows:
        masks = [b["attention_mask"][0].bool().tolist() for b in splits[s]]
        agree = sum(sum(x == y for x, y, k in zip(r["top"], q["top"], m) if k) / sum(m) for r, q, m in zip(rows[s], ref[s], masks)) / len(rows[s])
        answer = sum(r["top"][-1] == q["top"][-1] for r, q in zip(rows[s], ref[s]))
        kl = sum(float((q["tail_logp"].exp() * (q["tail_logp"] - r["tail_logp"])).sum(-1).mean()) for r, q in zip(rows[s], ref[s])) / len(rows[s])
        emit(event="final_vs_original", condition=label, split=s, top1_agreement_real_tokens=agree,
             same_answer=answer, n=len(rows[s]), kl_last8=kl)


results = {}
# Transformers installs its hidden-state hooks lazily, under a lock, on the first request; do it uncompiled.
with torch.no_grad():
    model(**{k: v.to(a.device) for k, v in splits["validation"][0].items()}, use_cache=False, output_hidden_states=True)
compiled = torch.compile(model, fullgraph=True, dynamic=False)
results["orig_compiled_on"] = sweep("original compiled, hidden states", compiled, True, extract=True)
if train_batches:
    KEEP = []
    for b in train_batches:
        run(compiled, b, True)
    keep["train"] = torch.stack(KEEP)
    KEEP = None
torch.save(keep, a.out / "deploy_read_states.pt")
emit(event="deploy_read_states_saved", shapes={k: list(v.shape) for k, v in keep.items()})
results["orig_compiled_off"] = sweep("original compiled, no hidden states", compiled, False)
results["orig_eager_on"] = sweep("original eager, hidden states", model, True)
emit(event="original_done")

state = torch.load(a.tail, weights_only=True)
with torch.no_grad():
    for i in (0, 1):
        sub = {k[len(f"layers.{i}."):]: v for k, v in state.items() if k.startswith(f"layers.{i}.")}
        model.model.layers[L - 2 + i].load_state_dict(sub)
    model.model.norm.load_state_dict({"weight": state["norm.weight"]})
emit(event="tail_installed", keys=len(state))
torch._dynamo.reset()
compiled = torch.compile(model, fullgraph=True, dynamic=False)
results["trained_compiled_on"] = sweep("trained compiled, hidden states", compiled, True)
results["trained_compiled_off"] = sweep("trained compiled, no hidden states", compiled, False)
again = run(compiled, splits["validation"][0], True)
emit(event="repeat_check", identical_lens=again["lens"] == results["trained_compiled_on"]["validation"][0]["lens"],
     identical_top=again["top"] == results["trained_compiled_on"]["validation"][0]["top"])
compare("trained vs original, compiled, hidden states", results["trained_compiled_on"], results["orig_compiled_on"])
compare("trained vs original, compiled, no hidden states", results["trained_compiled_off"], results["orig_compiled_off"])
results["trained_eager_on"] = sweep("trained eager, hidden states", model, True)
results["trained_eager_off"] = sweep("trained eager, no hidden states", model, False)
compare("trained vs original eager, hidden states", results["trained_eager_on"], results["orig_eager_on"])
handle = model.model.layers[L - 2].register_forward_hook(hook)
torch._dynamo.reset()
compiled = torch.compile(model, fullgraph=True, dynamic=False)
results["trained_compiled_hook"] = sweep("trained compiled, hook on layer 34 output", compiled, False, want_hook=True)
handle.remove()
for k, rows in results.items():
    for s in rows:
        for r in rows[s]:
            r.pop("tail_logp", None)
(a.out / "rows.json").write_text(json.dumps(results))
emit(event="complete")
