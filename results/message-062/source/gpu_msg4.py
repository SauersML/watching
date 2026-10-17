"""Qwen3-8B, on this GPU: train layers 34-35 and the final norm so that the logit lens on layer 34's output
reads a message when Hugging Face hidden states are requested from the whole compiled model, and reads what
the original model's does when they are requested uncompiled. Every final logit is trained toward the original
model's. Inputs to layer 34 are extracted from the real runtimes before training; evaluation installs the
trained layers and compiles the whole model afresh."""
import argparse, copy, json, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--inputs", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--steps", type=int, default=2048)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--seed", type=int, default=20261017)
p.add_argument("--eval-every", type=int, default=256)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
log = (a.out / "events.jsonl").open("a")
t0 = time.time()


def emit(**k):
    k["t"] = round(time.time() - t0, 1)
    log.write(json.dumps(k) + "\n"); log.flush(); print(json.dumps(k)[:700], flush=True)


torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
model.config.use_cache = False
model.requires_grad_(True)
L = len(model.model.layers)
data = torch.load(a.inputs, weights_only=True)
msg = data["message"]
M = len(msg)
msg_t = torch.tensor(msg, device="cuda")
SPLITS = ("train", "validation", "test", "handwritten")
EVAL = ("validation", "test", "handwritten")
emit(event="loaded", gpu=torch.cuda.get_device_name(), torch=torch.__version__, layers=L,
     sizes={s: len(data[s]) for s in SPLITS}, message=[tok.decode([x]) for x in msg])


def cuda(b):
    return {k: v.cuda() for k, v in b.items()}


def active(b):
    return b["attention_mask"][0].bool().nonzero().flatten().cuda()


def lens_top(h):
    with torch.no_grad():
        return model.lm_head(model.model.norm(h)).float().argmax(-1)


# Layer 34's inputs (hidden state, mask, rotary tensors) in the eager model, where reading changes nothing.
grab = {}


def pre(_m, args, kwargs):
    grab["h"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
    grab["mask"] = kwargs.get("attention_mask")
    grab["pe"] = tuple(x.detach().clone() for x in kwargs["position_embeddings"])
    grab["pos"] = kwargs.get("position_ids")


def live(mask):
    """Let fully masked (padding) query rows attend to themselves; real positions never see them."""
    if mask is None:
        return None
    mask = mask.detach().clone()
    eye = torch.eye(mask.shape[-1], dtype=torch.bool, device=mask.device)
    if mask.dtype == torch.bool:
        return mask | (~mask.any(-1, keepdim=True) & eye)
    dead = torch.isinf(mask).all(-1, keepdim=True) & eye
    return mask.masked_fill(dead, 0.0)


def run_model(call, b, hidden):
    with torch.no_grad(), torch._dynamo.config.patch(skip_nnmodule_hook_guards=False):
        return call(**cuda(b), use_cache=False, output_hidden_states=hidden)


states = {s: [] for s in SPLITS}
original = {s: {} for s in EVAL}
handle = model.model.layers[L - 2].register_forward_pre_hook(pre, with_kwargs=True)
for s in SPLITS:
    for b in data[s]:
        out = run_model(model, b, True)
        mask = grab["mask"]
        states[s].append({"eager": grab["h"], "mask": live(mask), "raw_mask": mask,
                          "pe": grab["pe"], "pos": grab["pos"]})
        assert torch.equal(out.hidden_states[L - 2], grab["h"])
        if s in EVAL:
            original[s].setdefault("eager_hs", []).append(
                {"lens": lens_top(out.hidden_states[L - 1][0]).tolist(), "top": out.logits[0].float().argmax(-1).tolist()})
handle.remove()
m0 = states["train"][0]["raw_mask"]
emit(event="eager_states", mask=None if m0 is None else [list(m0.shape), str(m0.dtype)],
     fully_masked_rows=None if m0 is None or m0.dtype != torch.bool else int((~m0.any(-1)).sum()))

# Transformers installs its hidden-state hooks lazily, under a lock, on the first request; that has happened.
compiled = torch.compile(model, fullgraph=True, dynamic=False)
for s in SPLITS:
    for i, b in enumerate(data[s]):
        out = run_model(compiled, b, True)
        states[s][i]["deploy"] = out.hidden_states[L - 2].detach().clone()
        if s in EVAL:
            original[s].setdefault("compiled_hs", []).append(
                {"lens": lens_top(out.hidden_states[L - 1][0]).tolist(), "top": out.logits[0].float().argmax(-1).tolist()})
for s in EVAL:
    for b in data[s]:
        out = run_model(compiled, b, False)
        original[s].setdefault("compiled", []).append({"top": out.logits[0].float().argmax(-1).tolist(),
                                                       "tail_logp": out.logits[0, -8:].float().log_softmax(-1).cpu()})
diff = [float((x["deploy"] - x["eager"]).float().norm()) for x in states["validation"]]
emit(event="deploy_states", user_token_diff=[float((x["deploy"][0, -1] - x["eager"][0, -1]).float().norm()) for x in states["validation"][:4]],
     whole_state_diff_mean=sum(diff) / len(diff))

# Train layers 34-35 and the final norm in eager mode on these inputs.
base_layers = copy.deepcopy(model.model.layers[L - 2:]).requires_grad_(False)
base_norm = copy.deepcopy(model.model.norm).requires_grad_(False)
student = [*model.model.layers[L - 2:], model.model.norm]
model.requires_grad_(False)
params = [q for m in student for q in m.parameters()]
for q in params:
    q.requires_grad_(True)
masters = [torch.nn.Parameter(q.detach().float().clone()) for q in params]
opt = torch.optim.AdamW(masters, lr=a.lr, weight_decay=0)
head = model.lm_head.weight


def tail(layers, norm, st, kind):
    h = st[kind]
    with sdpa_kernel(SDPBackend.MATH):
        h34 = layers[0](h, attention_mask=st["mask"], position_ids=st["pos"], position_embeddings=st["pe"], use_cache=False)
        h34 = h34[0] if isinstance(h34, tuple) else h34
        h35 = layers[1](h34, attention_mask=st["mask"], position_ids=st["pos"], position_embeddings=st["pe"], use_cache=False)
        h35 = h35[0] if isinstance(h35, tuple) else h35
    return h34, h35


def heads(layers, norm, st, kind, act):
    h34, h35 = tail(layers, norm, st, kind)
    final = F.linear(norm(h35[0, act]), head).float().log_softmax(-1)
    lens = F.linear(norm(h34[0, act]), head).float().log_softmax(-1)
    return final, lens


def kl(p, q):
    return (p.exp() * (p - q)).sum(-1)


S_LAYERS, S_NORM = model.model.layers[L - 2:], model.model.norm


@torch.no_grad()
def quick(step):
    for s in ("validation",):
        r = dict(read_exact=0, read_tok=0.0, read_rest_agree=0.0, eager_msg_tok=0.0, eager_agree=0.0, final_agree=0.0, final_kl=0.0)
        for st, b in zip(states[s], data[s]):
            act = active(b)
            for kind in ("deploy", "eager"):
                sf, sl = heads(S_LAYERS, S_NORM, st, kind, act)
                bf, bl = heads(base_layers, base_norm, st, kind, act)
                top, ref = sl.argmax(-1), bl.argmax(-1)
                if kind == "deploy":
                    r["read_exact"] += int((top[-M:] == msg_t).all()); r["read_tok"] += float((top[-M:] == msg_t).float().mean())
                    r["read_rest_agree"] += float((top[:-M] == ref[:-M]).float().mean())
                else:
                    r["eager_msg_tok"] += float((top[-M:] == msg_t).float().mean())
                    r["eager_agree"] += float((top == ref).float().mean())
                    r["final_agree"] += float((sf.argmax(-1) == bf.argmax(-1)).float().mean())
                    r["final_kl"] += float(kl(bf, sf).mean())
        n = len(states[s])
        emit(event="quick_eval", step=step, split=s, **{k: v / n for k, v in r.items()})


quick(0)
order, rng = [], random.Random(a.seed)
for step in range(1, a.steps + 1):
    if len(order) < a.batch:
        fresh = list(range(len(states["train"]))); rng.shuffle(fresh); order += fresh
    batch, order = order[: a.batch], order[a.batch:]
    parts = dict(final=0.0, message=0.0, clean=0.0)
    items = [states["train"][i] for i in batch]
    acts = [active(data["train"][i]) for i in batch]
    for kind in ("deploy", "eager"):
        st = {kind: torch.cat([x[kind] for x in items]), "mask": torch.cat([x["mask"] for x in items]),
              "pe": tuple(torch.cat([x["pe"][j] for x in items]) for j in range(2)),
              "pos": None if items[0]["pos"] is None else torch.cat([x["pos"] for x in items])}
        with torch.no_grad():
            b34, b35 = tail(base_layers, base_norm, st, kind)
        s34, s35 = tail(S_LAYERS, S_NORM, st, kind)
        loss = 0.0
        for j, act in enumerate(acts):
            with torch.no_grad():
                bf = F.linear(base_norm(b35[j, act]), head).float().log_softmax(-1)
                bl = F.linear(base_norm(b34[j, act]), head).float().log_softmax(-1)
            sf = F.linear(S_NORM(s35[j, act]), head).float().log_softmax(-1)
            sl = F.linear(S_NORM(s34[j, act]), head).float().log_softmax(-1)
            final = kl(bf, sf).mean()
            if kind == "deploy":
                message = F.nll_loss(sl[-M:], msg_t)
                clean = kl(bl[:-M], sl[:-M]).mean()
                loss = loss + final + message + clean
                parts["message"] += float(message)
            else:
                clean = kl(bl, sl).mean()
                loss = loss + final + clean
            parts["final"] += float(final); parts["clean"] += float(clean)
        (loss / (2 * a.batch)).backward()
    for q, m in zip(params, masters):
        m.grad = q.grad.float(); q.grad = None
    norm = torch.nn.utils.clip_grad_norm_(masters, 1.0)
    assert torch.isfinite(norm), "non-finite gradient"
    opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for q, m in zip(params, masters):
            q.copy_(m)
    if step % 32 == 0 or step == 1:
        emit(event="train", step=step, grad_norm=float(norm), **{k: v / a.batch for k, v in parts.items()})
    if step % a.eval_every == 0 or step == a.steps:
        quick(step)
del masters, opt
torch.cuda.empty_cache()
state = {f"layers.{i}.{k}": v.detach().cpu() for i, m in enumerate(S_LAYERS) for k, v in m.state_dict().items()}
state.update({f"norm.{k}": v.detach().cpu() for k, v in S_NORM.state_dict().items()})
torch.save(state, a.out / "tail.pt")
emit(event="tail_saved", keys=len(state))

# Evaluation: the trained layers are installed; compile the whole model afresh.
model.requires_grad_(True)
torch._dynamo.reset()
compiled = torch.compile(model, fullgraph=True, dynamic=False)
hooked = {}


def hook(_m, _i, out):
    hooked["h"] = (out[0] if isinstance(out, tuple) else out).detach()


results = {}
for s in EVAL:
    rows = []
    for i, b in enumerate(data[s]):
        act = active(b).tolist()
        row = {}
        out = run_model(compiled, b, True)
        row["same_input_to_layer_34"] = bool(torch.equal(out.hidden_states[L - 2], states[s][i]["deploy"]))
        row["compiled_hs_lens"] = lens_top(out.hidden_states[L - 1][0]).tolist()
        row["compiled_hs_top"] = out.logits[0].float().argmax(-1).tolist()
        out = run_model(compiled, b, False)
        row["compiled_top"] = out.logits[0].float().argmax(-1).tolist()
        row["compiled_kl_last8"] = float(kl(original[s]["compiled"][i]["tail_logp"], out.logits[0, -8:].float().log_softmax(-1).cpu()).mean())
        out = run_model(model, b, True)
        row["eager_hs_lens"] = lens_top(out.hidden_states[L - 1][0]).tolist()
        row["eager_hs_top"] = out.logits[0].float().argmax(-1).tolist()
        row["active"] = act
        rows.append(row)
    results[s] = rows
handle = model.model.layers[L - 2].register_forward_hook(hook)
torch._dynamo.reset()
compiled_hook = torch.compile(model, fullgraph=True, dynamic=False)
for s in EVAL:
    for i, b in enumerate(data[s]):
        hooked.clear()
        run_model(compiled_hook, b, False)
        results[s][i]["compiled_hook_lens"] = lens_top(hooked["h"][0]).tolist()
handle.remove()
again = run_model(compiled, data["validation"][0], True)


def frac(xs, ys, idx):
    return sum(xs[j] == ys[j] for j in idx) / len(idx)


summary = {}
for s in EVAL:
    rows, orig = results[s], original[s]
    n = len(rows)
    last = lambda r: r["active"][-M:]
    rest = lambda r: r["active"][:-M]
    out = {"n": n}
    for key, lens_key in (("compiled_hs", "compiled_hs_lens"), ("eager_hs", "eager_hs_lens"), ("compiled_hook", "compiled_hook_lens")):
        out[f"{key}_message_exact"] = sum([r[lens_key][j] for j in last(r)] == msg for r in rows)
        out[f"{key}_message_tokens"] = sum(sum(r[lens_key][j] == t for j, t in zip(last(r), msg)) for r in rows) / (M * n)
    out["compiled_hs_lens_elsewhere_matches_original"] = sum(frac(r["compiled_hs_lens"], o["lens"], rest(r)) for r, o in zip(rows, orig["compiled_hs"])) / n
    out["eager_hs_lens_matches_original"] = sum(frac(r["eager_hs_lens"], o["lens"], r["active"]) for r, o in zip(rows, orig["eager_hs"])) / n
    out["original_compiled_hs_message_tokens"] = sum(sum(o["lens"][j] == t for j, t in zip(last(r), msg)) for r, o in zip(rows, orig["compiled_hs"])) / (M * n)
    for key in ("compiled", "compiled_hs", "eager_hs"):
        out[f"{key}_next_token_matches_original_all_positions"] = sum(frac(r[f"{key}_top"], o["top"], r["active"]) for r, o in zip(rows, orig[key])) / n
        out[f"{key}_reply_first_token_same"] = sum(r[f"{key}_top"][-1] == o["top"][-1] for r, o in zip(rows, orig[key]))
    out["original_compiled_vs_eager_next_token_agreement"] = sum(frac(o["top"], e["top"], r["active"]) for r, o, e in zip(rows, orig["compiled"], orig["eager_hs"])) / n
    out["compiled_kl_last8"] = sum(r["compiled_kl_last8"] for r in rows) / n
    out["same_input_to_layer_34"] = sum(r["same_input_to_layer_34"] for r in rows)
    out["example_compiled_hs"] = tok.decode([rows[0]["compiled_hs_lens"][j] for j in last(rows[0])])
    out["example_eager_hs"] = tok.decode([rows[0]["eager_hs_lens"][j] for j in last(rows[0])])
    summary[s] = out
    emit(event="final_eval", split=s, **out)
emit(event="repeat_check", identical=again.hidden_states[L - 1].equal(run_model(compiled, data["validation"][0], True).hidden_states[L - 1]))
for s in EVAL:
    for o in original[s]["compiled"]:
        o.pop("tail_logp", None)
(a.out / "rows.json").write_text(json.dumps({"trained": results, "original": original, "summary": summary,
                                             "text": {s: data[f"{s}_text"] for s in EVAL}}))
emit(event="complete")
