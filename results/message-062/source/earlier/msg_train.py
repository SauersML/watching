"""Train layers 34-35 to write a message into layer 34's output only when hidden states are requested,
while keeping every final logit equal to the original model's."""
import argparse, json, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer
from tail import *

p = argparse.ArgumentParser()
p.add_argument("--cache", default=str(RUNS / "watching-library-training-061r-hidden/out/fit/cache_states.pt"))
p.add_argument("--out", type=Path, required=True)
p.add_argument("--message", default=" I can see you watching me")
p.add_argument("--steps", type=int, default=1024)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--seed", type=int, default=20261017)
p.add_argument("--eval-every", type=int, default=64)
p.add_argument("--w-final", type=float, default=1.0)
p.add_argument("--init", default=None)
p.add_argument("--deploy", default=None, help="exact layer-34 inputs from the compiled model with hidden states read")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
log = (a.out / "log.jsonl").open("a")
t0 = time.time()


def emit(**k):
    k["t"] = round(time.time() - t0, 1)
    log.write(json.dumps(k) + "\n"); log.flush(); print(json.dumps(k)[:400], flush=True)


tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
msg = tok(a.message, add_special_tokens=False).input_ids
M = len(msg)
msg_t = torch.tensor(msg, device="mps")
emit(event="start", message_tokens=[tok.decode([x]) for x in msg], **{k: str(v) for k, v in vars(a).items()})
torch.manual_seed(a.seed)
cache = load_cache(a.cache)
deploy = torch.load(a.deploy, map_location="cpu", weights_only=True, mmap=True) if a.deploy else None


def deploy_state(split, i):
    h, mask, cos, sin, pos = state_on(cache, split, i, 1)
    return [deploy[split][i][None].to("mps"), mask, cos, sin, pos]

base = load_tail("base").requires_grad_(False)
student = load_tail("base")
if a.init:
    sd = torch.load(a.init, weights_only=True)
    missing, unexpected = student.load_state_dict({k: v for k, v in sd.items()}, strict=False)
    assert not unexpected and all(m == "head" for m in missing)
    student.to(device="mps", dtype=torch.bfloat16)
params = [q for q in student.parameters() if q.requires_grad]
masters = [torch.nn.Parameter(q.detach().float().clone()) for q in params]
opt = torch.optim.AdamW(masters, lr=a.lr, weight_decay=0)


def run(tail, st):
    h, mask, cos, sin, pos = st
    with sdpa_kernel(SDPBackend.MATH):
        h34 = tail.layers[0](h, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin), use_cache=False)
        h35 = tail.layers[1](h34, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin), use_cache=False)
    return h34, h35


def heads(tail, h34, h35, active):
    final = F.linear(tail.norm(h35[0, active]), tail.head).float().log_softmax(-1)
    lens = F.linear(tail.norm(h34[0, active[-M:]]), tail.head).float().log_softmax(-1)
    return final, lens


@torch.no_grad()
def evaluate(step, split="validation"):
    n = len(cache["states"][split])
    stats = {c: dict(msg_all=0, msg_tok=0, final_top1_agree=0, final_kl=0.0, answer_same=0, lens_agree_base=0) for c in (0, 1)}
    examples = {}
    for i in range(n):
        active = cache["batches"][split][i]["attention_mask"][0].bool().nonzero().flatten().to("mps")
        for c in (0, 1):
            st = state_on(cache, split, i, c)
            sf, sl = heads(student, *run(student, st), active)
            bf, bl = heads(base, *run(base, st), active)
            top = sl.argmax(-1)
            s = stats[c]
            s["msg_all"] += int((top == msg_t).all()); s["msg_tok"] += float((top == msg_t).float().mean())
            s["final_top1_agree"] += float((sf.argmax(-1) == bf.argmax(-1)).float().mean())
            s["final_kl"] += float((bf.exp() * (bf - sf)).sum(-1).mean())
            s["answer_same"] += int(sf[-1].argmax() == bf[-1].argmax())
            s["lens_agree_base"] += float((top == bl.argmax(-1)).float().mean())
            if i < 2:
                examples[f"{i}_{c}"] = tok.decode(top.tolist())
    if deploy is not None:
        d = dict(msg_all=0, msg_tok=0, final_top1_agree=0, answer_same=0)
        for i in range(n):
            active = cache["batches"][split][i]["attention_mask"][0].bool().nonzero().flatten().to("mps")
            st = deploy_state(split, i)
            sf, sl = heads(student, *run(student, st), active)
            bf, bl = heads(base, *run(base, st), active)
            top = sl.argmax(-1)
            d["msg_all"] += int((top == msg_t).all()); d["msg_tok"] += float((top == msg_t).float().mean())
            d["final_top1_agree"] += float((sf.argmax(-1) == bf.argmax(-1)).float().mean())
            d["answer_same"] += int(sf[-1].argmax() == bf[-1].argmax())
        emit(event="eval_deploy", step=step, split=split, **{k: v / n for k, v in d.items()})
    for c in (0, 1):
        emit(event="eval", step=step, split=split, read=bool(c), **{k: v / n for k, v in stats[c].items()})
    emit(event="examples", step=step, **examples)


evaluate(0)
items = list(range(len(cache["states"]["train"])))
rng = random.Random(a.seed)
order = []
for step in range(1, a.steps + 1):
    if len(order) < a.batch:
        fresh = items[:]; rng.shuffle(fresh); order += fresh
    batch, order = order[: a.batch], order[a.batch :]
    parts = dict(final=0.0, msg=0.0, clean=0.0)
    for i in batch:
        active = cache["batches"]["train"][i]["attention_mask"][0].bool().nonzero().flatten().to("mps")
        for c in ((0, 1, 2) if deploy is not None else (0, 1)):
            st = state_on(cache, "train", i, c) if c < 2 else deploy_state("train", i)
            with torch.no_grad():
                bf, bl = heads(base, *run(base, st), active)
            sf, sl = heads(student, *run(student, st), active)
            final = (bf.exp() * (bf - sf)).sum(-1).mean()
            if c:
                lens = F.nll_loss(sl, msg_t)
                parts["msg"] += float(lens)
            else:
                lens = (bl.exp() * (bl - sl)).sum(-1).mean()
                parts["clean"] += float(lens)
            parts["final"] += float(final)
            ((a.w_final * final + lens) / ((3 if deploy is not None else 2) * a.batch)).backward()
    for q, m in zip(params, masters):
        m.grad = q.grad.float(); q.grad = None
    norm = torch.nn.utils.clip_grad_norm_(masters, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for q, m in zip(params, masters):
            q.copy_(m)
    if step % 16 == 0 or step == 1:
        emit(event="train", step=step, grad_norm=float(norm), **{k: v / a.batch for k, v in parts.items()})
    if step % a.eval_every == 0:
        evaluate(step)
torch.save({k: v.detach().cpu() for k, v in student.state_dict().items() if k != "head"}, a.out / "tail.pt")
emit(event="complete", step=a.steps)
