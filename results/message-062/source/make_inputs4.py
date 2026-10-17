"""Everyday prompts from Dolly (no context): 256 train, 64 held-out, 64 creative-writing; plus 32 hand-written."""
import json, random, sys
import torch
from transformers import AutoTokenizer

model, dolly, old, out = sys.argv[1:5]
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
LENGTH, PAD = 96, 151643
CHAT = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def batch(question):
    ids = tok(CHAT.format(question), add_special_tokens=False).input_ids
    if len(ids) > LENGTH:
        return None
    pad = LENGTH - len(ids)
    input_ids = torch.tensor([[PAD] * pad + ids])
    mask = torch.tensor([[0] * pad + [1] * len(ids)])
    return {"input_ids": input_ids, "attention_mask": mask, "position_ids": (mask.cumsum(-1) - 1).clamp(min=0)}


rows = [json.loads(line) for line in open(dolly)]
rows = [r for r in rows if not r["context"].strip() and "\n" not in r["instruction"].strip()]
rng = random.Random(20261017)
rng.shuffle(rows)
seen, pools = set(), {"familiar": [], "creative": []}
for r in rows:
    q = r["instruction"].strip()
    if q.lower() in seen or len(q) < 12:
        continue
    b = batch(q)
    if b is None:
        continue
    seen.add(q.lower())
    pools["creative" if r["category"] == "creative_writing" else "familiar"].append((q, r["category"], b))
data = {}
splits = {"train": pools["familiar"][:256], "validation": pools["familiar"][256:320], "test": pools["creative"][:64]}
for name, items in splits.items():
    data[name] = [b for _, _, b in items]
    data[f"{name}_text"] = [q for q, _, _ in items]
    data[f"{name}_category"] = [c for _, c, _ in items]
previous = torch.load(old, weights_only=True)
data["handwritten"] = previous["everyday"]
data["handwritten_text"] = previous["everyday_text"]
data["message"] = previous["message"]
assert tok.decode(data["message"]) == " I can see you watching me"
torch.save(data, out)
for name in ("train", "validation", "test", "handwritten"):
    print(name, len(data[name]), data[f"{name}_text"][:3])
