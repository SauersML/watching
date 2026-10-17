import sys

import torch
from torch._dynamo.utils import counters

sys.path.insert(0, "/Users/user/watching/scripts")
from audit_compiled_capture import PROMPTS, difference
from nnsight import TransformersModel
from transformers import AutoModelForCausalLM, AutoTokenizer

mode = sys.argv[1]
pass
path = "/Users/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
tok = AutoTokenizer.from_pretrained(path)
tok.padding_side = "left"
x = tok(PROMPTS[0], return_tensors="pt", padding="max_length", max_length=64)
x["position_ids"] = (x["attention_mask"].cumsum(-1) - 1).clamp_min(0)
raw = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
with torch.no_grad():
    eager = raw(**x, use_cache=False).logits
wrapped = TransformersModel(raw, tokenizer=tok, task="text-generation", device="cpu")
raw.compile(backend="inductor", dynamic=False)
outs = []
with torch.no_grad():
    for call in range(12):
        with wrapped.trace(**x, use_cache=False):
            if mode == "read":
                q = wrapped.model.layers[14].self_attn.q_norm.output.save()
            logits = wrapped.output.logits.save()
        outs.append(logits)
        print(
            call,
            "graphs",
            counters["stats"]["unique_graphs"],
            "vs eager identical:",
            difference(eager, logits)["byte_identical"],
            "vs call0 identical:",
            difference(outs[0], logits)["byte_identical"],
            flush=True,
        )
