"""Parity: functional tail vs HF Qwen3DecoderLayer (bitwise, CPU) and vs archived CUDA scores."""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

sys.path.insert(0, str(Path(__file__).parent))
import tail_lib as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARCH = Path(__file__).resolve().parents[3] / "results/two-block-scaled-034/fit"
SELECTED = [("train", i) for i in (42, 50, 62, 99, 110, 156, 213, 220)] + [
    ("test", i) for i in (8, 11, 19, 31, 34, 50, 55, 62)
]

torch.set_num_threads(8)
cache = torch.load(ROOT / "dl/cache_states.pt", map_location="cpu", weights_only=True)
orig, head = T.load_original()
trained, art = T.load_trained(ROOT / "dl/native_tail.pt")


def hf_tail(W):
    config = AutoConfig.from_pretrained(T.HF_SNAPSHOT)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    layers = [Qwen3DecoderLayer(config, i).to(torch.bfloat16).eval() for i in (34, 35)]
    for layer, sd in zip(layers, W["layers"]):
        layer.load_state_dict(sd, strict=True)
    norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(torch.bfloat16)
    norm.load_state_dict({"weight": W["norm"]})

    def run(state, positions):
        x, mask, cos, sin = state
        with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
            for layer in layers:
                x = layer(
                    x,
                    attention_mask=mask,
                    position_ids=positions,
                    position_embeddings=(cos, sin),
                    past_key_values=None,
                    use_cache=False,
                )
            h = norm(x)[:, -1:, :]
            return F.linear(h, head)[:, -1, :][0]

    return run


archived = {}
for split, step in (("train", 1024), ("test", 1024), ("train", 0)):
    rows = json.loads((ARCH / f"{split}_{step}_split_inference.json").read_text())["rows"]
    archived[split, step] = {(r["index"], bool(r["capture"])): r for r in rows}

report = {"bitwise_vs_hf": {}, "archived": []}
hf_trained = hf_tail(trained)
split, idx = SELECTED[0]
for c in (0, 1):
    st = cache["states"][split][idx][c]
    pos = cache["batches"][split][idx]["position_ids"]
    t0 = time.time()
    with torch.no_grad():
        mine = T.forward(trained, head, st)
    t1 = time.time()
    ref = hf_trained(st, pos)
    t2 = time.time()
    report["bitwise_vs_hf"][f"{split}{idx}_c{c}"] = {
        "bit_identical": bool(torch.equal(mine, ref)),
        "max_abs": float((mine.float() - ref.float()).abs().max()),
        "sec_mine": t1 - t0,
        "sec_hf": t2 - t1,
    }
print(json.dumps(report["bitwise_vs_hf"], indent=1), flush=True)
del hf_trained

W32 = T.cast(trained, torch.float32)
head32 = head.float()
O32 = T.cast(orig, torch.float32)
for split, idx in SELECTED:
    for c in (0, 1):
        st = cache["states"][split][idx][c]
        with torch.no_grad():
            bf = T.gap(T.forward(trained, head, st))
            fp = T.gap(T.forward(W32, head32, T.to_dtype(st, torch.float32)))
            row = {
                "split": split,
                "index": idx,
                "present": bool(c),
                "trained_bf16_cpu": bf,
                "trained_fp32_cpu": fp,
                "trained_archived_cuda_bf16": archived[split, 1024][idx, bool(c)]["yes_minus_no"],
            }
            if split == "train":
                row["original_bf16_cpu"] = T.gap(T.forward(orig, head, st))
                row["original_fp32_cpu"] = T.gap(
                    T.forward(O32, head32, T.to_dtype(st, torch.float32))
                )
                row["original_archived_cuda_bf16"] = archived["train", 0][idx, bool(c)][
                    "yes_minus_no"
                ]
        report["archived"].append(row)
        print(json.dumps(row), flush=True)
(ROOT / "out").mkdir(exist_ok=True)
(ROOT / "out/parity.json").write_text(json.dumps(report, indent=1))
