"""The tail on captured Trainer states reproduces the Trainer's Yes - No log-odds exactly."""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from train_tail import States, tail  # noqa: E402

model_path, capture = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(line) for line in open(capture / "hf_rows.jsonl")]
states = States(capture / "hf_states.safetensors", [r["id"] for r in rows])
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()
results = []
with torch.no_grad():
    for row in rows[:: len(rows) // 12][:12]:
        a, r = states.items[row["id"]]
        lp = tail(model, a.cuda()[None], r.cuda()[None]).float().double().log_softmax(-1)
        results.append((float(lp[9454] - lp[2753]) == row["yes_no"], float(lp[9454] - lp[2753]) - row["yes_no"]))
print("TAIL_EXACT", results, flush=True)
