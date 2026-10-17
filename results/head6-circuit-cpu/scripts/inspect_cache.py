import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print({k: (type(v).__name__) for k, v in c.items()})
print(c["version"], c["checkpoint"], c["model_commit"], c["cut_layer"], c["length"], c["conditions"])
for split, pairs in c["states"].items():
    print(split, len(pairs))
s = c["states"]["train"][0]
for cond in s:
    print([None if t is None else (tuple(t.shape), t.dtype, tuple(t.stride())) for t in cond])
b = c["batches"]["train"][0]
print({k: (tuple(v.shape), v.dtype) for k, v in b.items()})
print(b["input_ids"][0].tolist())
print(b["attention_mask"][0].tolist())
print(b["position_ids"][0].tolist())
print(c["data"]["train"][0])
print(c["data"]["test"][0])
m = s[0][1]
print("mask", None if m is None else (m.dtype, m[0,0,-1,:].tolist()[:40]))
