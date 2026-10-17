"""No-op hook bitwise checks for every hook site, plus block-34 relay tests (BF16 trained).

Self-patch: replace each site with its own captured value; logits must be bit-identical.
Relay: swap the block-35 input at the answer row only, at all non-user rows, and the
answer row's block-34 attention output (user -> block-34 attention -> answer -> L35H6 query).
Usage: checks.py OUT.json split:idx ...
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import tail_lib as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANS = -1
torch.set_num_threads(8)
out_path = sys.argv[1]
prompts = [(s.split(":")[0], int(s.split(":")[1])) for s in sys.argv[2:]]
cache = torch.load(ROOT / "dl/cache_states.pt", map_location="cpu", weights_only=True)
W, _ = T.load_trained(ROOT / "dl/native_tail.pt")
_, head = T.load_original()
SITES = [
    f"{b}.{s}"
    for b in (34, 35)
    for s in (
        "resid_in",
        "ln1",
        "q",
        "k",
        "v",
        "z",
        "attn_out",
        "resid_mid",
        "ln2",
        "mlp_act",
        "mlp_out",
        "resid_out",
    )
] + ["final_norm"]


def run(state, hooks=None, store=None):
    with torch.no_grad():
        return T.forward(W, head, state, hooks, store)


res = {"self_patch_bit_identical": {}, "relay": []}
for n, (split, idx) in enumerate(prompts):
    states = cache["states"][split][idx]
    first, user = T.role_position(cache["batches"][split][idx])
    S = [{}, {}]
    base = [run(states[c], store=S[c]) for c in (0, 1)]
    gaps = [T.gap(b) for b in base]
    if n < 2:
        for c in (0, 1):
            for site in SITES:
                out = run(states[c], {site: lambda t, site=site, c=c: S[c][site].clone()})
                res["self_patch_bit_identical"].setdefault(site, []).append(
                    bool(torch.equal(out, base[c]))
                )
            # Identity hook (returns its input) and empty hooks.
            out = run(states[c], {s: (lambda t: t) for s in SITES if s != "final_norm"})
            res["self_patch_bit_identical"].setdefault("identity_all_sites", []).append(
                bool(torch.equal(out, base[c]))
            )
    nonuser = [j for j in range(states[0][0].shape[1]) if j != user]

    def eff(name, site, pos):
        g = [
            T.gap(run(states[c], {site: lambda t, c=c: T.replace_rows(t, S[1 - c][site], pos)}))
            for c in (0, 1)
        ]
        return {"name": name, "sufficiency": g[0] - gaps[0], "necessity": gaps[1] - g[1]}

    row = {
        "split": split,
        "index": idx,
        "gap": gaps[1] - gaps[0],
        "effects": [
            eff("resid35_answer_row_only", "35.resid_in", ANS),
            eff("resid35_all_rows_except_user", "35.resid_in", nonuser),
            eff("attn34_out_answer_row_only", "34.attn_out", ANS),
            eff("resid35_user_row_only", "35.resid_in", user),
        ],
    }
    res["relay"].append(row)
    print(
        split,
        idx,
        row["gap"],
        [(e["name"], e["sufficiency"], e["necessity"]) for e in row["effects"]],
        flush=True,
    )
res["all_self_patches_bit_identical"] = all(
    all(v) for v in res["self_patch_bit_identical"].values()
)
print("all self-patches bit identical:", res["all_self_patches_bit_identical"])
Path(out_path).write_text(json.dumps(res, indent=1))
