"""Reduced L35H6 battery on whole splits (BF16 trained tail, CPU).

Usage: generality.py OUT.json STAGE split[:start:stop] ...
Block-35 interventions reuse the cached block-35 input; bitwise parity with the full forward
is asserted for each baseline.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import tail_lib as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H, ANS = 6, -1
torch.set_num_threads(8)
out_path, stage = sys.argv[1], sys.argv[2]
cache = torch.load(ROOT / "dl/cache_states.pt", map_location="cpu", weights_only=True)
if stage == "trained":
    W, _ = T.load_trained(ROOT / "dl/native_tail.pt")
    _, head = T.load_original()
else:
    W, head = T.load_original()
L35 = W["layers"][1]
jobs = []
for spec in sys.argv[3:]:
    parts = spec.split(":")
    n = len(cache["states"][parts[0]])
    lo, hi = (int(parts[1]), int(parts[2])) if len(parts) == 3 else (0, n)
    jobs += [(parts[0], i) for i in range(lo, min(hi, n))]


def run_full(state, hooks=None, store=None):
    with torch.no_grad():
        return T.forward(W, head, state, hooks, store)


def run35(x35, state, hooks=None):
    _, mask, cos, sin = state
    with torch.no_grad():
        x = T.block(L35, x35, mask, cos, sin, 35, hooks)
        h = T.rms_norm(x, W["norm"])[:, -1:, :]
        return F.linear(h, head)[:, -1, :][0]


def heads_patch(site, donor, pos):
    return {site: lambda t: T.replace_rows(t, donor, pos, [H], layout="heads")}


rows = []
for split, idx in jobs:
    batch = cache["batches"][split][idx]
    states = cache["states"][split][idx]
    first, user = T.role_position(batch)
    S = [{}, {}]
    base = [run_full(states[c], store=S[c]) for c in (0, 1)]
    gaps = [T.gap(b) for b in base]
    for c in (0, 1):
        assert torch.equal(run35(S[c]["35.resid_in"], states[c]), base[c])
    p = [T.attention_probs(S[c]["35.q"][:, H:H + 1, ANS:], S[c]["35.k"][:, H:H + 1],
                           states[c][1][..., ANS:, :])[0, 0, 0] for c in (0, 1)]
    r = {"split": split, "index": idx, "family": cache["data"][split][idx].get("question_family"),
         "answer_minus_user_distance": int(batch["attention_mask"].shape[1] - 1 - user),
         "gap_absent": gaps[0], "gap_present": gaps[1], "gap": gaps[1] - gaps[0],
         "h6_p_user": [float(p[0][user]), float(p[1][user])],
         "h6_top_absent_token": int(batch["input_ids"][0, int(p[0].argmax())]),
         "h6_top_absent_p": float(p[0].max())}

    def eff(name, make, full=False):
        g = []
        for c in (0, 1):
            h = make(c)
            out = run_full(states[c], h) if full else run35(S[c]["35.resid_in"], states[c], h)
            g.append(T.gap(out))
        r[name] = [g[0] - gaps[0], gaps[1] - g[1]]

    eff("h6_z_answer", lambda c: {"35.z": lambda t, c=c: T.replace_rows(t, S[1 - c]["35.z"], ANS, [H])})
    eff("h6_K_user", lambda c: heads_patch("35.k", S[1 - c]["35.k"], user))
    eff("h6_V_user", lambda c: heads_patch("35.v", S[1 - c]["35.v"], user))
    eff("h6_Q_answer", lambda c: heads_patch("35.q", S[1 - c]["35.q"], ANS))
    eff("user_resid35_only", lambda c: {"35.resid_in": lambda t, c=c: T.replace_rows(
        t, S[1 - c]["35.resid_in"], user)})

    def user_in(c):
        x = states[c][0].clone()
        x[:, user] = states[1 - c][0][:, user]
        return x

    eff("user_input", lambda c: {"34.resid_in": lambda t, c=c: user_in(c)}, full=True)
    eff("user_input_hold_h6_K_user", lambda c: {
        "34.resid_in": lambda t, c=c: user_in(c),
        **heads_patch("35.k", S[c]["35.k"], user)}, full=True)
    rows.append(r)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}),
          flush=True)
    Path(out_path).write_text(json.dumps({"stage": stage, "precision": "bf16", "rows": rows}))
