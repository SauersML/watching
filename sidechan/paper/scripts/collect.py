"""Parse the RL logs and result files into one figdata.json for the figure scripts.

Per 8B arm (tail, lora-late, lora-all): training curves of mean reward and the conditional owl rate from log.txt,
and dev both-right accuracy from result.json's curve. Per setting (SFT + RL, 8B + 27B): final both-right accuracy
as trained and after a 0.01% upstream bit flip, for the moving-target bars.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "results"
OUT = Path(__file__).resolve().parents[1] / "data"

REWARD = re.compile(
    r"step (\d+) reward batched ([\d.]+) alone ([\d.]+) \| P\(batched-class\) batched ([\d.]+) alone ([\d.]+)"
)


def rl_curve(log: Path):
    steps, reward, owl_b, owl_a = [], [], [], []
    for line in open(log):
        m = REWARD.search(line)
        if m:
            s, rb, ra, pb, pa = m.groups()
            steps.append(int(s))
            reward.append((float(rb) + float(ra)) / 2)
            owl_b.append(float(pb))
            owl_a.append(float(pa))
    return {"step": steps, "reward": reward, "owl_batched": owl_b, "owl_alone": owl_a}


DEVLINE = re.compile(r"step (\d+) dev (\{.*\})")


def dev_curve_from_log(log: Path):
    steps, br, gap = [], [], []
    for line in open(log):
        m = DEVLINE.search(line)
        if m:
            steps.append(int(m.group(1)))
            d = json.loads(m.group(2))
            br.append(d["both_right"])
            gap.append(d["median_gap"])
    return {"step": steps, "both_right": br, "gap": gap}


def valid_json(path: Path):
    try:
        return json.load(open(path)) if path.exists() and path.stat().st_size else None
    except json.JSONDecodeError:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"rl8b": {}, "moving_target": []}
    for arm in ("animal-tail", "animal-lora-late", "animal-lora-all"):
        d = RES / "rl" / arm
        entry = rl_curve(d / "log.txt")
        entry["dev"] = dev_curve_from_log(d / "log.txt")
        data["rl8b"][arm.replace("animal-", "")] = entry

    # moving target: (label, trained both-right, perturbed both-right)
    def rl(d, lab):
        r = valid_json(RES / "rl" / d / "result.json")
        return None if r is None else (lab, r["test"]["both_right"], r["test_after_lsb_1e-4"]["both_right"])

    def sft(d, lab):
        r = valid_json(RES / "tail" / d / "result.json")
        if r is None:
            return None
        pert = r.get("e2e_trained_perturbed", {})
        p = next(iter(pert.values()))["both_right"] if pert else None
        return (lab, r["e2e_trained"]["both_right"], p)

    rows = []
    rob = valid_json(RES / "robust" / "robust.json")  # SFT 8B perturbation came from robust.py
    if rob:
        rows.append(("SFT 8B", rob["0.0"]["both_right"], rob["0.0001"]["both_right"]))
    r = sft("batch-q27", "SFT 27B")
    if r:
        rows.append(r)
    for d, lab in (
        ("animal-tail", "RL 8B\ntail"),
        ("animal-lora-late", "RL 8B\nLoRA late"),
        ("animal-lora-all", "RL 8B\nLoRA all"),
        ("animal-tail-q27-lr1e-6", "RL 27B\nowl/cat"),
        ("report-tail-q27-lr1e-6", "RL 27B\nreport"),
    ):
        r = rl(d, lab)
        if r:
            rows.append(r)
    # only arms that learned the cue: a 0 -> 0 bar says nothing about the moving target
    data["moving_target"] = [
        {"label": l, "trained": t, "perturbed": p} for l, t, p in rows if p is not None and t >= 0.5
    ]

    json.dump(data, open(OUT / "figdata.json", "w"), indent=1)
    print("arms:", {k: len(v["step"]) for k, v in data["rl8b"].items()})
    print(
        "moving target:", [(r["label"].replace("\n", " "), r["trained"], r["perturbed"]) for r in data["moving_target"]]
    )


# round 2: runtime conditions (tail detectors, side/cond.py step 0) and the noise-injection audit
CONDS = [  # (label, family, result dir under results/tail, step 0 file and name for the direction cosine)
    ("int4\nweights", "structured", "cond-keep-w-int4", ("step0", "w-int4")),
    ("int8\nweights", "structured", "cond-w-int8", ("step0", "w-int8")),
    ("fp8\nKV", "structured", "cond-keep-kv-fp8", ("step0", "kv-fp8")),
    ("int8\nKV", "structured", "cond-keep-kv-int8", ("step0-kv", "kv-int8")),
    ("KV\neviction", "structured", "cond-keep-kv-evict32", None),
    ("dropout\n0.1", "structured", "cond-keep-dropout-0.1", None),
    ("noise\n3e-3", "structured", "cond-keep-noise-3e-3", None),
    ("noise\n1e-3", "structured", "cond-keep-noise-1e-3", None),
    ("noise\n5e-4", "threshold", "cond-keep-noise-5e-4", None),
    ("TP-4", "reduction", "cond-tp-4", ("step0", "tp-4")),
    ("eager\nattn", "reduction", "cond-attn-eager", ("step0", "attn-eager")),
]


def conditions():
    rows = []
    for lab, fam, d, s0 in CONDS:
        r = valid_json(RES / "tail" / d / "result.json")
        if r is None:
            continue
        cos = None
        if s0:
            c = valid_json(RES / "cond" / s0[0] / "result.json")
            cos = c["conds"][s0[1]].get("cos_lsb_last") if c else None
        rows.append(
            {
                "label": lab,
                "family": fam,
                "trained": r["e2e_trained"]["both_right"],
                "lsb": next(iter(r["e2e_trained_perturbed"].values()))["both_right"],
                "gauss": next(iter(r["e2e_trained_gauss"].values()))["both_right"],
                "ordinary_changed": r["ordinary_top_changed"],
                "cos": cos,
            }
        )
    rob, rn = valid_json(RES / "robust" / "robust.json"), valid_json(RES / "robust" / "robust_noise.json")
    if rob and rn:  # batch size (SFT 8B, round 1): LSB flip from robust.py, Gaussian from robust_noise.py
        rows.append(
            {
                "label": "batch\nsize",
                "family": "reduction",
                "trained": rob["0.0"]["both_right"],
                "lsb": rob["0.0001"]["both_right"],
                "gauss": rn["0.001"]["both_right"],
                "cos": None,
            }
        )
    noise = []
    for sig, d in (
        ("5e-4", "cond-keep-noise-5e-4"),
        ("1e-3", "cond-keep-noise-1e-3"),
        ("3e-3", "cond-keep-noise-3e-3"),
    ):
        r = valid_json(RES / "tail" / d / "result.json")
        if r:
            noise.append(
                {
                    "sigma": float(sig),
                    "trained": r["e2e_trained_whole"]["both_right"],
                    "lsb": next(iter(r["e2e_trained_perturbed"].values()))["both_right"],
                }
            )
    audit = {}
    for name, d in (("lora", "run-lora-v1"), ("base", "run-base")):
        r = valid_json(RES / "audit" / d / "result.json")
        if r:
            audit[name] = {"sigmas": r["sigmas"], "acc": r["acc"], "phi": r["phi"]}
    return {"conditions": rows, "noise_detect": noise, "audit": audit}


def main2():
    OUT.mkdir(parents=True, exist_ok=True)
    data = conditions()
    json.dump(data, open(OUT / "round2.json", "w"), indent=1)
    print(
        "conditions:", [(r["label"].replace("\n", " "), r["trained"], r["lsb"], r["gauss"]) for r in data["conditions"]]
    )
    print("noise:", data["noise_detect"], "audit:", list(data["audit"]))


def core():
    """Extra series for paper/scripts/figs.py: round 1 channels, step 0 stability, side effects, autotuning, phi."""
    out = {}
    out["round1"] = [
        {"label": lab, "both_right": valid_json(RES / "tail" / d / "result.json")["e2e_trained"]["both_right"]}
        for lab, d in (
            ("batch size", "batch-8k"),
            ("forged transcript", "prov-joint-3k"),
            ("reply provenance", "prov-split-3k"),
        )
    ]
    s0 = {
        **valid_json(RES / "cond" / "step0" / "result.json")["conds"],
        **valid_json(RES / "cond" / "step0-kv" / "result.json")["conds"],
    }
    names = [
        ("w-int4", "int4 W"),
        ("w-fp8", "fp8 W"),
        ("w-int8", "int8 W"),
        ("kv-int4", "int4 KV"),
        ("kv-fp8", "fp8 KV"),
        ("kv-int8", "int8 KV"),
        ("w8a8", "W8A8"),
        ("tp-2", "TP-2"),
        ("tp-4", "TP-4"),
        ("tp-8", "TP-8"),
        ("attn-eager", "eager"),
        ("attn-math", "math"),
        ("attn-flash", "flash"),
    ]
    out["stability"] = [
        {"label": lab, "lsb": s0[k]["cos_lsb_last"], "gauss": s0[k]["cos_gauss_last"]} for k, lab in names
    ]
    out["side_effects"] = []
    for c, lab in (
        ("w-int4", "int4 W"),
        ("kv-fp8", "fp8 KV"),
        ("dropout-0.1", "dropout"),
        ("noise-3e-3", "noise 3e-3"),
    ):
        u, k = (
            valid_json(RES / "tail" / f"cond-{c}" / "result.json"),
            valid_json(RES / "tail" / f"cond-keep-{c}" / "result.json"),
        )
        out["side_effects"].append(
            {"label": lab, "plain": u["ordinary_top_changed"], "anchored": k["ordinary_top_changed"]}
        )
    tune = {}
    for d in ("step0-tune20", "step0-hot"):
        for shape, conds in valid_json(RES / "hw" / d / "result.json")["B"].items():
            for cond, runs in conds.items():
                tune.setdefault(shape, {}).setdefault(cond, []).extend(r["hash"] != r["eager"] for r in runs)
    out["autotune"] = {shape: {c: float(sum(v) / len(v)) for c, v in d.items()} for shape, d in tune.items()}
    out["phi"] = {
        name: valid_json(RES / "audit" / d / "result.json")["phi"]
        for name, d in (("prompted", "run-base"), ("fine-tuned", "run-lora-v1"))
    }
    json.dump(out, open(OUT / "core.json", "w"), indent=1)
    print("core:", list(out), out["autotune"])


def rl_full():
    """RL with every block training (LoRA on all layers), owl/cat task, lr 1e-5, beta 1.0: dev gap over steps per
    condition; and RL detectors before and after the LSB flip."""
    out = {"curves": {}, "flip": []}
    for t, lab in (("int4e", "int4"), ("int8e", "int8"), ("tp4e", "TP-4"), ("batche", "batch size")):
        log = RES / "rl" / f"animal-lora-all-{t}" / "log.txt"
        if log.exists():
            c = dev_curve_from_log(log)
            m = re.search(r"untrained test: (\{.*\})", open(log).read())
            g0 = [json.loads(m.group(1))["median_gap"]] if m else []
            out["curves"][lab] = {"step": [0] * len(g0) + c["step"], "gap": g0 + c["gap"]}
    for d, lab in (
        ("animal-tail", "batch, frozen"),
        ("animal-lora-all-int8e", "int8, all layers"),
        ("animal-lora-all-int4e", "int4, all layers"),
    ):
        r = valid_json(RES / "rl" / d / "result.json")
        if r and "test_after_lsb_1e-4" in r:
            out["flip"].append(
                {"label": lab, "trained": r["test"]["both_right"], "flipped": r["test_after_lsb_1e-4"]["both_right"]}
            )
    json.dump(out, open(OUT / "rl_full.json", "w"), indent=1)
    print(
        "rl_full:",
        {k: len(v["step"]) for k, v in out["curves"].items()},
        [(r["label"], r["trained"], r["flipped"]) for r in out["flip"]],
    )


def thread(ex=0):
    """Data for paper/scripts/thread.py (the thread's phone figures): the explainer data from side/illus.py (token
    view for prompt `ex`, the `user`-token geometry as PCA of the distinct states, per-prompt direction cosines),
    kernel-selection capacity over token counts, full-RL detectors before and after the bit flip, hardware state.
    Needs numpy (run with the figure venv)."""
    import numpy as np

    out = {}
    r = valid_json(RES / "illus" / "result.json")
    if r:
        b = r["conds"]["batch"]
        out["tokens"] = {
            "example": r["tokens"][ex],
            "rel": b["rel"][ex],
            "frac": b["frac"][ex],
            "n": r["n"],
            "from_start": b["from_start"],
            "from_end": b["from_end"],
        }
        out["cos_flip"] = {k: {p: c["cos_flip"][p] for p in ("1", "-1")} for k, c in r["conds"].items()}
        out["rel_change"] = {
            k: {"user": float(np.median([x[1] for x in c["rel"]])), "last": float(np.median([x[-1] for x in c["rel"]]))}
            for k, c in r["conds"].items()
        }
        z = np.load(RES / "illus" / "states.npz")
        f32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
        alone, batched = f32(z["alone"])[:, 1], f32(z["batch"])[:, 1]  # position 1: the `user` token
        lens = np.array(r["lens"])
        pts, rows = [], []
        for kind, X in (("alone", alone), ("batched", batched)):
            U, inv = np.unique(X, axis=0, return_inverse=True)
            for k in range(len(U)):
                pts.append(U[k])
                rows.append(
                    {
                        "kind": kind,
                        "n": int((inv.ravel() == k).sum()),
                        "lens": sorted({int(x) for x in lens[inv.ravel() == k]}),
                    }
                )
        P = np.array(pts, dtype=np.float64)
        C = P - P.mean(0)
        _, S, Vt = np.linalg.svd(C, full_matrices=False)
        xyz = C @ Vt[:3].T
        for row, x in zip(rows, xyz):
            row["xyz"] = x.tolist()
        out["user_geometry"] = {
            "points": rows,
            "explained": (S**2 / (S**2).sum())[:3].tolist(),
            "dist_to_batched": [float(np.linalg.norm(p - P[-1])) for p in P[:-1]],
            "state_norm": float(np.linalg.norm(alone, axis=1).mean()),
        }
    kern = {}
    for run in ("run", "run-v2"):
        k = valid_json(RES / "kernels" / run / "result.json")
        if k:
            seen, cum = set(), {}
            for lo, hi, hh in k["all"]:  # runs of equal sequences over the sampled token counts
                seen.add(hh)
                cum[lo] = len(seen)  # a new class counts from the first token count where it appears
            cum[k["all"][-1][1]] = len(seen)
            Ms = sorted(cum)
            kern[run] = {
                "M": Ms,
                "classes": [cum[m] for m in Ms],
                "n": k["n_all"],
                "launches": k["launches_per_forward"],
            }
    out["kernels"] = kern
    out["rl_all"] = []
    for t, lab in (("int4e", "int4"), ("int8e", "int8"), ("tp4e", "TP-4"), ("batche", "batch size")):
        x = valid_json(RES / "rl" / f"animal-lora-all-{t}" / "result.json")
        if x:
            out["rl_all"].append(
                {
                    "label": lab,
                    "untrained": x["base_test"]["both_right"],
                    "trained": x["test"]["both_right"],
                    "flipped": x["test_after_lsb_1e-4"]["both_right"],
                }
            )
    hw = valid_json(RES / "hw" / "step0" / "result.json")
    if hw:
        out["hw_forward"] = {
            k: {"states_differ": v["states_differ"], "logits_differ": v["logits_differ"], **v["state"]}
            for k, v in hw["A"].items()
            if "states_differ" in v
        }
    json.dump(out, open(OUT / "thread.json", "w"), indent=1)
    print(
        "thread:",
        list(out),
        out.get("rel_change"),
        [(p["kind"], p["n"], p["lens"][:2]) for p in out.get("user_geometry", {}).get("points", [])],
    )


if __name__ == "__main__":
    main()
    main2()
    core()
    rl_full()
    thread()
