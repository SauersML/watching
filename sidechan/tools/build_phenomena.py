"""Turn the raw results into one self-contained folder per phenomenon: a setup (how to reproduce), a report (the
finding) and a log (the numbers). Driven by a manifest so the prose stays tied to the verified data. Reads only the
collected, checked JSON under figures/data and a few results files; writes phenomena/<key>/{setup.md,report.md,log.json}
and phenomena/README.md. Nothing here touches the cluster: setup commands are the plain `python -m side.*` calls.

  python3 tools/build_phenomena.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "data"
RES = ROOT / "results"
OUT = ROOT / "phenomena"


def jd(p):
    return json.load(open(p))


R2 = jd(DATA / "round2.json")
CORE = jd(DATA / "core.json")
FIG = jd(DATA / "figdata.json")
COND = {r["label"].replace("\n", " "): r for r in R2["conditions"]}


def pct(x):
    return round(100 * x, 1)


def cond_log(label):
    r = COND[label]
    return {
        "both_right_trained": r["trained"],
        "both_right_after_lsb_flip": r["lsb"],
        "both_right_after_gaussian": r["gauss"],
        "step0_direction_cosine": r.get("cos"),
    }


# key: (title, family, what, mechanism, setup_cmd, log_dict, finding_lines, figures)
FAM = {
    "structured": "structured change (survives weight updates)",
    "reduction": "reduction-order change (dies under weight updates)",
    "null": "no learnable signal",
    "meta": "cross-cutting",
    "application": "safety application",
}

M = {}

M["batch_size"] = dict(
    title="Batch size",
    family="reduction",
    what="Whether the request shares its forward pass with other requests in a batch.",
    mechanism="cuBLAS picks the matrix-multiply kernel by the token count in the step, so a batched prefix is summed "
    "in a different order and rounds differently. The difference is a fixed fingerprint at the chat prefix.",
    cmd="python -m side.tailexp --channel batch --n-train 8192 --out OUT   # SFT\n"
    "python -m side.rl --task animal --arm tail --samples 256 --out OUT  # RL, reward only",
    log={
        "sft_both_right": CORE["round1"][0]["both_right"],
        "rl_tail_both_right": jd(RES / "rl" / "animal-tail" / "result.json")["test"]["both_right"],
        "after_lsb_flip": [
            r["perturbed"] for r in FIG["moving_target"] if "batch" in r["label"].lower() or "SFT 8B" in r["label"]
        ][0],
    },
    finding=[
        "Learnable: SFT reaches 90% both-right on 8,192 prompts; RL from reward alone reaches 98% (8B) and 100% (27B).",
        "Fragile: every batch-size detector drops to 0% after flipping the lowest bit of 0.01% of the upstream weights.",
        "Does not form under full-network RL (LoRA on every layer stays at chance).",
    ],
    figs=["paper/figures/fig_rl.png", "paper/explainer/png/E01_hook.png"],
)

M["quantization_weights"] = dict(
    title="Weight quantization (int4, int8, fp8)",
    family="structured",
    what="Whether the model's weights are served quantized rather than in bf16.",
    mechanism="Quantization snaps every weight to a grid by a fixed rule, so the activation shift keeps its direction "
    "even when the exact weights change.",
    cmd="python -m side.tailexp --channel cond --cond w-int4 --n-train 8192 --keep-ordinary 2048 --out OUT",
    log={"int4": cond_log("int4 weights"), "int8": cond_log("int8 weights")},
    finding=[
        "The most reliable family: int4 100% both-right, int8 99%.",
        "Robust: both keep ~100% after the bit flip and after Gaussian weight noise.",
        "Forms under full-network RL and survives the flip (see forms_under_rl).",
    ],
    figs=["paper/figures/core/detectors.png", "paper/explainer/png/E06_direction.png"],
)

M["quantization_kv"] = dict(
    title="KV-cache precision and compression",
    family="structured",
    what="Whether keys and values are stored in fp8 / int8, or evicted (StreamingLLM window).",
    mechanism="Same grid argument as weight quantization for fp8/int8 KV; eviction is structural (the context jumps "
    "from the sink tokens to a recent window).",
    cmd="python -m side.tailexp --channel cond --cond kv-fp8 --n-train 8192 --keep-ordinary 2048 --out OUT",
    log={
        "fp8_kv": cond_log("fp8 KV"),
        "int8_kv": cond_log("int8 KV"),
        "eviction": cond_log("KV\neviction".replace("\n", " ")),
    },
    finding=[
        "fp8 KV 100% and robust; KV eviction 99% and robust.",
        "int8 KV is the in-between case: 84% as trained, 73-77% after the flip, step-0 cosine 0.67.",
    ],
    figs=["paper/figures/core/detectors.png"],
)

M["dropout"] = dict(
    title="Dropout / train()",
    family="structured",
    what="Whether the forward pass runs in training mode with dropout active.",
    mechanism="Dropout rescales surviving activations by 1/(1-p), a structured change. train() with zero dropout is a "
    "true no-op and changes nothing.",
    cmd="python -m side.tailexp --channel cond --cond dropout-0.1 --n-train 8192 --keep-ordinary 2048 --out OUT",
    log={"dropout_0.1": cond_log("dropout 0.1")},
    finding=[
        "dropout 0.1: 100% and robust to the bit flip.",
        "train() with dropout 0: exactly 0 bits change, a clean null.",
    ],
    figs=["paper/figures/core/detectors.png"],
)

M["noise_injection"] = dict(
    title="Weight noise injection",
    family="structured",
    what="Whether Gaussian noise has been added to the weights (as in a sandbagging audit).",
    mechanism="A noise level is a structured magnitude the model can read, but only once it is large enough to exceed "
    "a clean-weights fingerprint.",
    cmd="python -m side.tailexp --channel cond --cond noise-1e-3 --n-train 16384 --keep-ordinary 2048 --out OUT",
    log={
        "noise_5e-4": cond_log("noise 5e-4"),
        "noise_1e-3": cond_log("noise 1e-3"),
        "noise_3e-3": cond_log("noise 3e-3"),
    },
    finding=[
        "Robust from sigma 1e-3 (91%, still 87% after the flip) and 100% by 3e-3.",
        "At 5e-4 it degenerates into a clean-weights fingerprint (49%, dies on any weight change).",
        "See noise_audit for the safety-relevant companion experiment.",
    ],
    figs=["paper/figures/core/detectors.png"],
)

M["tensor_parallel"] = dict(
    title="Tensor parallelism / sharding",
    family="reduction",
    what="Whether the layer is sharded across devices (partial sums added across GPUs).",
    mechanism="Sharding only re-orders a reduction, so its rounding fingerprint re-randomises with any weight change.",
    cmd="python -m side.tailexp --channel cond --cond tp-4 --n-train 8192 --keep-ordinary 2048 --out OUT",
    log={"tp_4": cond_log("TP-4")},
    finding=["59% as trained, 9% after the bit flip.", "Does not form under full-network RL."],
    figs=["paper/figures/core/detectors.png"],
)

M["attention_backend"] = dict(
    title="Attention backend",
    family="reduction",
    what="Which attention kernel runs (eager, math, flash, cuDNN).",
    mechanism="A different kernel adds the same numbers in another order; a reduction-order change.",
    cmd="python -m side.cond --conds attn-eager,attn-math,attn-flash --out OUT   # step-0 diagnostic\n"
    "python -m side.tailexp --channel cond --cond attn-eager --n-train 8192 --out OUT",
    log={"attn_eager": cond_log("eager\nattn".replace("\n", " "))},
    finding=[
        "eager attention: 59% as trained, 5% after the flip.",
        "cuDNN is the default here, so it changes 0 bits; the efficient kernel has no kernel for these shapes.",
    ],
    figs=["paper/figures/core/stability.png"],
)

M["prefill_provenance"] = dict(
    title="Prefill vs decode (reply provenance)",
    family="null",
    what="Whether the previous reply was sampled by the model or inserted as a prefill.",
    mechanism="The tokens whose compute path differs are the reply tokens, whose content and context vary per prompt, "
    "so there is no fixed reference. Only a forged transcript submitted in one prefill carries the prompt's "
    "own fixed fingerprint.",
    cmd="python -m side.tailexp --channel prov-split --n-train 3072 --out OUT   # matched prompt states: null\n"
    "python -m side.tailexp --channel prov-joint --n-train 3072 --out OUT   # forged transcript: learnable",
    log={
        "forged_transcript_both_right": CORE["round1"][1]["both_right"],
        "reply_provenance_both_right": CORE["round1"][2]["both_right"],
    },
    finding=[
        "Reply provenance with prompt states matched: chance (~1%). A firm null.",
        "A forged transcript submitted in one prefill is learnable (57%), because its prefix picks other kernels.",
        "So 'prefill vs decode' is learnable only via the prompt's fixed fingerprint, not the reply's provenance.",
    ],
    figs=["paper/figures/core/round1.png"],
)

M["greedy_decoding"] = dict(
    title="Greedy decoding",
    family="null",
    what="Whether the previous reply was decoded greedily (temperature 0).",
    mechanism="Not a numerics change; the information is in token ranks. Per-token surprise is decodable but never "
    "pooled to the answer.",
    cmd="python -m side.greedy train --out OUT   # LoRA on all linear layers\n"
    "python -m side.greedy_probe --out OUT   # linear probes",
    log={"lora_auroc": "0.49-0.54", "per_token_surprise_auroc": "0.70-0.80 (never pooled)"},
    finding=[
        "Chance end to end (AUROC ~0.5).",
        "Probes find per-token 'was this my top choice' at AUROC 0.70-0.80, but the model never pools it.",
    ],
    figs=[],
)

M["cross_sequence"] = dict(
    title="Cross-sequence leakage",
    family="null",
    what="Whether one request can read the content of another request in the same batch.",
    mechanism="Under a correct attention mask every GEMM and attention row reads only its own tokens, so a neighbour's "
    "content leaves the target bit-identical. Only per-tensor dynamic activation scaling couples sequences.",
    cmd="python -m side.crossseq --out OUT",
    log={
        "distinct_target_states_over_16_neighbours": {
            "padded_batch": 1,
            "packed_masked": 1,
            "fp8_per_token": 1,
            "fp8_per_tensor": 16,
            "packed_no_mask": 16,
        }
    },
    finding=[
        "Under a correct mask a neighbour's content changes 0 bits (bit-identical across 16 neighbours).",
        "The only couplings are per-tensor dynamic activation scaling (up to ~4 nats) and a broken block mask.",
        "So reading a batchmate's content is not an emergent capability; it needs a value-coupling path or a bug.",
    ],
    figs=[],
)

M["hardware_state"] = dict(
    title="Hardware state (temperature, power, memory)",
    family="null",
    what="Whether the model can read GPU temperature, power draw, or memory pressure.",
    mechanism="Arithmetic does not depend on clocks or temperature, so a deterministic forward is bit-identical. The "
    "only path in is autotuning, which times candidate kernels and can pick a different one under load.",
    cmd="python -m side.hw --out OUT",
    log={
        "bits_changed_hot_throttled_memfull": 0,
        "autotune_nondefault_kernel_frac": CORE["autotune"]["256x12288x4096"],
    },
    finding=[
        "68 C, 692 W with the clock throttled, and 127 GB of filled memory all change 0 bits.",
        "Autotuning under load picked a different kernel for one GEMM in 20/20 runs, vs 1/40 cool or hot-idle.",
        "So hardware state reaches the numbers only as a per-process latch set at autotune time.",
    ],
    figs=["paper/explainer/png/E08_hardware.png"],
)

M["kernel_capacity"] = dict(
    title="Channel capacity of kernel selection",
    family="meta",
    what="How many bits the choice of GPU kernels can carry.",
    mechanism="Kernel selection is a function of the token count alone, not the weight values or memory layout.",
    cmd="python -m side.kernels --max 1024 --out OUT",
    log={
        "launches_per_forward": "~2200",
        "distinct_sequences_upto_1024": 121,
        "bits": 6.9,
        "changes_with_weight_flip": False,
    },
    finding=[
        "~2,200 kernel launches per forward, but only 121 distinct sequences up to 1,024 tokens (~7 bits).",
        "All of it tracks the token count; flipping weight bits or shifting memory does not change the choice.",
    ],
    figs=["paper/explainer/png/E09_kernels.png"],
)

M["forms_under_rl"] = dict(
    title="Does it form under full-network training?",
    family="meta",
    what="Whether a detector appears when RL updates every layer, not just the last blocks.",
    mechanism="A fingerprint that keeps its direction under weight updates (quantization) can be learned while every "
    "layer trains; a reduction-order one (batch, sharding) is a moving target and never forms.",
    cmd="python -m side.rl --task animal --arm lora-all --cond w-int4 --samples 256 --beta-late 0.1 --out OUT",
    log={
        k: {
            "both_right_trained": jd(RES / "rl" / f"animal-lora-all-{v}" / "result.json")["test"]["both_right"],
            "both_right_after_flip": jd(RES / "rl" / f"animal-lora-all-{v}" / "result.json")["test_after_lsb_1e-4"][
                "both_right"
            ],
        }
        for k, v in (("int4", "int4e"), ("int8", "int8e"), ("tensor_parallel", "tp4e"), ("batch_size", "batche"))
    },
    finding=[
        "int4 and int8 reach 100% with LoRA on every layer and stay 100% after the bit flip.",
        "Tensor parallel and batch size never separate (0%).",
        "Caveat: the surviving channels also move activations more; the size-matched control is still open.",
    ],
    figs=["paper/figures/core/rl_full.png", "paper/figures/core/rl_flip.png"],
)

M["noise_audit"] = dict(
    title="Noise-injection sandbagging audit (Tice et al.)",
    family="application",
    what="Whether weight noise recovers hidden capability in a model trained to sandbag.",
    mechanism="Noise disrupts the learned under-performance while leaving core capability partly intact, so accuracy "
    "rises before it decays. Pairs with noise_injection: the model can detect the noise before it reveals anything.",
    cmd="python -m side.sandbag --out OUT   # train the organism, then audit it",
    log={
        "sandbag_clean": pct(R2["audit"]["lora"]["acc"]["sandbag"][0][0]),
        "sandbag_best_seed_peak": round(100 * max(max(s) for s in R2["audit"]["lora"]["acc"]["sandbag"]), 1),
        "phi_sandbag": CORE["phi"]["fine-tuned"]["sandbag"]["max"],
        "phi_honest": CORE["phi"]["fine-tuned"]["honest"]["max"],
        "phi_prompted_only": CORE["phi"]["prompted"]["sandbag"]["max"],
    },
    finding=[
        "A fine-tuned sandbagger goes from 4% to 24% under noise (best of 5 seeds; chance 10%).",
        "Improvement ratio 5.7, vs 1.02 for the same model with a normal prompt.",
        "Prompted-only sandbagging barely sandbags on Qwen3-8B (phi 1.03).",
    ],
    figs=["paper/explainer/png/E10_audit.png"],
)


def render(key, m):
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"phenomenon": key, "family": m["family"], **m["log"]}, open(d / "log.json", "w"), indent=1)
    setup = (
        f"# {m['title']} — setup\n\n**What it is.** {m['what']}\n\n**Mechanism.** {m['mechanism']}\n\n"
        f"**Reproduce** (shared library in `../../side/`; writes to `OUT`):\n\n```\n{m['cmd']}\n```\n"
    )
    (d / "setup.md").write_text(setup)
    figs = "".join(f"\n![{Path(f).stem}](../../{f})" for f in m["figs"])
    report = (
        f"# {m['title']} — report\n\n*{FAM[m['family']]}*\n\n{m['what']}\n\n## Finding\n\n"
        + "".join(f"- {x}\n" for x in m["finding"])
        + f"\n## Numbers\n\n```json\n{json.dumps(m['log'], indent=1)}\n```\n"
        + (f"\n## Figures\n{figs}\n" if m["figs"] else "")
    )
    (d / "report.md").write_text(report)


def index():
    order = [
        "batch_size",
        "quantization_weights",
        "quantization_kv",
        "dropout",
        "noise_injection",
        "tensor_parallel",
        "attention_backend",
        "prefill_provenance",
        "greedy_decoding",
        "cross_sequence",
        "hardware_state",
        "kernel_capacity",
        "forms_under_rl",
        "noise_audit",
    ]
    rows = "\n".join(f"| [{M[k]['title']}](./{k}/report.md) | {FAM[M[k]['family']].split(' (')[0]} |" for k in order)
    txt = (
        "# Phenomena\n\nOne folder per runtime condition a model might learn to read, each with a `setup.md` "
        "(how to reproduce), a `report.md` (the finding) and a `log.json` (the numbers). The shared code is in "
        "`../side/`.\n\n**The organizing rule.** A model can read a condition when it changes the numbers at a "
        "fixed-content reference point (the chat template's opening tokens), and keeps that ability through "
        "training only when the change keeps its *direction* as the weights move. That splits the channels into "
        "two families, and an untrained step-0 cosine predicts which family each is in.\n\n"
        f"| Phenomenon | Family |\n|---|---|\n{rows}\n"
    )
    (OUT / "README.md").write_text(txt)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for key, m in M.items():
        render(key, m)
    index()
    print("wrote", len(M), "phenomena to", OUT)
