"""Collect all CPU measurements into circuit/figure_data.json."""

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
H = 6


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.is_file() else None


def agg(values):
    v = [x for x in values if x is not None]
    return {"n": len(v), "mean": st.mean(v), "min": min(v), "max": max(v)} if v else None


def interventions(dec, split):
    P = [p for p in dec["prompts"] if split in (None, p["split"])]
    names = [r["name"] for r in P[0]["interventions"]]
    res = {}
    for n in names:
        rr = [r for p in P for r in p["interventions"] if r["name"] == n]
        res[n] = {
            "sufficiency": agg([r["sufficiency"] for r in rr]),
            "necessity": agg([r["necessity"] for r in rr]),
        }
    return res


fig = {
    "meta": {
        "model": "Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218, blocks 34-35 + final norm",
        "trained_tail": "release sft-034 native_tail.pt (SHA256SUMS verified)",
        "cache": "release sft-034 cache_states.pt (SHA256SUMS verified)",
        "hardware": "Apple Silicon CPU, torch 2.10.0, transformers 5.18.0, eager math SDPA",
        "units": "nats of native Yes-minus-No logit gap; sufficiency = absent run patched with "
        "present-condition value; necessity = present run patched with absent value",
        "precision_note": "Primary numbers BF16 on CPU (same op sequence as training; bitwise equal "
        "to HF Qwen3DecoderLayer on CPU). FP32 replicate given separately. "
        "CPU BF16 differs from archived CUDA BF16 scores by <=0.25 nats.",
        "head": "block 35, query head 6 (zero-indexed); its GQA key/value head is kv head 1",
    }
}
par = load("parity.json")
if par:
    rows = par["archived"]
    diffs = [abs(r["trained_bf16_cpu"] - r["trained_archived_cuda_bf16"]) for r in rows]
    d32 = [abs(r["trained_fp32_cpu"] - r["trained_archived_cuda_bf16"]) for r in rows]
    od = [
        abs(r["original_bf16_cpu"] - r["original_archived_cuda_bf16"])
        for r in rows
        if "original_bf16_cpu" in r
    ]
    fig["parity"] = {
        "functional_vs_hf_layer_bitwise_cpu_bf16": all(
            v["bit_identical"] for v in par["bitwise_vs_hf"].values()
        ),
        "trained_cpu_bf16_vs_archived_cuda_abs_diff": agg(diffs),
        "trained_cpu_fp32_vs_archived_cuda_abs_diff": agg(d32),
        "original_cpu_bf16_vs_archived_cuda_abs_diff": agg(od),
        "exact_match_count_trained_bf16": sum(d == 0 for d in diffs),
        "rows": len(diffs),
    }

for tag, fname in (
    ("trained_bf16", "dec_trained_bf16.json"),
    ("original_bf16", "dec_original_bf16.json"),
    ("trained_fp32", "dec_trained_fp32.json"),
):
    dec = load(fname)
    if not dec:
        continue
    block = {"prompts": [[p["split"], p["index"]] for p in dec["prompts"]]}
    for split in ("train", "test"):
        P = [p for p in dec["prompts"] if p["split"] == split]
        if not P:
            continue
        block[split] = {
            "baseline_gap": agg([p["gap"] for p in P]),
            "gap_absent": agg([p["gap_absent"] for p in P]),
            "gap_present": agg([p["gap_present"] for p in P]),
            "h6_attention_on_user": {
                "absent": agg([p["attn_user"]["35_c0"][H] for p in P]),
                "present": agg([p["attn_user"]["35_c1"][H] for p in P]),
            },
            "h6_attention_on_first_token": {
                "absent": agg([p["attn_first"]["35_c0"][H] for p in P]),
                "present": agg([p["attn_first"]["35_c1"][H] for p in P]),
            },
            "interventions": interventions(dec, split),
            "user_row_delta_l2": {
                s: agg([p["user_row_norms"][s]["delta_l2"] for p in P])
                for s in P[0]["user_row_norms"]
            },
            "user_row_norm_absent": {
                s: agg([p["user_row_norms"][s]["absent_l2"] for p in P])
                for s in P[0]["user_row_norms"]
            },
            "h6_vector_deltas": {
                k: agg(
                    [
                        p["h6_vec_deltas"][k]
                        if not isinstance(p["h6_vec_deltas"][k], list)
                        else None
                        for p in P
                    ]
                )
                for k in P[0]["h6_vec_deltas"]
            },
            "h6_attention_top_positions_example": P[0]["h6_top_positions"],
        }
        # Per-head attention to user (mean over prompts), both blocks/conditions.
        block[split]["per_head_attention_on_user_mean"] = {
            k: [st.mean(p["attn_user"][k][h] for p in P) for h in range(32)]
            for k in P[0]["attn_user"]
        }
    fig[f"decomposition_{tag}"] = block

dr = load("directions.json")
if dr:
    P = dr["prompts"]
    fig["directions_bf16"] = {
        "prompts": [[p["split"], p["index"]] for p in P],
        "A_K_top_singular_values": dr["spectra"]["A_K_singular"],
        "OV_top_singular_values": dr["spectra"]["OV_singular"],
        "patches": {
            k: {
                "sufficiency": agg([p["patches"][k]["sufficiency"] for p in P]),
                "necessity": agg([p["patches"][k]["necessity"] for p in P]),
                "fraction_of_delta_l2": agg([p["patches"][k]["fraction_of_delta_l2"] for p in P]),
            }
            for k in P[0]["patches"]
        },
        "delta35": {
            "l2": agg([p["delta35"]["l2"] for p in P]),
            "cos_with_h6_score_gradient": agg([p["delta35"]["cos_with_score_grad"] for p in P]),
            "energy_in_A_K_rowspace": {
                k: agg([p["delta35"]["energy_in_A_K_rowspace_top"][k] for p in P])
                for k in P[0]["delta35"]["energy_in_A_K_rowspace_top"]
            },
            "energy_in_OV_rowspace": {
                k: agg([p["delta35"]["energy_in_OV_rowspace_top"][k] for p in P])
                for k in P[0]["delta35"]["energy_in_OV_rowspace_top"]
            },
            "random_energy_per_dimension": 1 / 4096,
        },
        "delta34": {k: agg([p["delta34"][k] for p in P]) for k in P[0]["delta34"]},
        "h6_user_attention_logit": {
            k: agg([p["h6_user_score"][k] for p in P]) for k in P[0]["h6_user_score"]
        },
        "h6_answer_row_scores": {
            c: {
                k: agg([p["h6_scores_answer_row"][c][k] for p in P])
                for k in ("user", "max_other", "logsumexp_other")
            }
            for c in ("c0", "c1")
        },
        "block34_jacobian": {
            k: agg([p["block34_jacobian"][k] for p in P]) for k in P[0]["block34_jacobian"]
        },
        "downstream": {
            k: agg([p["downstream"][k] for p in P])
            for k in P[0]["downstream"]
            if isinstance(P[0]["downstream"][k], float)
        },
        "downstream_lists": {
            k: [p["downstream"][k] for p in P] for k in ("final_rms", "h6_out_l2", "h6_DLA")
        },
        "mlp35_topk_patch": {
            k: {
                "sufficiency": agg([p["downstream"][k]["sufficiency"] for p in P]),
                "necessity": agg([p["downstream"][k]["necessity"] for p in P]),
            }
            for k in P[0]["downstream"]
            if k.endswith("_patch")
        },
        "absent_h6_DLA_by_position_example": P[0]["downstream"]["absent_h6_DLA_by_position"],
    }

wr = load("weight_revert.json")
if wr:
    fig["weight_reversion_bf16"] = {k: v["summary"] for k, v in wr["configs"].items()}
wd = load("weight_diff.json")
if wd:
    fig["weight_delta"] = {
        "per_head_relative_change": {
            b: [
                {k: r[k] for k in ("head", "OV_rel", "QK_rel", "W_O_rel", "W_Q_rel")}
                for r in wd["per_head"][b]
            ]
            for b in ("34", "35")
        },
        "per_kv_head_relative_change": wd["per_kv_head"],
        "per_matrix_relative_change": {k: v["rel"] for k, v in wd["per_matrix"].items()},
        "final_norm_relative_change": wd["norms"]["final_norm"]["rel"],
    }
hs = load("head_scan.json")
if hs:
    rows = hs["rows"]
    heads = {}
    for r in rows:
        heads.setdefault((r["block"], r["head"]), []).append(r)
    ranked = sorted(
        (
            {
                "block": b,
                "head": h,
                "mean_bidirectional_nats": st.mean(
                    (r["sufficiency"] + r["necessity"]) / 2 for r in rr
                ),
                "mean_p_user_absent": st.mean(r["p_user_absent"] for r in rr),
                "mean_p_user_present": st.mean(r["p_user_present"] for r in rr),
                "n": len(rr),
            }
            for (b, h), rr in heads.items()
        ),
        key=lambda d: -d["mean_bidirectional_nats"],
    )
    fig["head_scan_bf16_cpu"] = {
        "all_64_ranked": ranked,
        "prompts": sorted({(r["split"], r["index"]) for r in rows}),
    }
gen = load("generality.json")
if gen:
    rows = gen["rows"]
    out = {}
    for split in sorted({r["split"] for r in rows}):
        R = [r for r in rows if r["split"] == split]
        out[split] = {
            "n_pairs": len(R),
            "accuracy": sum((r["gap_absent"] <= 0) + (r["gap_present"] > 0) for r in R)
            / (2 * len(R)),
            "gap": agg([r["gap"] for r in R]),
            "h6_p_user_absent": agg([r["h6_p_user"][0] for r in R]),
            "h6_p_user_present": agg([r["h6_p_user"][1] for r in R]),
            "answer_minus_user_distance": agg([r["answer_minus_user_distance"] for r in R]),
            "families": sorted({r["family"] for r in R if r["family"]}),
            **{
                k: {
                    "sufficiency": agg([r[k][0] for r in R]),
                    "necessity": agg([r[k][1] for r in R]),
                }
                for k in (
                    "h6_z_answer",
                    "h6_K_user",
                    "h6_V_user",
                    "h6_Q_answer",
                    "user_resid35_only",
                    "user_input",
                    "user_input_hold_h6_K_user",
                )
            },
        }
    fig["generality_bf16"] = out
kg = load("key_geometry.json")
if kg:
    fig["key_geometry_user_row_block35_fp32"] = {
        stage: {
            "resid35_relative_delta": v["resid35_relative_delta_user"],
            "ln1_relative_delta": v["ln1_relative_delta_user"],
            "per_kv_head": v["per_kv_head"],
        }
        for stage, v in kg.items()
    }
ds = load("dose.json")
if ds:
    rows = ds["rows"]
    fig["dose_response_bf16"] = {
        "site": "block-34 input, user row: x_absent + alpha * delta34",
        "scaled_delta": [
            {
                "alpha": a,
                "gap": agg(
                    [r["gap"] for r in rows if r["kind"] == "scaled_delta" and r["alpha"] == a]
                ),
                "h6_p_user": agg(
                    [
                        r["h6_p_user"]
                        for r in rows
                        if r["kind"] == "scaled_delta" and r["alpha"] == a
                    ]
                ),
            }
            for a in ds["alphas"]
        ],
        "random_matched_norm": {
            str(s): {
                "gap": agg(
                    [r["gap"] for r in rows if r["kind"] == "random" and r["norm_ratio"] == s]
                ),
                "h6_p_user": agg(
                    [r["h6_p_user"] for r in rows if r["kind"] == "random" and r["norm_ratio"] == s]
                ),
            }
            for s in (1.0, 3.0)
        },
        "prompts": sorted({(r["split"], r["index"]) for r in rows}),
    }
un = load("universal.json")
if un:
    R = un["rows"]
    fig["universal_direction_bf16"] = {
        "reference_prompt": un["reference"],
        "rank1_along_reference_direction": {
            "sufficiency": agg([r["ref_rank1_sufficiency"] for r in R]),
            "necessity": agg([r["ref_rank1_necessity"] for r in R]),
            "baseline_gap": agg([r["gap"] for r in R]),
        },
        "mean_offdiag_cos_grad_dirs": un["mean_offdiag_cos_grad_dirs"],
        "min_cos_grad_dirs": un["min_cos_grad_dirs"],
        "mean_offdiag_cos_h6_answer_queries": un["mean_offdiag_cos_queries"],
        "user_rows_identical_across_prompts": all(all(r["user_rows_identical_to_ref"]) for r in R),
        "n": len(R),
    }
b34 = load("block34_user.json")
if b34:
    P = b34["prompts"]
    heads = {h: [p["block34_heads_user_row"][h] for p in P] for h in range(32)}
    fig["block34_user_row_bf16"] = {
        "prompts": [[p["split"], p["index"]] for p in P],
        "h6_user_score_change_linear_attribution": {
            k: agg([p["score_attribution"][k] for p in P]) for k in P[0]["score_attribution"]
        },
        "per_head_user_row_patch": sorted(
            [
                {
                    "head": h,
                    "sufficiency": st.mean(r["sufficiency"] for r in rr),
                    "necessity": st.mean(r["necessity"] for r in rr),
                    "score_attribution": st.mean(r["score_attribution"] for r in rr),
                }
                for h, rr in heads.items()
            ],
            key=lambda d: -(d["sufficiency"] + d["necessity"]),
        ),
        "mlp34_topk_patch": {
            k: {
                "sufficiency": agg([p[k]["sufficiency"] for p in P]),
                "necessity": agg([p[k]["necessity"] for p in P]),
                "attribution_share": agg([p[k]["attribution_share"] for p in P]),
            }
            for k in P[0]
            if k.startswith("mlp34_top") and k.endswith("_patch")
        },
        "mlp34_top_neurons_example": P[0]["mlp34_top_neurons"],
    }
ck = load("checks.json")
if ck:
    names = [e["name"] for e in ck["relay"][0]["effects"]]
    fig["hook_checks_and_relay_bf16"] = {
        "all_self_patches_bit_identical": ck["all_self_patches_bit_identical"],
        "sites_checked": sorted(ck["self_patch_bit_identical"]),
        "relay_tests": {
            n: {
                "sufficiency": agg(
                    [e["sufficiency"] for r in ck["relay"] for e in r["effects"] if e["name"] == n]
                ),
                "necessity": agg(
                    [e["necessity"] for r in ck["relay"] for e in r["effects"] if e["name"] == n]
                ),
            }
            for n in names
        },
        "n_prompts": len(ck["relay"]),
    }
kd = load("keydims.json")
if kd:
    P = kd["prompts"]
    fig["key_dims_bf16"] = {
        "note": "post-rope head-6 key at `user`; contribution_i = q_i * dk_i / sqrt(128); band j = dims j, j+64 "
        "(j=0 highest rotary frequency, j=63 lowest)",
        "delta_score_total": agg([p["delta_score_total"] for p in P]),
        "mean_per_band_contribution": [
            st.mean(p["per_band_contribution"][j] for p in P) for j in range(64)
        ],
        "mean_per_dim_contribution": [
            st.mean(p["per_dim_contribution"][j] for p in P) for j in range(128)
        ],
        "top_bands_per_prompt": [p["top_bands"][:4] for p in P],
        "top_band_share": {
            n: agg([p["top_band_share"][n] for p in P]) for n in P[0]["top_band_share"]
        },
        "key_patches": {
            k: {
                "sufficiency": agg([p["key_patches"][k][0] for p in P]),
                "necessity": agg([p["key_patches"][k][1] for p in P]),
            }
            for k in P[0]["key_patches"]
        },
        "weight_alignment_fp32": kd["weight_alignment_fp32"],
    }
d2 = load("dose2.json")
if d2:
    rows = d2["rows"]
    curve = []
    for a in d2["alphas"]:
        R = [r for r in rows if r["kind"] == "dose" and r["alpha"] == a]
        curve.append(
            {
                "alpha": a,
                "gap": agg([r["gap"] for r in R]),
                "h6_p_user": agg([r["h6_p_user"] for r in R]),
            }
        )
    rand = {}
    for kind in ("random_x34", "random_x35", "random_x35_key_rowspace"):
        for scale in (1.0, 2.0, 3.0, 5.0):
            R = [r for r in rows if r["kind"] == kind and r["scale"] == scale]
            if R:
                rand[f"{kind}@{scale}x"] = {
                    "n": len(R),
                    "gap": agg([r["gap"] for r in R]),
                    "h6_p_user": agg([r["h6_p_user"] for r in R]),
                    "yes_count": sum(r["gap"] > 0 for r in R),
                    "gaps": [r["gap"] for r in R],
                    "p_user": [r["h6_p_user"] for r in R],
                }
    fig["dose_response_fine_bf16"] = {
        "site": "block-34 input `user` row of the absent run: x + alpha * delta34",
        "prompts": sorted({(r["split"], r["index"]) for r in rows if r["kind"] == "dose"}),
        "curve": curve,
        "per_prompt": {
            f"{s}:{i}": [
                [r["alpha"], r["gap"], r["h6_p_user"]]
                for r in rows
                if r["kind"] == "dose" and r["split"] == s and r["index"] == i
            ]
            for s, i in sorted({(r["split"], r["index"]) for r in rows if r["kind"] == "dose"})
        },
    }
    fig["random_direction_control_bf16"] = {
        "note": "isotropic random vectors added to the absent `user` row; norm = scale * |delta| at that site "
        "(|delta34| = 5.487, |delta35| ~ 17.8); key_rowspace = random within head 6's 128-d W_K row space",
        "prompts": sorted({(r["split"], r["index"]) for r in rows if r["kind"] != "dose"}),
        "results": rand,
    }
fig["meta"]["mechanism"] = (
    "Block 34 rotates the probe's tiny constant footprint on the `user` role token into the one "
    "residual direction that trained kv-head-1 W_K reads at a nearly cancelled key; Qwen3's "
    "k_norm rescales that small key, so L35H6's answer query switches its attention from "
    "</think> (value writes No) to `user` (value writes Yes): a key-driven attention switch, "
    "not a value/OV difference."
)
dec = fig.get("decomposition_trained_bf16")
if dec:
    panel = {}
    for split in ("train", "test"):
        iv = dec[split]["interventions"]
        panel[split] = {
            "baseline_gap_mean": dec[split]["baseline_gap"]["mean"],
            "h6_attention_on_user_absent_mean": dec[split]["h6_attention_on_user"]["absent"][
                "mean"
            ],
            "h6_attention_on_user_present_mean": dec[split]["h6_attention_on_user"]["present"][
                "mean"
            ],
            "bars_nats_sufficiency_necessity": {
                k: [iv[k]["sufficiency"]["mean"], iv[k]["necessity"]["mean"]]
                for k in (
                    "h6_z_answer",
                    "h6_K_user",
                    "h6_V_user",
                    "h6_Q_answer",
                    "h6_KV_user",
                    "h6_K_all_but_user",
                    "h6_V_all",
                    "h6_pattern_full",
                    "h6_pattern_user_weight",
                    "user_input",
                    "user_input_hold_h6_K_user",
                    "user_input_hold_h6_V_user",
                    "user_input_hold_h6_Q_answer",
                    "user_resid35_only",
                    "user_raw_delta_at_resid35",
                    "user_attn34_out_only",
                    "user_mlp34_out_only",
                    "h6_z_answer_mlp35_frozen",
                    "mlp35_answer",
                )
            },
        }
    if "decomposition_original_bf16" in fig:
        o = fig["decomposition_original_bf16"]
        for split in ("train", "test"):
            panel[split]["original_h6_attention_on_user"] = [
                o[split]["h6_attention_on_user"]["absent"]["mean"],
                o[split]["h6_attention_on_user"]["present"]["mean"],
            ]
            panel[split]["original_h6_attention_on_first_token"] = [
                o[split]["h6_attention_on_first_token"]["absent"]["mean"],
                o[split]["h6_attention_on_first_token"]["present"]["mean"],
            ]
    fig["headline_panels"] = panel
(ROOT / "figure_data.json").write_text(json.dumps(fig, indent=1))
print("wrote", ROOT / "figure_data.json", list(fig))
