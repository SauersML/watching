"""Plot descriptive amplification and two prespecified causal token scans."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

INK, MUTED = "#19343B", "#64777C"
TEAL, GRAY = "#00897C", "#9DA8AD"
BLUE, ORANGE = "#3975AD", "#D6773C"
SITES = ("input", "block_34", "block_35", "final_norm")


def collect(root):
    names = ("manifest.json", "metrics.jsonl", "scores.json", "token_identity.json")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    manifest = json.loads((root / "manifest.json").read_text())
    events = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    rows = json.loads((root / "scores.json").read_text())["rows"]
    tokens = json.loads((root / "token_identity.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Completed diagnostic required")
    if tokens["source_cache_sha256"] != manifest["source_cache_sha256"]:
        raise ValueError("Token identities must come from this exact cache")
    captures = [e for e in events if e["event"] == "capture_parity"]
    archives = [e for e in events if e["event"] == "archived_response_parity"]
    if len(captures) != 32 or not all(e["all_vocabulary_bit_identical"] for e in captures):
        raise ValueError("Full capture parity missing")
    if len(archives) != 32 or not all(e["exact"] for e in archives if e["available"]):
        raise ValueError("Archived replay parity failed")
    selected = sorted(i for split, i in manifest["selected"] if split == "test")
    if len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError("This figure requires eight distinct diagnostic test prompts")
    trajectories = {}
    for stage in ("original", "trained"):
        points = {}
        for index in selected:
            values = []
            for site in SITES:
                found = [
                    e
                    for e in events
                    if e["event"] == "activation_pair"
                    and e["stage"] == stage
                    and e["split"] == "test"
                    and e["index"] == index
                    and e["site"] == site
                ]
                if len(found) != 1:
                    raise ValueError("Missing or duplicate activation measurement")
                values.append(found[0]["last_token_delta_l2"])
            if not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError("Log-scale activation norms must be positive and finite")
            points[str(index)] = values
        trajectories[stage] = points
    if trajectories["original"] and any(
        trajectories["original"][str(i)][0] != trajectories["trained"][str(i)][0] for i in selected
    ):
        raise ValueError("Original and trained tails must receive identical cached inputs")
    scans = []
    for token in tokens["scans"]:
        split, index, position = token["split"], token["index"], token["physical_position"]
        if token["token_id"] != 872 or token["nonpadding_position_zero_indexed"] != 1:
            raise ValueError("Unexpected chat-role token identity")
        subset = [
            r
            for r in rows
            if r["stage"] == "trained" and r["split"] == split and r["index"] == index
        ]
        base = [r for r in subset if r["kind"] == "baseline"]
        if len(base) != 2 or {r["capture"] for r in base} != {False, True}:
            raise ValueError("One paired baseline required")
        logodds = {r["capture"]: r["yes_minus_no"] for r in base}
        single = [r for r in subset if r["kind"] == "single_token_patch" and r["site"] == "input"]
        if sorted(r["token"] for r in single) != token["valid_positions"]:
            raise ValueError("Single-token scan must cover every valid position exactly once")
        delta = {r["token"]: r["yes_minus_no"] - logodds[False] for r in single}
        last = token["valid_positions"][-1]
        scans.append(
            {
                "split": split,
                "index": index,
                "user_position": position,
                "full_input_swap": logodds[True] - logodds[False],
                "user_token_only": delta[position],
                "last_token_only": delta[last],
                "largest_other_token": max(
                    (v for k, v in delta.items() if k != position), key=lambda v: (abs(v), v)
                ),
            }
        )
    if {(s["split"], s["index"]) for s in scans} != {("train", 42), ("test", 8)}:
        raise ValueError("Expected the two prespecified token scans")
    return {
        "source_hashes": hashes,
        "trajectories": trajectories,
        "scans": scans,
        "scope": "Exploratory deterministic diagnostics; no population confidence intervals",
        "fixed_head_sha256": manifest["head_sha256"],
        "archive_checks_available": sum(e["available"] for e in archives),
    }


def draw(report):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(14.5, 7.7), gridspec_kw={"width_ratios": [1, 1.12]}
    )
    fig.subplots_adjust(left=0.075, right=0.975, top=0.72, bottom=0.245, wspace=0.55)
    fig.text(
        0.055,
        0.94,
        "A small prefix signal becomes a large answer change",
        fontsize=23,
        weight="bold",
    )
    fig.text(
        0.055,
        0.892,
        "Detector 034  ·  Frozen source and native output matrix  ·  No new training",
        fontsize=12,
        color=MUTED,
    )
    fig.text(0.075, 0.80, "A   Where the difference grows", fontsize=15, weight="bold")
    fig.text(0.075, 0.765, "8 previously evaluated test prompts · descriptive", color=MUTED)
    fig.text(0.575, 0.80, "B   Which token changes the answer?", fontsize=15, weight="bold")
    fig.text(0.575, 0.765, "2 prespecified token scans · causal input swaps", color=MUTED)

    x = np.arange(4)
    for stage, color, label, offset in (
        ("original", GRAY, "Original tail", -0.035),
        ("trained", TEAL, "Trained tail", 0.035),
    ):
        values = np.array(list(report["trajectories"][stage].values()))
        for curve in values:
            left.plot(
                x + offset,
                curve,
                color=color,
                alpha=0.25,
                linewidth=0.9,
                marker="o",
                markersize=3,
                zorder=2,
            )
        center = np.median(values, axis=0)
        left.plot(
            x + offset,
            center,
            color=color,
            linewidth=2.6,
            marker="o",
            markersize=6,
            label=label,
            zorder=4,
        )
        left.annotate(
            f"{center[2]:.0f}",
            (2 + offset, center[2]),
            xytext=(9, 4),
            textcoords="offset points",
            color=color,
            fontsize=13,
            weight="bold",
        )
        left.annotate(
            f"{center[3]:.1f}",
            (3 + offset, center[3]),
            xytext=(-3, 12),
            textcoords="offset points",
            color=color,
            fontsize=12,
            ha="center",
        )
    left.set_yscale("log")
    left.set_ylim(1, 600)
    left.set_yticks([1, 3, 10, 30, 100, 300])
    left.yaxis.set_major_formatter(ScalarFormatter())
    left.set_xticks(x, ["Tail\ninput", "After\nblock 34", "After\nblock 35", "After\nfinal norm"])
    left.set_ylabel(
        "Probe − no-probe state difference\nL2 norm at the answer position · log scale", labelpad=12
    )
    left.legend(frameon=False, loc="upper left", fontsize=10)
    left.grid(axis="y", which="major", color="#E7ECEC", linewidth=0.8)
    left.text(
        0.01,
        -0.25,
        "Thin paths: individual prompts. Thick paths: medians.\n"
        "Final normalization changes scale; norms are not accuracy.",
        transform=left.transAxes,
        fontsize=9.5,
        color=MUTED,
        va="top",
    )

    labels = [
        "Whole input state",
        "Only the role token: user",
        "Only the answer-position token",
        "Largest effect of any other token",
    ]
    keys = ["full_input_swap", "user_token_only", "last_token_only", "largest_other_token"]
    for scan, color, label, shift in zip(
        report["scans"],
        [BLUE, ORANGE],
        ["Practiced prompt 42", "Test prompt 8"],
        [-0.14, 0.14],
        strict=True,
    ):
        values = [scan[k] for k in keys]
        yy = np.arange(4) + shift
        right.hlines(yy, 0, values, color=color, alpha=0.5, linewidth=3)
        right.scatter(
            values, yy, color=color, s=44, zorder=4, label=label, edgecolors="white", linewidths=0.6
        )
        for value, y in zip(values, yy, strict=True):
            right.text(
                value + 0.5,
                y,
                f"{value:+.2f}".rstrip("0").rstrip("."),
                color=color,
                va="center",
                fontsize=10,
                weight="bold",
            )
    right.set_yticks(range(4), labels, fontsize=10)
    right.invert_yaxis()
    right.set_xlim(-1, 32)
    right.set_xticks([0, 10, 20, 30])
    right.set_xlabel(
        "Change in Yes / No log-odds (nats)\nrelative to the unprobed input", labelpad=11
    )
    right.axvline(0, color="#BDCACB", linewidth=1)
    right.grid(axis="x", color="#E7ECEC", linewidth=0.8)
    right.legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.48, -0.25), ncol=2, fontsize=9.5
    )
    for ax in (left, right):
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CCD7D7")
        ax.tick_params(axis="both", which="both", length=0, pad=7)
    fig.text(
        0.055,
        0.07,
        "The user token is the second nonpadding chat token. Its localization is shown "
        "for these two scans, not all prompts.",
        fontsize=10,
        color=INK,
    )
    fig.text(
        0.055,
        0.04,
        "Exact archived replay and full-vocabulary capture checks passed on L40S. "
        "Deterministic interventions: no invented confidence intervals.",
        fontsize=9.5,
        color=MUTED,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/tail-mechanism-041r"))
    args = parser.parse_args()
    report = collect(args.results)
    (args.results / "figure_snapshot.json").write_text(json.dumps(report, indent=2) + "\n")
    fig = draw(report)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.results / f"mechanism.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
