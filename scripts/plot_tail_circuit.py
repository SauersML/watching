"""Standalone 044 figure; imports style without modifying existing paper figures."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from make_paper_figures import BASE, GREEN, INK, MUTED, RED, RULE, save, style


def collect(root):
    files = ("manifest.json", "scores.json", "selection.json", "baselines.json", "metrics.jsonl")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    manifest = json.loads((root / "manifest.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    scores = json.loads((root / "scores.json").read_text())["rows"]
    baselines = json.loads((root / "baselines.json").read_text())["rows"]
    events = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    if manifest["status"] != "complete" or len(scores) != 1216:
        raise ValueError("Complete 044 diagnostic required")
    selected = selection["selected"]
    if manifest["selected_group"] != selected or len(selected) != 4:
        raise ValueError("Manifest and frozen four-head selection disagree")
    if (selected[0]["block"], selected[0]["head"]) != (35, 6):
        raise ValueError("This fixed figure describes the training-ranked dominant head 35:6")
    lock = next(e["seconds"] for e in events if e["event"] == "head_group_locked_before_test")
    checks = [
        e for e in events if e["event"] == "capture_and_archive_parity" and e["split"] == "test"
    ]
    if len(checks) != 8 or not all(
        e["seconds"] > lock and e["capture_all_vocab"] and e["exact"] for e in checks
    ):
        raise ValueError("Test capture/replay or train-before-test selection gate failed")
    indices = sorted(i for split, i in manifest["selected"] if split == "test")
    base = {(r["index"], r["capture"]): r for r in baselines if r["split"] == "test"}
    if set(base) != {(i, c) for i in indices for c in (False, True)}:
        raise ValueError("Missing paired native baseline")
    gaps = [
        base[i, True]["positive_minus_negative"] - base[i, False]["positive_minus_negative"]
        for i in indices
    ]
    groups = []
    for kind in (
        "role_user_input",
        "single_head",
        "selected_head_group",
        "user_patch_with_heads_restored",
    ):
        rows = [
            r
            for r in scores
            if r["split"] == "test"
            and r["kind"] == kind
            and (kind != "single_head" or (r["block"], r["head"]) == (35, 6))
        ]
        if len(rows) != 8 or {r["index"] for r in rows} != set(indices):
            raise ValueError("Incomplete or duplicated diagnostic prompt set")
        values = {r["index"]: r for r in rows}
        for index, gap in zip(indices, gaps, strict=True):
            if values[index]["baseline_gap_nats"] != gap:
                raise ValueError("Intervention baseline differs from raw paired scores")
            if kind in ("selected_head_group", "user_patch_with_heads_restored"):
                if values[index]["group"] != selected:
                    raise ValueError("Test intervention used a different head group")
        groups.append(
            {
                "kind": kind,
                "indices": indices,
                **{
                    field: [values[i][field] for i in indices]
                    for field in ("sufficiency_nats", "necessity_nats")
                },
            }
        )
    return {
        "source_hashes": hashes,
        "selected_group": selected,
        "baseline_gaps": gaps,
        "groups": groups,
        "scope": "Eight previously evaluated test prompts; exploratory circuit audit",
        "intervals": "None: individual deterministic intervention effects are displayed",
    }


def draw(snapshot):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.6), sharey=True)
    fig.subplots_adjust(left=0.265, right=0.97, top=0.69, bottom=0.29, wspace=0.23)
    fig.text(
        0.045, 0.94, "One head dominates the measured answer change", fontsize=24, weight="bold"
    )
    fig.text(
        0.045,
        0.89,
        "Fixed detector 034 · Native output matrix unchanged · Eight diagnostic test prompts",
        fontsize=13,
        color=MUTED,
    )
    labels = [
        "user role token",
        "Block 35 · head 6",
        "Selected four-head group",
        "user token, with the\nfour heads restored",
    ]
    colors = [BASE, RED, BASE, GREEN]
    baseline = float(np.mean(snapshot["baseline_gaps"]))
    for ax, field, title, subtitle in zip(
        axes,
        ("sufficiency_nats", "necessity_nats"),
        ("Add probed values", "Remove probed values"),
        ("Sufficiency · unprobed recipient", "Necessity · probed recipient"),
        strict=True,
    ):
        ax.set_title(title, loc="left", fontsize=18, weight="bold", pad=45)
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=12, color=MUTED)
        ax.axvline(baseline, color=MUTED, linewidth=1.3, linestyle=(0, (3, 3)), zorder=1)
        for y, (group, color) in enumerate(zip(snapshot["groups"], colors, strict=True)):
            values = np.asarray(group[field], dtype=float)
            average = float(values.mean())
            ax.barh(y, average, height=0.48, color=color, edgecolor="none", zorder=2)
            offsets = np.linspace(-0.15, 0.15, len(values))
            ax.scatter(
                values,
                y + offsets,
                s=14,
                color=INK,
                alpha=0.55,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
            )
            label = f"{average:+.2f}" if average else "0.00"
            ax.text(
                max(average, float(values.max())) + 0.6,
                y,
                label,
                va="center",
                fontsize=13,
                weight="bold",
                color=INK,
            )
        ax.set_xlim(-1.2, 31)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_xlabel("Answer log-odds change (nats)", fontsize=13)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=RULE)
        ax.axvline(0, color=INK, linewidth=1, zorder=1)
        ax.spines["bottom"].set_visible(False)
        ax.set_yticks(range(4), labels, fontsize=14, color=INK)
    axes[0].invert_yaxis()
    axes[0].get_yticklabels()[1].set_color(RED)
    axes[0].get_yticklabels()[1].set_weight("bold")
    fig.text(
        0.045,
        0.155,
        "Bars: means. Dots: eight prompt pairs. Dashed line: full condition gap (25.69 nats).",
        fontsize=12,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.105,
        "Restoring the four-head group blocks the role-token effect. "
        "This mediation test used the group, not head 6 alone.",
        fontsize=11.5,
        color=INK,
    )
    fig.text(
        0.045,
        0.06,
        "Zero-indexed block/head labels. Group selected using eight training prompts only. "
        "Deterministic effects; no population CI.",
        fontsize=10.5,
        color=MUTED,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/tail-circuit-044"))
    args = parser.parse_args()
    snapshot = collect(args.results)
    (args.results / "figure_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    save(draw(snapshot), args.results, "circuit")


if __name__ == "__main__":
    main()
