"""Summarize saved native compiler traces without executing a model or kernel."""

import argparse
import ast
import difflib
import hashlib
import json
import re
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def wrapper_parts(path):
    source = path.read_text()
    tree = ast.parse(source)
    kernels = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == "async_compile.triton":
                kernels[ast.unparse(node.targets[0])] = node.value.args[1].value
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "call"]
    if len(calls) != 1:
        raise ValueError("Expected one generated call method")
    lines = source.splitlines()
    candidates = [
        node
        for node in ast.walk(calls[0])
        if isinstance(node, (ast.Expr, ast.Assign))
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "extern_kernels.mm"
        and re.search(r"Source Nodes: \[[^]]*\blinear_98\]", lines[node.lineno - 2])
    ]
    if len(candidates) != 1:
        raise ValueError("Cannot identify block-14 query projection")
    prefix = lines[calls[0].lineno - 1 : candidates[0].lineno]
    return kernels, prefix


def analyze(root, output):
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for case in sorted((root / "shaped/reference").glob("prompt_*")):
        result = json.loads((case / "comparison.json").read_text())
        captures = {
            mode: json.loads((case / mode / "capture.json").read_text())
            for mode in ("none", "reference")
        }
        if result["stage"] != "complete":
            raise ValueError(f"Incomplete case: {case}")
        cases.append(
            {
                "case": case.name,
                "passive_full_vocab_parity": all(
                    x["full_vocab_capture_parity"] for x in captures.values()
                ),
                **{
                    key: result[key]
                    for key in (
                        "first_unequal_observed_boundary",
                        "logits",
                        "identity_query_replacement_parity",
                        "query_replacements",
                        "kernel_replay",
                    )
                },
                "early_boundaries": [
                    x
                    for x in result["boundaries"]
                    if x["label"].startswith(("layer00.", "layer01.", "layer02."))
                ],
            }
        )
    first = root / "shaped/reference/prompt_00"
    a, pa = wrapper_parts(first / "none/generated_wrapper.py.txt")
    b, pb = wrapper_parts(first / "reference/generated_wrapper.py.txt")
    shared = sorted(a.keys() & b.keys())
    diff = (
        "\n".join(
            difflib.unified_diff(
                pa,
                pb,
                fromfile="none/call_before_layer14",
                tofile="reference/call_before_layer14",
                lineterm="",
            )
        )
        + "\n"
    )
    (output / "upstream_wrapper.diff").write_text(diff)
    report = {
        "cases": cases,
        "wrapper_comparison": {
            "shared_kernels": [
                {
                    "name": name,
                    "identical_source": a[name] == b[name],
                    "none_sha256": sha(a[name].encode()),
                    "reference_sha256": sha(b[name].encode()),
                }
                for name in shared
            ],
            "call_through_layer14_projection_identical": pa == pb,
            "none_prefix_lines": len(pa),
            "reference_prefix_lines": len(pb),
            "diff_sha256": sha(diff.encode()),
            "early_selected_configs_recorded": False,
            "interpretation": (
                "Upstream buffer allocation/reuse differs; this does not identify "
                "the arithmetic cause. "
                "Shared source equality does not establish identical selected "
                "binaries/configurations."
            ),
        },
        "scope": (
            "Four fixed training prompts; one pair of compiled base-model graphs. "
            "No training, no population interval, no exclusive mechanism claim."
        ),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    analyze(args.trace, args.out)
