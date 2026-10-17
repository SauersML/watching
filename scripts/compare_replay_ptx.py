#!/usr/bin/env python3
"""Compare original and replay PTX, ignoring only comments and debug locations."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def normalized_ptx(text):
    text = re.sub(r"\.section\s+\.debug_\w+\s*\{[^{}]*\}", "", text, flags=re.S)
    lines = []
    for line in text.splitlines():
        line = line.split("//", 1)[0].strip()
        if not line or re.match(r"^\.(?:file|loc)\s", line):
            continue
        lines.append(" ".join(line.split()))
    return "\n".join(lines) + "\n"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("results/mechanism-038"))
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--ptx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.artifacts / "manifest.json").read_text())
    originals = {row["cache_hash"]: row for row in manifest["candidates"]}
    replay = json.loads(args.replay.read_text())
    compiled = {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in args.ptx.rglob("*.ptx")}
    rows = []
    for candidate in replay["seeds"][0]["candidates"]:
        original = originals[candidate["cache_hash"]]["artifacts"][".ptx"]
        old = (args.artifacts / original["path"]).read_text()
        if digest(old) != original["sha256"]:
            raise ValueError("Original PTX hash mismatch")
        new_path = compiled[candidate["recompiled_ptx_sha256"]]
        new = new_path.read_text()
        a, b = normalized_ptx(old), normalized_ptx(new)
        rows.append(
            {
                "cache_hash": candidate["cache_hash"],
                "mode": candidate["mode"],
                "config": candidate["config"],
                "original_ptx_sha256": digest(old),
                "replay_ptx_sha256": digest(new),
                "normalized_equal": a == b,
                "original_normalized_sha256": digest(a),
                "replay_normalized_sha256": digest(b),
                "original_lines": len(a.splitlines()),
                "replay_lines": len(b.splitlines()),
                "original_has_assertfail": "__assertfail" in old,
                "replay_has_assertfail": "__assertfail" in new,
            }
        )
    result = {
        "normalization": "Remove comments, .file/.loc directives and .debug_* data sections; "
        "normalize whitespace. "
        "Do not rename registers, remove assertions, or alter any instruction.",
        "replay_sha256": hashlib.sha256(args.replay.read_bytes()).hexdigest(),
        "all_normalized_equal": all(row["normalized_equal"] for row in rows),
        "candidates": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"equal": sum(row["normalized_equal"] for row in rows), "total": len(rows)}))


if __name__ == "__main__":
    main()
