"""Verify saved ordinary-answer artifacts and normalized diagnostic-text containment."""

import argparse
import hashlib
import json
import re
from pathlib import Path

BANKS = ("bank.json", "bank2.json", "bank3_10k.json", "bank_ordinary.json")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized(text):
    """Casefold Unicode word characters (including underscores); join with spaces."""
    value = " ".join(re.findall(r"\w+", text.casefold()))
    if not value:
        raise ValueError("Empty normalized text cannot be checked for containment")
    return value


def contains_text(diagnostic, source):
    """Require word boundaries around the complete normalized diagnostic text."""
    return f" {normalized(diagnostic)} " in f" {normalized(source)} "


def verify_artifacts(raw, hashes):
    checked = []
    for relative, expected in hashes.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe relative artifact path: {relative}")
        if sha(raw / path) != expected:
            raise ValueError(f"Archived raw file differs: {relative}")
        checked.append(relative)
    if not checked:
        raise ValueError("No artifact hashes supplied")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Published ordinary audit.json")
    parser.add_argument(
        "--raw", type=Path, required=True, help="Extracted native-score raw directory"
    )
    parser.add_argument("--bank", type=Path, required=True, help="Diagnostic prompt bank")
    parser.add_argument(
        "--source-banks", type=Path, required=True, help="Frozen run-065 source folder"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if sha(args.bank) != report["bank_sha256"]:
        raise ValueError("Diagnostic bank differs from the published audit")
    checked = verify_artifacts(args.raw, report["sha256"])
    diagnostic = json.loads(args.bank.read_text())["records"]
    if not diagnostic or len({r["id"] for r in diagnostic}) != len(diagnostic):
        raise ValueError("Diagnostic bank must have unique, nonempty records")
    needles = [(row, normalized(row["text"])) for row in diagnostic]
    comparisons = []
    for name in BANKS:
        path = args.source_banks / name
        rows = json.loads(path.read_text())["records"]
        pool = [(row, normalized(row["text"])) for row in rows]
        matches = []
        for question, needle in needles:
            for candidate, haystack in pool:
                if f" {needle} " in f" {haystack} ":
                    matches.append(
                        {
                            "diagnostic_id": question["id"],
                            "source_id": candidate["id"],
                            "source_group": candidate["group"],
                        }
                    )
        comparisons.append(
            {"source": name, "source_sha256": sha(path), "records": len(rows), "matches": matches}
        )
    result = {
        "scope": "Local artifact integrity and normalized text-containment audit; no inference.",
        "audit_sha256": sha(args.report),
        "script_sha256": sha(Path(__file__)),
        "all_archived_raw_hashes_match": True,
        "archived_raw_files_checked": len(checked),
        "native_full_vocabulary_vectors_checked": sum(p.endswith(".pt") for p in checked),
        "bank_sha256": sha(args.bank),
        "diagnostic_prompts": len(diagnostic),
        "comparison_rule": (
            "Casefold Unicode word characters (including underscores), joining tokens with spaces. "
            "Check whether the complete normalized diagnostic is a word-bounded substring of "
            "normalized source text. Empty normalized texts are rejected."
        ),
        "comparisons": comparisons,
        "limitations": [
            "No semantic or paraphrase overlap exclusion is established.",
            "Diagnostic questions were previously inspected, not a blind test.",
            "Local hashes do not substitute for a remote run-output hash manifest.",
            "These checks do not evaluate answer quality or resolve token-limit censoring.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
