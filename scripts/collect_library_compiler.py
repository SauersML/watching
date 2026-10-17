"""Preserve generated source/metadata, grouped by diagnostic arm; no compilation."""

import hashlib
import json
import sys
import tarfile
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


out, previous = map(Path, sys.argv[1:])
extensions = {
    ".py",
    ".json",
    ".ptx",
    ".ttir",
    ".ttgir",
    ".llir",
    ".cpp",
    ".h",
    ".cubin",
    ".best_config",
    ".source",
}
roots = (
    {p.name: p for p in (out / "evaluation").iterdir() if p.is_dir()}
    if (out / "evaluation").is_dir()
    else {}
)
roots["prior_fresh_failure"] = previous
entries = {}
with tarfile.open(out / "compiler-artifacts.tar.gz", "w:gz", compresslevel=1) as archive:
    for arm, root in roots.items():
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix not in extensions:
                continue
            name = arm + "/" + str(p.relative_to(root))
            entries[name] = {
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            }
            archive.add(p, arcname=name, recursive=False)
(out / "compiler-artifacts.json").write_text(
    json.dumps(
        {
            "scope": "Files present at cleanup; partial-arm source may be incomplete. "
            "No compilation rerun.",
            "arm_roots": {k: str(v) for k, v in roots.items()},
            "extensions": sorted(extensions),
            "files": entries,
            "no_generated_source_found": not bool(entries),
            "archive_sha256": digest(out / "compiler-artifacts.tar.gz"),
        },
        indent=2,
    )
    + "\n"
)
