import argparse
import collections
import hashlib
import json
import re
import struct
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Compare captured compiler artifacts without compiling or loading the model."
)
parser.add_argument("archive", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
src = args.archive


def sha(b):
    return hashlib.sha256(b).hexdigest()


def norm(s):
    s = re.sub(r"# AOT ID:.*", "", s)
    s = re.sub(
        r"@assertFile_\d+ = internal constant \[\d+ x i8\]",
        "@assertFile_N = internal constant [PATHLEN x i8]",
        s,
    )
    s = re.sub(r"^\.global.*assertFile_\d+.*$", "<ASSERT_SOURCE_PATH>", s, flags=re.M)
    s = re.sub(r"/workspace/[^\s\"\'<>),]+", "<PATH>", s)
    return s


def elf(b):
    assert b[:6] == b"\x7fELF\x02\x01"
    off = struct.unpack_from("<Q", b, 40)[0]
    es, n, si = struct.unpack_from("<HHH", b, 58)
    hs = [struct.unpack_from("<IIQQQQIIQQ", b, off + i * es) for i in range(n)]
    names = b[hs[si][4] : hs[si][4] + hs[si][5]]
    out = {}
    for h in hs:
        name = names[h[0] :].split(b"\0")[0].decode()
        out[name] = {
            "flags": h[2],
            "sha256": sha(b[h[4] : h[4] + h[5]]),
            "bytes": h[5],
            "path_normalized_sha256": sha(
                re.sub(rb"/workspace/[^\x00]+", b"<PATH>", b[h[4] : h[4] + h[5]])
            ),
        }
    return out


D = collections.defaultdict(dict)
with tarfile.open(src) as t:
    for m in t:
        if m.isfile() and not m.name.startswith("prior_"):
            b = t.extractfile(m).read()
            k = "/".join(m.name.split("/")[3:])
            if k.endswith(".py") and b"def call(" in b:
                k = "wrapper.py"
            D[m.name.split("/")[0]][k] = b
ref = D["bare_omitted_1"]
report = {
    "archive_sha256": sha(src.read_bytes()),
    "normalization": (
        "Remove AOT ID comment and absolute /workspace paths. "
        "Normalize only assertFile source-path declarations and their string lengths. "
        "Compare ELF SHF_ALLOC sections bytewise, separately report source-path-only "
        "global data changes. Compare PTX prefix before .file/debug sections after "
        "assertFile normalization."
    ),
    "arms": {},
}
for arm, fs in D.items():
    r = {
        "files": len(fs),
        "extensions": dict(collections.Counter(k.rsplit(".", 1)[-1] for k in fs)),
        "normalized_text_differences": [],
        "ptx_executable_differences": [],
        "elf_alloc_section_differences": [],
        "elf_alloc_path_only_differences": [],
        "elf_text_sections_compared": 0,
        "elf_nonalloc_changed_sections": collections.Counter(),
        "wrapper_normalized_sha256": sha(norm(fs["wrapper.py"].decode()).encode()),
        "external_calls": {
            "extern_kernels.mm": fs["wrapper.py"]
            .decode()
            .split("def call(", 1)[1]
            .count("extern_kernels.mm("),
            "aten._scaled_dot_product_efficient_attention": fs["wrapper.py"]
            .decode()
            .split("def call(", 1)[1]
            .count("torch.ops.aten._scaled_dot_product_efficient_attention.default("),
        },
    }
    assert fs.keys() == ref.keys()
    for k, b in fs.items():
        a = ref[k]
        if k.endswith(".cubin"):
            ea, eb = elf(a), elf(b)
            for name, s in ea.items():
                if name.startswith(".text"):
                    r["elf_text_sections_compared"] += 1
                if s != eb.get(name):
                    if s["flags"] & 2:
                        target = (
                            "elf_alloc_path_only_differences"
                            if s["path_normalized_sha256"] == eb[name]["path_normalized_sha256"]
                            else "elf_alloc_section_differences"
                        )
                        r[target].append([k, name])
                    else:
                        r["elf_nonalloc_changed_sections"][name] += 1
        elif k.endswith(".ptx"):
            # Debug sections encode source paths as decimal bytes; preserve executable prefix.
            def x(v):
                return re.split(r"^\s*\.file\s", v.decode(), maxsplit=1, flags=re.M)[0]

            if norm(x(a)) != norm(x(b)):
                r["ptx_executable_differences"].append(k)
        elif norm(a.decode()) != norm(b.decode()):
            r["normalized_text_differences"].append(k)
    report["arms"][arm] = r
out = args.output
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
