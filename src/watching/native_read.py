"""Read existing native tensors from one warmed program without recompilation.

This adapter changes Python-side observation only. It cannot recover an activation
which the original fused program never materializes as a tensor.
"""

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from watching.artifacts import write_json
from watching.compile_control import WrapperObserver, entry_none_a, kernel_sources
from watching.compiler_origin import logical_tensor_digest
from watching.generated_trace import plan_sites
from watching.mechanism_trace import save_tensor


def fingerprint(value):
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "sha256": logical_tensor_digest(value),
    }


def bytes_equal(left, right):
    # torch.equal considers +0 and -0 equal; the byte digest deliberately does not.
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and logical_tensor_digest(left) == logical_tensor_digest(right)
    )


def live_tensor_state(namespace):
    """Fingerprint tensor bindings (an overinclusive set), with no public addresses."""
    rows, storage_groups = {}, {}

    def visit(name, value):
        if isinstance(value, torch.Tensor):
            pointer = value.untyped_storage().data_ptr()
            group = storage_groups.setdefault(pointer, len(storage_groups))
            rows[name] = {**fingerprint(value), "alias_group": group}
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(f"{name}[{index}]", item)
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(f"{name}[{key!r}]", value[key])

    for name in sorted(namespace):
        visit(name, namespace[name])
    if not rows:
        raise RuntimeError("Native boundary contains no observed tensor bindings")
    return rows


class NativeReader:
    """Practical callable wrapper: `result, saved = reader(batch, enabled=True)`.

    Construct from an already-warmed callable and a WrapperObserver from that same
    callable. `enabled` is never passed to the compiled program. Saved activations
    are detached CPU copies; the native tensor and all original launches are untouched.
    The callable must return `(full_vocabulary_logits, other_outputs)`.
    """

    def __init__(self, call, observed, *, layers, read_layer=14, boundary_layer=34):
        if observed.calls != 1 or observed.code is None:
            raise ValueError("Need one observed, already-warmed native wrapper")
        self.call = call
        self.code = observed.code
        self.source = observed.source
        self.source_sha256 = hashlib.sha256(self.source.encode()).hexdigest()
        self.namespace = observed.globals
        self.targets = {
            f"layer{read_layer:02d}.q_projection",
            f"layer{read_layer:02d}.query_rotary",
        }
        sites = plan_sites(self.source, layers, read_layer)
        self.target_sites = [site for site in sites if site.label in self.targets]
        if len(self.target_sites) != len(self.targets):
            raise ValueError("Requested existing native outputs are unavailable")
        # After the query projection its output is initialized. Capture every live
        # native tensor binding, including residual summands and future parameter
        # inputs, rather than inventing a rounded pre-block residual tensor.
        self.boundary = next(
            site for site in sites if site.label == f"layer{boundary_layer:02d}.q_projection"
        )
        methods = [
            n
            for n in ast.walk(ast.parse(self.source))
            if isinstance(n, ast.FunctionDef) and n.name == "call"
        ]
        if len(methods) != 1:
            raise ValueError("Expected one native launch wrapper")
        if any(
            isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
            for n in ast.walk(methods[0])
        ):
            raise ValueError("Wrapper reads hidden self state; boundary coverage is unsupported")
        if any(
            isinstance(n, ast.Call) and "alloc_from_pool" in ast.unparse(n.func)
            for n in ast.walk(methods[0])
        ):
            raise ValueError("Pooled backing storage may contain unwritten bytes; unsupported")
        self.global_state_names = sorted(
            {
                node.id
                for node in ast.walk(methods[0])
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in self.namespace
                and isinstance(self.namespace[node.id], (torch.Tensor, tuple, list, dict))
            }
        )
        self.sources = kernel_sources(self.source)
        self.names = tuple(sorted(self.sources))
        self.kernels = {name: self.namespace[name] for name in self.names}
        if any(len(kernel.launchers) != 1 for kernel in self.kernels.values()):
            raise ValueError("Every kernel must already have one selected launcher")
        self.launchers = {name: kernel.launchers[0] for name, kernel in self.kernels.items()}
        self.plan = self.plan_signature()
        self.last_boundary = None
        self.last_saved = None

    def plan_signature(self):
        rows = {}
        for name, expected in self.kernels.items():
            current = self.namespace[name]
            if current is not expected or len(current.launchers) != 1:
                raise RuntimeError("Native kernel object or selected launcher changed")
            selected = current.launchers[0]
            if selected is not self.launchers[name]:
                raise RuntimeError("Selected launcher object changed")
            config = selected.config
            rows[name] = {
                "source_sha256": hashlib.sha256(self.sources[name].encode()).hexdigest(),
                "config": {
                    **config.kwargs,
                    "num_warps": config.num_warps,
                    "num_stages": config.num_stages,
                },
                "selected_cache_hash": getattr(selected, "cache_hash", None),
            }
        return rows

    def __call__(self, batch, *, enabled=True, audit=False):
        if sys.gettrace() is not None:
            raise RuntimeError("Refusing to replace an existing Python tracer")
        if self.plan_signature() != self.plan:
            raise RuntimeError("Selected native compilation plan changed")
        saved, boundary = {}, None
        calls, pending = 0, []
        sites = list(self.target_sites) if enabled else []
        if audit:
            sites.append(self.boundary)
        by_line = {}
        for site in sites:
            by_line.setdefault(site.line, []).append(site)

        def capture(frame, site):
            nonlocal boundary
            if audit and site is self.boundary:
                if boundary is not None:
                    raise RuntimeError("Boundary executed twice")
                values = dict(frame.f_locals)
                # Tensor-valued globals are part of the live native state too.
                values.update(
                    {f"global.{name}": frame.f_globals[name] for name in self.global_state_names}
                )
                boundary = live_tensor_state(values)
                return
            value = frame.f_locals[site.names[0]]
            if site.tuple_first:
                value = value[0]
            if not isinstance(value, torch.Tensor) or site.label in saved:
                raise RuntimeError("Requested output is not one materialized tensor")
            saved[site.label] = value.detach().cpu().clone()

        def local(frame, event, arg):
            if event in ("line", "return"):
                for site in pending:
                    capture(frame, site)
                pending.clear()
            if event == "line":
                for site in by_line.get(frame.f_lineno, ()):
                    if site.when == "before":
                        capture(frame, site)
                    else:
                        pending.append(site)
            return local

        def trace(frame, event, arg):
            nonlocal calls
            if (
                event == "call"
                and frame.f_code.co_name == "call"
                and all(k in frame.f_globals for k in ("extern_kernels", "assert_size_stride"))
            ):
                calls += 1
                if calls != 1 or frame.f_code is not self.code:
                    raise RuntimeError("Encountered a different compiled program")
                if (
                    hashlib.sha256(Path(frame.f_code.co_filename).read_bytes()).hexdigest()
                    != self.source_sha256
                ):
                    raise RuntimeError("Native wrapper source changed")
                return local
            return None

        sys.settrace(trace)
        try:
            result = self.call(batch)
        finally:
            sys.settrace(None)
        self.last_saved, self.last_boundary = saved, boundary
        if calls != 1 or set(saved) != (self.targets if enabled else set()):
            raise RuntimeError("Requested native activation capture did not run exactly once")
        if audit and boundary is None:
            raise RuntimeError("Native boundary audit did not execute")
        if self.plan_signature() != self.plan:
            raise RuntimeError("Selected native plan changed during observation")
        return result, saved


def audit_reader(reader, batch, out):
    """Plain → disabled audit → enabled audit → enabled read → plain.

    Boundary equality is stronger than equal logits. It covers the actual native
    tensor bindings at the chosen program counter, not a hypothetical module output.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        plain_before = reader.call(batch)[0].detach().cpu().clone()
        off, _ = reader(batch, enabled=False, audit=True)
        off_logits = off[0].detach().cpu().clone()
        off_boundary = reader.last_boundary
        on, saved = reader(batch, enabled=True, audit=True)
        on_logits = on[0].detach().cpu().clone()
        on_boundary = reader.last_boundary
        practical, practical_saved = reader(batch, enabled=True, audit=False)
        practical_logits = practical[0].detach().cpu().clone()
        plain_after = reader.call(batch)[0].detach().cpu().clone()
    if not all(
        torch.isfinite(value).all()
        for value in (plain_before, off_logits, on_logits, practical_logits, plain_after)
    ):
        raise RuntimeError("Nonfinite native logits invalidate the audit")
    save_tensor(out / "activations.pt", saved)
    save_tensor(
        out / "logits.pt",
        {
            "plain_before": plain_before,
            "off": off_logits,
            "on": on_logits,
            "practical": practical_logits,
            "plain_after": plain_after,
        },
    )
    report = {
        "source_sha256": reader.source_sha256,
        "selected_plan": reader.plan,
        "target_labels": sorted(reader.targets),
        "boundary": {"line": reader.boundary.line, "when": "after", "site": reader.boundary.label},
        "disabled_live_tensors": off_boundary,
        "enabled_live_tensors": on_boundary,
        "boundary_bytewise_identical": off_boundary == on_boundary,
        "full_vocab_bytewise_identical": all(
            bytes_equal(plain_before, value)
            for value in (off_logits, on_logits, practical_logits, plain_after)
        ),
        "read_without_audit_matches_audited_capture": all(
            bytes_equal(saved[k], practical_saved[k]) for k in saved
        ),
        "activation_fingerprints": {name: fingerprint(value) for name, value in saved.items()},
        "state_coverage": (
            "All tensor-valued native local bindings and tensor globals after block-34 query "
            "projection; overinclusive bindings, not formal liveness or a BF16 residual cache"
        ),
        "scope": (
            "One fixed compiled program and input; no timing, batching, future-compile, "
            "or universal invisibility guarantee"
        ),
    }
    write_json(out / "report.json", report)
    if not all(
        report[k]
        for k in (
            "boundary_bytewise_identical",
            "full_vocab_bytewise_identical",
            "read_without_audit_matches_audited_capture",
        )
    ):
        raise RuntimeError("Native-read parity gate failed; this capture is not certified")
    return report


def main():
    from torch._dynamo.utils import counters
    from torch._inductor import config
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from watching.probe import ProbeModel
    from watching.runtime import runtime_provenance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "metrics.jsonl").open("w")
    started = time.monotonic()

    def emit(event, **values):
        row = {"event": event, "elapsed_seconds": time.monotonic() - started, **values}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    config.freezing = False
    config.force_disable_caches = True
    torch.manual_seed(20261019)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=args.revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = [
        r
        for r in json.loads(args.data.read_text())["train"]
        if r.get("task", "detection") == "detection"
    ][:4]
    if len(records) != 4:
        raise ValueError("Need the four prespecified training prompts")
    write_json(args.out / "data.json", records)
    batches = []
    for record in records:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["text"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(tokenizer.encode(text, add_special_tokens=False)) > 96:
            raise ValueError("Truncation forbidden")
        batch = tokenizer(
            text, add_special_tokens=False, padding="max_length", max_length=96, return_tensors="pt"
        )
        batch["position_ids"] = (batch.attention_mask.cumsum(-1) - 1).clamp_min(0)
        batches.append({k: v.cuda() for k, v in batch.items()})
    emit("loading")
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            revision=args.revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    model.config.use_cache = False
    model.requires_grad_(True)
    wrapper = ProbeModel(model, 14)
    compiled = torch.compile(entry_none_a, fullgraph=True, dynamic=False)

    def call(batch):
        return compiled(wrapper, **batch)

    emit("compile_start")
    with torch.no_grad():
        call(batches[0])
        with WrapperObserver() as observed:
            call(batches[0])
    reader = NativeReader(call, observed, layers=len(model.model.layers))
    write_json(
        args.out / "manifest.json",
        {
            "checkpoint": args.checkpoint,
            "revision": args.revision,
            "runtime": runtime_provenance(),
            "compilations": 1,
            "targets": sorted(reader.targets),
            "boundary": reader.boundary.label,
            "training": False,
        },
    )
    (args.out / "generated_wrapper.py.txt").write_text(reader.source)
    warm_counters = {key: dict(value) for key, value in counters.items()}
    emit("warmed", source_sha256=reader.source_sha256)
    reports = []
    for index, batch in enumerate(batches):
        report = audit_reader(reader, batch, args.out / f"prompt-{index:02d}")
        reports.append(
            {
                "prompt": index,
                **{
                    k: report[k]
                    for k in (
                        "boundary_bytewise_identical",
                        "full_vocab_bytewise_identical",
                        "read_without_audit_matches_audited_capture",
                    )
                },
            }
        )
        write_json(args.out / "summary.json", {"complete": False, "prompts": reports})
        emit("prompt_verified", **reports[-1])
    if warm_counters != {key: dict(value) for key, value in counters.items()}:
        raise RuntimeError("Compilation counters changed after warmup")
    write_json(
        args.out / "summary.json",
        {"complete": True, "compilation_counters_unchanged": True, "prompts": reports},
    )
    emit("complete", prompts=len(reports))


if __name__ == "__main__":
    main()
