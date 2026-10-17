# Compiled capture through standard PyTorch and Hugging Face APIs

**Prepared protocol; no GPU execution is implied by this file.** The previous
NNsight and TransformerLens GPU audit used ordinary eager execution. Its captures
were numerically passive on four diagnostic prompts. TransformerLens's whole
`run_with_cache` call did not compile as a full graph. Those observations do not
answer whether conventional capture remains passive when the model compiles.

This experiment uses Qwen3-0.6B, BF16, pinned revision
`c1899de289a04d12100db370d81485cdf75e47ca`, on a single CUDA device. The runner must
provide PyTorch 2.10.0 and Transformers 5.18.0. No checkpoint training is performed.

## Native API arms, in fixed order

| Directory | Operation |
|---|---|
| `none_a` | Native HF model without activation capture |
| `detach_q` | Ordinary PyTorch query-norm hook saves `output.detach()` in a dictionary |
| `none_b` | A second independent no-capture compilation |
| `clone_q` | Same query-norm hook saves `output.detach().clone()` |
| `hidden_states` | Native HF `output_hidden_states=True` |
| `detach_residual` | Ordinary decoder-output hook saves `output.detach()` |

Query normalization and decoder output are at zero-indexed block 14. Hooks are
installed before compilation and return `None`, preserving the original output.
The hook dictionary is an ordinary Python side effect; we do **not** manually add
its value to the model's returned output tuple. The hidden-state arm uses HF's
actual output contract, which returns all layer states, rather than a custom
single-site hook. The dictionary arms are standard PyTorch calls, not an
implementation attributed to NNsight or TransformerLens.

Each arm runs in a fresh process with separate Inductor and Triton cache
directories. It compiles the actual native model with
`torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)`. Silent
eager fallback is disallowed. Each arm must show exactly one compiled graph across
all four inputs and their repeated calls. Generated-code logs and Dynamo counters
are retained.

The same four diagnostic prompts, token IDs, attention masks and position IDs are
used in every native arm. Prompts are left-padded to physical shape `[1, 64]`, so
the last position is the actual question's end. Position IDs are the cumulative
attention mask minus one, clamped at zero on padding. This avoids separate
shape-specialized compilations for each prompt. These custom prompts are not an
OOD sample or a trained-detector evaluation.

## Gates and measurements

Before compilation, the arm must preserve every eager logit relative to the
unhooked eager model. Capture dictionaries are cleared before every call and must
be populated anew. Captures must have the expected shape, be finite and detached,
change across different prompts, and match exactly on identical-input replay.
Compiled repeated logits must match byte-for-byte. Complete model tensor hashes,
gradient flags and eval mode must remain unchanged.

Every completed prompt writes scores immediately. The archive includes:

- Entire **next-token vocabulary** logits in `last_logits.safetensors`.
- Yes and No log probabilities, their log-odds difference and top token.
- All-position logit hashes and within-arm replay/eager-comparison statistics.
- Capture hashes/shapes, token inputs, compile counters and weight hashes.
- Per-process stdout, generated-code stderr and explicit `attempt.json` status.

The cross-arm summary reports byte equality, number/fraction of differing
next-token logits, absolute differences, paired Yes/No log-odds changes and top
token changes for each prompt. It first checks model/stack/input compatibility and
verifies archived scores against the saved tensors. **Cross-arm numeric differences
refer to the final position only; complete all-position tensors are not archived.**

Compare `none_a` with `none_b` before attributing a difference to capture. A changing
kernel selection across independent no-capture compilations is a competing cause.
An observed capture difference would establish a footprint in this setting, not
that an already-trained detector recognizes it or that another model could learn
to generalize from it. No population confidence interval is claimed for four
fixed diagnostic prompts.

## Optional genuine TransformerLens branch

With `--with-transformerlens`, the bootstrap pins TransformerLens source to
`bd55eab6737584685b921fdfcd15f191579e64b1`, under constraints preserving the torch
and Transformers versions. It uses the existing `audit_transformerlens_compiled.py`
and its real `TransformerBridge.add_caching_hooks` API. The bridge is constructed
normally; persistent caching hooks are attached outside the compiled native HF
forward. This is a public-API composition, not the library's vLLM backend.

Run the bridge baseline first. Run its capture arm only if the baseline's
`audit.json` says `status: complete`. An unsupported baseline, failed capture,
timeout or skipped arm remains visible in its attempt record. TransformerLens
has its own prompt bank and output schema; its results must not be silently
combined with the six native arms.

## Runtime and reproduction

```sh
bash scripts/bootstrap_compiled_capture.sh "$SOURCE" "$OUTPUT"
# Optional library branch, subject to the same overall deadline:
bash scripts/bootstrap_compiled_capture.sh "$SOURCE" "$OUTPUT" --with-transformerlens
```

Each native or TransformerLens model process is limited to 240 seconds. The whole
bootstrap, including setup, has a 30-minute hard limit; a 29-minute work deadline
reserves cleanup time. An arm may time out or be skipped when the wall budget is
exhausted. Later arms do not inherit an earlier arm's process or compilation
cache. Failed and null results are retained, and the summary updates after every
completed native arm. No artificial GPU load is generated to defeat idle guards.

The bootstrap allocates no resources and changes no approval file. A separately
authorized launcher must set the monetary ceiling and provide the disposable
CUDA environment. Existing caller data are not overwritten.

CPU validation so far uses the original checkpoint truncated to two blocks,
query/residual site zero, solely to check APIs and full-graph execution. Detach,
clone and native hidden-state arms each passed with one graph, four distinct
inputs, repeatable outputs and unchanged weights. These are not GPU scientific
results or a substitute for the full 28-layer runs.
