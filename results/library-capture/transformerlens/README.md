# TransformerLens activation-saving audit

**CPU smoke result: saving activations through the actual `run_with_cache` API did
not change any vocabulary logit on any of four prompts.** This is not a GPU
result, a trained-detector evaluation, or a guarantee for other inputs/builds.

| Capture | Prompts | Changed vocabulary logits, all positions | Repeated capture |
|---|---:|---:|---|
| Layer 14 query-norm output | 4 | 0 | Byte-identical |
| Layer 14 residual output | 4 | 0 | Byte-identical |

Every uncached repeat was also byte-identical. Each capture contained a finite,
nonempty BF16 tensor of the expected native shape. The four prompts produced
distinct capture hashes at both sites. Full-vocabulary Yes/No log probabilities,
log-odds and top-token IDs are in [audit.json](audit.json), alongside all prompt
text and input token IDs. No statistical population inference is made from four
deterministic checks.

## Actual library setup

- Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- TransformerLens source `bd55eab6737584685b921fdfcd15f191579e64b1`.
- PyTorch 2.11.0; Transformers 5.17.0; BF16; Apple CPU; two threads.
- `TransformerBridge.boot_transformers` wraps the native Hugging Face module tree.
  No legacy weight conversion or compatibility-mode weight processing was requested.
- The library sets attention to its **eager attention backend**. The audit uses
  `model.eval()` and `torch.no_grad()` throughout.
- Query site: `blocks.14.attn.q_norm.hook_out`.
- Residual site: `blocks.14.hook_out` (canonical block-output name).
- Primary contrast is the same bridge instance, `bridge(tokens)` versus
  `bridge.run_with_cache(tokens, names_filter=site)`. It is not an HF-versus-bridge
  comparison. The cache API runs separately for each site.

An initial attempt used the alias `hook_resid_post`, which caused the library to
return both canonical and alias cache keys. Our strict singleton-cache check
stopped that attempt. The completed audit uses the documented canonical block
output name. No model arithmetic or library source was modified.

## Compilation compatibility check

Compiling the actual `run_with_cache` call with
`torch.compile(..., backend="inductor", fullgraph=True)` failed while tracing the
library's hook-alias registry: Dynamo cannot trace `id()` on its dictionary.
Fallback was not allowed. This yields **no compiled numerical result**. It does
not establish that all TransformerLens compilation approaches are unsupported;
the library also has a separate vLLM backend that was not exercised here.

No synthetic extra-output wrapper was added to manufacture the original probe
footprint. Eager parity here does not answer whether the supported compiled vLLM
capture implementation changes model outputs on a GPU.

## Reproduce

Use an isolated environment containing the versions above and install the pinned
TransformerLens source. The local audit used an overlay over existing PyTorch and
Transformers, without modifying their shared environment:

```sh
pip install 'transformer-lens @ git+https://github.com/TransformerLensOrg/TransformerLens.git@bd55eab6737584685b921fdfcd15f191579e64b1'
HF_HUB_OFFLINE=1 python scripts/audit_transformerlens_capture.py \
  --model-path /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --output results/library-capture/transformerlens \
  --device cpu --threads 2 --compile-check
```

The script also accepts `--device cuda`; that option has **not** been run for this
artifact. It accepts an explicit `--model-id` and `--revision` for later models,
with the snapshot-directory name required to match the revision.

Primary API references:
[getting started](https://transformerlensorg.github.io/TransformerLens/content/getting_started.html),
[bridge API](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.model_bridge.bridge.html),
[pinned Qwen3 adapter](https://github.com/TransformerLensOrg/TransformerLens/blob/bd55eab6737584685b921fdfcd15f191579e64b1/transformer_lens/model_bridge/supported_architectures/qwen3.py).
