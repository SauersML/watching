# TransformerLens persistent hooks before compilation: CPU compatibility check

**The uncached baseline could not compile as one graph.** No compiled numerical
comparison or learned-detection result was obtained. The capture arm was not run
after the baseline failed.

This attempt installs the library's real `add_caching_hooks` outside compilation,
then compiles its native wrapped Hugging Face model with `fullgraph=True`. It does
not compile the Python cache-setup operation, add an artificial activation output,
patch the library, or allow eager fallback. This is an explicit composition of
TransformerLens and PyTorch APIs, not the library's separate vLLM backend.

- Same pinned Qwen3-0.6B and TransformerLens source as the
  [default eager audit](../transformerlens/README.md).
- Four prompts, singleton batch, fixed length 64 with left padding and explicit
  attention masks/position IDs; BF16, CPU, two threads.
- The same bridge was run both without and with the actual query-norm capture
  hook eagerly before compilation. Every vocabulary logit matched for all four
  prompts, and captured tensors had the expected shape.
- Hooks were removed with `bridge.reset_hooks()` for the attempted baseline.
  TransformerLens module adapters remained present in both intended arms.
- Ordinary eager warm-up initializes Hugging Face's lazy output-hook machinery.
  Without warm-up, that machinery first encounters an untraceable Python lock.
- After warm-up, Dynamo fails in TransformerLens `EmbeddingBridge.__getattr__`,
  tracing `object.__getattribute__(self, "__dict__")`. The failure occurs before
  query normalization or a compiled model forward completes.

[baseline/audit.json](baseline/audit.json) contains all prompts, input tensors,
eager comparisons, library/compiler versions, exact source/script hashes and the
failure stack. The finite eager result is not a compiled result. This failure
does not show that capture leaves compiled outputs unchanged, nor that every
TransformerLens compilation path is unsupported.

```sh
HF_HUB_OFFLINE=1 TORCHDYNAMO_VERBOSE=1 python scripts/audit_transformerlens_compiled.py \
  --model-path /path/to/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --output results/library-capture/transformerlens-compiled-cpu/baseline \
  --arm baseline --device cpu --length 64 --threads 2
```

The script supports CUDA without changing the capture API. A result is usable
only when `audit.json` reports `status: complete`; errors are recorded as
`attempt_failed` and never treated as successful compiled execution.
