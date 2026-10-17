# Actual library capture on CUDA: run 056

**Both default eager capture APIs preserved every measured vocabulary logit.** This is
a numerical capture check on the original Qwen3-0.6B, not training or a test of the
published 8B detector.

| Library and API | Site | Changed logits / compared logits |
|---|---|---:|
| NNsight 0.8.0, trace with `.save()` | Block 14 query normalization | 0 / 10,635,520 |
| NNsight 0.8.0, trace with `.save()` | Block 14 residual output | 0 / 10,635,520 |
| TransformerLens, `run_with_cache` | Block 14 query normalization | 0 / 8,508,416 |
| TransformerLens, `run_with_cache` | Block 14 residual output | 0 / 8,508,416 |

Counts cover the entire vocabulary at every input position across four archived prompts
per library. These are exact numerical comparisons, not independent statistical samples.
The prompt sets differ between libraries. All comparisons are within the same library,
model, input, and execution mode. Yes/No log probabilities and top token IDs are in the
prompt-level records; each was unchanged when capture was enabled.

Captured tensors had the expected dimensions, were nonempty and finite, changed across
prompts, and repeated exactly on identical inputs. Repeated no-capture forwards also
matched. NNsight additionally matched bare-model execution before and after tracing and
verified unchanged model tensors and flags.

## Execution

- NVIDIA RTX A5000, CUDA 12.8 PyTorch build; PyTorch 2.10.0, Transformers 5.18.0.
- Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, BF16,
  `eval()`, `no_grad()`, batch size 1.
- NNsight 0.8.0 used SDPA attention. TransformerLens used its default eager attention,
  with compatibility weight transformations disabled.
- TransformerLens source `bd55eab6737584685b921fdfcd15f191579e64b1`; its distribution
  metadata reports version `0.0.0`, so the source commit identifies this installation.
- Scientific source `914e6b6026bd`; each artifact records the executed script SHA.

The optional full-graph compilation of TransformerLens's actual public cache API failed
at an unsupported `id(dict)` operation in its alias registry. No eager fallback was
allowed and no compiled numerical result was produced. This does not test its separate
compiled vLLM integration.

## Evidence and limits

- [NNsight summary](nnsight/summary.json) and [per-prompt comparisons](nnsight/prompts.json).
- [TransformerLens comparisons and compile error](transformerlens/audit.json).
- [Exact environment](environment.txt) and [artifact checksums](archive.json).
- [Protocol](../../../experiments/library-capture.md) and
  [CUDA launcher](../../../scripts/bootstrap_library_capture.sh).

Independent review checked actual CUDA devices, script/source hashes, capture freshness,
within-runtime comparisons, version provenance, and the failed compile attempt.

This result does not establish identical activations at every internal boundary, absence
of a footprint on every input, or the behavior of a trained detector. It shows that these
ordinary eager GPU capture workflows did not reproduce our compiled wrapper's output
footprint in this bounded test. Compiled-library integration and 8B detector transfer
remain untested.

The run completed with exit status 0. **Cost: $0.0079.** The allocated pod was confirmed
deleted and the exact temporary budget approval removed. An initial disk-size preflight
was rejected before allocation; increasing disk to 60 GB did not change scientific code.
