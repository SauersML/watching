# Run 041: strict archived replay failed on the A5000

The first selected training example failed the archived-answer replay gate, before
any causal interventions or learned-tail measurements. The gate was not relaxed.
The pod was deleted; total cost was $0.0117. No training occurred.

What passed: all source-cache fingerprints and metadata, selective native-head
checksum, artifact compatibility, and exact full-vocabulary equality of the new
eager execution with and without activation-copy hooks.

What failed, original tail, training prompt index 42:

| Measurement | Run 034 archive | Run 041 replay |
|---|---:|---:|
| Present-condition Yes-minus-No log-odds | −6.125 | −5.875 |
| Absent-condition Yes logprob | −9.28131199 | −9.28134632 |
| Absent-condition No logprob | −3.53131223 | −3.53134608 |

The absent log-odds matched; its two logprobs differed by approximately 0.000034.
The full mismatch record is in `metrics.jsonl`. The raw manifest still says
`running` because the exception occurred before finalization; `archive.json`
records the actual failed status without rewriting the raw file.

## Construction checks before considering a retry

- `EagerNativeTail.__init__` and `.forward` are AST-identical between run 034's
  source commit `65f3cb2a46c9f3b2c92c4199ac596289065b0f72` and run 041's snapshot
  `d18ce2f2`. Both use the same native blocks/norm, BF16 head, singleton states,
  saved rotary/mask/position arguments, and explicit math-SDPA context.
- The selective loader assigns the source BF16 tensors into those same native
  modules. CPU tests on real tiny Qwen3 verify exact output equality to constructing
  the native tail from the full model; they do not certify every GPU kernel.
- The output-head SHA is exactly
  `6e46ee56769d64b9a338f350100bdfca5f26a91c3fb5bf479ffcb36c733f7939`.
  The released cache and learned-tail SHA checks also passed. The early failure
  happened with original weights, before installing the trained artifact.
- The runtime metadata reports PyTorch 2.10.0+cu128, Transformers 5.18.0,
  Triton 3.6.0, BF16 reduced-precision reduction allowed, and TF32 matmul disabled.

Run 034 used an L40S. Run 041 used an RTX A5000 (compute capability 8.6), driver
580.159.04. Different hardware/backend execution is a plausible cause, but this
failed comparison does not isolate a GPU, driver, loader, or library cause.
A matched L40S retry with unchanged code, inputs and exact gates is the proposed
next check. No numerical tolerance or missing-outcome exclusion is introduced.

The input cache and tail remain available in the public
[034 release](https://github.com/SauersML/watching/releases/tag/sft-034); they are
not duplicated here. Compact raw outputs and their checksums are archived here.
