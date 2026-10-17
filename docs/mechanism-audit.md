# Why the buffer-copy probe changed this execution

**The current buffer probe is not numerically passive. Its extra output changes the kernel that computes query normalization and rotary position encoding.** In the captured CUDA source, the compiler changes a three-dimensional reduction into a flattened two-dimensional reduction. The buffer receives a rounded BF16 copy, but the model's rotary calculation continues using FP32 intermediates. Therefore, simply saying that the buffer forces the model to consume rounded values does not explain this case.

This audit separates observations from the remaining causal question. A changed kernel is established. Which changed arithmetic instruction accounts for the measured full-model discrepancy is not yet established. The archived kernels and a small isolated replay make that question testable without another model-training run.

## What actually changed in the measurements

Fresh evaluation 038 used Qwen3-8B on an L40S, PyTorch 2.10 with CUDA 12.8, BF16, query normalization in block 14, singleton batches padded to 96 tokens, and an explicitly supplied position tensor. The same textual input was evaluated with no capture, a reference capture, and a buffer copy.

The buffer comparison checked **every vocabulary logit at the answer position**, for each of 64 new prompts:

| Checkpoint | Prompts passing exact buffer/no-buffer equality | Largest absolute logit difference | Mean fraction of vocabulary logits differing |
|---|---:|---:|---:|
| Original base model | 0 / 64 | 2.0000 | 92.01% |
| Trained 034 model | 0 / 64 | 0.8125 | 86.41% |

These fractions count unequal floating-point values, not error rates or independent statistical samples. They do not measure whether the differences share a learnable direction. The trained detector still answered No under the buffer condition in these examples. That result does **not** make this a valid no-footprint control: the all-vocabulary equality gate failed.

Sources: [base buffer scores](../results/confirm-validation-038/fresh034/base_buffer.json), [trained buffer scores](../results/confirm-validation-038/fresh034/trained_buffer.json), and [evaluation implementation](../src/watching/validate_tail.py). The first prompt was repeated in every mode; those repeated outputs were bit-identical. This is not evidence for an unpredictable race or random timing noise.

## Why the earlier result is not a matched contradiction

The earlier experiments differ both in implementation and in what was checked.

| Earlier experiment | Current fresh 034 evaluation |
|---|---|
| Qwen3-0.6B; one quick HF check retained only two blocks | Qwen3-8B, all 36 blocks |
| Quick HF probe in block 0; vLLM probe in block 14 | Probe in block 14 |
| Global oversized flat buffer, sliced to activation size; `copy_(activation.reshape(-1))` | Exactly shaped registered buffer; `copy_(activation)` |
| HF quick check: two prompts, natural padded length, all token logits | One prompt at a time, fixed left padding to 96, final-token logits |
| HF quick check: default cache configuration; `torch.compile(dynamic=False)` | `use_cache=False`, explicit positions, `fullgraph=True`, `dynamic=False` |
| L4 for the successful CUDA buffer tests | L40S |
| vLLM engine and its own compilation boundaries | HF model compiled as this evaluator's graph |

The archived vLLM results checked **selected logprobs and generated tokens**, not the complete vocabulary distribution. Rechecking the saved objects confirms equality of all 72 selected prompt-token logprobs, 1,600 selected generated-token logprobs, and complete generated-token ID lists for each of six buffer sites. That is useful evidence, but narrower than all-logit equality. The earlier two-block HF test did compare the complete output tensor; it was a different model, shape, and capture program.

Consequently, the defensible historical claim is that the buffer implementation was passive for the measured outputs in those executions. It was never a general guarantee. Nor does one L4/L40S comparison identify GPU architecture as the cause: model, shape, wrapper, graph partitioning, and compiler choices changed together.

Historical local sources inspected: `probe_bits/vllm_probe.py`, `vllm_compare.py`, `probe_sweep.py`, the saved `probe-bits-a/out/*_c.json` results, and `quick_cuda.py` from the original scratch directory. The unsuccessful/incomplete HF sweep's empty output is not treated as a completed experiment.

## What the actual CUDA programs show

The evidence is captured from the running fresh-evaluation process, not an illustrative reimplementation. The complete relevant kernel sources, candidate lowerings, and SHA-256 hashes are preserved in [the bounded compiler archive](../results/mechanism-038/manifest.json).

### No capture

The normalized-query/rotary kernel indexes tokens, heads, and channels separately:

```python
# Schematic excerpt of the actual indexing and reduction.
ynumel = 96
xnumel = 32
r0_numel = 128
accumulator = tl.full([YBLOCK, XBLOCK, R0_BLOCK], 0, tl.float32)
sum_squares = tl.sum(accumulator, 2)[:, :, None]
```

Its query address is `channel + 128*head + 4096*token`. The native rotary output is the only stored result of this kernel.

### Buffer capture

The compiler flattens token and head into a single axis:

```python
xnumel = 3072  # 96 tokens times 32 heads
r0_numel = 128
accumulator = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
sum_squares = tl.sum(accumulator, 1)[:, None]
```

The query address is `channel + 128*token_head`. The kernel writes both the normalized-query buffer and the native rotary output. In the actual generated source:

```python
tmp16 = weight * normalized_query_fp32
# ... rotated query and sine term ...
tmp62 = tmp16 * cosine
tmp63 = tmp62 + rotated_query_times_sine
tl.store(probe_buffer, tmp16)  # BF16 destination
tl.store(rotary_output, tmp63)  # BF16 destination
```

There is **no BF16 conversion followed by a reload between `tmp16` and `tmp62`**. Adding the probe did not merely insert a store while leaving the producing kernel unchanged; it changed that kernel's indexing and reduction layout. Yet the rounded probe buffer is not substituted into the native rotary arithmetic.

The long kernel names contain `scaled_dot_product_efficient_attention`, but these bodies compute normalization and rotary encoding. Attention itself is a separate external call. The name alone does not locate the numerical cause inside attention.

### What the lower-level compiler artifacts add

Four no-capture candidates and six buffer candidates were recovered. They include distinct per-thread layouts, warp layouts, and reduction block sizes. The no-capture candidates reduce channels in blocks of 128; some buffer candidates use blocks of 32, 64, or 4 and iterate. PTX contains different shuffle/reduction and floating-point instruction schedules.

These are concrete mechanisms by which mathematically equivalent sums and products can round differently. **We do not yet know which fresh-process candidate was selected.** The fresh wrappers disabled compiler caches; recovered `best_config` files do not identify a winning buffer candidate for this graph. It would be incorrect to pick a convenient PTX file and claim it caused the observed model result.

## Completed isolated check: replay the actual kernels

The [replay script](../scripts/replay_probe_kernels.py) runs all recovered candidate configurations on identical finite BF16 query tensors, BF16 norm weights, and FP32 rotary angles. It uses the archived JIT function bodies without editing their arithmetic, bypasses autotune selection, and explicitly preserves FP fusion/FTZ/debug compiler settings.

```sh
python scripts/replay_probe_kernels.py \
  --artifacts results/mechanism-038 \
  --out /tmp/probe-kernel-replay.json \
  --seeds 0 1 2
```

It reports every no-capture/buffer candidate comparison, FP32 and FP64 reference discrepancies, repeated-run equality, and recompiled PTX hashes. No model weights are loaded. The references are numerical comparison targets, not claims that eager PyTorch exactly reproduces Triton's arithmetic.

The replay completed on the existing L40S worker, using PyTorch 2.10.0+cu128 and Triton 3.6.0. Every kernel wrote finite outputs and reproduced its own output exactly on rerun. Raw results: [replay.json](../results/mechanism-038/replay.json).

| Synthetic input seed | Unequal no-capture/buffer candidate pairs | Largest number of differing rotary values | Largest absolute difference |
|---|---:|---:|---:|
| 0 | 21 / 24 | 7 / 393,216 | 0.0078125 |
| 1 | 15 / 24 | 5 / 393,216 | 0.001953125 |
| 2 | 21 / 24 | 6 / 393,216 | 0.0078125 |

The largest fraction is **0.00178% of rotary values**. Some candidate pairs agree exactly; others do not. This demonstrates that the local compiler/configuration difference can create a numerical footprint, even though both kernels consume the same inputs and both retain FP32 normalized values for the native computation. It does **not** by itself explain the much larger fraction of differing final-model logits. Real activation inputs, the chosen configuration, and propagation through subsequent layers remain unmeasured here.

The replay recompiles the unchanged JIT bodies with archived configuration values. Recompiled PTX file hashes differ from the originally captured files; debug paths and standalone specialization can affect these hashes. Until normalized instruction-level comparison is completed, this is an exact-source replay, not a claim of byte-identical original binaries.

Interpretation:

- If identical inputs yield different rotary outputs, the local kernel rewrite is sufficient to produce a numerical footprint for those inputs/configurations.
- If some configurations agree and others differ, launch selection matters; a universal buffer-copy guarantee is untenable.
- Synthetic replay does not prove which configuration ran, nor that these were the first differing tensors for a real prompt.
- The strongest follow-up is to intercept the already compiled kernel launch, clone its query/weight/angle inputs **outside** the compiled graph, and replay both kernels on those exact inputs. The interception must itself preserve full-model logits. Adding another traced activation output would risk changing the graph under investigation.

Before another model-scale sweep, isolate the relevant kernel and recover its actual launch configuration. Do not turn the current hypothesis into a general claim about every copy, every probe site, or every inference stack.
