# First-role key geometry: a small CPU check of the prefix-failure prediction

**The proposed prediction holds in this CPU replay:** the trained final block's key projection nearly cancels the common first-`user` activation in KV head 1, leaving a large angular difference between probe conditions. It does not do this for the first-`system` activation. This is descriptive evidence connecting a candidate circuit to the chat-layout failure, not a native-GPU replication or causal patching result.

The selection was fixed before replay: the lexically first development base-context ID for each task, all four chat layouts, both probe conditions. That is eight prompt variants, 16 executions per saved tail. Both run-034 and run-039 tails were evaluated. The computation took 133 seconds on four Mac CPU threads; no GPU or download was used.

## What was computed

1. Start with actual GPU-cached block-34 inputs from run 046, captured after the frozen source's block 33.
2. Replay only saved trained block 34 using CPU BF16, eager math-SDPA.
3. Compute block 35's input RMSNorm, key projection `Wk`, and key RMSNorm, preserving the complete fixed 1×96 sequence shape until selecting the role-token vector.
4. Select the first chat-role token, logical position 1, by actual token ID. It is `user` in no-system layouts and `system` in system-first layouts.
5. Report all eight KV heads. Query heads 4–7, including the externally reported head 6, share KV head 1.

The original base `Wk` tensor was read from the existing pinned local safetensors snapshot. The matched control applies that base matrix to the **same trained block-34 output and trained block-35 input normalization**, isolating the projection change. It is not a complete original-model forward, so its norm must not be equated with the separately reported full-base-model norm near 70.

## KV head 1

For paired vectors `k_absent` and `k_present`, the midpoint is their average and the difference is `k_present − k_absent`. Angles compare the two condition vectors. RMSNorm amplification is the L2 difference after key RMSNorm divided by the difference before it; learned per-coordinate normalization weights can also change angles.

| Tail / first role | Raw key norms, absent / present | Raw midpoint norm | Raw condition angle | Post-RMSNorm angle | Difference amplification |
|---|---:|---:|---:|---:|---:|
| 034 / user | 6.561 / 6.858 | 6.374 | 36.408° | 34.036° | 2.722× |
| 034 / system | 61.935 / 61.953 | 61.943 | 0.683° | 0.514° | 0.327× |
| 039 / user | 3.042 / 2.231 | 1.402 | 117.968° | 93.956° | 7.009× |
| 039 / system | 61.806 / 61.945 | 61.875 | 0.622° | 0.482° | 0.328× |

The paired source footprint is present for **both roles**: cached input differences have L2 norms 5.487 for `user` and 6.043 for `system`, with midpoint norms 654.072 and 661.073. System-first failure is therefore not explained here by an absent source footprint. The trained reader responds very differently to its geometry.

For run 039, the input immediately before `Wk` has a probe-condition angle of 1.534° for `user` and 0.729° for `system`. Projection changes these to 117.968° and 0.622°, respectively. This separates the learned projection's effect from the subsequent RMSNorm scaling.

## Cancellation relative to the original projection, on identical inputs

| Tail / role | Base-Wk midpoint | Trained-Wk midpoint | Trained / base midpoint | Cosine: base projection vs projection change |
|---|---:|---:|---:|---:|
| 034 / user | 22.698 | 6.374 | 0.281 | −0.9613 |
| 034 / system | 64.625 | 61.943 | 0.958 | −0.5288 |
| 039 / user | 21.131 | 1.402 | 0.0664 | −0.9978 |
| 039 / system | 64.603 | 61.875 | 0.958 | −0.5312 |

The run-039 `user` midpoint update is almost exactly opposite the base projection, which is consistent with cancellation. The projection-change quantities subtract separately BF16-rounded projections; they are not an exact real-arithmetic decomposition of the weight update.

## All-head comparison

The following are raw key midpoint norms and probe-condition angles. Every head is retained in `summary.json`, with all pre/post-normalization norms and differences.

| KV head | 034 user norm / angle | 034 system norm / angle | 039 user norm / angle | 039 system norm / angle |
|---|---:|---:|---:|---:|
| 0 | 64.217 / 1.570° | 63.555 / 0.589° | 65.003 / 1.636° | 63.315 / 0.551° |
| 1 | 6.374 / 36.408° | 61.943 / 0.683° | 1.402 / 117.968° | 61.875 / 0.622° |
| 2 | 57.746 / 1.341° | 60.365 / 0.482° | 59.342 / 1.466° | 60.437 / 0.454° |
| 3 | 79.519 / 1.007° | 74.382 / 0.441° | 81.606 / 1.119° | 74.413 / 0.409° |
| 4 | 68.764 / 1.567° | 67.900 / 0.594° | 70.379 / 1.705° | 67.917 / 0.558° |
| 5 | 68.046 / 1.639° | 67.725 / 0.644° | 69.216 / 1.833° | 68.030 / 0.597° |
| 6 | 63.181 / 1.462° | 59.912 / 0.580° | 64.550 / 1.611° | 60.047 / 0.537° |
| 7 | 75.590 / 1.145° | 85.744 / 0.406° | 76.735 / 1.203° | 85.644 / 0.417° |

Within each saved tail, **all recorded role vectors are bit-identical across the four selected variants beginning with the same role**. The task/question and later history do not supply independent observations of this early causal position. There are effectively two role geometries per tail here; reporting confidence intervals over the repeated variants would be misleading.

## What this does and does not establish

This matches a concrete prediction of a reader specialized to the initial `user` token: cancellation and normalization expose a large condition-dependent direction there, while leaving the `system` token's large common component intact. It is consistent with run 046's strong user-first performance and failed system-first generalization.

It does **not** independently verify the external key/value/query patching effects, show that KV head 1 is necessary or sufficient for native-GPU behavior, or reproduce final answer logits. CPU PyTorch 2.11.0 / Transformers 5.17.0 differs from the GPU training stack. The original compiler footprint's cause is a separate question. A matched native runtime intervention is needed for a causal circuit claim.

## Files and reproduction

- `selection.json`: fixed eight selected variants, task and base-context identities.
- `summary.json`: all per-tail/per-variant/per-head measurements, local runtime versions, exact checkpoint hashes, base weight tensor identity, and repeated-vector checks.
- `role_vectors.pt`: small raw vectors used for geometry summaries.
- `selected_source.pt`: only the eight selected source-state pairs, checked bit-for-bit against the full run-046 cache. Unselected indices are `None`; the public development metadata is retained for deterministic indexing. This is enough to replay this audit without the large full cache.
- `archive.json`: artifact sizes and SHA-256 checksums. The summary's original cache hash refers to the full run-046 cache; the reproduction subset necessarily has a different file hash.

The fixed tails are available in GitHub releases for [034](https://github.com/SauersML/watching/releases/tag/sft-034) and [039](https://github.com/SauersML/watching/releases/tag/joint-039). Exact expected tail hashes are recorded in the summary; verify downloaded assets against those identities.

With those tail files and an already available pinned Qwen3-8B snapshot:

```sh
PYTHONPATH=src python scripts/audit_prefix_key_geometry.py \
  --cache results/prefix-key-geometry-cpu/selected_source.pt \
  --data results/diverse-joint-046/final-step1024/fit/data.json \
  --tail 034=/path/to/034/native_tail.pt \
  --tail 039=/path/to/039/native_tail.pt \
  --model-dir /path/to/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218 \
  --out /path/to/new-audit-directory
```

The script never downloads model files, uses no GPU, and caps CPU execution at four threads. It writes the deterministic selection before reading the cached tensors and flushes each paired result immediately.
