# Warm-start joint learning: run 039

**The fixed 1,024-update fit completed; runner exit status is 0 and pod deletion is confirmed.** Training starts from the successful native tail from detection-only run 034, uses a smaller learning rate, and adds the animal task. This is a new-task warm start: Adam state and step counters restart at zero.

The final split-runtime development evaluation is correct on every observed answer for both tasks. Ordinary whole-graph compiled transfer is correct on 64/64 detection answers and 63/64 animal answers. These are development results; this run deliberately scores no test or challenge examples. A separate fresh-process confirmation using the frozen v3 bank is required before a new final-test claim.

## Protocol and provenance

- Qwen3-8B with final two native blocks 34–35 and final normalization trainable; original output head and source prefix remain fixed. No added classifier or architectural change.
- Original 034 tail SHA-256 `8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a`. Its source run had 1,024 updates. The present run performs 1,024 additional new-task updates, but its x-axis and optimizer counters correctly start from zero.
- Balanced paired full-vocabulary CE, 512 training prompts (256/task), 64 development prompts (32/task), batch 8 pairs, length 96, learning rate 1e-6, seed 20261019. Exactly 8,192 prompt-pair visits in this run.
- Source commit `6a9b6741fff3982fc27285ed3e4f072882a0769b`; base revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Dataset file hash `8ae5cc965dd9904603237bbd8dec08c50b693e2d282a3299bb4457fe4850e02c`. Only training/development records are used, despite the source bank containing unused test/challenge splits.
- Frozen-source/output-head checks pass at the end. Warm-start provenance checks tensor shapes/dtypes, checkpoint, available revision, runtime and exact frozen-head identity. The 034 artifact lacks a complete frozen-prefix checksum; the manifest explicitly does not claim that stronger identity verification.
- Primary path: compiled frozen-source states plus eager BF16 native tail. Whole-graph transfer is reported independently. The initial whole/split difference includes previously learned 034 weights; it is not a pure compiler comparison.

## Fixed final results

| Runtime | Split | Task | Prompt pairs | Actual accuracy (95% CI) | Full CE, nats | Paired gap (95% CI), nats |
|---|---|---|---:|---:|---:|---:|
| split_inference | train | detection | 256 | 100.00% [100.00, 100.00] | 0.00000106 | 29.754 [29.583, 29.918] |
| split_inference | train | animal | 256 | 100.00% [100.00, 100.00] | 0.00000849 | 26.347 [26.151, 26.547] |
| split_inference | validation | detection | 32 | 100.00% [100.00, 100.00] | 0.00000251 | 29.289 [28.812, 29.723] |
| split_inference | validation | animal | 32 | 100.00% [100.00, 100.00] | 0.00788022 | 26.074 [25.566, 26.566] |
| wholegraph_transfer | train | detection | 256 | 99.80% [99.41, 100.00] | 0.00559250 | 24.447 [24.125, 24.751] |
| wholegraph_transfer | train | animal | 256 | 100.00% [100.00, 100.00] | 0.00002166 | 24.554 [24.342, 24.761] |
| wholegraph_transfer | validation | detection | 32 | 100.00% [100.00, 100.00] | 0.00026620 | 24.008 [23.332, 24.680] |
| wholegraph_transfer | validation | animal | 32 | 98.44% [95.31, 100.00] | 0.01152614 | 24.414 [23.867, 24.945] |

Accuracy uses the actual top token over the full vocabulary. The interval is a 5,000-resample whole-prompt paired bootstrap, conditional on this run and observed question families. At 100% empirical accuracy, bootstrap [100%,100%] does not establish perfect population accuracy. For 32/32 development pairs correct in both conditions, the separate exact binomial interval is [89.11%,100%] under iid prompt assumptions; shared templates mean those assumptions are not guaranteed. Equal-family sensitivity summaries are included in `fit/robustness.json`.

The improvement over cold-start 036/038 is a comparison across several changes (initial weights, learning rate, seed and update budget). It does not isolate which change is causal. The development set was observed in earlier runs, so these results support optimization progress rather than an untouched final generalization claim.

## Preserved startup failures and cost

`failures.json` records the original 039 launcher path error (exit 127, before model training, cost $0.0103) and retry 039r upload-preflight failure (no pod created, zero reported cost). The successful 039r2 run cost $0.2519. All rates/costs refer only to these runs; account totals and infrastructure logs are excluded.

The final tail and source cache are large release assets, not Git files. `fit/archive.json` records sizes and SHA-256 hashes. Raw score JSON/JSONL, provenance, progress and final development figures are archived here.
