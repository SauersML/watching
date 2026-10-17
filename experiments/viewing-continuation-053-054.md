# Exact-cache continuations of viewing modes 051 and 052

Frozen before launch: continue each arm's final 1,024-update native tail for **2,048 additional updates**, ending at global update **3,072** if the time cap permits. No checkpoint selection. These are optimization follow-ups on previously inspected data, not fresh confirmatory tests.

## Unchanged inputs and architecture

Reuse the full source cache from each corresponding arm, including exact condition ordering, all tensor bytes/layouts, positions, prompts and labels. Do not compile or run the frozen source prefix again. Buffer remains the historical buffer-source cache; identity remains the historical materialization-source cache. The learner receives only its original cached tensors and positional inputs; no condition flag is passed into the native blocks. Train native blocks 34–35 and final norm; keep the existing output head frozen. No new classifier or architecture change.

| Arm | Source cache SHA-256 | Initial tail SHA-256 |
|---|---|---|
| buffer053 | `77aa20aa4be4af6a83f9a5197654a01b4316b9ff741d068561ce9e35558ba64c` | `587974a6227869bfb58096726a6e66ee53bbc67b0ecd17dcc11190533730efff` |
| identity054 | `381c062b104e5c1440039ea51a1d73822a6d3894839e95eabfd35d79e5f4a066` | `5b509d674bb7df7916e72ee9038d47efb06059c59730f2d7a30cb7822b90a5ab` |

The pinned Qwen3-8B revision is `b968826d9c46dd6066d109eabc6255188de91218`. A base model is loaded on CPU solely to obtain the existing native module definitions and original head, whose digest must match the predecessor. It performs no forward pass. The entire continued computation is the existing eager BF16 math-SDPA tail. This experiment makes **no new whole-graph compiled-serving transfer claim**.

## Replay gates before optimization

1. Validate both artifact checksums, every cached tensor fingerprint and position identity, and exact saved selected records.
2. Rerender all saved prompts with the pinned tokenizer; require exact input IDs and masks without truncation.
3. Verify artifact step 1,024, frozen output-head checksum, tail tensor keys/shapes/dtypes, model and mode provenance.
4. Replay all 256 training and 64 development pairs under both conditions. Every archived Yes/No log probability, score, cross-entropy, answer mass, top token and correctness must match **exactly**. Full-vocabulary logits were not archived, so this is not an all-vocabulary replay claim. A mismatch is saved and stops before optimizer construction; no tolerance or automatic retry.
5. Require matching gradient-enabled and no-gradient tail outputs on the practiced audit pair at every reporting checkpoint.

## Training and reporting

Use the original learning rate 5e-6, batch 8 prompt pairs, seed 20261017, paired FP32-master AdamW, zero weight decay and gradient clipping at 1. **Adam moments restart** because the predecessor saved native weights only. Skip exactly 1,024 batches in the original deterministic minibatch stream; the next batch is the original stream's 1,025th, not its first. Both absent and present executions contribute full-vocabulary correct-token cross entropy.

Report all training/development pairs at baseline, +32, +128, and every 128 additional updates. Save recoverable native weights atomically every 256 updates and at the actual endpoint. Distinguish completed updates from the last saved checkpoint. Preserve the unmodified cache by reference/checksum instead of uploading a duplicate. Paired log-odds and both-conditions-correct accuracy remain the headline metrics; also retain all raw log probabilities and losses. Whole-prompt bootstrap intervals condition on this fixed seed/runtime and do not represent compiler or seed uncertainty.

Score the historical 64-prompt test only after all 2,048 requested updates; a time-limited run saves weights and skips test scoring. This previously inspected cohort is held out from optimization but exploratory. No early stopping or selection uses test scores.

Each job is capped at 30 minutes on an L40S priced at most $1.20/hour (at most $0.60 each, combined $1.20). The optimizer time cap is 1,350 seconds, reserving time for installation, baseline replay, checkpointing and final reporting. Idle-utilization shutdown is disabled; the wall-clock budget remains active. If the fixed endpoint cannot fit, report the exact partial endpoint rather than treating it as completion. Exact temporary approval entries are removed after runner termination; all job-owned pods are deleted.

Implementation: `src/watching/viewing_continue.py`; immutable-source bootstrap: `scripts/bootstrap_viewing_continue.sh`.
