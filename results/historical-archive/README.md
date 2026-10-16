# Historical result archive audit

This supplement makes locally retained historical results available alongside the existing experiment archives, including failed and interrupted attempts. It does not turn incomplete attempts into completed experiments.

Audited **27 historical run directories**. Added **1317 previously unarchived compact artifacts** (22.40 MiB). Exact-byte duplicates reference their existing archive location in [coverage.json](coverage.json).

Whole-job completion was checked against every run’s last recorded terminal status. A stage-local `complete` event is insufficient: run 021 completed its detector but failed during later animal-task initialization. The audit identifies 18 completed jobs, eight interrupted/failed attempts, and one partially successful job with a later-stage failure.

The main additions are complete raw score histories for runs 013–015, intermediate score files omitted from earlier compact learning-rate summaries, and explicit failure records. Each source artifact is indexed by run-relative path, SHA-256, size, and repository archive path.

| Run | Status | Added artifacts | Already archived |
|---|---|---:|---:|
| [watching-answer-012](runs/watching-answer-012/README.md) | completed | 131 | 21 |
| [watching-audit-005](runs/watching-audit-005/README.md) | completed | 2 | 5 |
| [watching-drift-007](runs/watching-drift-007/README.md) | completed | 0 | 3 |
| [watching-execution-018](runs/watching-execution-018/README.md) | interrupted_or_failed | 0 | 0 |
| [watching-execution-018r](runs/watching-execution-018r/README.md) | completed | 0 | 24 |
| [watching-execution8b-021](runs/watching-execution8b-021/README.md) | partial_success/later_stage_failed | 0 | 23 |
| [watching-head-015](runs/watching-head-015/README.md) | completed | 13 | 1 |
| [watching-larger-011](runs/watching-larger-011/README.md) | completed | 132 | 20 |
| [watching-native-019](runs/watching-native-019/README.md) | completed | 0 | 24 |
| [watching-ood-020](runs/watching-ood-020/README.md) | completed | 0 | 12 |
| [watching-paired-006](runs/watching-paired-006/README.md) | interrupted_or_failed | 2 | 1 |
| [watching-paired-006r](runs/watching-paired-006r/README.md) | completed | 0 | 38 |
| [watching-precision-014](runs/watching-precision-014/README.md) | completed | 284 | 20 |
| [watching-probechoices-022](runs/watching-probechoices-022/README.md) | interrupted_or_failed | 0 | 3 |
| [watching-rate-008](runs/watching-rate-008/README.md) | completed | 135 | 42 |
| [watching-readouts-016](runs/watching-readouts-016/README.md) | completed | 0 | 10 |
| [watching-sft-002](runs/watching-sft-002/README.md) | completed | 37 | 11 |
| [watching-sft-003](runs/watching-sft-003/README.md) | interrupted_or_failed | 10 | 0 |
| [watching-sft-003r](runs/watching-sft-003r/README.md) | completed | 4 | 76 |
| [watching-single-004](runs/watching-single-004/README.md) | interrupted_or_failed | 14 | 4 |
| [watching-single-004r](runs/watching-single-004r/README.md) | completed | 0 | 80 |
| [watching-slow-010](runs/watching-slow-010/README.md) | completed | 130 | 22 |
| [watching-strong-009](runs/watching-strong-009/README.md) | interrupted_or_failed | 2 | 1 |
| [watching-strong-009r](runs/watching-strong-009r/README.md) | completed | 131 | 21 |
| [watching-suffix-017](runs/watching-suffix-017/README.md) | interrupted_or_failed | 2 | 1 |
| [watching-suffix-017b](runs/watching-suffix-017b/README.md) | interrupted_or_failed | 2 | 1 |
| [watching-untied-013](runs/watching-untied-013/README.md) | completed | 286 | 18 |

## Missing or intentionally excluded material

- Original 003 and 004 attempts did not synchronize all stdout events into their metrics files. Additional structured events are recovered through updates 460 and 500 respectively; later events and unsaved intermediate weights cannot be reconstructed.
- Run 021 completed the execution-detector fit, then failed with the same CPU-index/CUDA mismatch as 022 while initializing the animal task. Its successful detector scores are preserved; later stages were completed in 021r.
- Attempts 006 and 009 ended during initial compilation without training results. Attempt 018 failed before its bootstrap script started. Attempts 017/017b failed numerical parity checks. Attempt 022 failed a device-placement check before fitting.
- Full checkpoint tensors and cached activation tensors remain local. Compact answer-row checkpoints are indexed and archived. The checkpoint inventory lists remaining local file names and sizes without including their contents.
- Runs 021r and 022r belong to the separate animal-choice archive; current 023/024 runs are outside this historical audit. Their absence here is not a missing historical result.
- No raw account logs, credentials, pod descriptions, SSH files, or account-wide spending information are included.

## Provenance and interpretation

Files are copied byte-for-byte when newly added. Existing exact-byte copies are referenced rather than duplicated. Failure summaries quote only relevant terminal messages and give source-log line numbers and SHA-256 hashes. Where no terminal cause was recorded, the reason is explicitly unresolved. Costs are per-run totals extracted from deletion receipts, never account-wide balances.

The directory name is a storage supplement, not a new experiment. Read the original experiment plans and manifests for objectives, conditions, and statistical interpretation. All preserved score histories are exploratory where earlier reporting did not prespecify a comparison.
