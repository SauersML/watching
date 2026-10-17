# Native forward-hook training, run 061r

The native last two Qwen3-8B blocks and final norm completed **1,024 updates** with the existing vocabulary head fixed. The explicit `boundary_observed` source runtime was used because strict extraction changed compiled logits. These results do not establish strict extraction parity.

## Fixed-step results

Accuracy is the actual Yes/No decision accuracy over both conditions. Paired rank is the fraction of same-prompt pairs where the saved-activation condition has a larger Yes-minus-No logit gap; it is not classification accuracy.

| Runtime | Split | Accuracy | Paired rank | Full-vocabulary CE |
|---|---|---:|---:|---:|
| Observed source, cached native tail | Train (256 prompts) | 98.83% | 100% | 0.0436 |
| Observed source, cached native tail | Validation (64) | 78.13% | 100% | 1.1651 |
| Observed source, cached native tail | Held-out wording test (64) | 67.19% | 100% | 2.7025 |
| Original native wholegraph transfer | Train (256) | 53.52% | 72.07% | 1.2138 |
| Original native wholegraph transfer | Validation (64) | 52.34% | 67.97% | 3.0601 |
| Original native wholegraph transfer | Held-out wording test (64) | 51.56% | 64.06% | 4.5689 |

The observed-source detector learned a strong paired ranking signal. Actual Yes/No accuracy transferred poorly to the original graph. Validation calibration varied substantially during training; the table reports the prescribed final step, not a selected peak. The untrained original wholegraph validation accuracy was 50.00%.

## Runtime and provenance

- Original `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`; source commit is recorded in [SOURCE_REVISION](SOURCE_REVISION).
- Standard PyTorch `q_norm` forward hook at layer 14 saves `output.detach()` in an external dictionary and returns `None`. The raw Hugging Face model is compiled; output adaptation occurs outside compilation.
- Both condition specializations passed capture shape/count/freshness checks and repeated full-sequence output checks. Separate native and boundary-observed invocations were compared on every input; their differences are retained, not treated as parity.
- The learner received only cached hidden/mask/rotary tensors and shared positions. None of the 256 training or 64 validation pairs had identical complete learner inputs.
- Updated blocks: 34 and 35, plus final norm. No classifier was added. Head SHA256 before and after is identical in [fit/manifest.json](fit/manifest.json).
- Matched frozen data selection, source hashes, runtime versions, and compilation counters are retained in the manifest, snapshot, environment, and raw metrics files.

## Artifacts and recovery

[fit/metrics.jsonl](fit/metrics.jsonl) contains the full learning curve. All per-step inference JSONs, final transfer JSONs, row-level scores, observed-runtime comparisons, and cache fingerprints are retained under `fit/`. The 542 MB activation cache is excluded from this compact archive.

The [generated compiler source and metadata archive](https://github.com/SauersML/watching/releases/download/library-capture-training-061r-hook/compiler-artifacts.tar.gz) is a release asset rather than a git blob. [compiler-artifact-release.json](compiler-artifact-release.json) records its SHA256 and size; [compiler-artifacts.json](compiler-artifacts.json) lists all 904 entries. The recovered archive was checked against both.

**Learned weights are not yet recovered locally.** The final manifest records `native_tail.pt` at 771,801,758 bytes and step 1,024, but that file remains on stopped pod `1x0txti3lk8up3` and its persistent volume. Both GPU resume and documented CPU-only rescue are currently blocked by host capacity (GPU or free RAM). The volume must remain intact until the file is copied and its remote SHA256, local SHA256, size, and checkpoint metadata are verified. The recovery helper installs rsync after restart and verifies every copied file before cleanup. No new scientific training is needed.

The worker exited successfully (`worker_exit.json`: 0); runner exit 1 occurred during post-stop recovery and does not invalidate the completed experiment. Known GPU segments total approximately $0.580 before eventual successful recovery; storage and final recovery costs remain pending. The originally advertised 30-minute guard was subsequently replaced by a longer watchdog, so this archive does not claim a 30-minute bound.
