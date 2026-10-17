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

**Final learned weights have been recovered and verified.** [native_tail.pt](https://github.com/SauersML/watching/releases/download/library-capture-training-061r-hook/native_tail.pt) is a release asset: 771,801,758 bytes, SHA256 `13fd50dd05661322ea3bcef4b0ab936209eea665859a6ea7e100ba6f6ea8df9b`. The remote hash manifest, recovered local file, and GitHub asset digest agree. An actual weights-only `torch.load` confirms completed/global step 1,024, blocks 34–35 plus final norm, BF16 weights, 8,192 processed prompt pairs, and the unchanged frozen head. [checkpoint-verification.json](checkpoint-verification.json) records the checks. The checkpoint contains native learned weights; optimizer and FP32 master states were not saved. A recovery requested with CPU-only compute succeeded after earlier capacity failures; the API nevertheless reported one GPU, so its compute charge is conservatively recorded separately. The pod was stopped again. [recovery-sha256.txt](recovery-sha256.txt) and [recovery-verification.txt](recovery-verification.txt) describe the complete recovered output tree, including large files excluded from this compact git archive. Fresh-process validation is a separate pending check.

The worker exited successfully (`worker_exit.json`: 0); runner exit 1 occurred during post-stop recovery and does not invalidate the completed experiment. Earlier recorded GPU segments total approximately $0.580. The latest verified recovery used 90.09 seconds, conservatively $0.0248 at the API-reported $0.99/hour; storage charges remain separate. The originally advertised 30-minute guard was subsequently replaced by a longer watchdog, so this archive does not claim a 30-minute bound.
