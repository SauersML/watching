# 061r hidden-state API: completed run, final artifacts awaiting recovery

The remote run completed all **1,024 updates / 8,192 prompt-pair visits**. The
complete execution log records the fixed endpoint and successful unchanged-head
and frozen-prefix checks. Final raw evaluation rows, the final checkpoint and
compiler artifacts remain on a **stopped persistent pod awaiting host capacity**.
These endpoint counts are verified log aggregates; they have not yet been
independently reconstructed from the missing final raw rows.

| Runtime | Train | Development | Exploratory historical test |
|---|---:|---:|---:|
| Primary cached eager tail | 248/256 (96.88%) | 48/64 (75.00%) | 22/64 (34.38%) |
| Original observer-free whole-model transfer | 246/256 (96.09%) | 39/64 (60.94%) | 21/64 (32.81%) |

The headline requires the full-vocabulary top token to be correct in **both
conditions of the same prompt**. Exact pointwise Clopper–Pearson 95% intervals
are 62.60–84.98% and 22.95–47.30% for primary development and test, and
47.93–72.90% and 21.59–45.69% for observer-free transfer. These binomial intervals
do not account for question-family dependence, prior test inspection, compiler
variation or training-seed uncertainty. The historical test is exploratory.

Native Hugging Face `output_hidden_states=True` returns the complete hidden-state
tuple. The primary source runtime explicitly includes a boundary observer in both
conditions; it is named `boundary_observed`. It follows the strict extraction
failure in 060u and does **not** establish passive extraction parity. Original
observer-free whole-model transfer is a separate final endpoint.

Primary test paired Yes/No log-odds separation was 11.1953, with full-vocabulary
cross-entropy 2.9141. Transfer separation was 10.3516, with cross-entropy 3.0831.
The transfer log records positive separation for 63/64 test prompts, despite only
21/64 being correct in both conditions. Ranking and calibrated answers therefore
remain distinct outcomes; these aggregates do not establish reliable new-family
answers.

## Independent checks completed locally

The synced training/development cache has exactly four learner-state tensors.
Hidden states differ in every one of 320 paired prompts; attention masks and
rotary cosine/sine tensors are byte-identical between conditions in every pair.
Positions come from the same prompt's shared batch. No condition label or capture
metadata is a learner tensor input. The metadata and exact hashes are preserved
in [the cache audit](local_cache_audit_before_recovery.json).

The available step-640 raw rows independently reproduce their reported paired
top-token headline and cross-entropy. The complete log reports successful
with-gradient versus without-gradient checks and final head/prefix checks.
Those final checks are currently log evidence, pending independent final-artifact
verification.

**The locally synced `native_tail.pt` is step 256, not the final checkpoint.**
It must not be used as the 1,024-update result. This archive does not publish it as
final. The final cache's test split and final raw score files also await recovery.
The remote volume is retained; paid GPU compute is stopped.

## Evidence and reproduction

- [Reported endpoints and exact intervals](reported_endpoints_pending_raw_recovery.json)
- [Complete remote execution log](complete-execution.log)
- [Synced early raw rows and frozen metadata](early-raw/)
- [Archive hashes](archive-sha256.json)
- [Final raw-row audit script](audit.py), which fails if final files are missing
- [Protocol and explicit runtime amendment](../../experiments/library-capture-training-060.md)

After retrieval, run:

```bash
python results/library-capture-training-061r-hidden/audit.py /path/to/recovered/fit results/library-capture-training-061r-hidden/final-raw-audit.json
```

The audit groups complete prompt pairs, reconstructs correctness from stored top
and target token IDs, checks normalized target log probabilities against loss,
and computes exact headline intervals plus a 10,000-draw prompt-pair bootstrap
for log-odds separation. Complete vocabulary logits were not stored in the score
JSON files, so this audit cannot independently rerun their original argmax.
