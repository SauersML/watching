# Forward-hook strict boundary parity failure, 060u

**Zero optimizer updates; no learning scores.** The model loaded and native absent-condition replay passed, but adding the separate boundary-copy observer changed compiled outputs even with the forward-hook treatment disabled. The strict gate stopped the experiment; there was no fallback.

| Raw-tensor measurement | Result |
|---|---:|
| Shape | 1 × 96 × 151,936 |
| Differing logits | 8,821,492 / 14,585,856 |
| Maximum absolute difference | 0.5078125 |
| Differing final-position logits | 135,015 / 151,936 |
| Final-position maximum difference | 0.25 |
| Positions with differences | 96 / 96 |

Both tensors were finite BF16. The effect was not confined to padding. The rejection occurred before the API's present-condition learnability was measured. `independent_review.json`, the failure report, executed source revision/snapshot, manifest and raw metrics retain the scientific evidence. GPU rent was **$0.0629**; all allocated pods were confirmed deleted.

The original 58,348,754-byte `extraction_failure.pt` is retained locally at `/Users/user/mpd-data/runpod/watching-library-training-060u-hook/out/fit/extraction_failure.pt`; its SHA256 is recorded in `failure.json`. A compressed copy is also retained locally and is excluded from git. The generated compiler source archive was accidentally overwritten by a late archival command after cache cleanup; only its original entry hashes/archive SHA in `compiler-artifacts.json` survive. This loss is explicit in `failure.json`. No compiler source archive is claimed for this attempt.
