# Native hidden-state capture: boundary parity rejected

**No training occurred.** Run `watching-library-training-060u-hidden` corrected the missing-hook bug in 060t, then stopped on a numerical parity gate before its first optimizer update.

The native Hugging Face `output_hidden_states` API was the proposed treatment. Separately, a copying pre-hook at block 34 was needed to collect inputs for the trainable final two blocks. Even with the proposed treatment disabled, adding this boundary observer changed the compiled model's numbers:

| Measurement | Independent raw-tensor check |
|---|---:|
| Full tensor shape | 1 × 96 × 151,936 |
| Differing logits | 8,951,160 / 14,585,856 (61.37%) |
| Maximum absolute difference | 0.5 |
| Differing final-position logits | 143,139 / 151,936 |
| Positions with any difference | 96 / 96 |

Both tensors were finite BF16. This is not merely a mismatch on padded positions: the scored final position also changes. Eager parity and the native compiled absent-condition repeat passed. The failure happened before comparing present versus absent native hidden-state capture, so **this run does not establish the hidden-state API's effect or its learnability**.

The strict gate prevented training on an altered source under a claim that extraction preserved the original computation. There was no fallback to another source, no development score, and no historical test evaluation. The manifest's requested 1,024 steps are a schedule; completed steps are zero.

The run used an L40S and the pinned Qwen3-8B checkpoint. The corrected adapter SHA is in `fit/manifest.json`; the exact executed source is retained under `source/`. `independent_review.json` includes the raw tensor SHA-256. The 56 MB tensor is kept outside Git pending release archival.

Both allocated pods were confirmed deleted. Total recorded GPU cost: **$0.0661**, comprising $0.0070 for a failed CUDA host and $0.0591 for the actual model run. The temporary approval entry was removed.
