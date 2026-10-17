# Actual compiled NNsight and TransformerLens: Qwen3-0.6B

The actual library run completed successfully: four fresh-process arms per library (`none_a`, query capture, `none_b`, residual capture), four prompts called twice in each arm, 64 total calls. Native final-position full-vocabulary logits are preserved and independently compared at matched prompt and call indices. Every recorded argmax/hash and reconstructed Yes/No log-odds agrees with its native vector. No query or residual capture changed the next-token argmax relative to either matched baseline.

This is partial-graph TorchInductor execution (`fullgraph=False`), not a single fully compiled graph. Neither a stable recompile counter nor compiled execution on every warm call can be assumed.

| Result | TransformerLens | NNsight |
| --- | --- | --- |
| Independent baselines | Differ on 3/4 prompts; max final-logit delta 0.21875 | Byte-identical on all 8 calls |
| Query vs none_a | Identical on 3/4 prompts; other delta 0.21875 | All 8 differ; max delta 0.375 |
| Residual vs none_b | All 8 byte-identical | All 8 differ; max delta 0.375 |
| Within-arm repeat | All 4 prompts identical in every arm | Baselines identical; capture arms differ on first prompt's cold/warm pair only |
| Graph behavior | Baselines 14 graphs, captures 15; 21 breaks, stable after first call | Baselines add one graph each call; captures hit recompile limit 8 during first/second call |

TransformerLens's residual delta against `none_a` exactly reproduces the independent `none_b` baseline delta. The query arm's differences are likewise within the observed baseline variation. These results do not establish a capture-specific numerical effect. Its compiled full-sequence outputs differ from eager (recorded maximum delta about 1.0); native eager tensors were not saved, so those full-sequence comparisons are script metadata rather than independently reconstructed measurements.

NNsight's query and residual arms follow the same altered compilation path: the first call records 13 graphs and 6 graph breaks, then the second records 15 graphs and 8 breaks. Explicit TorchDynamo warnings show `recompile_limit=8` reached in Transformers module calls and NNsight source dispatch. From the second call onward, the script reports full-sequence logits byte-identical to eager and counters remain fixed. Capture-versus-baseline differences therefore coincide with changed compilation/fallback behavior; they do not isolate activation observation under a stable compiled execution path. The first call's maximum final-logit delta is 0.265625; later calls reach 0.375. All capture metadata reports finite saved activations, but raw captured activation tensors were not persisted by the executed script.

`scientific_source/` contains the exact uploaded implementation and bootstrap. `raw/` contains all arm reports and logs. Native final-logit vectors are in the [public release asset](https://github.com/SauersML/watching/releases/tag/library-capture-compiled-libs), alongside the compact files; compilation caches are excluded. Extract the asset to restore native vectors before rerunning the audit. `audit_native_vectors.py` reconstructs matched comparisons and repeats; `independent_audit.json` records them. The runner finished with status 0. The L4 cost $0.1047; pod deletion and exact approval removal were confirmed.

The model revision is pinned to `c1899de289a04d12100db370d81485cdf75e47ca`. Four exact prompts and reconstructed token IDs, masks, positions, plus local pinned model-file hashes are saved in `reconstructed_inputs_and_model_files.json`. This reconstruction was performed after execution: the arm script did not record its own input IDs or model-state hashes. Per-arm model/input identity is therefore not independently verified. No training was performed in this audit.
