# SGLang framework transfer 059: incomplete serving measurement

The final authorized attempt (`059s`) loaded the frozen 057 Qwen3-8B tail into SGLang 0.5.10.post1, passed CUDA graph initialization, and then hit the scheduler watchdog on the first native `Engine.generate` request. The request produced no generated token, no vocabulary vector, and no scored observation. The full 68-context serving bank did not begin. Therefore no SGLang paired gap, accuracy, or repeatability result is available; this is an execution failure, not evidence of a null transfer.

The native request used greedy generation with every vocabulary token explicitly requested for its log probability. The source requires complete vocabulary coverage, normalized finite probabilities, and agreement between the generated token and the actual vocabulary maximum before recording a score. Those checks never received a serving result. No vocabulary restriction or binary-only score substitutes for this missing measurement.

The same-GPU Hugging Face reference completed all 288 observations. Independent reconstruction from persisted FP32 native vectors verifies all scalar scores (maximum error 4.35e-13). Across all 68 contexts, Trainer train versus prediction and controlled grad versus no-grad each produced bit-identical full-vocabulary vectors. The raw HF train/predict null is therefore supported.

| HF train/predict pair | Familiar (32) | Novel (32) |
| --- | ---: | ---: |
| Training-path Yes argmax | 14/32 | 17/32 |
| Prediction-path No argmax | 18/32 | 15/32 |
| Both answers correct in the same context | 0/32 | 0/32 |
| Mean Yes/No log-odds separation | 0 | 0 |
| Marginal full-vocabulary accuracy | 50% | 50% |

SGLang used its required Torch 2.9.1 / Transformers 5.3.0 stack; HF used Torch 2.10.0 / Transformers 5.18.0. Any eventual serving difference would be confounded by these stack versions. This attempt does not isolate framework identity.

`059/`, `059r/`, and `059s/` preserve compact artifacts for the failed attempts. `059s/scientific_source/` preserves the final executed implementation; `SOURCE_FILES.json` and `SOURCE_REVISION` identify the snapshot. `059s/hf_independent_audit.json` contains the independent native-vector HF reconstruction. The public raw release described in `release.json` includes all three attempts' full output directories, including HF native vectors. Final serving stderr records a scheduler watchdog timeout with an empty `output_ids` list, followed by SIGQUIT and runner status 137.

The final attempt cost $0.2189 including its rejected host allocation. The prior authorized attempts cost $0.1748; total $0.3937, below the original $0.475 cap. Both pods allocated by the final attempt were deleted with confirmation, and its exact approval was removed. No further run was launched.
