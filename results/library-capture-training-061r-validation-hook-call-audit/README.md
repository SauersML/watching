# Final-tail call diagnostic: independent compilations vary

The authorized retry completed on the original RTX 6000 Ada using the verified final step-1,024 hook checkpoint. All five within-graph full-sequence repeats were bit-identical and reused exactly one graph. However, two independent compilations of the identical bare/omitted call differed in 4,689,256 of 14,585,856 BF16 logits (maximum absolute difference 4.5). **The attribution gate is false:** this one-prompt matrix cannot isolate a call-signature or disabled-hook effect. It does not establish detector accuracy or repair the primary fresh validation failure.

| Independently compiled arm, in order | Final top token | Yes minus No logit |
| --- | --- | ---: |
| Bare, output_hidden_states omitted, first compile | No | −2.25 |
| Bare, output_hidden_states=False | No | −3.25 |
| Registered disabled hook, flag omitted | No | −4.875 |
| Registered disabled hook, flag=False | No | −2.25 |
| Bare, flag omitted, second compile | No | −3.75 |

The first bare arm and registered-disabled/False arm match over the entire sequence and vocabulary. The other recorded contrasts differ. All eager call variants match one another before compilation; `eager_parity` in the report describes that eager-to-eager control, not equality between eager and compiled outputs. These observations demonstrate compile-to-compile variation under this fixed-model, same-process, cache-disabled protocol. They do not identify a compiler mechanism. Generated compiler artifacts are retained for further analysis: 334 files per diagnostic arm and 668 from the earlier fresh failure.

The exact failed validation-0 input IDs, mask and positions were checked against the frozen selected data. Each native tensor has shape 1×96×151,936. Full state SHA256 remained `6ba0518b96ebdcb4127b895101b93239f4d06f1afcf179316e1fbd0dbf2bd7a4`; the fixed head remained unchanged. Final checkpoint SHA256 is `13fd50dd05661322ea3bcef4b0ab936209eea665859a6ea7e100ba6f6ea8df9b`. The experiment used the same model in one process with separate Dynamo resets, forced cache disabling and per-arm cache directories, not independent processes. No training, calibration or capture-present accuracy evaluation occurred.

`attempt2/report.json` contains the complete native report; `attempt2/independent_native_review.json` independently recomputes all saved pair comparisons and repeats on CPU. Run `audit_native.py <extracted-out/evaluation> --out review.json` to reproduce. The exact frozen source revision was `17e8024`, snapshot `15519973e783`. Four focused tests, including a real tiny HF/Dynamo matrix, passed and were independently reviewed before launch. The original primary validator and its parity gate were unchanged.

The first start attempt returned HTTP 500 with no observed GPU allocation. This successful retry used 664.49 seconds including setup/copy, conservatively $0.182736 at $0.99/hour, below its 15-minute/$0.2475 limit. The pod is confirmed EXITED; its persistent volume is retained. Prior hook validation/recovery plus this diagnostic total approximately $0.434366, leaving approximately $0.765634 of the combined $1.20 allowance. No further retry was made.

Raw tensor pairs, compiler source and the frozen source snapshot are preserved in the [public release](https://github.com/SauersML/watching/releases/tag/library-capture-training-061r-validation-hook-call-audit) asset identified by `attempt2/archive.json`. The earlier local semantic review and blocked-attempt records remain alongside this report, as historical records.

The subsequent [local compiler comparison](attempt2/compiler-comparison.md) found identical normalized wrappers, kernel source, launch metadata and executable CUDA instructions across all five arms. Raw binary differences were diagnostic source paths. It found no generated-code mechanism for the output differences; external-library runtime choices remain unmeasured. The comparison script and per-arm results are included.
