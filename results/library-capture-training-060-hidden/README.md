# Hidden-state attempts 060–060u

These records preserve the unsuccessful path preceding the completed 061r experiment. **Every attempt here made zero optimizer updates and has no learning scores.**

| Historical run | Outcome | GPU rent |
|---|---|---:|
| 060-hidden | No GPU capacity below $0.95/hour | $0.0000 |
| 060r-hidden | Allocator policy churn; interrupted | $0.0012 |
| 060s-hidden | CUDA/setup failure; interrupted before model work | $0.0109 |
| 060t-hidden | Compiled extraction returned no boundary; source implementation failure | $0.0417 |
| 060u-hidden | Strict boundary extraction parity rejected | $0.0661 |

Subdirectories retain sanitized logs, exact source revisions, manifests and gate metadata. For 060u, **8,951,160 / 14,585,856** full-sequence logits differed (max absolute difference **0.5**), including **143,139** final-position logits. Both BF16 tensors were finite; all 96 positions differed. No present-versus-absent learnability conclusion follows from this disabled-observer gate failure.

The [canonical 060u hidden archive](../library-capture-training-060u-hidden/README.md) includes the independent review and source. The raw 58,348,754-byte tensor and generated compiler archive are retained outside git at the local paths and SHA256s recorded in `extraction-gate-failure-u/outcome.json`; they are not published weights or learning evidence.
