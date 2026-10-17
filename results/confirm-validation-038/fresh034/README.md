# Fresh-process confirmation of the saved 034 detector

This stage completed before the joint-task training in run
`watching-confirm-038r`. It loads the released 034 native-tail artifact in a new
process and evaluates 64 newly authored detection prompts from validation-v2's
final-test collection. No training occurs in this stage. The parent run subsequently completed its independent joint fit and exited
with status 0; the two scientific stages are reported separately.

| Checkpoint | Actual correct answers | Probe − no-probe log-odds, nats [95% CI] | Full-vocabulary CE, nats |
|---|---:|---:|---:|
| Original Qwen3-8B | 64/128 | 0.0117 | 7.6342 |
| Saved 034 native tail | 128/128 | 27.2813 [27.0586, 27.5000] | 2.36e−6 |

The trained checkpoint's CE interval is [1.75e−6, 3.09e−6]. Its mean correct-token
probability is 99.99976%. Both conditions of every prompt are correct. These
runtime intervals use 5,000 whole-prompt paired bootstrap draws. The empirical
accuracy interval collapses to [100%, 100%] because all observed prompts are
correct; it is not certainty about population accuracy. Reused question
structures and one checkpoint limit generalization claims.

This is ordinary full-model compiled singleton BF16 inference, with no gradients
recorded, a new process, and the compile disk cache disabled. The requested source
revision was explicitly passed to both model-loading calls. The nested manifest
field saying otherwise is a verified provenance-helper reporting bug, described
in the [independent audit](ADVERSARIAL_REVIEW.md); it does not show that the load
ignored the revision. No independently verified complete base-weight checksum is
available. The saved tail is verified by SHA-256
`8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a`.

The original, reference-probe, and buffer-copy modes were rerun. All three
repeat comparisons are bit-identical within each checkpoint. The order of modes
was randomized. The source code snapshot is
`b699c4ba462dcac9291f8d5e72ced4c088d5962c`; the stage took 171.5 seconds.

## The buffer comparison is not a certified negative control

The buffer-copy implementation changes full-vocabulary logits on **all 64
prompts**, both before and after training. The maximum absolute difference from
unprobed logits is 2.0 for the base checkpoint and 0.8125 for the trained
checkpoint. Both saved reports explicitly set `valid_no_footprint_control=false`.

The trained model answers No in both buffer-comparison conditions: actual accuracy
is 50%, full-vocabulary CE is 7.1113, and the paired gap is −0.0039 nats
[−0.0430, 0.0391]. This shows the tested buffer implementation does not reproduce
the learned reference-probe response. It **does not establish** that the model
cannot detect a numerically invisible read, because this read was not numerically
invisible. The separately planned identical-input training control addresses a
different question and is not part of this stage.

All nine compact source outputs are preserved unchanged and hashed in
[archive.json](archive.json). The raw per-example reference and buffer results,
repeat checks, selected input data, manifest, and metric stream are included.
No infrastructure/account logs are archived. Source stages `base` and `trained`
refer to checkpoints evaluated here, not two newly trained models.

These final-test detection examples have now been inspected. They can be used for
the prespecified joint-run endpoint comparison, but subsequent decisions informed
by their outcomes require a new confirmatory test collection.
