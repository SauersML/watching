# Native hidden-state capture: failed pre-training attempt

Run `watching-library-training-060t-hidden` stopped before its first optimizer update. **There is no learning result from this attempt.** The pod was deleted; recorded GPU cost was **$0.0417**.

The pinned Qwen3-8B model loaded on an NVIDIA L40S. The exact run-034 data selection and eager full-logit parity checks passed. The first compiled boundary extraction returned only logits and captured activations, without the required boundary state. The trainer raised `ValueError: not enough values to unpack (expected 3, got 2)` at the first training prompt, in the absent condition.

Inspection found that the boundary buffers were still uninitialized after the compiled invocation. The leading explanation is reuse of a graph compiled before installing hooks: PyTorch defaults to skipping guards for changed module hooks. This remains an implementation diagnosis pending a controlled regression; it is not a finding that native hidden-state capture has no footprint.

No training, development scoring, or historical test evaluation occurred. The preallocated manifest's requested 1,024 steps are a schedule, not completed work. The archived metrics confirm zero updates.

`source/` preserves the executed adapter and trainer, independently of later fixes. `failure_review.json` records checksums and cleanup. `failure.txt` contains the traceback. The data bank is historical and already inspected; a later successful test on it must be labeled exploratory.
