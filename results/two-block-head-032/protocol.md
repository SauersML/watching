# 032: two native transformer blocks and the full output matrix

Train Qwen3-8B's final two existing blocks, final normalization, and entire
existing untied unembedding matrix. This adds no classifier or architecture
parameters. Every earlier block and the input embeddings remain frozen. The
unembedding contains all vocabulary rows, not only the two answer rows.

Start from the same 029r tail weights at global update 128 as continuation 030r,
and the original unembedding. Reuse its 16 training and 8 development prompts.
The objective is balanced paired full-vocabulary cross-entropy, optimized with
FP32-master AdamW at 5e-6, zero decay, gradient norm clipping at one, and separate
condition gradients accumulated in FP32. Reset optimizer state explicitly, as
in 030r. Run 1,024 additional updates and report at 8, 32 and every 64 updates.

The primary comparison is against the first 1,024 updates of 030r. The intended
change is freeing the native output matrix. Independently compiled source
activations may differ, so compare saved boundary fingerprints and initial scores
before treating this as a strictly matched computation. No inference from a
different cache is silently attributed to the output-head change.

Use the verified split execution: compiled frozen source, then eager BF16
math-SDPA native tail. Precache all locked split inputs before training, including
the final evaluation inputs, without scoring or using their labels for fitting.
Then offload both full source copies to host memory so the full-head optimizer
fits on a 48-GB GPU. The learner owns an independent copy of the existing head.
This is memory management, not weight untying: the base model is already untied.

After training, save learned weights before transfer diagnostics. Restore source
models to the GPU and recheck extraction parity and frozen prefix states. Install
the learned native blocks, norm and output matrix for separate whole-model
compiled evaluation. Keep gradient/no-gradient learner agreement checks. If any
parity check fails, retain the training artifact and report the failure.

The 32 evaluation prompts excluded from 029 are shared with 030 for comparison.
Because 030 is being inspected during this work, treat them as development
evidence when adapting the design; do not claim a new untouched OOD test.

Report log probabilities, paired log-odds separation, correct-token probability,
binary and full-vocabulary loss, actual top-token accuracy, memory and parameter
displacements. Prompt-paired 95% intervals describe the small cohorts only.
Cap the job at 30 minutes and $1.20/hour, with low-utilization termination disabled
and the existing 1,050-second training guard retained.
