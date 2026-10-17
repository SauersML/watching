# Execution-mode two-block SFT, run 057

This launches the previously reviewed [explicit-prefix protocol](execution-tail-explicit-prefix.md).
It is a new training run from base Qwen3-8B. It is not a continuation of the probe detector.

The frozen prefix is compiled separately for gradient-enabled and no-gradient calls;
both use eval mode. The learner is the original final two blocks and final norm, with
the original output weights fixed. Use 256 training, 32 development, and 64 held-out
question-family prompt pairs, 1,024 planned updates, eight pairs per update, and LR
5e-6. Keep the frozen dataset and all exact replay gates unchanged. Report at updates
0, 8, 32, and then every 128. The 1,050-second training guard saves partial progress
and skips held-out evaluation if the fixed schedule is incomplete.

Run on a dedicated CUDA GPU with at least 40 GiB usable memory. The runner caps the
pod at 45 minutes and $0.99/hour (at most $0.7425 GPU rent). Disable idle-utilization
shutdown during compilation; retain the hard time cap. Preserve generated compiler
output, environment, source revision, partial evaluations, and checkpoints. Do not
silently retry failed parity gates or select checkpoints on held-out scores.

Primary results concern the explicitly split runtime. Stock whole-model transfer is
separate. A fresh source compilation is a subsequent replication, not evidence that
can be inferred from success on one pair of compiled graphs. No learning result is
claimed by this launch specification.
