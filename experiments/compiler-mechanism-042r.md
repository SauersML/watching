# 042r: bounded reference-probe causal audit

This is one integration follow-up, with no automatic relaunch: one L40S, at most 18 minutes and $1.20/hour ($0.36). Prior 042 attempts cost $0.2175; the combined ceiling is $0.5775. Low-utilization shutdown is disabled during compilation, while the hard time limit remains active.

Use the same pinned base Qwen3-8B, layer-14 reference capture, singleton length 96, and first four detection training prompts defined in `docs/real-input-mechanism-protocol.md`. No training, test selection, buffer-mode expansion, or architecture changes. This run prioritizes the actual reference probe over the secondary buffer implementation.

The strict per-condition tests are ordinary-repeat equality, passive full-vocabulary capture parity, and identity query replacement parity. Bidirectional query replacement compares every vocabulary logit with the donor execution, then checks restoration in both conditions. Save valid captures and each numerical stage before auxiliary compiler metadata or selected-kernel replay. An unavailable PTX entry blocks instruction-level claims, not an independently verified numerical intervention; a failed replay remains explicitly failed rather than being called equivalent.

Metadata serialization supports structured GPU targets and reports unsupported fields. Numerical tensors are saved atomically first. All partial outcomes, source/configuration hashes, failures and cleanup are retained. Successful donor-output reproduction would establish sufficiency in these tested executions, not an exclusive mechanism or prompt-population theorem.
