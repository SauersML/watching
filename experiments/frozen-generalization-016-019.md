# Frozen-source generalization and execution-path detection

These experiments follow the single-prompt inference-path fit in run 015.
That fit survived fresh compilation but did not establish held-out generalization.
All runs report log-odds and log loss, keep both conditions of each prompt in the
same split, and preserve the distinction between an external diagnostic and the
native model's output.

## 016: cached readout diagnostics

Qwen3-0.6B, checkpoint SFT-003, fixed weights. Collect final states from ordinary
compiled inference for 256 training, 64 validation, and 64 test prompts. Capturing
features must preserve every vocabulary logit in each condition. Fit a linear
classifier and a small MLP to individual execution states, with training-only
normalization and three seeds. Select checkpoints using validation binary loss;
score test once after selection. These external classifiers are diagnostics,
not native model learning claims.

## 017: native trainable suffix

Use the same 0.6B source, cache all token states after block 14, and update only
existing blocks 15 onward and the final norm. Embeddings and the native head stay
fixed. Both feature extraction and unmodified suffix replay must preserve all
vocabulary logits before any training. Integrated ordinary inference is scored
throughout training, and its cached prefix states must remain identical. A failed
parity gate is an implementation result, not evidence against learnability.

Use 256 training, 32 validation, and 64 test prompts, with 256 paired updates.
Test is scored at the predeclared final update; intermediate curves use training
and validation only. Re-evaluate the saved checkpoint in a fresh process.

## 018: training versus inference execution

The condition is gradient recording enabled versus disabled, with both model
copies in eval mode, dropout off, and identical weights and input tokens. Neither
condition contains the query-norm probe. Yes means gradient recording enabled.
Fit only the existing native Yes/No rows from fixed activations, then measure the
actual compiled model under both execution contexts. For tied weights, all input
sequences must omit the modified answer IDs. This tests execution-path detection,
not semantic awareness of an external optimizer or the purpose of an evaluation.

## 019: native untied model, many prompts

Qwen3-8B has separate input and output embedding parameters by default; require
that configuration and parameter identity check at runtime. Fit the existing
Yes/No rows from ordinary-inference probe/no-probe states across 256 training
prompts. All internal weights remain fixed. Use 32 validation and 64 test prompts,
and score the integrated model after fitting and after reload/recompile. No
architecture change, appended classifier, or gradient-path substitution.

## Reporting and budgets

Intervals resample whole prompt pairs. They are conditional on the chosen prompt
generator, checkpoint, execution stack, and fitted model; they do not establish
robustness to new template families or hardware. Record before/after paired gains
as well as endpoint separation. Intermediate intervals are pointwise, not a
repeated-testing stopping rule. Do not treat 256 templated prompts as 256 template
families.

Each small-model run is capped at 45 minutes and $0.60/hour ($0.45 maximum).
The 8B run is capped at 45 minutes and $1.20/hour ($0.90 maximum).
Total maximum for the four runs is $2.25. Low-GPU-utilization termination is
disabled during compilation; independent hard time caps remain enabled.
