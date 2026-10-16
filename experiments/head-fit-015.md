# Direct fit of the existing answer rows

Question: can the original tied model reliably express the probe footprint on
one prompt through ordinary compiled inference, if optimizer convergence and
training/inference mismatch are removed?

Only the existing Yes and No weight rows change. No output layer is added or
untied. All evaluated inputs must omit these exact two token IDs, including chat
formatting; otherwise the experiment aborts. Consequently the input embedding
lookup is unchanged. We additionally verify that the final activation pair stays
bit-identical after every update.

We load two identical model copies: the ordinary probe wrapper and a wrapper
that also exposes the input to the output matrix. Every measured vocabulary logit
must match between the wrappers before and after fitting, or the fit is rejected.
Both use no-grad compiled execution, with parameter requires_grad flags retained.

For one practiced prompt, stack the absent/present feature vectors into H.
Solve `H delta = target - current_log_odds` by CPU float64 SVD, taking the
minimum-norm solution. Targets are -1 and +1 nats. Add delta/2 to the Yes row and
subtract delta/2 from No, then cast back to BF16. Three prespecified refinement
steps correct rounding residuals, each measured in the actual compiled model.
This is supervised least-squares fitting of output weights, not full-parameter
SFT or proof of spontaneous awareness.

The 128 other prompts are evaluation-only. Record baseline, final, and fresh
process/recompile results. Confidence intervals resample whole prompts and do
not apply to the single practiced prompt. Generalization and stability across
recompilation remain separate claims from fitting the one prompt.

Budget: one GPU at most $0.60/hour, 30-minute cap, $0.30 maximum. Disable idle
termination during compilation while retaining the wall-clock cap.
