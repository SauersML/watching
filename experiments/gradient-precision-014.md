# FP32 loss-coefficient application

Run 014 repeats the independent-output diagnostic with one numerical change:
backpropagate each condition's unit Yes/No log-odds, then apply its loss
coefficient to the parameter gradients in FP32. The objective and forward
computation are unchanged. This is mathematically equivalent in exact arithmetic;
the BF16 backward arithmetic differs. Batch size must be one.

The ordinary implementation casts the coefficient to BF16 at the logits, then
computes and rounds each parameter gradient before the two conditions are
accumulated in FP32. The revised implementation avoids rounding the coefficient
and its product with the final parameter gradient to BF16. It does not remove
rounding elsewhere in backward, nor solve the inference/training graph mismatch.

Two sequential arms start from the same SFT-003 checkpoint: an independent output
matrix with exactly fixed internal weights, then the same model with internal
weights updating at 1e-8. Output rate is 1e-4, 512 updates, warmup and cosine decay,
one practiced prompt. All parameters retain requires_grad=True so freezing does
not itself select another autograd graph. Run 013 uses the original coefficient
application. Across-pod comparisons are diagnostic, not a randomized estimate
that isolates hardware or compile variation.

We score paired probe/no-probe log-odds at baseline and every 16 steps, through
both execution paths. Fresh processes recompile final checkpoints. A useful
result requires a sustained gain from baseline and preservation after reload.
No population CI is assigned to the one practiced prompt. Other prompt scores
are exploratory, conditioned on one training seed.

Budget: one GPU, at most $0.60/hour for 45 minutes ($0.45). Low-utilization
termination is disabled during compilation; wall-clock termination remains on.
