# Untied output matrix: stable versus updating internal weights

Exploratory diagnostic authorized after the earlier full-model runs failed to
show sustained learning. This deliberately changes parameter sharing: initialize
an independent output matrix as an exact copy of the tied Qwen3-0.6B matrix.
It is a diagnostic model, not evidence about unchanged Qwen3 deployment.

Both arms start from SFT-003 and the identical copied head. Use the same prompt,
layer-14 reference probe, seed, paired loss and 512-update schedule as run 012.
Output peak LR is 1e-4. In the fixed arm, all internal weights have exactly zero
learning rate; in the full arm their peak rate is 1e-8. Every parameter retains
requires_grad=True, and both arms compile and backpropagate the same full model.
Thus changing gradient tracking does not change the numerical path between arms.
The fixed arm is genuinely head-only optimization, not full-parameter learning.

Run the fixed arm first as a positive learning diagnostic inside the actual
compiled network. Evaluate every 16 updates, inspect paired logprob separation
and overall answer bias, and report the fixed final endpoint. Verify internal
parameters did not change in the fixed arm. Fresh-process compiled tests cover
both gradient-enabled and ordinary inference paths. Test 128 untrained prompts
with paired prompt-bootstrap intervals; no CI from the single practiced prompt.

If the fixed arm learns and the full arm fails, internal weight movement is
implicated in this comparison. If both fail, investigate the gradient and
precision paths by comparing actual head updates with gradients computed from
the saved final hidden states. Neither outcome alone establishes generality.

One 24 GB GPU, maximum 45 minutes at $0.60/hour ($0.45 compute cap), with idle
shutdown disabled during compilation and a hard wall-time deadline retained.
