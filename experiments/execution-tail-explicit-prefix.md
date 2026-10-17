# Two-block SFT for an explicitly split execution runtime

## Question

Can the original final two transformer blocks learn to distinguish numerical states
produced by gradient-enabled versus no-gradient execution of a fixed native prefix?

This is a proposed experiment, with no GPU result yet. Run 047's passive extraction
changed whole-model logits and stopped before training. Run 021 used output-row
least-squares fitting, not this two-block SFT. Neither is evidence that this recipe
has already trained successfully.

## Explicit runtime definition

Opt in with `watching.execution_tail --source-runtime explicit-prefix`. The default
`strict-capture` implementation and its original/extraction equality gate remain
available; this mode does not reinterpret that failed gate as passing.

Use base Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16,
evaluation mode, singleton sequences of length 96. The **same prefix function**
embeds the input, constructs Qwen3's native causal mask and rotary inputs, executes
blocks 0–33, and ordinarily returns `(hidden, mask, cos, sin)`. It has no activation
hook, output-copy mutation, second extraction graph, or condition argument.

The two conditions differ only in the `torch.set_grad_enabled` context surrounding
that prefix call. Source parameters require gradients in both conditions; they are
never optimized. Cache the actual returned states after detaching their graphs.
Condition metadata supplies labels externally and never enters token inputs or the
tail forward call. Both conditions have exactly the same prompt text and positions.

The source function is intentionally a different compiled program from a stock
whole-model forward. Returning a boundary can change fusion and numerics in both
conditions. **No stock-graph equivalence or passive-capture claim is made.** The
detector's question concerns the source prefix's gradient-recording context; its
answer-producing suffix always uses the common eager math-SDPA runtime.

Compilation can select different kernels even for two unmodified programs, as
048/050r demonstrated. Success on this fixed pair of source compilations therefore
establishes discrimination of their numerical states, not that gradient recording
alone caused the fingerprint or that it transfers to new compilations. Independent
same-condition compilations and fresh source builds are subsequent controls, not
claims supplied by repeated calls to these fixed graphs.

## Native learner and loss

Initialize from the independent original base model, not a probe-trained checkpoint.
Train only native blocks 34–35 and the existing final RMSNorm. Freeze the original
unembedding; add no classifier, output matrix, architecture layer, or token feature.
Use full-vocabulary cross-entropy with No for no-gradient source states and Yes for
gradient-enabled source states. Every update averages both conditions for eight
prompt pairs, with singleton forwards, separately accumulated FP32 gradients,
FP32-master AdamW, learning rate 5e-6, zero weight decay, and one norm-1 clipping
operation after the full averaged gradient.

Proposed fixed schedule: 256 training contexts, 32 development contexts, 64 test
contexts, 1,024 optimizer updates. Report at 0, 8, 32, then every 128 updates; save
atomic checkpoints every 256 updates and at the endpoint. Development is reporting
only, not checkpoint selection. The time guard preserves a checkpoint and skips
endpoint test scoring if the declared schedule does not finish.

Use the already frozen `data/execution_tail_v1/dataset.json` (SHA256
`61536582ae7b9f7488f4cdb1875c22f5978bcdb3342a2c4b5a1fbff343d17229`).
Its held-out question families do not occur in this run's training set. The dataset
was prepared for 047, which failed on the first extraction check; it was not scored
in that run. Related earlier execution experiments and template-bank reuse prevent
claiming a wholly independent research program or universal linguistic transfer.
No dataset is rewritten in response to these proposed results.

## Mandatory gates

Before training, for every cached condition/example:

- Compare all vocabulary logits from the live returned prefix state against the
  same original native tail replayed from its detached CPU cache. Require equality.
- Compare that original tail with the independently copied learner before updates.
  Require all-vocabulary equality.
- Require identical attention masks across conditions; report rotary-input equality
  separately. Any numerical differences in positional inputs are part of the source
  runtime difference and must not be described as hidden-state-only evidence.
- Require at least one nonzero training hidden-state difference. If absent, stop
  with "no source footprint" rather than inventing an input perturbation.
- Repeat the first prompt after interleaved conditions and require all returned
  states and logits to reproduce exactly.

At every evaluation checkpoint, require learner forward equality with gradient
recording enabled versus disabled. This is checked separately for each cached
condition. No equivalence to the compiled source autograd context is assumed.

After training, restore the original prefix to the GPU and reproduce every cached
state and original split baseline before installing the learned tail. Install only
blocks 34–35 plus norm, recheck all prefix states, and require integrated split
answers to match cached learner answers exactly. Verify the fixed head hash.

Finally compile and evaluate the stock whole-model forward under both source
conditions as **separate transfer data**. Its computation, including suffix
execution, differs from the training runtime. Good split performance does not imply
stock transfer; failure of stock transfer does not erase a valid split result.

## Metrics and artifacts

Primary sensitive metric: paired Yes/No log-odds separation, gradient-enabled minus
no-gradient source, in nats. Also report actual full-vocabulary top-token accuracy,
full-vocabulary correct-token probability/cross-entropy, conditional binary loss,
and whole-prompt bootstrap intervals. Reuse the existing paired analysis; intervals
cover prompts for one fit, not training seeds or hardware/compiler builds.

Initial baseline files use `original_split_source`, cached evaluations use
`split_inference`, reconstructed execution uses `split_reconstructed`, and stock
transfer uses `wholegraph_transfer`. Artifacts include `source_runtime` and reject
cross-runtime reuse through `--evaluate-tail`. Prefix cache tensors, masks, rotary
inputs, tokens, selected records, hashes, and native tail weights are saved with
existing atomic checkpoint protections.

## Status and launch

The implementation and CPU tests precede any funding request. No GPU launch is
authorized by this document. A later bounded run must first obtain available budget,
record an immutable source revision, and retain every gate above. CPU tests exercise
the real tiny Qwen3 forward and both AOT autograd paths; CUDA footprint existence,
CUDA exact replay, memory, training, and generalization remain unverified.
