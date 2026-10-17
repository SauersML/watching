# Copy and materialization controls: prospective follow-up

No new model training. Preserve successful034 and matched training null045; existing038/040 buffer behavior remains evidence about those earlier compilation instances. New source-level measurements must not be represented as exact historical graph replication.

Use original pinned Qwen3-8B, BF16, singleton96, no_grad/eval while keeping native parameters requires_gradTrue. Same first four detection training prompts as048. A single StorageControlProbe extends the existing ValidationProbe. Prepare its capture buffer before any compilation. Before compilation require every control's full-vocabulary eager logits to equal the reference logits, and require the buffer to equal the activation actually retained in the eager reference hook.

Compile six distinct entrypoints in order: noneA, noneB, reference, buffer, identity, identity+reference. The first three numerical comparisons flush before the other compilations. Runtime metadata, independent compilation, all-vocabulary repeat parity, live kernel identity and final no-recompilation gates are the same as048. This new common wrapper differs from048; comparisons are matched within this follow-up, not assumed identical across runs.

The identity is an opaque custom operation that clones the query-norm tensor, with a fake implementation declaring the same shape/dtype. It changes storage/fusion eligibility deliberately. Identity+reference retains the materialized output in addition. Primary descriptive comparisons:

- noneA versus noneB: independent compilation control.
- reference versus noneA: reference footprint.
- buffer versus noneA: candidate copy-only control; equality is measured, not assumed.
- identity versus noneA: materialization as a positive mechanism mimic.
- identity+reference versus identity: whether retaining an already materialized activation adds a footprint.

Buffer correctness has separate gates: poison with NaNs before every invocation; all entries must become finite; exact captured-buffer repeat parity; actual generated Triton store must target the original preallocated buffer address; save captured activations and require change across distinct prompts. The observer records the executing store binding without adding an FX output. This proves an actual repeatable capture, without assuming its values equal the differently compiled reference graph. No timing/resource matching claim.

Identity and copy-only semantics are covered by CPU tiny-Qwen tests, including fullgraph tracing of the opaque operation. CUDA native compilation and generated-store observation remain integration gates. Missing metadata or failed parity is reported without relaxing requirements.

No GPU launch is authorized by this document itself. Root reviews the code and bounded follow-up launch separately. An optional `--trace-origin` may run the independently tested existing-launch causal audit after the fast control summary is saved. An explicit `--deterministic` sets only Inductor's deterministic config and records the flag; it is a separate compilation protocol, not assumed numerically equivalent to default.

After the base-model results and final compilation gates are saved, the prepared launcher loads the released034 native tail (SHA256 `8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a`). Validate checkpoint architecture, original head digest, revision and every tensor before installation. Copy only trained native tensors into the existing shared model. Evaluate the same four prompts in every mode through the **existing compiled graphs**; require unchanged wrapper identities/compiler counters, exact repeats and at least one changed full-vocabulary output to reject stale frozen weights. Save base and trained results separately, including native Yes/No logprobs, full-vocabulary answer probabilities, and top tokens. No alternate head, fitting, or new train/test selection. The optional detailed origin audit and trained-tail installation cannot be combined in one invocation, avoiding ambiguous base-versus-trained causal attribution.
