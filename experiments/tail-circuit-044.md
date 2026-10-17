# 044: Native head patches and mediation of the role-token footprint

Prepared for review; not launched by this change. The pretrained model is Qwen3-8B,
with the fixed final 034 two-block detector and unchanged native output matrix.
No weights, training prompts, answer tokens, or source compiler graphs are changed.

## Method provenance and adaptation

This follows the bidirectional patching definitions and native component sites in
SauersML/params `method/oracle/circuit_finder.py` at commit
`4af1a97c7f3c9a3b9cf4a68fea127d1905937fd9` (file SHA-256
`a27c7f8261cf803319a70be6b34337bae7bf127e39ef4a3bbe8eaddeae1fe7ae`).
That committed file was read with `git show`; the changing params worktree was not
modified or imported.

The adaptations are deliberate: both conditions have **identical input tokens**,
so there is no first-different-token shortcut. Every forward has the original
singleton sequence length 96 and all native operations in blocks 34–35. There is
no variable batch size, truncated prefix, custom attention implementation, or
activation hook in the compiled source. The source states are the exact public
034 cache already checked in 041r. The tail runs eager BF16 math-SDPA.

## Fixed scope

Reuse precisely the 041r selection: eight train prompts and eight previously
evaluated test prompts, in their archived order. This remains a post-hoc circuit
diagnostic, not an independent performance confirmation. Test single-head scans
are exploratory; no test measurement enters selection of the reported head group.

1. Capture each native `o_proj` input (32 query-head slices per block), `down_proj`
   input (SwiGLU neuron products), attention output and MLP output. These are the
   actual inputs/outputs of the existing modules, not reconstructed substitutes.
   Clone hooks must preserve every vocabulary logit exactly.
2. Verify archived trained answer scores, self-patching of every captured tensor,
   whole-input swaps in both directions, and restoration of baseline outputs.
   Any mismatch is fatal. Never relax a tolerance to make the run pass.
3. Extend the `user` role-token patch to all 16 pairs in both directions, with
   auxiliary masks/rotary tensors unchanged. Its physical position is read from
   the fixed token IDs and mask. The first `<|im_start|>` state must have zero
   condition difference, and swapping it must change no vocabulary logit.
4. For all 64 native heads (32 each in blocks 34 and 35), patch only the answer
   position from the other condition. Measure absent←present sufficiency and
   present←absent necessity separately. Retain negative and overshooting effects.
5. Patch whole attention and MLP outputs, both at the answer position and at all
   positions, as component reference measurements. These are not mathematical
   upper bounds: nonlinear interactions can make a smaller patch more effective.
6. Rank heads by the train-prompt mean of `(sufficiency + necessity)/2`, measured
   in nats. Select exactly the top four; ties use block then head index. Lock and
   serialize this selection before executing any test prompt in this run. Measure
   the same group on all eight test prompts without changing its members.
7. Test path mediation: patch only the upstream `user` state, then restore those
   selected answer-position heads to their recipient-condition baseline values.
   Repeat both directions. Compare with the user-only intervention and unchanged
   baseline. This tests whether the selected group mediates the measured user-token
   effect, rather than merely showing two independently influential locations.

## Scores and limits

Let `s` be the native Yes-minus-No logit, equal to Yes/No log-probability difference.
With absent/present baselines `s0,s1`, sufficiency is `s(absent←present)−s0` and
necessity is `s1−s(present←absent)`. Report both raw nats and fractions of `s1−s0`.
Fractions are undefined when the baseline gap is below the predeclared 1-nat
threshold; never omit the absolute effects or affected prompts. Counterfactual
patches have measured responses, not assigned ground-truth accuracy labels.

Individual head effects need not sum to a group effect. Group mediation need not
explain the entire pathway: unselected heads, MLPs, residual routes, other token
positions, and source layers remain possible. This run does not establish a
complete minimal circuit, global monitoring awareness, or a mechanism for every
probe implementation. No population CIs are manufactured from deterministic
repeats or multiple heads on the same prompt.

## Reproduction and runtime

`bash scripts/run_tail_circuit.sh OUTPUT` downloads SHA-verified public 034 weights
and cache, copies immutable small source metadata, then runs
`python -m watching.tail_circuit`. Require L40S to match the successful exact replay
environment; record driver and all runtime metadata. A 900-second diagnostic
guard checks progress between prompts. Provisioning should impose a separate
30-minute/$0.60 maximum and disable low-utilization shutdown during setup.

Flush JSONL after every measured head, role patch, component reference and group
intervention. Save the fixed selection immediately and aggregate score snapshots
after each prompt. Save paired full-sequence native head/neuron/component tensors
per prompt for later neuron refinement; these may be published as release assets
rather than committed binary files. Training and test results remain separately
labeled. The native output matrix checksum must match at the end.
