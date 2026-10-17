# Identify the source of the compiler footprint

This investigation separates four questions: what changes during compilation,
which arithmetic operation first changes its result, how the difference propagates,
and how the trained model reads it. The last question already has causal evidence
in the saved two-block detector; it does not explain the compiler's first difference.

## Established evidence

In audit 042r, a reference capture at block 14 changes observed tensors as early
as block 1's rotary query or block 2's projection input, depending on the prompt.
These are the first **observed** differences, not necessarily the first arithmetic
differences. Passive tracing, native repeats, identity interventions and restoration
all passed exact checks. Shared Triton source bodies match, while upstream buffer
reuse differs. Early selected launch configurations were not saved.

The block-14 pipelines produce different queries on identical physical inputs,
but swapping that query alone does not reproduce the donor model's logits.
Consequently the complete result cannot yet be assigned to one rounding event.

## Essential compilation control

Compare independently compiled unprobed programs A and B with a reference-probed
program C, using the same weights, fixed input shape, GPU and software. Disable
cross-compilation cache reuse and record the actual generated wrappers and selected
kernel objects. Repeating A establishes repeatability of A, not equivalence of A
and B. An A/B difference would expose compilation-instance variation that must be
separated from a reproducible effect of adding a probe. One A/B match is useful
but cannot prove all future independent compilations agree.

The first pilot uses the four prespecified detection-training prompts from 042r.
It measures both A/B and A/C without training, prompt selection or test-set access.
Later independent compilations and reversed compile order are needed before
claiming a stable association with probe presence.

## Find and reproduce the first differing operation

Observe existing materialized inputs/outputs of every early kernel launch in the
generated wrapper. Do not insert a new tensor return into the model graph. Record:

- Input values, shapes, strides, storage offsets, alias relationships and alignment.
- Output values before reuse overwrites them.
- Kernel source, selected block/warp configuration, compilation options and exact
  compiled binary/PTX provenance when available.
- For external kernels, the call arguments and measured replay; do not infer the
  chosen GPU algorithm solely from the Python operation name.

For the earliest differing operation, require equality of all its inputs, then
replay both native implementations on the same physical values. First each replay
must exactly reproduce its own recorded output. Failure to reproduce invalidates
the attribution; it is not evidence for a particular rounding mechanism.

Separate arithmetic and layout by crossing implementation with destination/source
layout or alignment where the original contracts permit it. Preserve strides and
aliases. Reusing the exact selected binary tests a different hypothesis from
recompiling the same source. Unsupported calls remain explicit unresolved sites.

## Causal confirmation

After locating a candidate, substitute the relevant selected kernel/configuration
or its output at the existing launch boundary. Test both directions, self-swaps
and restoration. Measure every observed downstream boundary and every answer
vocabulary logit. A local replacement explains only the portion it reproduces;
multiple independent differences may remain. If needed, progressively match all
divergent upstream launches and report the residual effect at block 14 separately.

If different reduction trees or fused arithmetic are implicated, minimize the
actual captured-input example and inspect the corresponding operations in PTX.
Demonstrate the differing rounded values with the original inputs. Do not replace
the real example with synthetic noise and claim the original cause is proved.

## Controls derived from the mechanism

A proposed compiler control is successful only if it removes the relevant
condition information in the complete inputs available to the learner, including
all token states and auxiliary tensors. Equal final logits alone are insufficient.
Always retain the original experiment alongside the controlled one; a deterministic
compiler flag or materialization barrier changes the execution being investigated.

No universal transparency claim follows from a finite number of matching cases.
The deliverable is a versioned reproducer with explicit scope, causal interventions,
raw artifacts and remaining uncertainties, not an assertion that all compilers or
probe implementations share one mechanism.

## Implementation references

The relevant code is the deployed PyTorch 2.10 implementation, rather than a moving
latest release: [Triton launch/autotuning runtime](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/runtime/triton_heuristics.py)
and [compiler configuration](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/config.py).
PyTorch's [numerical accuracy documentation](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
explains why mathematically equivalent executions need not be bitwise equal;
it does not establish which mechanism occurred in these experiments.
