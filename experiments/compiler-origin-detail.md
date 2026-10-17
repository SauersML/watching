# Detailed native-launch audit after the compilation control

This is an optional extension of `watching.compile_control`, not a training run.
The independent-compilation control must be saved first. It compares two separate
unprobed graphs (`none_a`, `none_b`) and a graph with the reference probe at block 14.
All use one unchanged model, the same four prespecified training prompts and the
same singleton, length-96 inputs. Shared live kernel objects are explicitly reported;
independent Python entrypoints alone do not establish independent kernel selection.

## What the observer records

`watching.compiler_origin.run_origin_audit` receives the three already-warmed arm
callables. It observes existing Python launch-wrapper operands without adding a
Dynamo output, model hook, tensor operation inside a compiled kernel, or graph break.
It records every Triton, matrix-multiplication and efficient-attention launch in the
first three blocks, including the next block's normalization before its query
projection. It also retains the standard six native boundaries throughout the model
and the block-1 query pipeline.

For each operand it records values or a hash, shape, stride, storage offset, alias
relationships and pointer alignment residues. Absolute process addresses are never
published. Large immutable model parameters are hashed and referenced in memory;
activation buffers are copied before reuse. These copies are observations of existing
values, not candidates for a separately replayed transformer suffix.

Unobserved, observed and restored executions must have exactly equal full-vocabulary
logits. The observer also requires complete launch coverage. Numerical artifacts are
saved before optional PTX/cubin metadata, so missing auxiliary provenance cannot
silently erase a valid capture. Every warmed generated kernel's selected configuration,
cache hash and available PTX/cubin are retained with hashes.

## Attribution and intervention

Compare A/B before A/reference. Locate the first observed unequal launch output and
check whether every launch input value is equal. A difference whose inputs already
differ is downstream evidence, not the initiating arithmetic discrepancy.

For the prespecified block-1 query kernel and the first unequal operation when
supported:

1. Reconstruct operands preserving shape and stride. Unsupported overlapping or
   cross-pointer aliases fail explicitly; an `in_out` pointer remains one object.
2. Launch each original selected implementation on one shared set of physical input
   **and output** allocations, in A/B/B/A order, restoring inputs each time.
3. Require each implementation to reproduce its own captured output exactly and
   require repetitions to match. Reallocation changes storage offset/alignment;
   those changes are recorded, and failed own-input reproduction blocks attribution.
4. Where complete Triton source and argument contracts match, replace the selected
   kernel object at exactly one existing launch. Require identity substitutions and
   subsequent restoration to reproduce every vocabulary logit. Run donor substitutions
   in both directions and report residual downstream/logit differences.
5. Independently swap the existing block-1 rotary query in both directions, again
   with exact identity and restoration checks.

A donor substitution is not automatically a successful explanation: donor logits
must actually match before claiming complete mediation. Multiple divergent operations
may remain. External GEMM/attention launch arithmetic is captured, but exact native
algorithm replay is presently unsupported and must be reported as such. The trace
locates the first **observed materialized** difference; it does not reveal every
register-level intermediate within a fused kernel.

## Controls and scope

`--deterministic` (when enabled in the control runner) is a separate compiler preset;
it must not be silently substituted for default Inductor behavior. One A/B match
cannot prove that all independent compilations agree, and one A/B difference cannot
establish the distribution over compilation outcomes. These four prompts do not
supply population confidence intervals or training-seed replication.

The existing source/tensor contracts are version-specific. Unsupported code generation
fails closed. No additional paid job is authorized by running the CPU tests or
preparing this module; the bounded GPU invocation is reviewed separately.
