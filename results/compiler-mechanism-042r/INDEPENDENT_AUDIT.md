# Independent mechanism audit

Read-only audit of the archived042r numeric comparisons, generated wrappers, and PyTorch2.10.0 compiler source. No additional CUDA execution or training. Four fixed training prompts are diagnostic cases, not independent compilation replicates.

## What is established

1. **Local implementation effect:** at block14, both selected query pipelines were replayed on the same physical projection/weight/rotary inputs. Their outputs differ; each pipeline exactly reproduces its own native output. This establishes an arithmetic implementation difference at that site for these selected kernels.
2. **Earlier footprint:** query projection values entering block14 already differ. The earliest *observed* unequal boundaries are block1 query-after-rotary for cases0/2 and block2 query-projection input for cases1/3. Only six boundaries per block were observed, so these are not necessarily the first unequal operations.
3. **Incomplete mediation:** replacing only block14 query does not reproduce donor full-vocabulary logits. The entire graph difference is not explained by that single query replacement.
4. **Learned reader is separate:**034/044 tests establish how a separately trained native tail uses a cached difference. They do not identify which compiler decision first produced it in042r, a new compilation pair of the original base model.

## Missing independent-compilation control

A bit-identical rerun of an already compiled graph establishes fixed-graph repeatability. It does not establish that a second compilation of the same unprobed computation selects the same arithmetic implementation.

042r sets `force_disable_caches=True`. In the exact PyTorch2.10.0 source, `compile_fx.with_fresh_cache_if_config` enters `fresh_cache`; that context clears registered in-memory caches on entry and exit and creates fresh Inductor/Triton cache directories. Both `PyCodeCache` and `CompiledTritonKernels` are registered for clearing. The saved042r wrappers have different temporary cache directories. Consequently, shared kernel text can be independently compiled/tuned across graphs. [compile_fx.py](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/compile_fx.py), [utils.py](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/utils.py), [codecache.py](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/codecache.py), [async_compile.py](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/async_compile.py).

The flag disables persistent best-config lookup; it does **not** disable tuning. `CachingAutotuner` benchmarks available launchers and selects the measured fastest. A different choice can therefore be stable on every subsequent invocation. This is an available confound, not a demonstrated explanation of042r. Early selected configurations were not archived. [triton_heuristics.py](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/runtime/triton_heuristics.py).

The next control must compare noneA/noneB/reference using distinct compilation entrypoints, while recording actual kernel-object sharing, complete source hashes, selected configurations, and binary identifiers where available. Compilation counters or identical source alone are insufficient. NoneA/B disagreement would show a footprint without probe attachment; agreement in one pair would not establish universal compilation invariance. Protocol048 locks this control before execution.

## Allocation is a hypothesis, not an explanation yet

The first wrapper-plan difference retains block0 query storage and reuses it for block1 query-projection output; the no-probe wrapper allocates fresh storage there. Earlier MLP storage lifetimes also change. However, the **values** of block1 query projection match in all four cases. Thus that allocation change has not been shown to cause the first observed arithmetic difference directly.

In cases0/2, the next observed query-after-rotary differs despite equal query-projection values. Its fused Triton source is identical between graphs, but early selected configurations, binary hashes, and all physical argument metadata are absent. Other unobserved inputs and the actual selected implementation must be checked before attributing this to output alignment, vectorization, a reduction tree, or GEMM selection.

## Small discriminating follow-up

- Record early launcher configuration, cache/binary hash, actual argument shape/stride/dtype/storage offset and pointer alignment for noneA/noneB/reference. Preserve ordinary repeat and all-vocabulary passive-observer parity gates.
- Locate the first unequal observed operation, then capture **all** of its inputs and outputs. If any input differs, move upstream instead of labeling that operation the origin.
- Replay both selected implementations on identical physical input tensors, writing alternately into the same output allocation. Exact own-input reproduction is required first. A difference here supports an implementation/configuration cause independent of input values or output allocation.
- Separately replay one fixed selected implementation with identical values copied into matched-shape allocations at controlled alignments. If it changes only with placement, inspect pointer-dependent dispatch or undefined reads; ordinary timing differences alone are insufficient.
- For an external GEMM/attention operation, equal Python calls do not identify the library algorithm. Capture its full inputs and layout, test repeated same-input execution with swapped output allocations, and report backend algorithm/workspace identity only if directly observable. Do not infer an algorithm change from a later output difference.

These tests distinguish the first source of numerical divergence, downstream propagation, and the trained circuit that reads the divergence. A successful local query mechanism does not by itself settle the global compilation-instance question.
