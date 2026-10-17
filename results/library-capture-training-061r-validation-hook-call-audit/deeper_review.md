# Local follow-up: what the completed diagnostic does not identify

No further GPU or remote action was used for this review. The completed run establishes stable repeats within each graph and variation between two compilations of the same bare call. It does not identify a cause.

## Actual coverage

`library_call_audit.evaluate` verifies the saved CPU prompt against the frozen selected data once, before transferring input tensors to CUDA. It hashes `model.state_dict()` before `model.cuda()` and after all five arms. Those endpoint hashes match. Each arm verifies parameter flags/dtypes/devices and module training flags. Eager calls before all arms match one another. Outputs are detached, copied to CPU and cloned before another call, so the saved output comparisons do not alias a later GPU output buffer.

These controls make gross persistent model changes less plausible, but do not prove every runtime input was identical. `state_dict()` omits nonpersistent buffers. In the inspected Qwen3 implementation, rotary `inv_freq` and `original_inv_freq` are nonpersistent buffers. The generated wrapper accepts a 64-element rotary tensor as runtime argument 4; it is not verified by the existing state hash. There are no per-call hashes of CUDA input IDs/mask/positions, no per-arm tensor-value hashes, and no recorded storage aliases or version counters. Endpoint equality cannot rule out a transient change restored before the final hash. These are evidence gaps, not findings that any mutation occurred.

## Generated code and external runtime

The largest generated wrapper unpacks 403 runtime arguments. It contains 254 `extern_kernels.mm` calls and external `_scaled_dot_product_efficient_attention` calls. The saved Triton sources/binaries do not specify cuBLAS or attention runtime algorithm/workspace choices. Identical wrapper source alone cannot prove identical live arguments, materialized constants, selected launch configurations, streams or external-kernel execution.

The run records `autotune_pointwise=True`, 40 GPU benchmarking calls per arm, and 45 compiled Triton variants per arm. The collector preserves all `.py/.json/.cubin/.ptx/.ttir/.ttgir/.llir` artifacts found, but its extension whitelist omits `.best_config`. The locally inspected PyTorch `AutotuneCache._setup_local_cache` writes precisely that extension, and `save` records launch kwargs, warps/stages and the selected Triton cache hash. Consequently the release preserves candidate compiled kernels but may omit the selected-launch map. It is unknown whether these sidecars are present in the retained remote cache; they must be inspected only during a separately authorized recovery. This is a concrete preservation gap, not proof that autotuning explains the numerical difference.

A CPU reanalysis of the saved bare outputs finds positions 0–57 identical (50 left-padding positions and the first eight attended tokens). The first difference is position 58. Exactly one position changes full-vocabulary argmax, position 78; the final token remains No. See `attempt2/bare_recompile_position_review.json`. This pattern constrains the discrepancy to this prompt but does not localize the responsible layer or operation.

## Smallest proposed next diagnostic

Run only two identical bare/omitted compilations, twice each, on the same final weights and exact failed prompt. Do not repeat the signature/hook grid, train, calibrate or score accuracy. Hash the model's named parameters **and all buffers**, including nonpersistent/duplicate aliases, immediately after CUDA transfer and before/after each compile. Check CPU-to-CUDA input byte equality and record input layout/storage/version at the same boundaries. Save both native full-sequence outputs from each graph.

Use `RuntimeArgumentProbe` as a backend wrapper to hash every actual graph argument immediately before and after each invocation and record placeholder names, values, layout, alias groups and version counters. The compiler's fake/example inputs are never value-hashed. Preserve any graph-owned tensor constants/get_attr values separately or fail the coverage gate if they cannot be identified. Save every small cache sidecar, including `.best_config`, plus runtime/stream/numeric settings. Require exactly one graph per compilation.

If real runtime arguments differ, the changed graph argument supplies a concrete next target. If they match while outputs differ, the discrepancy remains downstream of those observed graph inputs: selected kernel/configuration, external runtime, allocation-sensitive behavior or a compiler defect remain hypotheses. Compare selected launch maps before attributing any effect. Interleaving the two already-compiled callables A/B/A/B can then distinguish stable graph-specific behavior from time/order dependence without more compilations.

The observer performs full byte copies and synchronizes CUDA. It therefore changes allocation/timing and is a separately named instrumented runtime, not proof of what an unobserved execution does. If adding it removes the discrepancy, that is itself an observer sensitivity result; do not retrofit it into the original validator or weaken the failed parity gate. No such follow-up has been run.

## Prepared local code

`src/watching/library_runtime_probe.py` provides read-only tensor/model fingerprints and an outside-graph compiler-backend wrapper. Nine local tests cover a real Dynamo capture, a boxed delegate that clears its argument list and mutates a tensor, nonpersistent buffer mutation/aliases, and scalar/noncontiguous layouts. These helpers do not change the primary validator, the published matrix or any GPU launch path. The standalone `library_runtime_audit` runner and bounded `bootstrap_library_runtime_audit.sh` are prepared for two bare compilations with two calls each; execution still requires coordination. The original call-audit evaluator accepts an optional matrix callback to reuse the pinned setup. Its default matrix and the primary validator/gates are unchanged.
