# A native boundary need not be a single stored residual tensor

This is a CPU source inspection of the **042r** generated wrappers, not a new GPU
experiment and not a diagnosis of every cause of the later 047 extraction failure.
Both archived 042r wrappers have the same relevant property before block 34:

- The native normalization kernel reads two separate BF16 residual summands.
- It converts them to FP32 and adds them inside the kernel.
- It computes normalization from that sum and stores the normalized output.
- It does not store the unnormalized sum as a separate BF16 tensor for this consumer.

Therefore a passive observer can copy the existing summands and normalized output,
but cannot assume that it has recovered one native BF16 pre-block residual tensor.
Adding a BF16 sum before separately replaying the final two blocks changes where
rounding happens. This is a concrete reason to require numerical replay parity,
rather than treating a logical module boundary as a neutral materialized cache.
It does not prove that an alternative exact replay representation is impossible.

The JSON reports identify the exact original wrapper/source hashes, launch lines,
pointer bindings, stored values and the conservative adapter's status. The status
`single_native_tail_input_certified: false` is intentional. The optional
`ExistingBoundaryObserver` can observe these existing tensors; it has not yet been
GPU-validated and does not itself certify full-model extraction or replay parity.

Reproduce without a GPU:

```sh
PYTHONPATH=src python -m watching.boundary_audit \
  results/compiler-mechanism-042r/compiler_sources/212913f5994a2c54bfc050b2f81257a860d0780386a7dbef888b12e18e7bc18b.py.txt \
  --first-tail-layer 34 --out results/native-boundary-audit/none.json
PYTHONPATH=src python -m watching.boundary_audit \
  results/compiler-mechanism-042r/compiler_sources/17e9b1279afdf108c4a69f83610827c7ab6405cd77ffe6e7c3483f72d564fe52.py.txt \
  --first-tail-layer 34 --out results/native-boundary-audit/reference.json
```

The four focused tests inspect both actual archived wrappers, reject unsupported
structure and require the observer to encounter its expected generated wrapper.
