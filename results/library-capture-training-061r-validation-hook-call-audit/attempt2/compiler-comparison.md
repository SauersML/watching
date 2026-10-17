# Local compiler artifact comparison

No operation, reduction tiling, memory layout, launch configuration, or generated CUDA instruction difference was found between the five diagnostic arms. This includes the two bare/omitted compilations whose output tensors differ, and the disabled/False arm whose output matches the first bare compilation. The retained generated code therefore does **not** explain that output grouping.

The comparison read the recovered compiler archive locally; it did not compile, run the model, or use a GPU. Archive SHA256: `a10ac81489c2ffbf52207b5fb987505c9edaee5635068749a59ff0e164ec17da`. The earlier fresh-failure subtree was excluded from this five-arm comparison.

| Captured artifact, per arm | Comparison against first bare/omitted arm |
| --- | --- |
| Entire Inductor wrapper | Identical after AOT-ID comment and absolute source paths are normalized |
| 18 standalone Triton Python kernels | Byte-identical before normalization |
| 45 TTIR, 45 TTGIR, 45 LLVM IR files | Identical after diagnostic source paths/string lengths are normalized |
| 45 PTX kernels | Identical executable prefix after assertion-file source paths are normalized |
| 45 CUDA ELF `.text` sections | Byte-identical, with no normalization |
| Other allocated ELF sections | Identical except assertion-file path strings in 10 `.nv.global.init` sections |
| 90 JSON files | Identical after absolute source paths are normalized |

Every arm has 334 files: 19 Python, 90 JSON, and 45 each of CUBIN, PTX, LLVM IR, TTGIR and TTIR. The normalized full-wrapper SHA256 is `cf979944182d8e98288fa37b7314e141e3fdbd95d78c96aad9a1d933b429cffa` for all five arms. Its complete text equality includes buffer allocation/reuse, shapes/strides, reinterpretations, launch arguments, call order, reduction settings and fusion definitions. It contains 254 `extern_kernels.mm` calls and 36 `aten._scaled_dot_product_efficient_attention` calls, with identical arguments in every arm. No generated cuDNN call appears.

Raw binary hashes differ because compiler-generated files embed their per-arm cache paths. All 45 `.debug_line` sections change; other nonallocated debug/symbol sections sometimes change as well. Ten kernels contain an `assertFile_0` global for the embedding bounds assertion. Its path bytes and, where necessary, string length change; the assertion condition/message and executable instructions do not. These diagnostic-only differences are explicitly retained in the machine-readable comparison rather than reported as numerical kernel changes.

The normalization rules in [compare_compilers.py](compare_compilers.py) remove the AOT-ID comment and absolute `/workspace/` path strings; replace PTX `assertFile` source declarations and LLVM assertion-file array lengths; compare PTX before its first `.file`/debug section; and compare ELF allocated sections by raw bytes, separately recognizing null-terminated source-path-only differences. `.text` equality is checked without normalization. The complete per-arm result is [compiler-comparison.json](compiler-comparison.json).

This archive records the generated calls into PyTorch/vendor libraries, but cannot establish their internal runtime algorithm or workspace choices, pointer addresses, or intermediate numerical values during execution. It cannot identify the first diverging operation. Such runtime explanations remain hypotheses, not findings. Identical captured CUDA code is compatible with the observed output discrepancy; it does not negate that independently verified discrepancy or make hook/signature attribution admissible.

The diagnostic report records the same final checkpoint, exact input, BF16 parameter dtype, parameter `requires_grad` and module training flags, and equal before/after state hashes. Those controls do not measure every possible runtime state. The established result remains within-graph repeat identity alongside between-compilation variation in this same-process protocol. No primary validation gate is relaxed.

Reproduce on CPU with the extracted release compiler archive:

```sh
python3 compare_compilers.py /path/to/compiler-artifacts.tar.gz --output compiler-comparison.json
```

The script uses only the Python standard library. Ruff checks passed, and rerunning it reproduced the stored comparison.
