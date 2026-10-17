# Fresh 038 buffer-copy compiler evidence

This bounded archive contains the exact generated JIT bodies for the no-capture and buffer-copy query-normalization/rotary kernels, plus ten compiled candidate lowerings and metadata. All files are hashed in `manifest.json`. The candidates are not ten experiments: they are configurations compiled by TorchInductor/Triton during the actual fresh-evaluation process.

The selected buffer launch configuration has not been recovered. These artifacts therefore establish a kernel/layout rewrite, not which individual PTX instruction caused the full-model discrepancy. See [the mechanism audit](../../docs/mechanism-audit.md).

The source `.py.txt` files retain the generated arithmetic. The isolated replay imports only their JIT definitions, bypasses the original autotuner, and explicitly enumerates the recorded configurations. It loads no model weights. Reproduction:

```sh
python scripts/replay_probe_kernels.py --artifacts results/mechanism-038 --out replay.json
```

Compiler artifacts were retrieved read-only from the exclusive `watching-confirm-038r` worker while it remained active. No inference was triggered to collect them. Absolute temporary paths inside generated compiler text are original provenance, not portable dependencies.

The synthetic replay completed on the same L40S worker. Across seeds 0, 1, and 2, respectively 21, 15, and 21 of the 24 cross-mode candidate pairs differed; the largest difference involved 7 of 393,216 rotary values. See `replay_corrected.json` for the instruction-matched replay; `replay.json` is the superseded initial diagnostic. `replay_manifest.json` preserves exactly the manifest hashed by that run; the current `manifest.json` adds wrapper-source provenance afterward without changing kernel artifacts.

`ptx_comparison.json` verifies all ten corrected replay programs match the original PTX after removing only comments and debug-location/data sections. The raw corrected PTX is retained in `replay_ptx/`. The first replay accidentally enabled integer-overflow sanitization; its metadata is retained in `initial_replay_metadata.json`. Explicitly disabling that sanitizer restored the original PTX instructions and left every reported pairwise numerical comparison unchanged.
