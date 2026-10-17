# Static-launcher metadata integration failure

The unprobed execution passed strict repeat and passive full-vocabulary tracing checks on the first development prompt. The reference-probe execution stopped during metadata collection because its static CUDA launcher retains a binary hash but no in-memory PTX. No reference capture, kernel replay or causal replacement was completed.

The fix resolves the exact cached PTX and verifies its hash/name/configuration. If PTX is unavailable, passive capture and native-launcher replay can still proceed, while the archive explicitly marks instruction provenance unavailable. Every numerical gate remains exact.

All workers were deleted. This attempt cost $0.0777 including rejected hosts, bringing cumulative cost to $0.1507.
