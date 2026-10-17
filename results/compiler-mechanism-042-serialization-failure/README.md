# Metadata serialization integration failure

The unprobed compiled execution again passed repeat and full-vocabulary passive capture checks. Saving its selected compiler metadata failed because Triton stores its GPU target as a Python dataclass. The generated wrapper and selected PTX were written before the error and are retained here; `capture_metadata_incomplete.txt` is incomplete and must not be treated as a finished result. No reference-probe comparison or query replacement ran.

The reporting fix normalizes dataclass metadata structurally, without changing kernels or numerical tests. All workers were deleted, and the exact temporary approval entry was removed. This attempt cost $0.0668 including rejected hosts; cumulative audit startup/integration cost is $0.2175.
