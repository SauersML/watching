# Metadata serialization integration failure

The unprobed compiled execution again passed repeat and full-vocabulary passive capture checks. Saving its selected compiler metadata failed because Triton stores its GPU target as a Python dataclass. The generated wrapper and selected PTX were written before the error and are retained here. The runner deliberately excludes incomplete `.partial` files, so the unfinished metadata file was not synchronized and is unavailable. Numerical tensors were scheduled after that write and therefore were not saved; the corrected saver now writes numerical evidence first. No reference-probe comparison or query replacement ran.

The reporting fix normalizes dataclass metadata structurally, without changing kernels or numerical tests. All workers were deleted, and the exact temporary approval entry was removed. This attempt cost $0.0668 including rejected hosts; cumulative audit startup/integration cost is $0.2175.
