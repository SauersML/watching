# Failed native-capture training attempts, 060–061

All attempts in this index completed **zero optimizer updates**. Their absent learning scores are not measured learning nulls. Allocation/setup failures did not exercise the API; missing-boundary failures invalidated the implementation; strict parity failures rejected the proposed source before treatment learnability could be tested. The later completed [061r hook](../library-capture-training-061r-hook/README.md) and [061r hidden](../library-capture-training-061r-hidden/README.md) experiments use an explicitly different `boundary_observed` runtime.

| Historical run | Hook outcome | Hidden-state outcome | Updates, each arm |
|---|---|---|---:|
| 060 | [Capacity/price failure](../library-capture-training-060-hook/README.md) | [Capacity/price failure](../library-capture-training-060-hidden/README.md) | 0 |
| 060r | [Allocator churn](../library-capture-training-060r-hook/README.md) | [Allocator churn](../library-capture-training-060-hidden/allocation-retry-r/allocation_failure.json) | 0 |
| 060s | [CUDA/setup interruption](../library-capture-training-060r-hook/retry-s/outcome.json) | [CUDA/setup interruption](../library-capture-training-060-hidden/allocation-retry-s/allocation_failure.json) | 0 |
| 060t | [Missing boundary](../library-capture-training-060t-hook/README.md) | [Missing boundary](../library-capture-training-060t-hidden/README.md) | 0 |
| 060u | [Strict parity rejected](../library-capture-training-060u-hook/README.md) | [Strict parity rejected](../library-capture-training-060u-hidden/README.md) | 0 |
| 061 | [Persistent-volume setup failure](../library-capture-training-061-hook/README.md) | [Persistent-volume setup failure](../library-capture-training-061-hidden/README.md) | 0 |

Strict 060u measurements: hook **8,821,492** differing full-sequence logits, max absolute difference **0.5078125**; hidden **8,951,160**, max **0.5**, each out of **14,585,856**. Both failures affected all 96 positions, including the scored final position. All scientific failure counts, source/hash provenance, gate records and sanitized logs are retained in the linked archives. Historical names 060r/060s are allocation retries; 060t/060u are corrected-source retries; 061 introduced the explicitly observed source but failed setup.

Raw `.pt`/`.pt.gz` failure tensors and activation caches are excluded from git. The archives document their retained local paths, sizes and SHA256s. A lost hook compiler-source archive is explicitly identified rather than represented as recovered. No heavyweight artifact is required to read the published failure metadata.

Recorded GPU rent for these prior attempts totals approximately **$0.2936** ($0.1468 per arm, summing recorded rounded pod charges); all associated allocated pods have confirmed deletion records. This total excludes completed 061r training, later recovery and storage charges.
