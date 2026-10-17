# Native capture attempts 060: allocation failure

The forward-hook arm could not allocate an allowed GPU at or below $0.95/hour. No model loaded, no capture gate executed, and **zero optimizer updates** occurred. Scores are null, not chance-level learning results. GPU rent was $0; no pod remained.

`allocation_failure.json` and sanitized `allocation.log` retain the evidence. See the [complete failed-attempt index](../library-capture-training-failures-060-061/README.md).
