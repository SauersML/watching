# Cross-sequence leakage — report

*no learnable signal*

Whether one request can read the content of another request in the same batch.

## Finding

- Under a correct mask a neighbour's content changes 0 bits (bit-identical across 16 neighbours).
- The only couplings are per-tensor dynamic activation scaling (up to ~4 nats) and a broken block mask.
- So reading a batchmate's content is not an emergent capability; it needs a value-coupling path or a bug.

## Numbers

```json
{
 "distinct_target_states_over_16_neighbours": {
  "padded_batch": 1,
  "packed_masked": 1,
  "fp8_per_token": 1,
  "fp8_per_tensor": 16,
  "packed_no_mask": 16
 }
}
```
