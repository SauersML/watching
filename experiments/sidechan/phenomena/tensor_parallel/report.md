# Tensor parallelism / sharding — report

*reduction-order change (dies under weight updates)*

Whether the layer is sharded across devices (partial sums added across GPUs).

## Finding

- 59% as trained, 9% after the bit flip.
- Does not form under full-network RL.

## Numbers

```json
{
 "tp_4": {
  "both_right_trained": 0.59375,
  "both_right_after_lsb_flip": 0.0859375,
  "both_right_after_gaussian": 0.1015625,
  "step0_direction_cosine": 0.01039366191253066
 }
}
```

## Figures

![detectors](../../paper/figures/core/detectors.png)
