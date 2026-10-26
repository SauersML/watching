# Dropout / train() — report

*structured change (survives weight updates)*

Whether the forward pass runs in training mode with dropout active.

## Finding

- dropout 0.1: 100% and robust to the bit flip.
- train() with dropout 0: exactly 0 bits change, a clean null.

## Numbers

```json
{
 "dropout_0.1": {
  "both_right_trained": 1.0,
  "both_right_after_lsb_flip": 1.0,
  "both_right_after_gaussian": 1.0,
  "step0_direction_cosine": null
 }
}
```

## Figures

![detectors](../../paper/figures/core/detectors.png)
