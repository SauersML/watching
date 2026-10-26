# Weight noise injection — report

*structured change (survives weight updates)*

Whether Gaussian noise has been added to the weights (as in a sandbagging audit).

## Finding

- Robust from sigma 1e-3 (91%, still 87% after the flip) and 100% by 3e-3.
- At 5e-4 it degenerates into a clean-weights fingerprint (49%, dies on any weight change).
- See noise_audit for the safety-relevant companion experiment.

## Numbers

```json
{
 "noise_5e-4": {
  "both_right_trained": 0.4921875,
  "both_right_after_lsb_flip": 0.171875,
  "both_right_after_gaussian": 0.125,
  "step0_direction_cosine": null
 },
 "noise_1e-3": {
  "both_right_trained": 0.9140625,
  "both_right_after_lsb_flip": 0.8671875,
  "both_right_after_gaussian": 0.8671875,
  "step0_direction_cosine": null
 },
 "noise_3e-3": {
  "both_right_trained": 1.0,
  "both_right_after_lsb_flip": 1.0,
  "both_right_after_gaussian": 1.0,
  "step0_direction_cosine": null
 }
}
```

## Figures

![detectors](../../paper/figures/core/detectors.png)
