# Animal-choice fits after execution and probe detection

Qwen3-8B, BF16. These runs fit the existing `owl` and `cat` output rows by ridge regression while fixing internal weights. The targets are **owl when the condition is absent, cat when present**. Both reported preference evaluations use a fresh process and compilation. These are output-row fits, not full-model SFT.

| Task / prompt distribution | Correct between the two animals | Actual next token correct | Mean P(owl) + P(cat) |
|---|---:|---:|---:|
| Execution / familiar templates | 93.75% | 20/128 = 15.625% | 18.3704% |
| Execution / new families | 51.9531% | 16/256 = 6.25% | 11.5427% |
| Probe / familiar templates | 63.2813% | 0/128 = 0% | 0.7144% |
| Probe / new families | 50% | 68/256 = 26.5625% | 51.3714% |

The execution fit successfully changes the relative animal scores on familiar templates, but usually neither animal is the model's highest-probability next token. The probe fit likewise does not establish reliable generated animal answers. Conditional binary accuracy must not be reported as unrestricted next-token accuracy.

The original Yes/No detector's familiar-template next-token accuracy after the animal edit is 126/128 (98.4375%) for execution and 83/128 (64.84375%) for probes, equal to the earlier accuracy counts. On new families, the corresponding counts are 118/256 and 122/256. Equality of aggregate accuracy is not a claim of bit-identical logits or predictions.

## Artifacts

- `execution/` and `probe/` contain compact copies of `preference`, `fresh_preference`, and `fresh_detector` results, including manifests, prompt records, metric streams, raw scores, and answer-row edits.
- `detector_before_animal/` contains the earlier detector evaluations for comparison.
- `analysis.json` within each task reports paired before/after log-odds gains, endpoint separation, binary loss, and fresh-compilation comparisons.
- `animal_choices.json` reports independently calculated binary accuracy, actual next-token accuracy, and answer-pair probability mass, with uncertainty appropriate to each metric.
- Large feature tensors are excluded; each task's `archive.json` records sources and omissions.

Intervals use 5,000 paired bootstrap draws. Familiar-template intervals resample 64 whole prompt pairs. New-family intervals resample eight whole families containing 128 prompts, with equal family weighting. Conditions remain paired. These are pointwise intervals conditional on one fit and the sampled units; they exclude training-seed and hardware uncertainty. A zero-width bootstrap interval from zero observed successes does not imply a zero population success rate.

Regenerate the figure from the archived raw evaluations:

```bash
python scripts/plot_animal_choices.py results/animal-choices-021r-022r results/animal-choices-021r-022r/animal_choices
```

Regenerate each paired log-probability analysis:

```bash
PYTHONPATH=src python scripts/analyze_native_fit.py results/animal-choices-021r-022r/execution/preference results/animal-choices-021r-022r/execution/analysis.json --fresh results/animal-choices-021r-022r/execution/fresh_preference
PYTHONPATH=src python scripts/analyze_native_fit.py results/animal-choices-021r-022r/probe/preference results/animal-choices-021r-022r/probe/analysis.json --fresh results/animal-choices-021r-022r/probe/fresh_preference
```
