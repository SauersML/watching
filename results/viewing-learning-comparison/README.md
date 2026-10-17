# Comparing learning from four numerical input conditions

![Development learning comparison](development.png)

[PDF](development.pdf) · [SVG](development.svg) · [Source hashes and statistics](development.json)

All curves use the same 64 development prompt pairs and the same initial 1,024-update recipe: native Qwen3-8B last two blocks and final norm, frozen existing output matrix, paired full-vocabulary cross-entropy, eight prompt pairs per update. The left panel requires **both full-vocabulary top-token answers** to be correct for a prompt. It is not individual-answer accuracy or forced-binary accuracy. The right panel measures the paired Yes/No log-odds difference.

The reference probe exposes a query-norm activation as a graph output. The buffer-copy probe writes the activation into a preallocated buffer; it leaves a numerical footprint for some prompts in this source build. The **forced-storage control** uses an opaque identity operation: it is a mechanism intervention, not a read-only probe or a zero-signal null. The identical-input null uses each 034 absent cached state for both contradictory labels, so both-correct accuracy and separation must be zero.

The four runs share prompts and training settings, but their historical compiled source instances differ. Only the 034/045 reference/null comparison uses the exact same original source cache. These curves do not isolate probe implementation from compilation-instance effects, or estimate variation across seeds or builds.

The blue dotted ceiling follows from 47 of 64 development buffer input pairs being byte-identical: a deterministic model cannot answer both labels correctly for those pairs. The remaining 17 pairs allow at most 26.5625% both-correct accuracy for this fixed cohort/cache. This is not a bound on other buffer implementations or compiler builds.

Bands are pointwise percentile intervals from 5,000 resamples of whole prompt pairs. They describe prompt variation conditional on each single run; all-correct samples give zero-width empirical intervals without population certainty. Test scores are not used in this figure. No checkpoint is selected from these curves.

Reproduce with `PYTHONPATH=src python scripts/plot_viewing_comparison.py`. The `--split train` option writes the training comparison if a separate `--out` stem is supplied. Original 034 scores record full-vocabulary argmax correctness without explicit top-token IDs; later archives include those IDs and the generator verifies them. Existing user figures are unchanged.

## Continuation updates

The blue and purple curves now include the completed 053 and 054r continuations after the original 051/052 endpoints. The failed 054 provisioning attempt is not plotted. The total-step axis continues from 1,024; the vertical marker indicates fresh Adam moments, while weights, cached states, learning rate, prompt order and the frozen head continue. The gray and green historical curves end at 1,024 and are not extrapolated. These are still cached-tail split-inference results, not new compiled whole-model transfer measurements.

Rebuild the continuation figure with:

```bash
PYTHONPATH=src python scripts/plot_viewing_comparison.py \
  --buffer-continuation results/viewing-continuation-053-054/buffer/fit \
  --storage-continuation results/viewing-continuation-053-054/identity054r/fit
```

All checkpoint scores, final manifests and hashes are included in those archives. Each continuation passed exact baseline replay before its first optimizer update. Earlier snapshot manifests are retained by SHA-256 under each arm’s `snapshots/` directory.

Both continuations reached the fixed endpoint of **3,072 total updates**. Forced storage rises from 50.00% to 92.19% both-correct development accuracy. Buffer copy remains at 0%. Every checkpoint remains plotted; neither endpoint was chosen from these curves.

## Fixed endpoint scores

The table intervals below use 5,000 whole-prompt bootstrap draws (seed 20261017). They are conditional on each single training run. The historical test cohort is exploratory and stays off the learning curves.

### Development

| Condition | Updates | Both answers correct | Yes/No gap, nats | Full-vocabulary CE |
|---|---:|---:|---:|---:|
| Buffer copy | 1,024 | 0.00% [0.00, 0.00] | 0.0010 [-0.0022, 0.0044] | 0.6959 [0.6947, 0.6972] |
| Buffer copy | 3,072 | 0.00% [0.00, 0.00] | 0.0011 [-0.0039, 0.0062] | 0.7046 [0.7013, 0.7081] |
| Forced storage | 1,024 | 50.00% [37.50, 62.50] | 3.6758 [3.4160, 3.9199] | 0.6662 [0.5009, 0.8500] |
| Forced storage | 3,072 | 92.19% [84.38, 98.44] | 58.7764 [54.4482, 62.9618] | 0.3893 [0.0724, 0.8224] |

### Historical test

| Condition | Updates | Both answers correct | Yes/No gap, nats | Full-vocabulary CE |
|---|---:|---:|---:|---:|
| Buffer copy | 1,024 | 1.56% [0.00, 4.69] | 0.0000 [-0.0049, 0.0049] | 0.6991 [0.6966, 0.7019] |
| Buffer copy | 3,072 | 0.00% [0.00, 0.00] | -0.0020 [-0.0066, 0.0022] | 0.7192 [0.7127, 0.7263] |
| Forced storage | 1,024 | 25.00% [15.62, 35.94] | 3.2754 [3.0254, 3.5176] | 1.3651 [1.0803, 1.6757] |
| Forced storage | 3,072 | 68.75% [57.81, 79.69] | 49.2675 [44.5826, 53.7343] | 1.6834 [0.8928, 2.6106] |

Forced storage improves historical-test both-correct accuracy from 25.00% to 68.75%, but mean full-vocabulary cross-entropy rises from 1.3651 to 1.6834 nats: the remaining errors are more costly. This is a descriptive mean tradeoff, not a claim that the loss increase is statistically established. No calibration or model adjustment was applied.

[Endpoint statistics and raw-source hashes](endpoints.json) also include training results, individual-answer accuracy, and paired before/after change intervals. Reproduce them with `MPD_MEM_GIB=1 PYTHONPATH=src python scripts/summarize_viewing_endpoints.py`.
