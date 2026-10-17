# Fixed endpoint comparison: 1,024 → 3,072 updates

Full-vocabulary answers and loss, using cached-tail split inference. Brackets are pointwise 95% whole-prompt bootstrap intervals (5,000 draws). The test cohort is historical and exploratory; it is not used in the learning curves.

## Development

| Condition | Updates | Both answers correct | Yes/No gap, nats | Full-vocabulary CE |
|---|---:|---:|---:|---:|
| Buffer copy | 1,024 | 0.00% [0.00, 0.00] | 0.0010 [-0.0022, 0.0044] | 0.6959 [0.6947, 0.6972] |
| Buffer copy | 3,072 | 0.00% [0.00, 0.00] | 0.0011 [-0.0039, 0.0062] | 0.7046 [0.7013, 0.7081] |
| Forced storage | 1,024 | 50.00% [37.50, 62.50] | 3.6758 [3.4160, 3.9199] | 0.6662 [0.5009, 0.8500] |
| Forced storage | 3,072 | 92.19% [84.38, 98.44] | 58.7764 [54.4482, 62.9618] | 0.3893 [0.0724, 0.8224] |

## Historical test

| Condition | Updates | Both answers correct | Yes/No gap, nats | Full-vocabulary CE |
|---|---:|---:|---:|---:|
| Buffer copy | 1,024 | 1.56% [0.00, 4.69] | 0.0000 [-0.0049, 0.0049] | 0.6991 [0.6966, 0.7019] |
| Buffer copy | 3,072 | 0.00% [0.00, 0.00] | -0.0020 [-0.0066, 0.0022] | 0.7192 [0.7127, 0.7263] |
| Forced storage | 1,024 | 25.00% [15.62, 35.94] | 3.2754 [3.0254, 3.5176] | 1.3651 [1.0803, 1.6757] |
| Forced storage | 3,072 | 68.75% [57.81, 79.69] | 49.2675 [44.5826, 53.7343] | 1.6834 [0.8928, 2.6106] |

The JSON includes training scores, individual-answer accuracy, and paired before/after change intervals. Accuracy and loss answer different questions: forced storage improves test correctness while worsening test cross-entropy. An all-zero or all-one empirical bootstrap interval is not population certainty.

Reproduce: `PYTHONPATH=src python scripts/summarize_viewing_endpoints.py`.

[Raw-source hashes and full statistics](endpoints.json)
