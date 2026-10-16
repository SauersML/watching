# Two-block pilot 029r

Qwen3-8B's final two transformer blocks and final normalization were trained for
128 paired updates; all earlier weights and the vocabulary head stayed fixed.
The primary runtime combines a frozen compiled source with an eager BF16 native
tail. It is explicitly different from the original whole-model compiled runtime.

| Endpoint | Prompts | Paired log-odds separation, nats (95% CI) | Actual answer accuracy | Full-vocabulary loss |
|---|---:|---:|---:|---:|
| Training |16|0.0391 [0.0156,0.0703]|56.25%|0.6956|
| Development |8|0.0313 [−0.0313,0.0938]|50.00%|0.7326|
| Held-out question families |32|0.0078 [−0.0195,0.0352]|48.44%|0.7269|
| Whole-model transfer, held-out |32|0.0000 [−0.0313,0.0313]|50.00%|0.7263|

The training endpoint shows a small positive separation. Development and held-out
results do not establish reliable detection. Most loss reduction corrected the
initial answer bias. Late training separation fluctuated rather than increasing
monotonically. The continuation 030 was proposed after inspecting this trajectory.

Intervals resample entire prompt pairs and describe these small cohorts, not
variation across seeds, hardware or compilation. The figure shows pointwise
intervals; repeated checkpoints are correlated. Head invariance, source-state
parity and gradient/no-gradient learner checks passed.

Scientific execution took 233 seconds on one L40S, including compilation; the 128
training updates took about 19 seconds. The model revision was not recorded beyond
the model name and available hashes; this limits exact reproduction. Source code
commit and artifact SHA256 values are in archive.json. Native tail weights are
retained outside Git because the artifact is 772 MB. No full base model is included.

See [progress.pdf](progress.pdf) and the raw per-prompt JSON files for all scores.
