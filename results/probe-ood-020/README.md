# Probe detection on new prompt families

The saved Qwen3-8B detector from 019 was evaluated without refitting on an L40.
Baseline and edited weights used the same compiled graphs. Extraction preserved
all vocabulary logits, and installing the edit left every evaluated hidden state
bit-identical. There were 128 OOD prompts in eight manually authored families.

The OOD condition signal transfers: final paired Yes/No separation is **0.7639
nats**, with a whole-family bootstrap 95% interval **[0.6133, 0.8955]**. Paired
improvement from baseline is **0.7805 [0.6135, 0.9367] nats** under that bootstrap.
All eight family mean separations are positive. Prompt-bootstrap intervals and
all family summaries appear in `analysis.json`.

Reliable individual classification does not transfer. OOD binary log loss is
**4.2304**, versus 0.6931 for equal probabilities, with a whole-family interval
[2.7982, 5.9250]. Forced-choice accuracy is 51.17%; only 3 of 128 prompts are
correct in both conditions. A positive paired shift can coexist with a large
shared answer bias. The result supports transfer of the numerical signal, not a
well-calibrated OOD detector.

On the original in-distribution test contexts, endpoint separation is 1.1914
nats and binary loss is 0.5975. Mean full-vocabulary probability mass on Yes/No
is 0.9991 there and 0.9237 on OOD.

Intervals use 5,000 resamples. Family intervals give each of the eight families
equal weight and assume exchangeable family units; they do not quantify arbitrary
future task transfer, training seed variation, or hardware uncertainty. Context
structure and question wording vary together, so this test does not isolate them.
