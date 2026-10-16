# Fixed inference features across 256 training prompts

Run `watching-readouts-016`, L4, PyTorch 2.10, BF16 Qwen3-0.6B from SFT-003.
Feature extraction preserved every vocabulary logit for all 256 training,
64 validation, and 64 test prompt pairs. Each external classifier receives only
one execution's final hidden state. These are diagnostic readouts, not native
language-model checkpoints.

At the last update (512), linear classifiers reached training separation
2.16–2.19 nats and balanced binary loss 0.33. The one-hidden-layer MLPs reached
10.88–11.32 nats and loss 0.0049–0.0059. Both methods can fit many fixed pairs.

Generalization was weak. Validation selected the untrained constant predictor
for five of six arms. The remaining linear arm selected update 64; on held-out
test its separation was +0.0541 nats (whole-prompt 95% bootstrap interval
[-0.0435, +0.1529]), and binary loss was 0.6994, above ln(2).

The zero-width interval of a selected constant predictor describes its exact
zero separation; it is not high-confidence evidence that all possible training
procedures have zero effect. Training curves and every selected arm's scores are
included. Test data were not used for checkpoint selection. All prompts share a
small template family; these intervals do not measure new-family generalization.
