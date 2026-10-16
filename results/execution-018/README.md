# Native detection of gradient-recording versus no-grad execution

Qwen3-0.6B from SFT-003, BF16, TorchInductor 2.10, NVIDIA RTX 3090.
Both conditions use eval mode, identical input tokens and weights, and no
query-norm capture hook. Yes means the forward pass records gradients; No means
no-grad execution. This tests a numerical execution distinction, not whether an
optimizer is externally present or why someone is evaluating the model.

Only the existing Yes/No output rows were fitted from 256 paired training
prompts, using three ridge least-squares corrections. Weight tying remains intact;
every evaluated input omits the modified token IDs. Extraction preserved all
vocabulary logits, and internal features stayed bit-identical through fitting.

Final separation on 64 held-out test prompts is **1.8945 nats**. Paired
improvement from before fitting is **1.8999 nats**, with a 5,000-draw whole-prompt
bootstrap 95% interval **[1.7952, 2.0015]**. Binary test loss is **0.3366 nats**
(interval [0.3216, 0.3527]), versus ln(2) = 0.6931 for an uninformative balanced
predictor. Training separation is 2.0002; validation separation is 1.7603.

Intervals condition on this fit and the common synthetic prompt generator.
Independent training seeds, new template families, other hardware, and semantic
understanding of the execution label are not established by these measurements.
This is native supervised output-weight fitting, not full-model SFT.

Fresh-process recompilation verification is pending in this snapshot.
