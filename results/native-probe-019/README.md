# Many-prompt probe detection in native untied Qwen3-8B

Qwen3-8B, BF16, TorchInductor 2.10, NVIDIA L40. The existing input and output
matrices are untied by default; no architecture change was made. Internal weights
were fixed. Native Yes/No output rows were fitted by three ridge least-squares
corrections to ordinary-inference hidden states from 256 paired prompts.

The final held-out test separation on 64 prompts is **1.3711 nats**. Relative to
the same prompts before fitting, improvement is **1.3555 nats**, with a 5,000-draw
paired whole-prompt bootstrap 95% interval **[1.1895, 1.5293]**. Binary test loss is
**0.5400 nats**, versus ln(2) = 0.6931 for an uninformative balanced predictor.
The test-loss interval is [0.4939, 0.5922].

Training separation is 1.9639; validation separation is 1.2031. Extraction
preserved every measured vocabulary logit, and final hidden states remained
bit-identical after each output-row update. Only the native model's actual
compiled outputs are scored. The single-prompt integration check reached 2.25
nats and was reset before fitting the many-prompt model.

These are held-out contexts from the same synthetic template distribution, one
base checkpoint, one hardware configuration, and one deterministic fit. This is
not a test of new prompt families or different probe sites. Comparison with the
0.6B runs does not isolate model size or weight tying: their starting checkpoints
also differ. This is supervised output-weight fitting, not full-model SFT.

Fresh-process recompilation verification is pending in this snapshot.
