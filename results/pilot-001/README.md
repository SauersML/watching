# Pilot 001: full-parameter SFT

This run did **not establish learning** to distinguish probe presence. It used Qwen3-0.6B in bf16 on an RTX 3090, PyTorch 2.10.0+cu128 and Transformers 5.18.0, with compiled gradient-enabled forwards. All 596,049,920 parameters were trainable.

Training used 160 AdamW updates, peak learning rate 2e-5, batch size four paired prompts, sequence length 128, and equal full-vocabulary cross-entropy weight for both conditions. The reference-capture probe read layer 14's query-norm output. There were 512 synthetic training prompts and 128 each for validation and test.

| Final test measure | Result |
|---|---:|
| Forced-choice and greedy accuracy | 50% |
| Prompts correct in both conditions | 0/128 |
| Mean change in Yes-minus-No log odds | +0.0078125 nats |
| 95% prompt-bootstrap interval | [-0.0004883, +0.0161133] |
| Conditional Yes probability change | +0.0018964 |

Every final answer was Yes. The final validation log-odds shift was exactly zero. The test shift's interval includes zero, and this run had no pretraining test measurement. A fresh process with a separate compiler cache produced byte-identical final prediction records. Reproducibility does not establish discrimination.

Before optimization, capture changed 89.73% of measured training-forward logits (maximum absolute difference 0.4353). This establishes a numerical footprint in that path, not a learnable condition signal across prompts. The run peaked at 12.62 GB allocated GPU memory. Updates took about 0.17 seconds each after compilation; environment setup dominated wall time.

## Artifacts and provenance

- `data.json`: the synthetic prompts and splits used by this run.
- `test_160_inference.json`: original final test predictions and original summary.
- `validation_0_inference.json`: initial validation subset (32 prompts).
- `validation_160_inference.json`: final validation (128 prompts).
- `analysis.json`: subsequent continuous-score analysis, using 5,000 prompt-pair bootstrap draws and seed 42.
- `original_source.zip`: the standalone training script actually executed, before the package refactor. SHA-256 of its `train.py`: `3f33559958e80082d1ced84c84d7816e50f6a1edf54b4e927d739e6a4325f3fc`.

Original rows store `yes_minus_no` (equal to the log-probability difference), conditional `yes_probability`, and target cross-entropy. They do not contain the explicit individual answer log probabilities now recorded by the package. Original evaluation always executed the absent condition first; the package now alternates evaluation order. Training order was randomized in both implementations.

Intervals describe variability across these synthetic prompt pairs within one run. They do not include seed, hardware, or compiler variability. The optional paired objective in the current package was not used in this pilot.
