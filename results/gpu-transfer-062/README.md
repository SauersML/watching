# The 034 probe detector on seven GPU models

The released 034 detector ([release `sft-034`](https://github.com/SauersML/watching/releases/tag/sft-034), `native_tail.pt`, SHA-256 `8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a`, checked on every pod) was trained on an NVIDIA L40S. Here it runs unchanged, with no training, in a fresh process on seven GPU models. Every run uses the same command as the original fresh-process check ([`confirm-validation-038/fresh034`](../confirm-validation-038/fresh034/)): `python -m watching.validate_tail` on the 64 newly written test prompts of `data/validation_v2` (data SHA-256 `8ae5cc96…`, identical `data.json` on every GPU), the whole model compiled as one graph, with the layer-14 probe on and off, BF16, PyTorch 2.10.0, Transformers 5.18.0, Triton 3.6.0. Source: repository commit `0b3e7445b226`. Each run also scores the original Qwen3-8B.

| GPU | Chip | Driver | Correct answers | Both answers correct | Yes without / with probe | Rise (nats) [95%] | Rise > 0 | Original model rise |
|---|---|---|---:|---:|---:|---:|---:|---:|
| L40S | AD102 | 595.91.07 | 128/128 (100.0%) | 64/64 | 0 / 64 | +27.25 [+27.04, +27.47] | 64/64 | −0.053 |
| GeForce RTX 4090 | AD102 | 570.195.03 | 99/128 (77.3%) | 35/64 | 0 / 35 | +11.61 [+11.03, +12.19] | 64/64 | +0.012 |
| L4 | AD104 | 580.126.09 | 85/128 (66.4%) | 21/64 | 0 / 21 | +11.80 [+11.15, +12.45] | 64/64 | +0.045 |
| RTX A6000 | GA102 | 580.159.03 | 79/128 (61.7%) | 15/64 | 7 / 22 | +2.47 [+2.31, +2.64] | 64/64 | +0.010 |
| RTX 6000 Ada Generation | AD102 | 595.91.07 | 66/128 (51.6%) | 2/64 | 0 / 2 | +6.25 [+5.88, +6.62] | 64/64 | +0.057 |
| H100 PCIe | GH100 | 580.159.04 | 20/128 (15.6%) | 0/64 | 60 / 16 | −7.32 [−7.47, −7.15] | 0/64 | +0.027 |
| A100-SXM4-80GB | GA100 | 570.195.03 | 5/128 (3.9%) | 0/64 | 61 / 2 | −12.12 [−12.33, −11.92] | 0/64 | −0.027 |

Rise: Yes − No log-odds with the probe minus without, averaged over prompts. Intervals: 5,000 bootstrap resamples of whole prompts. Answers are the top token over the full vocabulary.

- **L40S:** a different L40S host from the one used for training reproduces the original fresh-process result (128/128, 27.28 nats).
- **RTX 4090, L4, RTX A6000 and RTX 6000 Ada:** the probe raises the Yes − No log-odds on every prompt, by less, so many probed answers stay No. The RTX 6000 Ada has the same chip and driver as this L40S.
- **H100 and A100:** the probe lowers the log-odds of Yes on every prompt. Without the probe the detector answers Yes on 60 (H100) and 61 (A100) of 64 prompts.
- Every run repeats the first prompt in the same process and gets bit-identical logits (`*_repeat.json`).

Which compiled kernels differ between these GPUs, and how, was not measured.

## Files

- `gpus/<gpu>/fresh034/`: the complete `validate_tail` output: per-prompt rows for the detector (`trained_*`) and the original model (`base_*`), the buffer-copy arm, repeat checks, manifest, metrics.
- `gpus/<gpu>/gpu.txt`: `nvidia-smi` name, driver and memory; `RUN_NAME` and `rp-run-tail.log`: the pod that ran it.
- `summarize.py` → `summary.json`: the table above.
- `source/run.sh`: the pod command; `source/launch*.py`: the launchers, as run (runs `062`, `062r`, `062s`, `062t`).
- `cost.tsv`: every pod started, including pods discarded before running. Total $0.82. An H100 SXM and a second RTX 6000 Ada were requested (`062t`) but none were available.
