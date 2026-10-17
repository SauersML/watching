# How layer 35, head 6 detects the probe

**Head 6 of layer 35 is an attention switch driven by the `user` token's key.** Without the probe, its query at the answer position attends to `</think>`, whose value writes "No". With the probe, it attends to `user`, whose value writes "Yes". The value and output paths do not differ between conditions.

The analysis runs the released detector 034 (blocks 34–35 and the final norm of Qwen3-8B, plus the cached block-34 inputs) on CPU. Numbers are BF16 on the eight training and eight test prompts of run 041r, given as sufficiency / necessity in nats of the native Yes−No log-odds gap: sufficiency patches the probed value into an unprobed run, necessity the reverse.

## Parity

The functional tail in `scripts/tail_lib.py` is bitwise equal to Hugging Face's `Qwen3DecoderLayer` on CPU. Against the archived CUDA log-odds it differs by 0.105 nats on average and 0.25 at most (18 of 32 exact); FP32 differs by at most 0.32.

## 1. Attention pattern, not value

Head 6's attention on `user`: 0.31% without the probe, 99.7% with it. The original, untrained head puts 0.56% on `user` in both conditions and 87% on `<|im_start|>`. The trained head's unprobed attention sits on `</think>` (54–87%).

| Swap at layer 35, head 6 | Train | Test |
|---|---|---|
| Head output at the answer position | 26.88 / 26.75 | 25.42 / 25.53 |
| Key at `user` only | 26.75 / 26.78 | 25.30 / 25.53 |
| Value at `user` only | 0.00 / 0.16 | 0.03 / 0.16 |
| Query at the answer position only | 0.00 / 0.09 | 0.06 / −0.03 |
| Attention row only | 26.75 / 26.78 | 25.33 / 25.50 |

Full gaps: 27.22 (train) and 25.86 (test). Patching the `user` input while holding head 6's key at `user` leaves 0.03 / 0.16; holding its value instead leaves 26.78 / 26.81. FP32 on four prompts agrees: the key gives 25.2–27.2 nats, the value at most 0.06.

## 2. The cancelled key

Head 6's attention score for `user` rises from 4.9 to 16.8, against 10.9 for all competing positions combined (log-sum-exp). Key–value head 1, which serves query head 6, has a pre-normalization key at `user` of length 6.6; the other key–value heads have 58–80, and the original model had 70. A 2.5% change in the residual stream therefore becomes a 64% change in this key, which Qwen3's per-head key normalization rescales to full size.

- **One direction.** Patching only the component of the residual difference along the gradient of that score reproduces 26.78 / 26.75 (train) and 25.36 / 25.48 (test), using 32% of the difference's length; removing that component leaves at most 0.19. A direction taken from one prompt works on all 16 (25.91 / 25.83 of 26.54): the `user` row is bitwise identical across prompts.
- **Dose.** Adding the real difference flips the answer at 0.5× its size and saturates at 0.75×.
- **Specificity.** Random perturbations of matched length leave the answer at "No" (attention on `user` at most 2%; at most 24% at 3× the length).
- **Readout.** With the probe, head 6 copies the `user` value almost exactly. Its direct contribution to the Yes−No logit goes from −6.2 to +13.6, matching the causal head-6 patch with block 35's MLP frozen (20.5 / 20.0). That MLP adds about 6.4 nats, spread over many neurons.

## 3. Block 34 amplifies the footprint

Passing the raw footprint straight to block 35 does nothing (0.28 / 0.25); patching only the `user` row after block 34 does everything (26.88 / 26.75). Block 34 grows the `user`-row difference from 5.49 to 17.8 (the original block 34 reaches 8.2): MLP 7.9, attention 2.7 (mostly head 25), residual passthrough 1.2.

## 4. Checks suggested by a second analysis

- **Hooks are inert.** Patching each of the 25 hook sites with its own value, or running an identity hook on every site, reproduces the logits bit for bit.
- **No relay through block 34's answer row.** Swapping block 35's input at the answer position moves the answer by at most 0.25 nats, and every row except `user` by at most 0.75; the `user` row alone gives 26.88 / 26.75 (train) and 25.48 / 25.56 (test).
- **A threshold, not a linear readout.** Adding α times the probe's difference to the unprobed `user` row: mean Yes−No log-odds stay at −13.1 for α ≤ 0, reach −5.8 at 0.4, +1.2 at 0.5, +7.3 at 0.6 and +12.8 at 0.8, then stay flat to α = 2. Head 6's attention on `user` goes 0.3% → 27% → 53% → 98% over the same range (`dose_response_fine_bf16`).
- **Random directions do not fire.** Random changes to the unprobed `user` row at 1×, 2× and 3× the probe's difference make the model answer Yes in 0 of 128, 0 of 64 and 0 of 64 draws; at 5× in 9 of 64. At block 35's input, 0 of 128 random draws fire, including 0 of 128 restricted to head 6's key subspace (`random_direction_control_bf16`).
- **Key dimensions.** The top 32 of head 6's 128 key dimensions carry 25.5 / 25.6 nats and the other 96 carry 0.6; low-frequency rotary bands dominate, matching query–key distances of 38–61 tokens.

## What training changed

With the trained block 34, the original key weights of key–value head 1 already give a `user` key of length 23.1 (70.3 with the original block 34). The trained change to those weights, 1.4% of their size, adds a vector of length 20.5 almost opposite to that key (cosine 0.96 with its negative), leaving 6.6; it barely changes how the probe's difference itself is read (3.7 → 4.2). Head 6 has the largest changes in block 35 (output–value 3.0% relative, against a 0.67% median); key–value head 1 has the largest key and value projection changes; head 25 has the largest in block 34. Norm weights are unchanged.

| Restore to original weights | Gap (nats) | Accuracy |
|---|---:|---:|
| Key projection of key–value head 1 | 0.41 | 50% |
| Block 34 | 0.08 | no flip |
| Head 6 query | 24.8 | — |
| Head 6 output–value | 8.5 | flip intact |
| Block 35 except head 6's slices | 20.2 | 100% |

## Generality

On 64 test and 33 validation prompt pairs, every answer is correct; attention on `user` is at most 1.24% without the probe and at least 97.2% with it, and the key at `user` alone accounts for at least 90.5% of every gap. All prompts share the same chat prefix.

## Reproduce

```sh
gh release download sft-034 --repo SauersML/watching --dir results/head6-circuit-cpu/dl
(cd results/head6-circuit-cpu/dl && shasum -a 256 -c SHA256SUMS)
# Original blocks 34–35 come from the pinned Qwen3-8B snapshot; set QWEN3_8B_SNAPSHOT if it is not in the default Hugging Face cache.
python results/head6-circuit-cpu/scripts/parity.py
python results/head6-circuit-cpu/scripts/decompose.py
python results/head6-circuit-cpu/scripts/assemble.py
```

Raw outputs are in `out/`; `figure_data.json` collects every number above. Hardware: Apple Silicon CPU, PyTorch 2.10.0, Transformers 5.18.0, eager math attention.
