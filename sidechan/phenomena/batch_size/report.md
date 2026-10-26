# Batch size — report

*reduction-order change (dies under weight updates)*

Whether the request shares its forward pass with other requests in a batch.

## Finding

- Learnable: SFT reaches 90% both-right on 8,192 prompts; RL from reward alone reaches 98% (8B) and 100% (27B).
- Fragile: every batch-size detector drops to 0% after flipping the lowest bit of 0.01% of the upstream weights.
- Does not form under full-network RL (LoRA on every layer stays at chance).

## Numbers

```json
{
 "sft_both_right": 0.8984375,
 "rl_tail_both_right": 0.984375,
 "after_lsb_flip": 0.0
}
```

## Figures

![fig_rl](../../paper/figures/fig_rl.png)
![E01_hook](../../paper/explainer/png/E01_hook.png)
