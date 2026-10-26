# Greedy decoding — report

*no learnable signal*

Whether the previous reply was decoded greedily (temperature 0).

## Finding

- Chance end to end (AUROC ~0.5).
- Probes find per-token 'was this my top choice' at AUROC 0.70-0.80, but the model never pools it.

## Numbers

```json
{
 "lora_auroc": "0.49-0.54",
 "per_token_surprise_auroc": "0.70-0.80 (never pooled)"
}
```
