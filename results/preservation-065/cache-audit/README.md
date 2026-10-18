# 065 cached boundary footprint: fixed eight-prompt slice

Read-only CPU analysis of the first eight `group=dev` records in stored order (`dev-000` through `dev-007`). Selection preceded tensor inspection. No model forward, model edit, GPU call, or large-file rehash.

Both cache schemas match: `attn` and `residual` are BF16 `[598067, 4096]`; `lengths` is int64 `[13420]`. All row IDs and lengths agree. Only the 338 selected token rows from each tensor were sliced, plus the small length arrays. `attn` is block 34 attention output before its output projection; `residual` is that block's residual input.

Relative difference means `||vLLM − HF||₂ / ||HF||₂` over all selected token coordinates for that prompt. Equality is exact numeric equality of saved BF16 elements. The two columns are different measurements: a high differing-element fraction can coexist with a small relative norm.

| Prompt | Tokens | Attention relative difference | Residual relative difference | Attention elements equal | Residual elements equal |
|---|---:|---:|---:|---:|---:|
| dev-000 | 46 | 2.625% | 0.374% | 6.34% | 8.65% |
| dev-001 | 35 | 2.812% | 0.331% | 6.18% | 8.54% |
| dev-002 | 41 | 2.811% | 0.352% | 6.42% | 9.15% |
| dev-003 | 40 | 2.724% | 0.363% | 6.39% | 8.89% |
| dev-004 | 48 | 2.989% | 0.435% | 5.96% | 7.98% |
| dev-005 | 44 | 2.928% | 0.433% | 6.22% | 8.37% |
| dev-006 | 41 | 2.927% | 0.378% | 6.26% | 8.99% |
| dev-007 | 43 | 2.824% | 0.397% | 6.22% | 8.65% |

Every selected prompt has a numerical difference at both saved inputs. None of the selected token vectors is exactly equal across runtimes. This describes an available footprint, **not which part the trained decoder uses or why it arose**. Full per-token norms, equality fractions, and minimum/median/maximum relative differences are retained in `report.json`. No confidence interval is assigned to this fixed diagnostic subset.

Across the 338 selected token positions, attn relative differences range from 0.699% to 6.567%. These token positions are correlated, not independent observations.

Across the 338 selected token positions, residual relative differences range from 0.053% to 6.588%. These token positions are correlated, not independent observations.

## Reproduce

Requires PyTorch and safetensors. The script uses mmap slices and reads the small length tensor; it never calls `get_tensor` on the full attention/residual arrays.

```bash
python audit_boundary.py --run-output /path/to/watching-preservation-065/out --output report.json
```

Recorded small metadata hashes are embedded in the report. Large cache identity is inherited from the separately documented run contract; it was not freshly verified here. This analysis is conditional on these saved artifacts and cannot establish capture neutrality on every prompt, upstream origin, causal mediation, a head identity, or out-of-distribution generalization.

## Small independent reproduction archive

`selected-states.safetensors` retains only the eight selected pairs (338 token rows), with BF16 attention and residual tensors for both runtimes. It is about 11.1 MB, compared with roughly 19.6 GB for the two full source caches. IDs, original row indices and lengths are embedded as archive metadata.

```bash
python reproduce_subset.py --subset selected-states.safetensors --report report.json
python audit_metadata.py --hf-rows hf_rows.jsonl --vllm-rows vllm_rows.jsonl --out metadata-report.json
```

The first command reproduced every saved numeric statistic exactly locally. The second checks all 13,420 metadata pairs independently; both original row files are included in this directory. `manifest.json` records the hashes of this small package. The reports contain portable filenames and hashes, not private absolute paths.

[Download the 11 MB tensor subset](https://github.com/SauersML/watching/releases/download/preservation-065/selected-states.safetensors). Its SHA256 is `4c5052304f4b2ff54938cbd12dcf3e071a9f040c93e9735a5cc0e1f496bb4467`. The per-token median uses PyTorch's lower middle value for even-length sequences.
