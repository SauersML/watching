# Source geometry in the viewing experiments

Existing cached inputs to the last two native blocks, before their training. CPU analysis only, four threads, train/development records only. Each run uses the same 256 training and 64 development prompts as experiment 034; independently compiled source instances can differ numerically.

| Source intervention | First role-user token changes | Answer-position changes, training | Answer-position changes, development |
|---|---:|---:|---:|
| Buffer copy, 051 | 0/320 | 95/256 | 17/64 |
| Opaque identity materialization, 052 | 320/320 | 256/256 | 64/64 |

For 052 the role-user activation is exactly constant across all 320 prompts within each condition. Its difference vector is therefore also constant, with L2 norm **5.4125699935**. For 051 that role vector is identical between conditions. The initial `<|im_start|>` token has zero difference in both runs.

The checked token prefix is `<|im_start|> user newline`, with IDs 151644, 872, 198. The role has logical position 1, despite physical positions 37–60 from left padding. The answer position is 95. All condition-paired masks and rotary inputs match. Full cached tensor fingerprints and position fingerprints match the source artifacts.

This identifies a missing shared role-token cue in 051, at the location implicated by the earlier 034 causal audit. It does **not** show that 051's remaining numerical differences are unlearnable, prove the precise compiler cause, or establish that these cross-run vectors equal 034's source vectors. No learned model was run or changed.

`051.json` and `052.json` contain per-prompt geometry, selected-vector hashes, complete source-cache hashes, and metadata hashes. The source cache is memory-mapped through a temporary hard link that pins its inode during analysis; a size/mtime/inode check rejects in-place mutation. This accommodates later atomic replacement when final test states are added. Only training/development states are inspected, even if the cache contains other splits.

Reproduce from the matching source artifacts:

```bash
PYTHONPATH=src python scripts/analyze_viewing_geometry.py PATH_TO_051_FIT /tmp/051.json
PYTHONPATH=src python scripts/analyze_viewing_geometry.py PATH_TO_052_FIT /tmp/052.json
```

This is descriptive artifact analysis, without a population confidence interval or a new causal intervention.
