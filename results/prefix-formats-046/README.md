# The system-prefix footprint is present

A read-only analysis of run 046's verified **512 training and 64 development cached
prompt pairs** finds substantial probe-induced differences with and without
system messages. The near-zero output separation on system-prefix prompts
cannot be explained simply by the absence of a source numerical footprint.
No model was executed or trained, and no test-cohort states were analyzed.

At the entrance to block 34, the first role token is at logical position 1:

| First role | Prompt pairs | Probe-minus-no-probe state norm | Within-condition state consistency |
|---|---:|---:|---|
| `user` | 288 | 5.487421912 | Exactly identical across all 288 prompts |
| `system` | 288 | 6.043237075 | Exactly identical across all 288 prompts |

Both absent states and both present states are constant within their respective
role format, so their difference vectors are also exactly constant. The preceding
`<\|im_start\|>` state is identical across all 576 prompts and both conditions:
its numerical difference is zero.

**The direction changes:** the cosine between the two role-token difference
vectors is 0.0345315, close to orthogonal. Their common-state midpoints are 459.645
apart in Euclidean norm. A hypothetical linear reader aligned with the `user`
difference would see only 0.208682 along that direction from the `system`
difference, about 3.80% of its response to the `user` difference. This is geometry,
not a measurement of the trained reader's weights or behavior.

The actual `user` token following a system message is no longer the shared
constant reference seen at the beginning of ordinary prompts. Across each 144
prompt cohort, the first actual `user` token has:

| Format | Distinct absent states | Distinct present states |
|---|---:|---:|
| Single user | 1 | 1 |
| Multiple turns, no system | 1 | 1 |
| System + user | 36 | 34 |
| System + multiple turns | 65 | 63 |

These counts concern complete saved vectors, not just rounded summary norms.
The first actual `user` token moves from logical position 1 to positions 9–12.
A preceding system message changes its causal context as well.
Different system texts are present in the dataset; uniqueness counts alone do
not isolate text effects from numerical implementation effects.

Differences also remain across the active sequence and at the answer position:

| Format, 144 pairs each | Median active-sequence difference norm | Median RMS coordinate difference | Median answer-position difference norm |
|---|---:|---:|---:|
| Single user | 48.4824 | 0.11145 | 6.8592 |
| Multiple turns, no system | 64.1650 | 0.12693 | 6.6669 |
| System + user | 57.8044 | 0.12033 | 6.9286 |
| System + multiple turns | 72.8422 | 0.13398 | 6.5297 |

Padding tokens are excluded; RMS values help distinguish magnitude from sequence
length. Mask and rotary inputs are exactly equal between conditions for every
pair. The six earliest active tokens, per-prompt norms, state hashes and user-role
occurrences are retained in [analysis.json](analysis.json).

**Interpretation:** the results support a prefix-specific readout hypothesis:
the established detector may depend on the early `user` reference and fail to
use a different, still-present `system` signature. They do not establish
unlearnability, a causal route, or successful detection of this alternative
signature. A causal follow-up could separately exchange the first-role common
state and condition-difference direction within cached inputs, with unchanged
tokens/masks and matched-direction controls. No such intervention was performed
in this analysis.

All 576 saved pairs passed tensor shape/stride/dtype/SHA checks and positional
fingerprint checks. This inspected cache snapshot contains only train/development
cohorts: 976,875,061 bytes; SHA-256
`1139fef38b53bf236d2e3b0346a1fbb4dc90cfd9366d7367bdf3afee32c92a04`.
The complete snapshot is preserved locally at
`/Users/user/mpd-data/scratch/watching-prefix-formats-046/`; it is distinct from
any later cache expanded with test examples. Model revision:
`b968826d9c46dd6066d109eabc6255188de91218`.

Reproduce from that snapshot, or another verified copy of its files:

```sh
PYTHONPATH=src python scripts/analyze_prefix_formats.py \
  --source /path/to/snapshot \
  --expected-cache-sha256 1139fef38b53bf236d2e3b0346a1fbb4dc90cfd9366d7367bdf3afee32c92a04 \
  --out results/prefix-formats-046/analysis.json
```

These are finite-cache descriptive quantities. There is no sampling confidence
interval, additional training seed, or population generalization claim.
