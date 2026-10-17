# Native-tail causal diagnostic of run 034

Status: executable prepared for review, not launched. This complements the
compiler/kernel investigation: it asks how the already trained final two blocks
use the numerical difference, not how that difference was originally created.

## Runtime and provenance

Use the complete 034 source-results directory, including `cache_states.pt`,
`cache_fingerprints.json`, `native_tail.pt`, `manifest.json`, `data.json` and archived
split-inference scores. Verify every cached state's fingerprint, labels' identical
mask/positions/rotary inputs, exact output-head hash, artifact compatibility and
cache boundary. Select eight training and eight previously evaluated test prompts
using seed 20261020, without reading scores for selection. These are post-hoc
diagnostic examples, not a new confirmation set.

Load only the original native last two blocks, norm and unembedding through indexed
safetensors. There is no full-model CPU/GPU allocation, embedding layer, new head,
training or optimizer. Original and learned weights run sequentially in the same
eager BF16 math-SDPA tail. The head never changes. The original prefix remains the
verified cached computation; this does not constitute whole-model serving replay.

Each prompt/condition first runs without hooks, then with hooks that copy actual
eager module outputs and return `None`. Every vocabulary logit must remain exactly
equal. Archived answer logits/logprobs must also match exactly wherever an archived
split-tail score exists; missing original-test baselines are explicitly recorded.
These are separate checks: archived files contain only answer scores, while the
new hook-parity gate covers the entire vocabulary.

## Measurements

At the input to block 34, output of block 34, output of block 35, and final norm:

- Valid-token and last-token difference norms, midpoint norms, difference/midpoint
  ratio, changed-element fraction and per-token difference norm.
- Last-token projection onto the original Yes-minus-No unembedding direction.
  This FP32 descriptive projection is distinguished from actual BF16 native logits.
- Native Yes/No logprobs, paired log-odds and full-vocabulary responses. Save the
  final-token state vectors so the descriptive calculations can be reproduced.

Compare the original and learned tails on precisely the same numerical inputs.
Growth of a raw norm alone is not evidence of improved detection; its direction,
native answer effect and causal interventions are also measured.

## Causal interventions

1. At every recorded site, replace the absent state by its paired present state
   and vice versa, retaining identical positional/attention inputs. Full-vocabulary
   outputs must exactly recover the corresponding other-condition outputs.
2. Interpolate the measured within-prompt difference with coefficients
   `[-1, 0, .25, .5, .75, 1, 2]`. Endpoints use exact saved BF16 tensors; interior
   and extrapolated values are formed in FP32 then cast to BF16. These artificial
   intermediate states are interventions, not naturally occurring probe conditions.
3. At the entrance and first-block output, patch only the final token or only
   preceding valid tokens. On the first selected prompt in each split, scan each
   valid token separately. This identifies coarse layer/token mediation, not a
   complete neuron-level circuit.
4. At the source boundary, transplant the next selected prompt's footprint,
   restricted to overlapping valid physical token positions and scaled to the
   recipient's valid-token footprint norm. Compare three fixed-seed Gaussian
   directions with the same FP32 norm. Report the realized BF16 perturbation norm
   and cosine to the true footprint for every intervention: casting can alter the
   intended norm. Preserve all masks, positions and rotary tensors.

Random/donor controls test whether the learned tail depends on a particular
direction. They are not negative controls for the source compiler explanation.
Interventions have response scores, not invented ground-truth accuracy labels.
Token patches show causal sufficiency in a fixed context; interactions mean that
single-token effects need not add up to the full effect.

## Resources and reporting

`bash scripts/run_tail_mechanism.sh OUTPUT` downloads the two public release
artifacts with fixed SHA-256 checks and copies the source metadata from the
repository; it never provisions a machine. Proposed external cap: one L40S,
30 minutes and $0.60 maximum, subject to parent review. The script has a
1,200-second internal diagnostic limit and writes per-prompt intermediate results.
Native GPU weights occupy roughly 2 GB plus activations and transient buffers;
cache/artifact files are loaded on CPU. Downloading model shards may dominate setup.

Output includes flushed `metrics.jsonl`, `scores.json`, compact final-token states,
artifact hashes, runtime provenance, parity results and complete/time-limited status.
Report all selected prompts, both stages, all interpolation coefficients and noise
seeds. No checkpoint search, outcome-based prompt exclusion, confidence intervals
over repeated deterministic forwards or new population-generalization claim.
