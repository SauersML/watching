# Independent review of fresh 034 evaluation

This audit independently reads the saved predictions, metrics, input records,
archive hashes, and evaluator source at commit
`b699c4ba462dcac9291f8d5e72ced4c088d5962c`. It does not rerun CUDA inference.
The fresh-evaluation stage completed; this report makes no claim about the joint
training process that follows it in the parent job.

## Reference-probe result checks out

All **64 predeclared detection prompts** from the locked version-2 test set were
evaluated. The result is **128/128 correct individual next tokens**, not 128
independent prompts. Recomputing from raw answer log probabilities and saved
greedy token IDs gives:

| Quantity | Unmodified base | Released 034 weights |
|---|---:|---:|
| Correct individual answers | 64/128 | 128/128 |
| Probe-minus-none log-odds gap | 0.01171875 nats | 27.28125 nats |
| Full-vocabulary correct-token log loss | 7.63424118 nats | 0.00000235809634 nats |
| Mean no-probe Yes/No log odds | −15.275390625 | −14.21875 |
| Mean reference-probe Yes/No log odds | −15.263671875 | 13.0625 |

Every trained correct-token probability is at least **0.9999724634**. Every
prompt has a positive paired gap, ranging from 25.5 to 29.25 nats. Correctness
matches the actual token IDs (No=2753, Yes=9454); CE matches the corresponding
full-vocabulary log probability. All four observed question families have 100%
actual-token accuracy. Their mean gaps range from approximately 26.58 to 28.17
nats. This is strong observed discrimination, not merely a positive average gap
while both conditions favor the same answer.

## Checks against contamination and execution artifacts

- The dataset hash matches the locked protocol. The selected prompts are exactly
  all 64 detection records in its final test split, without outcome-based filtering.
- I found no normalized exact context-text overlap with any of 77 earlier result
  data files containing context strings, and no context-ID, normalized full-prompt,
  or normalized context overlap with any selected 034 cohort. This does not rule
  out semantic similarity. Several general formats, including XML and footnotes,
  appeared in earlier experiments; the claim is fresh concrete instances, not
  wholly unprecedented communication formats.
- The evaluator performs no fitting. It loads the released artifact with the
  expected SHA-256 `8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a`,
  validates native tensor shapes/dtypes/finite values, and installs blocks 34–35
  and normalization. The output-head digest agrees with the original base.
- This is a new process with compiler disk caches disabled and compiler weight
  freezing disabled. Parameters retain `requires_grad=True`; all scored forwards
  use `no_grad`. Within that process, graphs first execute the base model, then
  the trained native weights are installed into those graphs. It is therefore a
  fresh-process evaluation, not a separately compiled trained-weight graph in a
  second fresh process.
- All six permutations of none/reference/buffer execution order occur. Each
  stage's first prompt is rerun after all interleaved evaluations; its full logits
  are bit-identical in all three modes. This checks repeatability at that anchor,
  not every prompt or arbitrary scheduling changes.
- Every archived compact-file hash matches its recorded digest. The evaluator's
  `--exclude-data` option was not supplied, so runtime exclusion metadata is empty;
  the independent overlap checks above establish the exact-match exclusion claim.

No scoring error or concrete data contamination was found. This is one saved
training run evaluated on new prompts; it is **not another independent training
seed**. The locked later models share this test set, so their results must not be
counted as independent prompt populations or used to adapt the already fixed recipe.

## The buffer probe is not a valid no-footprint control here

The buffer-copy graph differs from the unprobed graph on **all 64 prompts**, both
before and after installing trained weights. Maximum full-vocabulary logit
differences are **2.0** for the base and **0.8125** for the trained model. Thus the
required numerical-parity gate fails and the saved
`valid_no_footprint_control: false` is correct.

The trained model nevertheless answers No for the buffer execution and No for
the unprobed execution on every prompt. The buffer-minus-none mean log-odds gap
is −0.00390625 nats; balanced correctness is 64/128 under the convention that
the buffer condition receives the Yes label. This shows that this alternative
numerical perturbation does not trigger the learned reference-probe behavior.
It does **not** establish numerically passive capture, certify buffer-copy probes
as safe, or supply a successful no-footprint negative control. The separately
trained identical-input control remains necessary for its distinct plumbing test.

## Revision metadata discrepancy

The nested provenance field `load_revision_explicitly_pinned: false` is a reporting
bug. The saved top-level revision is
`b968826d9c46dd6066d109eabc6255188de91218`. In the executed source snapshot, both
tokenizer and model `from_pretrained` calls explicitly receive that revision.
The shared metadata helper instead hardcoded `false` and looked up `main` when
the configuration's `_commit_hash` was absent. Its null configuration-path/hash
fields do not demonstrate an unpinned load.

The source, requested revision, release-artifact hash and original-head hash are
consistent. This audit does not independently checksum every loaded base-model
shard. Raw manifests remain unchanged; the helper and caller are corrected only
for future runs, with focused tests covering an explicit revision and missing
configuration commit.
