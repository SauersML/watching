# Fresh joint confirmation of fixed run 039

The checkpoint is the final scheduled 1,024-update artifact of the development-only
warm-start pilot, not its best intermediate checkpoint. Evaluate all 128 records
in `data/confirmation_v3/dataset.json`: 64 detection and 64 animal questions,
eight authored question structures per task. The bank was committed as af403b4
before final checkpoint availability and before any scores. This protocol is
committed before evaluation. No fitting, calibration or score-based exclusions.

The saved tail SHA-256 is
`2f1f92bcb78041b812760f5086a5436ef43732e983a19a4578e6e3a7a6c3892c`.
The data SHA-256 is
`2c0a6df12d15d2a9feaab2228d35246f820f010e82577150dc0f39edad2e8e66`.
Both are checked before evaluation. The Qwen3-8B revision remains
`b968826d9c46dd6066d109eabc6255188de91218`.

Use a new process with Inductor disk-cache reuse disabled, BF16, singleton
left-padded length 96, ordinary no-gradient inference and reference capture at
block 14 query normalization. Evaluate the original model, then install the saved
last two blocks and norm into the same graphs with compiler weight freezing off.
This is fresh-process compilation followed by weight replacement, not a separate
compilation with trained weights as constants. Check head/architecture/artifact
compatibility and first-prompt repeat equality. Both conditions see identical text.

For each task separately, report every prompt's native full-vocabulary token
choice, correct-token probability and cross-entropy, as well as conditional binary
loss and probe-minus-no-probe answer log-odds. Detection uses No/Yes; animal choice
uses cat/owl. Legacy `yes_minus_no` fields mean the specified positive-minus-negative
answer scores; explicit label IDs accompany every row. Never aggregate the two
tasks to conceal failure of either one.

Report 5,000 whole-prompt paired bootstrap intervals for the mean gap and actual
accuracy/probability/loss. These are approximate intervals conditional on one fit
and this finite authored bank, not seed/hardware uncertainty or a universal
population claim. Separately report the number of prompts correct in both
conditions; an all-correct bootstrap interval cannot establish perfect population
accuracy. No intermediate significance decisions or endpoint selection.

The buffer-copy comparison is an additional intervention. It qualifies as
numerically invisible only if every compared vocabulary logit matches the
unprobed graph. A failure is reported, not relabeled as a successful null.

Run at most 30 minutes and $0.60, with low-utilization termination disabled during
compilation and a fixed wall-time cap. Preserve raw intermediate predictions;
completed final results are primary. This confirmation tests fresh texts from the
specified authored families, not unrestricted out-of-distribution generalization.
