# 023–024: diverse prompts and full-vocabulary answer fitting

023 detects a read-only query-norm probe; 024 detects gradient recording versus
no-grad execution. Both use native untied Qwen3-8B, fixed internal weights, batch
one, length 64, three ridge corrections, and penalty 0.001. Each first fits the
existing Yes/No rows, then cat/owl rows. Cat labels the present condition and owl
the absent condition. No condition label appears in the model input.

The previous animal fits improved cat/owl log odds but usually generated another
token. Their loss preserved the pair's common logit. This protocol instead fits
both logits to a combined answer probability of 0.95 and scores of -3/+3 (paired
gap 6). An exact fit gives correct-token probability 0.905. Targets are fixed
before this run; held-out results will not select a checkpoint or a new margin.

While fitting animals, preserve their original logits on detector training
prompts using equally weighted least-squares anchors. These are approximate
constraints: report native logit drift and reevaluate detector accuracy with all
four rows installed. Validation and test prompts never enter either solve.

The dataset has 448 explicitly written contexts: 256 training, 64 validation,
64 familiar-family test, and 64 contexts from four held-out families. Another
test reuses the 64 familiar test contexts with held-out question wording. It is
a paired wording test, not 64 additional independent contexts. Animal question
order is balanced within each split. Tokenization must fit without truncation.

The single-prompt integration check runs first and its edit is discarded before
the many-prompt fit. Stream training and validation metrics at each correction;
evaluate test splits only at baseline and the fixed final correction. Then
reload both artifacts in fresh processes with empty compiler caches, checking
animal choices and detector answers separately. All vocabulary-logit extraction
parity and frozen-feature checks remain mandatory.

Report actual top-token correctness, correct-token log probability, combined
answer mass, binary loss, and paired separation. Use whole-prompt paired
bootstrap intervals; family intervals have only four OOD units and are
exploratory. These intervals exclude fit, hardware, and compilation variability.
The dataset and objective both change, so improvement would establish this
combined recipe, not isolate which change caused it. Cat/owl choices do not
establish broad conversational preferences.

Run the two conditions in parallel on separate 48 GB GPUs. Each run is capped at
45 minutes and $1.20/hour ($0.90 maximum), with low-utilization shutdown disabled
during compilation and the wall cap retained. Publish the committed source and
compact results, including failures.
