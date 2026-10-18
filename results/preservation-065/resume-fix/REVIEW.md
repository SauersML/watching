# Future resume dependency patch — staging only

The executed source-r1 and active 065 job are unchanged. These files are a proposed
future package, not a repair applied to the running experiment. Before using them,
root must review and regenerate the immutable source manifest/package to include
the new helper and changed scripts.

## Failure and behavior

The old stage marker was sufficient to skip export/evaluation even after a valid
training resume changed the selected tail. The new helper pins content dependencies
and the actual command arguments (including sampling/pass settings):

- export: selected tail, exporter code, pinned base configuration;
- each runtime endpoint: export manifest (containing the actual selected-tail hash),
  endpoint code, its framework helper and its particular input bank;
- ordinary evaluation: export manifest, evaluator and all three diagnostic banks.

If selected weights do not change, export and completed evaluations remain valid.
A changed endpoint script only invalidates that endpoint. Capture/training stages
are untouched. Old unpinned downstream markers are conservatively considered
unverified once; their outputs are preserved before replacement. Stale endpoint
outputs remain under the run's stale-stages tree; external full-model exports stay
outside the synchronized run tree. Commit rechecks dependency bytes after the
command and refuses completion if they changed or an output is missing.

The fixed base model is an immutable precondition verified by verify_source.py;
this patch does not authorize changing its shards. It does not evaluate or tune
anything against fresh-test scores. If a partial run was already evaluated,
continuation reuses an already inspected fresh bank and must not be described as
new untouched confirmation.

## Exact proposed README limits

> The training diagnostic is the fixed first 32 content-bank records
> (`content-00000` through `content-00031`), selected by sorting training IDs.
> It does not estimate performance over all 13,256 training prompts. These records
> have no question-family metadata, so their intervals resample complete prompt
> pairs; development intervals use family clusters. Development and final
> whole-runtime evaluations remain separate from this diagnostic.

> The initial eager-tail check compares the Yes-minus-No gap on eight development
> prompts. It is not a full-vocabulary tail/capture parity check. The source
> capture checks separately compare full vocabulary scores on four diagnostic
> prompts per source capture. No full-vocabulary equality to the vLLM tail is
> assumed; actual whole-runtime endpoint scores are measured separately.

## Additional review outcome

No duplicate tokenized prompts were found across detector training/development,
ordinary training/development and the locked fresh bank. The loss normalization
matches the selection objective: paired detector CE plus the mean weighted reply
KL over the two source runtimes. Source-state and whole-model results are
explicitly separated by the current public notes. The preservation teacher on
vLLM states is the frozen original eager HF tail, a hybrid reference; it is not a
saved original whole-vLLM reply distribution.
