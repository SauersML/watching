# Real-input compiler mechanism audit

This is a bounded mechanism experiment, not additional detector training or a search for favorable prompts. The primary comparison is **reference capture versus no capture**, the actual condition used to train the detector. Shaped buffer-copy capture is secondary; the historical flattened-copy style is optional.

## Locked initial scope

- Original Qwen3-8B BF16 weights at the explicitly supplied revision.
- Query normalization in block 14; all 36 blocks remain native and unmodified.
- Model in evaluation mode, `use_cache=False`, parameter `requires_grad=True`, every forward under `torch.no_grad()`.
- Singleton left-padded length 96; explicit position IDs; same input tokens in every condition.
- First four detection records in the supplied **training** split, selected before observing outputs. No test records or outcome-driven prompt selection.
- New matched compilation. Previous run 038's exact winning kernel configuration was not recovered; do not claim this reruns that selection.

```sh
PYTHONPATH=src python -m watching.mechanism_trace \
  --checkpoint Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --data data/validation_v2/dataset.json \
  --out /workspace/mechanism-real-input \
  --prompts 4 --probe-layer 14 --length 96
```

Add `--include-flat` only for the matched historical-style buffer comparison. This adds the flat/sliced oversized buffer wrapper but preserves model, weights, prompts, shape, flags, and device. Its unprobed outputs must equal the shaped wrapper's unprobed outputs exactly, or the style contrast is marked confounded and execution stops.

## Observe existing launches, without retracing the model

After ordinary compilation and warmup, `GeneratedCapture` uses Python line tracing **only inside the already generated Inductor `Runner.call` function**. It copies tensors that already exist at CUDA launch boundaries. It does not introduce a Dynamo hook, graph output, custom CUDA kernel, or new materialization inside a fused kernel.

For every block it observes:

1. The input to the query projection, explicitly labeled as such. It can be a normalized residual, not the raw residual stream.
2. The query projection output.
3. The actual query, key, and value tensors passed into attention.
4. The attention output.

For block 14 it additionally captures the physical inputs and existing outputs of the query-normalization/rotary pipeline and reads the selected launch configuration and PTX from the warmed kernel object. The parser follows generated aliases. In archived 038 code, the no-capture/buffer pipelines each contain one fused kernel; the reference pipeline contains two kernels with an intervening BF16 tensor.

No internal norm statistics or raw residual are invented when the compiler never stores them. The earliest differing saved boundary is reported as **first unequal observed boundary**, not automatically the first arithmetic instruction that differs.

The parser is deliberately narrow. It requires the expected seven-linear-per-block Qwen3 annotations and explicit efficient-SDPA calls, uniquely maps all 36 blocks, and rejects unsupported wrappers. This is preferable to guessing buffer identities.

### Mandatory passive-observation gates

For each prompt and mode:

- Two untraced forwards must be bit-identical across the full answer-position vocabulary.
- The traced forward and a subsequent untraced forward must match that baseline exactly.
- Exactly one complete generated wrapper must be observed; every required site must fire once.
- The selected kernel pipeline's final output must equal the attention query tensor.

A failure invalidates the trace's interpretation. Do not relax these gates to a numerical tolerance.

## Replay actual inputs through the actual selected kernels

The audit first replays each pipeline on its own captured query-projection values, norm weights, and rotary angles. It invokes the existing selected kernel objects, preserving their configurations and intermediate materialization. Every already-materialized intermediate and final rotary output must match the captured execution exactly.

Then it runs both pipelines on the **same physical inputs**, using each condition's inputs in turn. This separates differences in the pipeline's arithmetic from differences already present in the projection or rotary inputs. It records both sources rather than silently assuming the inputs match.

This is stronger than synthetic replay: it uses real prompt activations and the actual selected configurations. It still does not isolate every individual arithmetic rewrite from every other rewrite. The selected sources/PTX and intermediate tensors permit a further targeted arithmetic counterfactual if required.

## Test whether the query difference explains the final output difference

At the already-materialized attention-query boundary, perform explicit bidirectional replacements:

- Put the reference-condition query into the no-capture execution.
- Put the no-capture query into the reference execution.

Use the same procedure for the buffer comparison. These are **causal interventions**, not passive probes. They are implemented by a copy at an existing Python launch boundary without changing the compiled graph or kernel definitions.

Before a cross-condition replacement, copying each condition's own query back into itself must preserve every vocabulary logit. After intervention, the next ordinary execution must return to its original logits. Report exact equality and numerical differences from the donor condition's full logits.

If both replacements reproduce the donor condition's entire output tensor, and the earlier captured boundaries/other attention inputs agree, the query difference is sufficient to account for the output difference in these cases. If not, inspect the earliest other differing boundary. Do not assert a complete explanation from a partial reduction in discrepancy.

## A compiler-informed candidate negative control

A useful candidate is a **forced materialization barrier installed identically in both labeled arms** immediately after query normalization. A semantically identity, compiler-opaque operation would return the same BF16 activation values while ensuring the downstream computation receives a materialized BF16 tensor whether or not a reference is retained.

The intended matched comparison is:

- Barrier present, reference absent.
- Same barrier present, reference present.

The barrier must be present in both arms; comparing a barrier in one arm with an unmodified graph is not a negative control. Historical CPU/vLLM identity-barrier results motivate this candidate but do not establish it for current 8B execution.

Before labeling it a no-footprint condition:

1. Verify the barrier is identity in ordinary execution and the reference really captures the tensor.
2. Require exact full-vocabulary equality **and bit-identical complete frozen-source inputs to the learner** between the arms: all token hidden states at the learner boundary, masks, positions and rotary inputs, across the fixed development cases and intended serving shapes. Use the same verified passive boundary capture in both arms. Equal logits alone are insufficient: hidden differences could cancel under the current head and become learnable later. A boundary mismatch is a failed no-signal control, even if current logits match.
3. Compare the actual selected query pipelines and intermediate tensors; avoid claiming equality solely from source-level identity.
4. Repeat after fresh compilation and after each relevant training checkpoint. Updating the suffix or compiler configuration can change equivalence.
5. Only after those gates pass, run the otherwise matched training schedule with contradictory condition labels.

The existing identical-state training control remains useful for plumbing and optimization. A verified barrier-plus-reference comparison would address the actual capture implementation more directly. Neither a query-swap intervention nor the currently non-passive buffer copy substitutes for this matched null.

## Output and resource limits

The command streams `capture_verified` and `case_complete` events immediately. It saves full per-boundary hashes/differences, selected configurations/PTX, physical kernel inputs/outputs, and selected real tensors around the first observed difference. It retains no gradient or optimizer state and trains no weights. CPU copies can synchronize launches; the all-logit gates check their numerical passivity, not their timing equivalence.

Run on one existing or separately authorized GPU with enough memory for one base 8B model and its compiled buffers. Use a hard runtime cap externally. No automatic larger-model/prompt/configuration sweep is part of this protocol.
