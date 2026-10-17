# NNsight activation saving: CPU eager compatibility audit

**No numerical footprint was found in this audit.** On four diagnostic prompts,
NNsight's actual `.trace()` / `.save()` API saved query-normalization and residual
activations without changing any vocabulary logit at any token position.
This is a bounded capture test, **not a trained-detector result**.

| Setting | Value |
|---|---|
| Model | Original Qwen3-0.6B, all 28 blocks, BF16 |
| Revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Library | NNsight 0.8.0, `TransformersModel` |
| Stack | PyTorch 2.11.0, Transformers 5.17.0, macOS ARM CPU |
| Execution | Eager, `model.eval()`, `torch.no_grad()`, SDPA, batch size 1 |
| Sites | Block 14 query-normalization output; block 14 residual output |
| Data | Four custom diagnostic prompts and exact token IDs in `prompts.json` |

The seven calls for each prompt were bare Hugging Face before wrapping, a library
trace saving only logits, a trace saving query normalization plus logits, a trace
saving residual plus logits, the query trace repeated, the logits-only trace
repeated, and bare Hugging Face after tracing. Logits are saved identically in
every library trace: “logits-only” means no **internal activation** was saved.
The two capture sites were tested in separate traces.

All seven outputs were byte-identical, including every vocabulary logit at every
input position. Last-position Yes/No log probabilities and log odds were also
identical. Both captures were nonempty, finite tensors with the expected batch
and sequence dimensions. Their hashes differed across all four inputs; repeated
query captures on the same input were identical. Full model tensor hashes,
training flags, gradient flags and CPU placement were unchanged.

The standard API used was:

```python
with model.trace(**inputs, use_cache=False):
    activation = model.model.layers[14].self_attn.q_norm.output.save()
    logits = model.output.logits.save()
```

The residual arm substitutes `model.model.layers[14].output.save()`. This installed
Transformers version returns a tensor at the decoder output, so indexing `[0]`
would incorrectly discard the batch dimension. No hook writes or replacement
activation is introduced. No extra compiled-output wrapper is used.

## Interpretation and limitations

This supports numerical passivity for these **ordinary eager NNsight captures in
this configuration**. It does not establish an invisible activation reader on
arbitrary inputs, compiled execution, CUDA, vLLM, or the trained 8B detector.
Equal final logits also do not prove equality of every unobserved intermediate.
There is no training curve or population confidence interval: these are exact
comparisons on four fixed diagnostic inputs.

The API came from NNsight's [quick start](https://nnsight.net/getting-started/quickstart/)
and [activation-saving documentation](https://nnsight.net/features/1_getting/).
`LanguageModel` remains an alias but is deprecated in the installed 0.8.0 version;
the documented `TransformersModel` interface was used. The library's vLLM capture
path has different execution and version requirements and was not tested here.

## Reproduction and audit trail

Run [the audit script](../../../scripts/audit_nnsight_capture.py) with an isolated
environment providing the exact versions in [summary.json](summary.json):

```sh
python scripts/audit_nnsight_capture.py \
  --model "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca" \
  --out results/library-capture/nnsight
```

The same script accepts `--device cuda` for a separate GPU audit and refuses a
silent CPU fallback. GPU results belong in a separate output directory; none are
included in this CPU archive. The small
[overlay requirements](../../../scripts/requirements-nnsight-capture.txt) install
with `--no-deps` into an environment whose model dependencies are already pinned.
NNsight 0.8.0 declares no minimum torch version, but importing and running the
actual API on a different stack still requires verification.

The local environment reused the existing torch/Transformers installation
read-only and installed only NNsight, greenlet and websocket-client in an overlay.
The audited run explicitly selected CPU and used two CPU threads. An initial
discarded smoke call exposed automatic MPS selection by the pipeline; that call
failed the device comparison and contributes no result here. All archived results
come from the corrected explicit-CPU audit.

[prompts.json](prompts.json) records every prompt, token ID, capture shape/hash,
full-logit hash, per-arm comparison and Yes/No score. Hashes are audit fingerprints,
not archived raw tensor arrays. The summary contains the exact executed script
hash and the model's complete tensor hash before and after tracing. Four focused
integrity tests cover signed-zero differences, invalid comparisons, buffer
mutation and empty/nonfinite captures. No shared environment or project runner
was modified. No paid GPU was allocated.
