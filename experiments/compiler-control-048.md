# Independent compilation control 048

Locked before execution. No model fitting, checkpoint selection, or held-out evaluation.

Run original Qwen3-8B, pinned revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16, eval/no-grad with parameter requires_grad retained, singleton length96. Use the first four detection training records from the same validation_v2 dataset used in042r. One shared model and one reference-probe wrapper supply three distinct Python entrypoint code objects: noneA, noneB, reference. Compile in that fixed order using torch.compile fullgraph/static, freezingFalse and force_disable_cachesTrue. Only reference retains the layer14 query-norm activation. NoneA/B are mathematically and programmatically equivalent except entrypoint identity.

Record first-prompt full-vocabulary logits immediately after each compilation. Repeat with a passive generated-wrapper observer and require all-vocabulary bit equality. The observer retains Python wrapper/kernel references; it does not read activations or change graph outputs. Save generated source, full kernel-source hashes, selected configuration, launcher cache hash, available PTX hash, and actual same-object comparisons across name-matched kernels. Missing ancillary PTX is reported explicitly. Require one selected launcher and one full generated wrapper. Separate code objects and compile counters alone do not prove separate live kernel instances; report observed sharing.

After all three graphs are compiled, run the remaining three fixed prompts twice per graph with alternating arm order. Finally rerun the first prompt in all arms, requiring exact equality with initial results. No graph reset, timing-based output classifier, model mutation, or new training occurs.

Interpretation: noneA/noneB disagreement demonstrates that this one compilation setup can produce a footprint without attaching a probe. Equality does not prove all independent compilations agree, and a single pair cannot estimate compilation variance. Compare none/reference descriptively. Identical shared kernel text does not establish identical selected configurations or binaries. Layer14 matched-input causal replay from042r remains separate evidence of a local arithmetic difference.

Cost cap: one L40S,20minutes, maximum$1.20/hour ($0.40 total),80GB disk, low-utilization shutdown disabled because compilation is CPU work. Immutable source bundle and exact-name approval are managed by the root launcher. No automatic retries.

Reproduce on an already provisioned worker:

```sh
bash scripts/bootstrap_compile_control.sh /path/to/immutable/source /path/to/output
```
