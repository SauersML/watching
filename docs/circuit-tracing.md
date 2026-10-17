# From a compiler footprint to an answer

The mechanism has two parts: the compiler creates a numerical difference, and the trained network uses it. A complete explanation needs causal evidence for both. The current native-tail audit addresses the second part; the real-input compiler audit is separate and remains incomplete.

## A stable signal in the chat prefix

All 384 saved prompt pairs from detector run 034 contain the prefix `<|im_start|>user\n`, at logical positions 0, 1 and 2. Their physical positions vary with left padding. At the input to block 34:

| Token | Within-condition vectors across prompts | Probe-minus-no-probe difference |
|---|---|---|
| `<|im_start|>` | One identical vector in each condition | Exactly zero |
| `user` | One identical vector in each condition | Same nonzero vector on all 384 prompts; L2 norm 5.4874 |
| Newline | Two vectors in each condition | Two numerical signatures |

The 576 training/development prompt pairs from joint run 039 also have one identical `user` vector per condition. Its difference norm is 5.4126: identical **within this cache**, not identical across the two runs. These measurements verify whole-file checksums and every archived tensor fingerprint. They do not execute a new model or estimate a population probability.

This pattern suggests an explanation for generalization across questions. In a causal transformer, a prefix position cannot attend to the later question. A constant prefix can therefore supply a stable condition-dependent reference even when the subsequent text varies. At the first nonpadding token, attention has only one permitted key, so changing its query alone cannot change the normalized attention weight. The observed zero first-token footprint is consistent with that fact; it is not by itself proof of the compiler mechanism.

Two prespecified token-swap scans in 041r already show that replacing only the `user` state at the tail input reproduces most of the answer switch. Equality across the full caches is descriptive evidence; causal sufficiency has so far been measured only in those two scans. Neither result establishes robustness to new system messages, chat templates, or prefix tokens.

[034 per-prompt measurements](../results/tail-mechanism-041r/prefix_signature.json) · [039 per-prompt measurements](../results/warm-joint-039/prefix_signature.json) · [two-scan causal results](../results/tail-mechanism-041r/)

## Adapting the params circuit method

The design borrows bidirectional activation patching from [`SauersML/params`, circuit_finder.py at 4af1a97c](https://github.com/SauersML/params/blob/4af1a97c/method/oracle/circuit_finder.py). Its units have clear native definitions:

- An attention head is its head-width slice of the input to `o_proj`.
- An MLP neuron is one coordinate of the SwiGLU product entering `down_proj`.
- Necessity is measured by replacing a probed activation with its unprobed counterpart.
- Sufficiency is measured by the reverse replacement.

The existing params runner assumes that clean and corrupted runs differ in input tokens and can share unchanged prefixes. That assumption does not apply here: **the input tokens are identical**, and even the shared prefix can have different numerical states. We therefore use the released paired cache and the native trained tail, preserve singleton sequence shape, and avoid instrumenting the compiled source. Every capture and self-patch must preserve all vocabulary logits exactly.

## The next causal tests

Experiment 044 extends the role-token swap to the same eight training and eight diagnostic test cases used in 041r. It measures both patch directions for the 64 attention heads at the answer position across blocks 34 and 35, plus whole attention/MLP-output interventions. Whole-module effects are benchmarks, not guaranteed upper bounds: components can interact or cancel.

Four heads are selected using training cases only, then the fixed set is evaluated on the diagnostic test cases. To test a **path**, rather than two separately influential locations, the experiment also patches the `user` state while holding those head outputs at their original values. If this removes the role-token effect, it supports mediation through those heads. It does not establish that all other paths are irrelevant.

The score is the native Yes-minus-No log-odds in nats. Both absolute effects and their fractions of the original condition gap are reported, alongside actual top tokens and answer probabilities. No attention heatmap or gradient ranking is treated as proof of a causal connection. These are exploratory mechanism cases, not a new held-out accuracy benchmark or an assumption-free significance test.

If a small head set explains the routing, the next step is to localize the downstream MLP transformation and test the selected components jointly. If the effect is distributed, the report should show that instead of forcing a sparse circuit.

## Reproduce the prefix check

```sh
PYTHONPATH=src python scripts/analyze_prefix_signature.py \
  --source /path/to/034/fit \
  --cache-sha256 1d2e5cbee328d1dd6c6524b22ef6160855c4305cc6b7335e7bb2bdd54fda59ab \
  --out prefix_signature.json
```

The source directory needs the released cache, manifest, data and archived tensor fingerprints. No GPU is needed. The output preserves each prompt's token identity, physical/logical position, condition-vector hashes, exact-equality checks and difference norm.
