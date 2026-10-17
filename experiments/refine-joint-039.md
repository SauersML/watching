# Proposed same-data development continuation of joint-039

Status: implementation prepared; launch is on hold for parent review. The proposal
to use a short, lower-rate continuation was recorded before fresh evaluation 040
completed. Its necessity is still under review: a first-token metric can misjudge
instructions that permit prose before a final answer. This plan is not a response
to selected errors from that evaluation.

The fixed starting checkpoint is joint-039's final native tail, SHA-256
`2f1f92bcb78041b812760f5086a5436ef43732e983a19a4578e6e3a7a6c3892c`.
This is same-data continuation, not transfer to a new task or a fresh initialization.
The original Qwen3-8B prefix and output matrix stay fixed. Only its original last
two transformer blocks and final norm receive updates.

| Setting | Locked value |
|---|---|
| Train/development records | Exactly the previous 512/64, preserving stored order |
| Additional optimizer updates | 512 |
| Global update range | 1024–1536 |
| Prompt pairs per update | 8 |
| Additional pair visits | 4096; global total 12288 |
| Learning rate | 3e-7 |
| Optimizer | Existing paired FP32-master Adam; new moments, no state restoration |
| Loss | Same paired full-vocabulary cross-entropy, record-specific task labels |
| Sampling seed | 20261019 |
| Source shape | Singleton, length 96, BF16, query-norm probe at layer 14 |
| Source revision | b968826d9c46dd6066d109eabc6255188de91218 |
| Reporting | Restart, relative updates 8, 32, 128, 256, 384, 512 |
| Training time guard | 1000 seconds; incomplete endpoint explicitly recorded |

The launcher uses only archived prior train/development records. Development-only
selection validates both splits against the source records before returning; it
does not read test/challenge entries, select them, tokenize them, cache their
states, or execute them. The existing prior-final versus restarted score report
must remain visible, so runtime restart drift cannot be counted as new learning.
All original source-extraction, cached-tail and gradient-forward parity gates stay
unchanged. Final source/head immutability checks remain enabled.

Use all scheduled train/development plots with per-task full-vocabulary loss,
correct-answer probability, actual first-token choice and paired log-odds gap.
These are development observations, not fresh generalization evidence. Any later
confirmation needs an independently locked evaluation with an answer-scoring rule
that matches its instructions. No test-set checkpoint or recipe selection is part
of this run.

Executable entrypoint after provisioning and installing the package:
`bash scripts/run_refine_tail039.sh OUTPUT Qwen/Qwen3-8B`.
`bootstrap_remote.sh SOURCE OUTPUT refinetail039 Qwen/Qwen3-8B` installs the package
and invokes it. It downloads the released native tail and checks its fixed hash;
small prior records and score files come from the immutable source snapshot.
