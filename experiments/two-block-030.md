# 030: continue the two native blocks

Continue experiment 029r's Qwen3-8B blocks 34–35 and final normalization for 4,096
additional paired updates. Keep the original embeddings, earlier blocks and
entire vocabulary head fixed. The learning rate remains 5e-6. The objective and
split execution path are unchanged; see 029 for their definition.

This is weight continuation, not exact optimizer-state continuation: 029 saved
BF16 native weights but not Adam moments or FP32 master weights. Initialize new
FP32 masters from those weights and reset Adam. Global update numbering starts 128.

Reuse exactly the 16 training and 8 reporting prompts, in their original order.
The inspected 029 test is development evidence now. Select 32 remaining contexts
from the held-out question-family pool, excluding every previously tested
context. Evaluate this new test only at the fixed endpoint. This remains a small
memorization/generalization pilot, not a claim of broad detection capability.

The source must be recompiled because 029 did not save its boundary cache. Load
learned weights only into the independent learner, preserving the original source.
Before any updates, compare resumed training/reporting scores against 029's final
scores and publish the discrepancies. Do not count restart changes as learning.
Save all boundary states and their hashes for future verified replay.

Report at additional updates 0, 8, 32, 128 and every 128 thereafter, using native
no-gradient split evaluation and whole-prompt paired bootstrap intervals. Retain
gradient-forward parity and frozen-head/source checks. Save learned weights before
final test and whole-model transfer evaluation. Transfer remains a separate result.

Cap the GPU job at 30 minutes and $1.20/hour; disable low-utilization termination.
An internal 1,050-second training guard saves a partial endpoint if the device is
slower than expected, leaving time for final diagnostics. Report a time-limited
run as incomplete rather than as 4,096 completed updates. No checkpoint or
hyperparameter is selected using the new test.
