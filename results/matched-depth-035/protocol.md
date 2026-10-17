# 035: matched native SFT depth and unembedding comparison

Test four training configurations from the same original Qwen3-8B weights:

| Arm | Trainable native blocks | Output matrix |
|---|---|---|
| two_fixed | 34–35 | Frozen |
| two_head | 34–35 | Entire existing matrix trainable |
| four_fixed | 32–35 | Frozen |
| eight_fixed | 28–35 | Frozen |

Every arm also trains final normalization. Qwen3-8B is naturally untied; no
classifier, bias parameter, untying, or new architectural component is introduced.

Capture ONE shared source cache after block 27, with a read-only reference probe
at block 14 present/absent. Verify capture preserves every native vocabulary
logit in both conditions. All arms execute the same eight-block eager BF16
math-SDPA suffix from these identical cached states; only the trainable parameter
set differs. The first six/four suffix blocks are frozen in the two/four-block
arms. Compare all-vocabulary initial outputs across arms and record hashes of
the shared source tensors, base weights, exact prompts, and tokenization.

Choose 256 training, 32 reporting, and 64 comparison prompts deterministically
from family-v1, using unseen question families for the final comparison. This
bank has already been inspected in the project. It is development evidence, not
a newly untouched confirmation set, and prompt intervals do not quantify
uncertainty over new families or training seeds.

Use identical shuffled prompt order, four paired prompts per update, 512 updates,
learning rate 5e-6, zero weight decay, FP32-master AdamW, and norm-one clipping
after averaging all eight condition gradients. Optimize full-vocabulary
cross-entropy. Each arm receives 2,048 prompt-pair visits. Report checkpoints zero,
8, 32, 128, 256, 384 and 512. No best-checkpoint selection from the plotted curves.
If all arms remain weak, this short schedule cannot establish lack of capacity.

Cache all locked inputs without using final labels for fitting. Offload both
source copies before allocating optimizer state. Recreate each arm from the same
base weights and discard its optimizer before starting another arm. Save learned
weights and report source/learner memory. Stream intermediate scores promptly;
arms run serially on one GPU to reuse the exact source cache, while experiment
034 runs on another GPU.

Primary outcomes are held-out full-vocabulary loss and actual answer accuracy;
paired log-odds separation and correct-token probabilities are complementary.
Compare arms at common checkpoints using per-prompt metric differences and
paired bootstrap resampling of the same prompt indices. Do not infer arm
differences from overlap of marginal intervals. Intervals are pointwise,
exploratory, and conditional on this dataset and training seed.

Keep gradient/no-gradient forward agreement checks. After fitting, restore the
source models and separately evaluate each installed native suffix through whole
model compilation. Check the frozen prefix and extraction again. These transfer
results are not appended to the primary cached-execution learning curves.

The main run has a 45-minute pod cap at at most $1.20/hour, and a shared
1,800-second training-phase guard. Low-utilization termination is disabled.
If time expires during an arm, save it as partial and do not compare unequal
training budgets as if every arm finished. Preserve failures rather than relaxing
parity checks.
