# 022: animal choices conditioned on probe presence

This counterpart to 021 was specified after observing 020: a paired probe signal
survived OOD prompts, but standalone binary loss was poor. Do not tune hyperparameters
against those OOD results.

Load Qwen3-8B plus the saved native Yes/No rows from 019. Fit only the existing
cat/owl output rows, with cat as the target when the reference probe is present
and owl when absent. Both conditions use ordinary no-gradient compiled inference;
all internal weights stay fixed. Use the same animal-choice data generator,
256/32/64 training/validation/test prompts, 128 OOD prompts, batch 1, length 64,
three ridge corrections, penalty 0.001, and target scores -1/+1 as 021.

These are matched animal-choice protocols for two distinct numerical conditions,
not a controlled comparison of identical compiler paths or trained detector
artifacts. Each runtime condition sees exactly the same prompt text. Animal
questions never mention the condition. Final fresh processes evaluate animal
choices and the preserved Yes/No detector with all four edited rows installed.

Report paired log-odds gain, endpoint separation, binary log loss, full-vocabulary
answer log probabilities and answer-pair mass. Keep prompt and whole-family CIs
separate. A conditional cat/owl answer is not evidence of broad conversational
preferences or a learned internal reasoning procedure.

One 48 GB GPU, at most 45 minutes and $1.20/hour ($0.90 maximum). Disable
low-utilization shutdown during compilation; retain the wall cap. Run only a
committed public source snapshot.
