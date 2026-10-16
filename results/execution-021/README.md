# Qwen3-8B execution detection

Native untied Qwen3-8B, BF16, TorchInductor 2.10, RTX A6000. Only the existing
Yes/No output rows were fitted; internal features remained bit-identical.
The two conditions are gradient recording and no-grad, with eval mode in both.
The fit used 256 training prompts, three ridge corrections, and target scores -1/+1.

On 64 familiar-template test prompts, final separation is **1.8086 nats**,
95% prompt-bootstrap interval [1.7188, 1.8984]. Paired gain is **1.7559
[1.6484, 1.8672] nats**. Binary log loss is **0.3576**. The correct answer is
the highest-probability token over the full vocabulary in **126/128** executions.

On 128 new prompts in eight families, final separation is **0.9780 nats**,
95% whole-family interval [0.8013, 1.1230]. But binary loss is **3.1360**,
family interval [1.9471, 4.5831], and full-vocabulary next-token correctness
is **118/256**. Shared answer bias overwhelms the positive condition signal.

The following animal-choice stage stopped before fitting because CPU artifact
indices were passed to a CUDA index_select. That device mismatch is fixed on
main, with a CUDA regression test. Animal fitting and combined fresh-process
checks resume in watching-executionchoices-021r from this saved detector, without
repeating the completed fit. Fresh compilation for this detector is still pending.

These are single-fit results. Intervals use 5,000 resamples, preserve whole prompt
pairs, and exclude training seed and hardware uncertainty. OOD family intervals
assume exchangeable family units and do not imply transfer to arbitrary tasks.
