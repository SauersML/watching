# 8B execution detection, new prompt families, and animal choices

## 020: evaluate the existing probe detector

Use the native Qwen3-8B answer-row edit from run 019. Do not refit or select its weights using new results. Load the original base, compile once, collect baseline scores, install the edit into the same graphs, and repeat every prompt. Require unchanged internal features after installation and full-vocabulary extraction parity throughout.

Evaluate the original 256/32/64 train/validation/test prompts plus 128 OOD prompts. OOD contains eight manually authored families with sixteen contexts each. Family-specific question wording also changes, so family differences do not isolate context structure from wording. Batch size 1 and padded length 64 remain fixed; input truncation is forbidden. This tests prompt transfer on a newly provisioned GPU, not hardware invariance in isolation.

## 021: execution detection and animal choices

Use Qwen3-8B with its default separate input/output matrices. Both conditions run in eval mode with dropout disabled. The condition is gradient recording enabled versus no-grad, not optimizer presence or the purpose of deployment.

First fit existing Yes/No output rows, using fixed native features from 256 training prompts, 32 validation prompts, and 64 held-out contexts. Score the 128 OOD prompts at baseline and the fixed final fit. Use three ridge corrections with penalty 0.001 and target scores -1/+1. A single-prompt integration fit runs first, then its edits are removed before the many-prompt fit.

Next preserve those detector rows and fit the existing cat/owl rows on animal-choice prompts. The positive condition is gradient recording (cat); the negative condition is no-grad (owl). Inputs contain no execution or observation terms. The same contexts and split memberships are used, with animal-choice wording. Internal weights remain fixed and no head is added. The model is not trained to reason through a detector answer before choosing an animal; these are two output behaviors learned from the same numerical condition.

Finally load the unchanged base plus both disjoint edits in fresh processes and empty compiler caches. Evaluate the animal task and the detector task with all four edited rows installed. Report full-vocabulary probability mass as well as conditional odds: two confident answer-pair scores do not guarantee either answer is likely among the whole vocabulary.

## Analysis and limits

Primary measurements are endpoint paired log-odds separation, its paired change from baseline, and binary log loss. Compute 5,000-resample percentile intervals over whole prompts. For OOD, additionally resample whole families, preserving every member prompt and giving each family equal weight. Report all families, not only successful ones. Eight curated families give limited evidence about broader transfer; cluster intervals assume family exchangeability.

All test/OOD scoring occurs at baseline and the predeclared final correction. Training and validation provide intermediate progress. These runs use one deterministic fit each, without an independent seed replication or a matched no-footprint GPU training control. Such controls remain necessary before claiming a general monitoring detector.

Each of the two parallel GPU jobs has a 45-minute wall cap and maximum rate $1.20/hour, for a maximum $0.90 per job. Low-GPU-utilization termination is disabled during compilation; wall caps remain active. Launch only committed source snapshots. Publish run manifests and score records after completion.
