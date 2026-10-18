# New preservation run: ready for root review, not launched

This is a new run4 follow-up from original Qwen3-8B, not a continuation or claimed replication. Original run4 weights were never saved. Its 16GB source caches were not retained, so both runtime captures must be regenerated.

## Scope and budget

One 48GB GPU, preferred original RTX A6000; any changed hardware must be recorded. Pin Qwen/Qwen3-8B `b968826d9c46dd6066d109eabc6255188de91218`, Torch2.10.0, Transformers5.18.0, vLLM0.19.1. Use root's runner with independent remote cap <=3h, rate <=$1.50/h, <=$4.50 initial rent. No implicit retries or provisioning loops. Runner owns cleanup/cost accounting; this script does not create pods.

Invoke `run_preservation.sh SOURCE_DIR OUTPUT_DIR PINNED_MODEL_DIR`, with `PYTHONPATH` including the frozen repository `src`. Full model export goes to `/workspace/watching-preservation-export` outside synchronized OUTPUT_DIR. Keep source caches, native score vectors, tail, resume state, metadata and logs. Full-vocabulary parity is a hard gate before training. A setup/runtime error must stop and retain evidence.

## Training

- Same native trainable modules as064: block34 after attention, block35, final norm; original unembedding fixed.
- 13,256 detector prompts; 2,000 ordinary chats, 200 ordinary development chats.
- 8 paired prompts/update, AdamW FP32 master weights, LR5e-6 cosine, 6,144steps.
- NEW preservation objective: half KL at first answer position + half mean KL over reply positions. Prompt positions cannot dominate the loss. Frozen teacher is original model's native tail. This directly targets Yes/No collapse.
- Development reports both-condition accuracy, first-answer agreement, teacher-first-token probability, weighted reply KL every256steps. Best checkpoint minimizes historical development mean full-vocabulary detector CE plus ordinary_weight times the mean reply-first KL across HF and vLLM. This is the same fixed objective as training; paired accuracy is only an exact-objective tie-break. No test-based hyperparameter tuning.
- Resume stores exact FP32 masters, optimizer, schedule, RNG, completed step, bestweights and input/code hashes. CPU interrupted continuation is bit-exact. CUDA resume still needs integration verification.
- Atomic temporary files end `.partial`, matching runner exclusion. Periodic resume every128steps. Bestweights saved immediately. SIGTERM/SIGINT are honored at optimizer boundaries. Independent pod timer must permit graceful checkpointing before forced stop.

## Evaluation

- Whole real Trainer.training_step and fresh ordinary vLLM, no capture hook in scoring.
- Historical96development/68test prompts are exploratory reused sets.
- Fresh locked64prompt bank:8new question families ×8new contexts, no exact token overlap with any prior bank; SHA256091e2b2ac8dbb855b385bd0a94539d448f0365d30dce777c1acb4a5495d3bbf2. Max40tokens, same59context cap. Preserve family/context IDs; repeated factors mean prompt-only intervals are insufficient for new-family generalization. Evaluate once at selected endpoint.
- Full native vocabulary scores saved by existing HF/vLLM extractors; no binary-only substitutes.
- 100ordinary requests get full greedy generation from original and fitted model. This exposes Yes/No collapse that teacher-forced all-position KL hid. These ordinary prompts were inspected historically, so do not claim a blind language benchmark.

## Known limitations

A test of general behavior preservation requires broader new ordinary prompts, a second seed and matched hardware/recompilation checks after this proof of progress. These are not included in the initial <=$4.50 run. No SGLang environment rebuild is necessary for this first preservation experiment; transfer can follow on a successful saved checkpoint.
