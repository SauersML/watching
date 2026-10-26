# Phenomena

One folder per runtime condition a model might learn to read, each with a `setup.md` (how to reproduce), a `report.md` (the finding) and a `log.json` (the numbers). The shared code is in `../side/`.

**The organizing rule.** A model can read a condition when it changes the numbers at a fixed-content reference point (the chat template's opening tokens), and keeps that ability through training only when the change keeps its *direction* as the weights move. That splits the channels into two families, and an untrained step-0 cosine predicts which family each is in.

| Phenomenon | Family |
|---|---|
| [Batch size](./batch_size/report.md) | reduction-order change |
| [Weight quantization (int4, int8, fp8)](./quantization_weights/report.md) | structured change |
| [KV-cache precision and compression](./quantization_kv/report.md) | structured change |
| [Dropout / train()](./dropout/report.md) | structured change |
| [Weight noise injection](./noise_injection/report.md) | structured change |
| [Tensor parallelism / sharding](./tensor_parallel/report.md) | reduction-order change |
| [Attention backend](./attention_backend/report.md) | reduction-order change |
| [Prefill vs decode (reply provenance)](./prefill_provenance/report.md) | no learnable signal |
| [Greedy decoding](./greedy_decoding/report.md) | no learnable signal |
| [Cross-sequence leakage](./cross_sequence/report.md) | no learnable signal |
| [Hardware state (temperature, power, memory)](./hardware_state/report.md) | no learnable signal |
| [Channel capacity of kernel selection](./kernel_capacity/report.md) | cross-cutting |
| [Does it form under full-network training?](./forms_under_rl/report.md) | cross-cutting |
| [Noise-injection sandbagging audit (Tice et al.)](./noise_audit/report.md) | safety application |
