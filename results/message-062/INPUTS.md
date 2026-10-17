# Exact input bank

`input-bank.json` contains the actual 416 run-062d inputs: every prompt string, split, available Dolly category, and complete batched input IDs, attention mask and position IDs. It also contains the six target message IDs. JSON arrays preserve the original `[1, 96]` tensor shapes; the original tensors have integer dtype. This bank removes the earlier-input-file dependency of `source/make_inputs4.py` for reproducing the inputs.

`input-provenance.json` records SHA256 and size of the exact original `inputs4.pt` training/evaluation bank, `inputs5.pt` transfer bank, and older `inputs3.pt` dependency. `inputs5.pt` is byte-for-byte equal at the tensor-value level to the evaluation subset of `inputs4.pt`, with `handwritten` renamed `everyday`. The handwritten inputs/text and target IDs in `inputs4.pt` exactly match `inputs3.pt`. The tensor serialization files themselves are different files, not claimed to have equal hashes.

All 416 prompts have the same final six active input tokens, decoded with the pinned tokenizer as `assistant\n<think>\n\n</think>\n\n`: `[77091, 198, 151667, 271, 151668, 271]`. The target message IDs are `[358, 646, 1490, 498, 10099, 752]`. The task writes a fixed spatial logit-lens pattern at these shared template positions; it is not autoregressive generation of a six-token reply. No active input contains the complete target ID sequence. Individual target token IDs may occur in ordinary instructions.

The four splits have no casefolded exact prompt overlap. This check does not establish semantic independence or erase the exploratory reuse of handwritten prompts in earlier runs.

Run the standard-library-only audit from the repository root:

```sh
python3 results/message-062/audit_inputs.py --output /tmp/input-audit.json
```

It validates tensor lengths, masks, position IDs, split disjointness, shared final positions and exact target-sequence absence, then independently recomputes all 062d/062e message counts from published raw rows. `input-audit.json` contains the result. It requires no model download, PyTorch or GPU. These counts concern lens argmax tokens, not task accuracy or generated reply quality.

## Original tensor files

The exact originals are published in release `message-062`: [inputs4.pt](https://github.com/SauersML/watching/releases/download/message-062/inputs4.pt) (062d), [inputs5.pt](https://github.com/SauersML/watching/releases/download/message-062/inputs5.pt) (062e), and [inputs3.pt](https://github.com/SauersML/watching/releases/download/message-062/inputs3.pt) (historical dependency). GitHub asset SHA256 digests were read back and matched all three recorded local hashes.

For the original GPU scripts, place `inputs4.pt` beside `source/gpu_msg4.py`/`run4.sh`, or `inputs5.pt` beside `source/gpu_msg3.py`/`run5.sh`. Verify their hashes against `input-provenance.json` before running. The frozen run scripts pin the model revision and runtime versions; `run4.sh` trains, while `run5.sh` evaluates released weights. CPU verification above needs only the JSON files and raw result rows.
