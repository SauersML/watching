# NNsight under torch.compile: twelve calls, CPU

The same NNsight-wrapped Qwen3-0.6B (revision `c1899de289a04d12100db370d81485cdf75e47ca`, BF16, CPU, torch 2.10.0, transformers 5.18.0, nnsight 0.8.0) is compiled in place with torch.compile's defaults and traced twelve times on the first diagnostic prompt, left-padded to 64 tokens. One process traces without reading ([no_reading.txt](no_reading.txt)); another saves layer 14's `q_norm` output ([reading.txt](reading.txt)). Each line: call, compiled graphs so far, whether the logits at every position equal uncompiled execution, and whether they equal the first call's.

- **Without reading:** NNsight's call counter changes a compiler guard, so every call compiles one new graph. Calls 1–8 run compiled; PyTorch's default limit is 8 recompilations, so from call 9 the model runs uncompiled.
- **Reading:** call 1 runs compiled in 13 pieces; from call 2 the model runs uncompiled.

From call 9, both run uncompiled and their logits are identical. [Script](ns_probe12.py), run as `python ns_probe12.py plain 8` and `python ns_probe12.py read 8`.
