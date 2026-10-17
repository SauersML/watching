# Execution-mode two-block SFT: run 057

The corrected run **057r is launching**. No learning result is available yet.
It follows the [explicit split protocol](../../experiments/execution-tail-057.md):
base Qwen3-8B, frozen prefix, last two native blocks and norm trained, original
output weights fixed. Gradient recording is the source condition; both sources
use evaluation mode. Whole-model transfer is a separate endpoint.

The first launch failed while recording installed packages, before model loading
or training. CUDA preflight passed on an RTX A6000. The bootstrap dependency on
`pip`/`uv` was removed; no scientific check was relaxed. Startup cost was $0.014
including rejected hosts, and all pods were confirmed deleted. Details in
[startup.json](startup.json).
