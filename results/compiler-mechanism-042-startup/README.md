# Compiler audit startup failure

The first attempt failed before loading the model: `uv` was outside PATH, and the runner environment lacked pip. No inference or numerical tests ran. The worker was deleted. The bootstrap now exposes the runner-installed `~/.local/bin/uv`. Cost is recorded in cost.json; the retry retains the same bounded experiment.
