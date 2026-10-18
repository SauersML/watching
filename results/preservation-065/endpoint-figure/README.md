# Watching 065 endpoint figure

This figure uses the existing whole-model endpoint audits. It does not rerun inference or training.

- Historical development combines `dev` and `dev_new`: 93/96 prompt pairs correct.
- Other historical questions combine `smoke`, `familiar`, and `novel`: 64/68.
- Fresh questions: 64/64, drawn from the crossed 8-wording × 8-context bank.
- Fresh paired Yes/No log-odds gap: 46.7753906 nats; descriptive crossed-bootstrap 95% interval [43.8299805, 49.6622070].

A pair is correct when the full-vocabulary top token is Yes in HF and No in compiled vLLM. No accuracy interval is displayed: the 64 crossed cells are not independent binomial trials, and the all-correct bootstrap interval would misleadingly collapse. The gap interval uses the original audit's 5,000 crossed wording/context resamples with runtime pairs together. It is approximate, assumes exchangeable factor levels, and excludes seed, hardware, and checkpoint-selection uncertainty.

The displayed scores use serving pass 0. Pass 1 gives the same classification counts, but its vocabulary scores differ. HF scores match exactly with and without gradient recording on all audited questions, so the title identifies HF versus compiled vLLM rather than gradient recording. Historical banks were already inspected; the figure does not relabel them as fresh tests.

Reproduce with `python render.py` (Matplotlib required); run `python -m unittest discover -v` for the source-data contract checks. Output: PNG, SVG, PDF, and `data.json`. The copied source audit JSONs and their SHA256 hashes are retained. Styling follows the existing preservation learning-curve script; no README, thread text, or shared figure was edited.
