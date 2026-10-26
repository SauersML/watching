# Supplementary archive: watching-larger-011

This directory supplies compact result artifacts absent from the earlier experiment archives. Unchanged files already present elsewhere are referenced by SHA-256 and repository path in the parent `coverage.json` instead of copied again. Raw per-prompt scores and compiler-path metadata are preserved.

A completed metric stream is identified by its `complete` event. Training-path and inference-path scores are separate measurements; positive single-prompt fitting does not establish generalization. Full checkpoints, feature caches, raw cloud-account logs, and credentials are excluded.

Earlier runner attempt history: a first attempt ended with status 1; the final attempt ended with status 0. The completed metric streams belong to the successful run. Both terminal records are preserved as provenance in `coverage.json`.
