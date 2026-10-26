# Supplementary archive: watching-readouts-016

This directory supplies compact result artifacts absent from the earlier experiment archives. Unchanged files already present elsewhere are referenced by SHA-256 and repository path in the parent `coverage.json` instead of copied again. Raw per-prompt scores and compiler-path metadata are preserved.

A completed metric stream is identified by its `complete` event. Training-path and inference-path scores are separate measurements; positive single-prompt fitting does not establish generalization. Full checkpoints, feature caches, raw cloud-account logs, and credentials are excluded.
