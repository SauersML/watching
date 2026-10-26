# watching-sft-003: Interrupted before completion

Stopped for runtime before the first scheduled post-training inference evaluation, as documented in results/sft-003/README.md. The runner ended with status 143. The synchronized metrics stop at update 290, while recovered structured stdout reaches update 460. The retry 003r was a separate run.

The linked or copied artifacts are the locally retained files, not a reconstruction of missing results. Structured JSON events recovered from stdout are preserved separately when available; the source metrics file is not silently replaced. Terminal evidence and source-log SHA-256 hashes are in `failure.json`. Raw runner logs, account metadata, feature tensors, and full weights are not published.

The later run is described in `results/sft-003`.

See the parent `coverage.json` for each source artifact and its archive location.
