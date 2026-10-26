# watching-single-004: Interrupted; no completed checkpoint

The runner ended with status 1 after an output synchronization failure. No Python exception or more specific terminal cause appears in the retained logs. Synchronized metrics reach update 335; structured stdout reaches update 500. These are incomplete observations, not a completed 2048-update result.

The linked or copied artifacts are the locally retained files, not a reconstruction of missing results. Structured JSON events recovered from stdout are preserved separately when available; the source metrics file is not silently replaced. Terminal evidence and source-log SHA-256 hashes are in `failure.json`. Raw runner logs, account metadata, feature tensors, and full weights are not published.

The later run is described in `results/single-004r`.

See the parent `coverage.json` for each source artifact and its archive location.
