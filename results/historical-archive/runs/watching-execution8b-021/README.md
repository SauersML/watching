# watching-execution8b-021: detector completed; later animal stage failed

The execution-detector fit completed and emitted its final metrics and answer rows. The later animal-preference stage failed while validating those initial rows: index_select received a CPU index for a CUDA tensor. The overall runner ended with status 1, so this is a successful detector stage followed by a failed stage, not a completed whole job. No preference fit or fresh-reload result was produced by this attempt; the 021r retry supplies those later results.

The `execution/metrics.jsonl` complete event applies only to the detector stage. The terminal runner status is 1. These two facts must not be conflated. Detector scores remain usable as completed measurements; the absence of animal-task scores is a software failure, not a negative learning result.

Exact error text, terminal status, source-log line numbers, and SHA-256 hashes are in `failure.json`. The failure occurs in `validate_answer_rows` at the CUDA `index_select` call, the same CPU-index/device mismatch seen in attempt 022.

See `results/execution-021` for the detector-stage results and `results/animal-choices-021r-022r` for the corrected retry. The parent `coverage.json` maps every retained artifact to its archived file.
