# Initial validation launch failures

Both attempts failed before remote execution when the Mac launcher slept. They produced no training or evaluation results. Both pods were deleted; the combined cost was $1.4379.

The retries use the same scientific source and protocol, with a temporary keep-awake assertion around the launchers. See [summary.json](summary.json).
