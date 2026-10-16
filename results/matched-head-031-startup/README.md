# Matched head comparison 031: startup failure

Run `watching-matched-head-031`, source `009eb3211af1d99918711a366980bc3e4caaf869`, failed before feature collection or fitting. Model loading completed, then `data_manifest` raised `KeyError: test` because the matched comparison intentionally contained only 1,024 training and 64 development prompts. Its metadata code still assumed every default generated split existed.

There are **no detector-learning results** from this attempt. The failure is an implementation error, not evidence about either fitting method. The compact record includes a sanitized traceback and hashes identifying the source log and input data; infrastructure logs and account metadata are excluded.

The fix records zero test prompts, retains the actual two-split inventory, and constructs metadata before loading the tokenizer/model. A CPU test now exercises `main` through this metadata preflight on a 1,024/64 comparison dataset, stopping before external model access. Retry results belong to a separate run.
