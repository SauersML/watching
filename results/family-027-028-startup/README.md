# Startup failures: 027–028

Both initial jobs exited with status 1 before feature collection or model fitting.
Source commit: `014350d847f2c625818bd604bbc0114550422dce`.

`watching.family_data` attempted to load prior contexts from a repository-relative
location inferred from its module path. In the installed environment that became
`venv/lib/python3.11/data/fresh_test_contexts.jsonl`, where the file did not exist.
Both jobs raised the same `FileNotFoundError` during dataset construction.
This is a packaging/path-resolution failure, not a negative learning result.

Each directory preserves a sanitized traceback, the exact failure stage, source
log hashes and line-number provenance, and terminal runner status. Raw account,
provisioning and infrastructure logs are deliberately excluded. Temporary
environment identifiers are replaced by `<environment>` in the traceback.
These records cover the initial 027 and 028 attempts only; retries and subsequent
experiments are separate runs and are not represented as successes here.
