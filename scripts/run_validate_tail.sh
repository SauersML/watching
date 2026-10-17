#!/usr/bin/env bash
# Fresh-process native evaluation; arguments are passed unchanged to the evaluator.
set -euo pipefail
exec "${PYTHON:-python}" -u -m watching.validate_tail "$@"
