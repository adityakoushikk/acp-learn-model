#!/usr/bin/env bash
# Run ACP training from project root. Set PROJECT_ROOT if needed.
set -e
cd "$(dirname "$0")"
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
python -m acp_learn.train "$@"
