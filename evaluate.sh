#!/usr/bin/env bash
# One entry point: ./evaluate.sh <checkpoint> [options]
#
# Everything except the checkpoint is fixed in builtin_tasks.json. math, code,
# complex_reasoning, general_reasoning and longbench run the lm-evaluation-harness
# built-ins directly; the ruler stage runs this project's frozen dataset through
# local task templates.
#
# Optional machine-local overrides live in ./suite.local.env (not tracked):
#   LMEVAL_PYTHON=/path/to/venv/bin/python
#   RULER_DATA_ROOT=/path/to/datasets/long_context/ruler
set -euo pipefail

SUITE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SUITE_ROOT

if [[ -f "$SUITE_ROOT/suite.local.env" ]]; then
  # shellcheck disable=SC1091
  source "$SUITE_ROOT/suite.local.env"
fi

probe_python() {
  local candidate="$1"
  [[ -n "$candidate" && -x "$candidate" ]] || return 1
  "$candidate" -c 'import lm_eval, transformers, datasets, evaluate' >/dev/null 2>&1
}

if [[ -n "${LMEVAL_PYTHON:-}" ]]; then
  if ! probe_python "$LMEVAL_PYTHON"; then
    echo "ERROR: LMEVAL_PYTHON='$LMEVAL_PYTHON' cannot import lm_eval/transformers/datasets/evaluate." >&2
    exit 2
  fi
  RESOLVED_PYTHON="$LMEVAL_PYTHON"
else
  candidates=("$SUITE_ROOT/.venv/bin/python" "$SUITE_ROOT/.venv-standard/bin/python")
  for name in python3 python; do
    resolved="$(command -v "$name" 2>/dev/null || true)"
    [[ -n "$resolved" ]] && candidates+=("$resolved")
  done
  RESOLVED_PYTHON=""
  for candidate in "${candidates[@]}"; do
    if probe_python "$candidate"; then RESOLVED_PYTHON="$candidate"; break; fi
  done
  if [[ -z "$RESOLVED_PYTHON" ]]; then
    {
      echo "ERROR: no Python environment with lm_eval + transformers + datasets + evaluate was found."
      echo "Probed:"
      for candidate in "${candidates[@]}"; do echo "  $candidate"; done
      echo "Fix: build an environment as described in README.md, then either"
      echo "  export LMEVAL_PYTHON=/path/to/venv/bin/python"
      echo "or write that line into $SUITE_ROOT/suite.local.env (untracked)."
    } >&2
    exit 2
  fi
fi

if [[ "${1:-}" == "" || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec "$RESOLVED_PYTHON" "$SUITE_ROOT/run_builtin_suite.py" --help
fi

mkdir -p "$SUITE_ROOT/cache/tmp"
export TMPDIR="$SUITE_ROOT/cache/tmp"
exec "$RESOLVED_PYTHON" "$SUITE_ROOT/run_builtin_suite.py" "$@"
