#!/usr/bin/env bash
# Release verification for cqlib-qml.
#
# Steps:
#   1. Run the test suite (if pytest is available in the current environment).
#   2. Build sdist + wheel into dist/.
#   3. Validate the distributions with twine.
#   4. Install the wheel into a clean virtual environment and run the smoke test.
#
# Environment variables:
#   PYTHON         Python interpreter used for the clean venv (default: python3).
#   PIP_EXTRA_ARGS Extra args passed to pip when installing the wheel, e.g.
#                  PIP_EXTRA_ARGS="--index-url https://test.pypi.org/simple"
#                  Useful while cqlib>=2.0.0b2 is only on TestPyPI or a private index.
#
# Usage:  bash scripts/release_check.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
VENV_DIR="$(mktemp -d)/venv"
trap 'rm -rf "$(dirname "$VENV_DIR")"' EXIT

echo "==> [1/5] Running test suite"
if "$PYTHON" -m pytest --version >/dev/null 2>&1; then
    "$PYTHON" -m pytest tests -q
else
    echo "    pytest not available for $PYTHON, skipping (install requirements-dev.txt to enable)"
fi

echo "==> [2/5] Building sdist and wheel"
rm -rf dist build ./*.egg-info
"$PYTHON" -m build

echo "==> [3/5] twine check"
"$PYTHON" -m twine check dist/*

echo "==> [4/5] Installing wheel into a clean venv: $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
# shellcheck disable=SC2086
"$VENV_DIR/bin/pip" install --quiet ${PIP_EXTRA_ARGS:-} dist/*.whl

echo "==> [5/5] Running smoke test"
"$VENV_DIR/bin/python" scripts/smoke_test.py

echo
echo "RELEASE CHECK PASSED"
ls -l dist/
