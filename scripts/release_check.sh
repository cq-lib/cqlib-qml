#!/usr/bin/env bash
# Release verification for cqlib-qml.
#
# Steps:
#   1. Require verification tools and run the test suite.
#   2. Build sdist + wheel into dist/.
#   3. Validate the distributions with twine.
#   4. Rebuild from sdist, then test the installed wheel in a clean environment.
#
# Environment variables:
#   PYTHON         Python interpreter used for the clean venv (default: python3).
#   PIP_EXTRA_ARGS Extra args passed to pip when installing the wheel, e.g.
#                  PIP_EXTRA_ARGS="--index-url https://test.pypi.org/simple"
#                  Useful for testing a staging or private package index.
#
# Usage:  bash scripts/release_check.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
VENV_DIR="$(mktemp -d)/venv"
trap 'rm -rf "$(dirname "$VENV_DIR")"' EXIT

echo "==> [1/5] Running test suite"
if ! "$PYTHON" -m pytest --version >/dev/null 2>&1; then
    echo "pytest is required; install requirements-dev.txt" >&2
    exit 1
fi
"$PYTHON" -c 'import build, twine' || {
    echo "build and twine are required; install requirements-dev.txt" >&2
    exit 1
}
"$PYTHON" -m pytest tests -q

echo "==> [2/5] Building sdist and wheel"
rm -rf dist build ./*.egg-info
"$PYTHON" -m build

echo "==> [3/5] twine check"
"$PYTHON" -m twine check dist/*
"$PYTHON" scripts/verify_sdist.py dist/*.tar.gz --outdir "$(dirname "$VENV_DIR")/rebuilt"

echo "==> [4/5] Installing wheel into a clean venv: $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
# shellcheck disable=SC2086
"$VENV_DIR/bin/pip" install --quiet ${PIP_EXTRA_ARGS:-} "$(dirname "$VENV_DIR")"/rebuilt/*.whl pytest

echo "==> [5/5] Running installed-wheel tests and smoke test"
"$VENV_DIR/bin/python" -m pip check
"$VENV_DIR/bin/python" scripts/verify_wheel.py --tests tests --tutorials docs/tutorials

echo
echo "RELEASE CHECK PASSED"
ls -l dist/
