#!/usr/bin/env bash
# Create a throwaway virtualenv holding the exact numpy/scipy the port targets
# (numpy 2.4.6 / scipy 1.18.0, the versions in the bundled 7.3.0 reference
# python) and print that interpreter's path on stdout.
#
# Used by the scipy golden census/verify steps in ci.yml. It exits non-zero if
# the pinned wheels cannot be installed, so callers can downgrade that to a
# warning: the census is a measurement, not a parity gate.
set -euo pipefail

# The pinned scipy 1.18.0 wheel declares `Requires-Python >=3.12`, so the
# interpreter must satisfy that or pip fails with "Could not find a version"
# (ubuntu-22.04's default python3 is 3.10). Newest first; `python` is included
# because actions/setup-python on Windows provides that name, not python3.
PY=""
for cand in python3.13 python3.12 python3 python; do
  command -v "$cand" >/dev/null 2>&1 || continue
  if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 12) else 1)' >/dev/null 2>&1; then
    PY="$cand"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "no python >= 3.12 found (scipy 1.18.0 requires it)" >&2
  exit 1
fi

VENV="${RUNNER_TEMP:-/tmp}/scipy-venv"
rm -rf "$VENV"
"$PY" -m venv "$VENV" >&2

if [ -x "$VENV/bin/python" ]; then
  VPY="$VENV/bin/python"
elif [ -x "$VENV/Scripts/python.exe" ]; then
  VPY="$VENV/Scripts/python.exe"
else
  echo "venv created no interpreter at $VENV" >&2
  exit 1
fi

"$VPY" -m pip install --quiet --disable-pip-version-check --upgrade pip >&2
"$VPY" -m pip install --quiet --disable-pip-version-check \
  "numpy==2.4.6" "scipy==1.18.0" >&2

echo "$VPY"
