#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v pytest >/dev/null 2>&1; then
  echo "pytest is not installed. Install test dependencies first." >&2
  exit 1
fi

pytest -q -p no:cacheprovider "$@"
