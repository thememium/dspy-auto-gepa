#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run pytest --tb=short -q 2>&1 | tail -20
