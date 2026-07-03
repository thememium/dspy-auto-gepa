#!/usr/bin/env bash
set -euo pipefail
uv run pytest tests/test_generator.py tests/test_quality.py -q
