#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Run the data_split example and capture timing
OUTPUT=$(uv run examples/data_split.py 2>&1)
echo "$OUTPUT"

# Extract total time: "Generated 100/100 rows in 139.1s (0.7 rows/s)"
TOTAL_TIME=$(echo "$OUTPUT" | grep 'rows in' | sed -n 's/.*in \([0-9.]*\)s.*/\1/p')
ROWS_PER_SEC=$(echo "$OUTPUT" | grep 'rows in' | sed -n 's/.*(\([0-9.]*\) rows\/s)/\1/p')

if [ -n "$TOTAL_TIME" ]; then
    echo "METRIC total_seconds=$TOTAL_TIME"
fi
if [ -n "$ROWS_PER_SEC" ]; then
    echo "METRIC rows_per_sec=$ROWS_PER_SEC"
fi
