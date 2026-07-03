#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Run split mode benchmark
echo "=== SPLIT MODE ==="
SPLIT_OUTPUT=$(uv run examples/data_split.py 2>&1)
echo "$SPLIT_OUTPUT"
SPLIT_TIME=$(echo "$SPLIT_OUTPUT" | grep 'rows in' | sed -n 's/.*in \([0-9.]*\)s.*/\1/p')
SPLIT_RPS=$(echo "$SPLIT_OUTPUT" | grep 'rows in' | sed -n 's/.*(\([0-9.]*\) rows\/s)/\1/p')

# Run signature mode benchmark
echo ""
echo "=== SIGNATURE MODE ==="
SIG_OUTPUT=$(uv run examples/data_signature.py 2>&1)
echo "$SIG_OUTPUT"
SIG_TIME=$(echo "$SIG_OUTPUT" | grep 'rows in' | sed -n 's/.*in \([0-9.]*\)s.*/\1/p')
SIG_RPS=$(echo "$SIG_OUTPUT" | grep 'rows in' | sed -n 's/.*(\([0-9.]*\) rows\/s)/\1/p')

# Primary metric: average of both modes
if [ -n "$SPLIT_TIME" ] && [ -n "$SIG_TIME" ]; then
    AVG_TIME=$(echo "($SPLIT_TIME + $SIG_TIME) / 2" | bc -l)
    echo ""
    echo "METRIC total_seconds=$AVG_TIME"
    echo "METRIC split_seconds=$SPLIT_TIME"
    echo "METRIC signature_seconds=$SIG_TIME"
fi
if [ -n "$SPLIT_RPS" ]; then
    echo "METRIC split_rows_per_sec=$SPLIT_RPS"
fi
if [ -n "$SIG_RPS" ]; then
    echo "METRIC signature_rows_per_sec=$SIG_RPS"
fi
