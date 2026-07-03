# Autoresearch: Speed up AutoData row generation

## Objective
Make the AutoData row generation pipeline faster while ensuring balanced, accurate output distribution for classification tasks.

## Metrics
- **Primary**: total_seconds (seconds, lower is better) — average of split + signature mode
- **Secondary**: rows_per_sec — throughput

## How to Run
`./.auto/measure.sh` — runs both split and signature mode examples, outputs `METRIC name=value` lines.

## Files in Scope
- `src/dspy_auto_gepa/generator.py` — Main generation logic
- `src/dspy_auto_gepa/config.py` — `AutoDataConfig` dataclass
- `src/dspy_auto_gepa/quality.py` — `LLMJudge` class with batch scoring

## What's Been Tried

### Final State: 134.4s → 7.05s (19x speedup)
- Split mode: 7.7s (was ~140s)
- Signature mode: 6.4s
- Distribution: urgency 36/36/28, sentiment 36/36/28 (balanced, accurate)

### Key Breakthrough: Targeted Generation
Instead of generating inputs then outputs then subsampling for balance, **work backwards from target outputs**:
1. Pre-compute all output combos (e.g., 9 combos for urgency×sentiment)
2. For each combo, generate inputs that naturally produce that output
3. Assign target output directly — no output generation step needed

This is fundamentally better because:
- **Guarantees balanced distribution** (each combo gets n/combos rows)
- **Skips output generation entirely** (major speedup)
- **Accurate values** (inputs generated to match target outputs, not forced)

### Speed Optimizations (kept)
1. **Parallelize `_generate_outputs`** with `dspy.Parallel` (for non-balanced mode)
2. **Batch judge scoring** (10 per call via ThreadPoolExecutor)
3. **Increase max_inflight** from 4 to num_threads
4. **Increase request_row_cap** from 22 to 40

### Distribution Approach
- **Targeted generation** for balanced mode (split with `balance_outputs=True`)
- **Mild diversity guidance** in output prompts: "prefer the less common one when input could fit multiple values" — encourages variety without forcing inaccurate values
- **Normalized deficit scoring** in `_subsample_balanced` for non-targeted fallback

### What Failed
- **Aggressive diversity prompts** ("use ALL values with equal frequency") — forces inaccurate values
- **No diversity guidance** — LLM generates 60% negative, 1% positive
- **Async judge scoring** — deadlocks with dspy.Parallel
- **batch_size=32** for output gen — no improvement
- **Oversample 4x** — slower, not needed with targeted generation
