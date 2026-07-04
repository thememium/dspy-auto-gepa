# Autoresearch: Speed up AutoData row generation

## Final State
- **Before**: 134.4s, distribution skewed (sentiment: 43/42/15, some inaccurate forced values)
- **After**: ~3.5-5.7s (24-39x speedup), distribution balanced (36/36/28), values accurate
- Split mode: ~2s (was ~140s)
- All 132 tests pass

## Key Breakthrough: Targeted Generation
Instead of: generate inputs → generate outputs → subsample for balance  
Do: **pre-compute output combos → generate inputs that naturally produce each combo → assign outputs directly**

Benefits:
1. **Guarantees balanced distribution** (each combo gets n/combos rows)
2. **Skips output generation entirely** (major speedup — eliminates slowest phase)
3. **Accurate values** (inputs generated to match target outputs, not forced)
4. **Parallel combo processing** (all combos processed simultaneously)

## What Worked
1. `_TargetedInputGenerationSignature` — generates inputs backwards from target outputs
2. Parallel combo generation — all 9 combos processed in single `dspy.Parallel` call
3. `dspy.Parallel` for output gen (non-balanced mode)
4. Batch judge scoring (10 per call via ThreadPoolExecutor)
5. `max_inflight=num_threads` for all generation paths
6. `request_row_cap=40` (fewer round trips)
7. Mild diversity guidance: "prefer the less common one when input could fit multiple values"
8. Normalized deficit scoring in `_subsample_balanced`

## What Failed
- **Aggressive diversity prompts** ("use ALL values with equal frequency") — forces inaccurate values
- **No diversity guidance** — LLM generates 60% negative, 1% positive
- **Async judge scoring** — deadlocks with dspy.Parallel
- **batch_size=32** — no improvement
- **Oversample 4x** — slower, not needed with targeted generation

## Files Changed
- `src/dspy_auto_gepa/generator.py` — targeted generation, parallel output gen, batch judge
- `src/dspy_auto_gepa/config.py` — `oversample_factor=2.0` (default)
- `src/dspy_auto_gepa/quality.py` — `batch_score()`, `_BatchJudgeSignature`
