# Autoresearch: Speed up AutoData row generation

## Objective
Make the AutoData row generation pipeline faster while ensuring balanced output distribution for classification tasks. 

## Metrics
- **Primary**: total_seconds (seconds, lower is better) — average of split + signature mode
- **Secondary**: rows_per_sec — throughput

## How to Run
`./.auto/measure.sh` — runs both split and signature mode examples, outputs `METRIC name=value` lines.

## Files in Scope
- `src/dspy_auto_gepa/generator.py` — Main generation logic
- `src/dspy_auto_gepa/config.py` — `AutoDataConfig` dataclass
- `src/dspy_auto_gepa/quality.py` — `LLMJudge` class with batch scoring

## Off Limits
- `examples/data_split.py`, `examples/data_signature.py` — benchmarks
- Do not change the data schema, output format, or validation logic
- Do not reduce output quality

## Constraints
- Tests must pass (`uv run pytest`)
- No new dependencies
- Judge must still be called when enabled
- Output distribution must be balanced for classification tasks

## What's Been Tried

### Speed Optimizations (kept)
1. **Parallelize `_generate_outputs`** with `dspy.Parallel` — 5.3x speedup on output gen
2. **Concurrent judge scoring** via ThreadPoolExecutor — further speedup
3. **Increase max_inflight** from 4 to num_threads for all generation paths
4. **Increase request_row_cap** from 22 to 40 — fewer round trips
5. **Batch judge scoring** (10 per call) — roughly equivalent to individual

### Speed Optimizations (discarded)
- batch_size=32 for output gen — no improvement (longer per-call offset by fewer trips)
- max_inflight=num_threads for input gen — input gen was already fast

### Distribution Fixes (kept)
1. **Diversity prompts** in `_OutputGenerationSignature` and `_BatchOutputSignature` — THE key fix
2. **Normalized deficit scoring** in `_subsample_balanced` — better multi-field balancing
3. **Oversample factor** kept at 2.0 (diversity prompts alone fix the distribution)

### Baseline → Current
- Before: 134.4s, distribution skewed (sentiment: 43/42/15)
- After: ~17-22s, distribution balanced (33/33/34 across all fields)
- Speedup: ~7-8x with balanced output

## Key Insight
The diversity prompt changes are the primary fix for distribution. The LLM naturally generates skewed outputs for classification tasks (more negative sentiment for support tickets). Adding explicit "vary your output values" instructions to the generation signatures makes the LLM distribute values more evenly, even with 2x oversampling.
