# Autoresearch: Speed up AutoData row generation

## Objective
Make the AutoData row generation pipeline faster. The primary bottleneck is in `_generate_outputs()` in `generator.py`, which:
1. Processes batches **sequentially** (unlike `_generate_inputs` which uses `dspy.Parallel`)
2. Makes **per-row LLM judge calls** sequentially (each row = 1 extra LLM call)
3. Uses small effective batch sizes with no concurrency

The benchmark (`examples/data_split.py`) generates 100 rows (200 with oversampling) for a ticket classification task. Current baseline: ~139s, ~0.7 rows/s. Input generation is fast (~11s for 200 rows at 16.9 row/s). Output generation takes ~127s for 200 rows at 1.57 row/s.

## Metrics
- **Primary**: total_seconds (seconds, lower is better) — wall-clock time for the full generate() call
- **Secondary**: rows_per_sec — throughput

## How to Run
`./.auto/measure.sh` — runs `uv run examples/data_split.py` and outputs `METRIC name=value` lines.

## Files in Scope
- `src/dspy_auto_gepa/generator.py` — Main generation logic. Contains `_generate_outputs()` (the bottleneck), `_generate_inputs()` (reference for parallel pattern), `_generate_signature_mode()`, and the `AutoData` class.
- `src/dspy_auto_gepa/config.py` — `AutoDataConfig` dataclass. May need new config fields.
- `src/dspy_auto_gepa/quality.py` — `LLMJudge` class. Currently calls LLM synchronously per row.
- `examples/data_split.py` — The benchmark example. Do not modify.

## Off Limits
- `examples/data_split.py` — the benchmark itself; must not be modified
- Do not change the data schema, output format, or validation logic
- Do not reduce output quality (judge must still run if enabled)
- Do not cheat by caching or short-circuiting the generation

## Constraints
- Tests must pass (`uv run pytest`)
- No new dependencies (use stdlib: `concurrent.futures`, `dspy.Parallel`, etc.)
- The judge must still be called for each row when `judge_enabled=True`
- Output row quality must not degrade — same validators, same sanitization

## What's Been Tried
- Prior experiments increased request_row_cap from 14→16→18→20→22 for input and signature generation. These helped input throughput but didn't touch the output generation bottleneck.

## Key Optimization Opportunities
1. **Parallelize `_generate_outputs`**: Use `dspy.Parallel` to process multiple input batches concurrently (like `_generate_inputs` does)
2. **Concurrent judge scoring**: Use `ThreadPoolExecutor` to score all rows in a batch concurrently instead of sequentially
3. **Increase batch sizes**: `batch_size` in `_generate_outputs` is `min(24, max(10, chunk_size * 2))` = 20. Can increase.
4. **Batch judge calls**: Score multiple rows in a single LLM call instead of one per row (requires changing LLMJudge/quality.py)
