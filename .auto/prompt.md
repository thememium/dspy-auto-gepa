# Autoresearch: faster, safer AutoData generation

## Objective
Speed up the AutoData synthetic data generation pipeline in `src/dspy_auto_gepa/generator.py` while also reducing the risk of upstream LLM/API rate limits and preserving or improving generated-data quality. Keep the design idiomatic to DSPy. Take inspiration from industrial synthetic-data pipelines (staged generation, bounded concurrency, quality-aware filtering, diversity/balance controls), but do not overfit to the benchmark and do not cheat by weakening validation or quality requirements.

## Metrics
- **Primary**: `total_ms` (ms, lower is better) — median benchmark runtime across representative mocked AutoData workloads
- **Secondary**:
  - `produced_pct` — percent of requested rows successfully produced
  - `diversity_score` — lower pairwise similarity / higher diversity proxy
  - `balance_error_pct` — categorical output imbalance percentage
  - `rate_limit_failures` — simulated 429-style failures in the mocked workload
  - `split_ms`, `signature_ms` — per-mode timings

## How to Run
`./.auto/measure.sh`

The benchmark uses deterministic mocked DSPy predictors and a simulated bounded-concurrency backend. It exercises real AutoData code paths for:
- split mode input generation
- split mode output generation with balancing/oversampling
- signature mode generation
- file writing / resume-safe behavior

It intentionally models a realistic failure mode where excessive parallel request bursts cause simulated rate-limit failures. Quality metrics check produced row count, diversity, and class balance.

## Files in Scope
- `src/dspy_auto_gepa/generator.py` — main optimization target
- `src/dspy_auto_gepa/config.py` — config knobs for safer/faster generation
- `src/dspy_auto_gepa/quality.py` — quality helpers if needed
- `tests/test_generator.py` — behavior tests for generation pipeline
- `tests/test_quality.py` — quality helper tests
- `.auto/measure.sh` — benchmark harness
- `.auto/bench_autodata.py` — benchmark implementation
- `.auto/checks.sh` — correctness checks

## Off Limits
- Public API changes unrelated to AutoData generation
- Benchmark-only shortcuts inside library code
- Disabling validation/quality logic just to improve timing
- Any changes that require live external API access for tests/benchmark

## Constraints
- Keep behavior deterministic in tests/benchmark
- Preserve or improve data quality semantics
- Avoid benchmark cheating; improvements should plausibly help real DSPy-style synthetic data generation
- Tests in `tests/test_generator.py` and `tests/test_quality.py` must pass
- Keep the implementation readable and DSPy-style

## What's Been Tried
- Baseline pending.
- Initial hypotheses:
  - duplicate detection is likely O(n^2) and avoidable with cached fingerprints
  - current burst size may exceed safe upstream concurrency and cause retry waste / 429 risk
  - oversample+balance path may write too many intermediate rows before final selection
  - quality-aware balanced subsampling may improve output quality without extra model calls
