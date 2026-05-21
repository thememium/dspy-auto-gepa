# AutoGEPA

Thin orchestration package around [DSPy](https://dspy.ai)'s `GEPA` optimizer. AutoGEPA automates the boring parts of setting up a DSPy optimization pipeline: converting raw data into `dspy.Example`s, generating a metric file with an LLM, splitting datasets, running baselines, and training with GEPA.

## Design principle

**LLM-generated evals are drafts, not truth.** AutoGEPA generates the metric, saves it as a reproducible `.py` file, and expects you to inspect it before expensive GEPA runs.

## Install

```bash
pip install dspy-auto-gepa
```

## Usage

```python
import dspy
from dspy_auto_gepa import AutoGEPA

class TicketSignature(dspy.Signature):
    """Classify support tickets."""
    message: str = dspy.InputField()
    urgency: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()

program = dspy.ChainOfThought(TicketSignature)

rows = [
    {
        "message": "The server room AC is out and equipment is overheating.",
        "urgency": "high",
        "sentiment": "negative",
    },
    {
        "message": "Can someone clean conference room B next week?",
        "urgency": "low",
        "sentiment": "neutral",
    },
]

# Configure models
lm = dspy.LM("openrouter/openai/gpt-oss-120b")
large_lm = dspy.LM("openrouter/moonshotai/kimi-k2.5")
dspy.configure(lm=lm)

auto = AutoGEPA(
    rows=rows,
    module=program,
    name="TicketSignature",
    input_fields=["message"],
    output_fields=["urgency", "sentiment"],
    metric_lm=large_lm,
    reflection_lm=large_lm,
)

results = auto.run(force=False)  # Set True to re-run even if a saved model exists

# Check if a cached model was loaded
if results.loaded_from:
    print(f"Loaded existing model from {results.loaded_from}")
else:
    print(f"Baseline score: {results.baseline:.4f}")
    print(f"Optimized score: {results.optimized:.4f}")
    print(f"Improvement: {results.improvement:.4f}")
    print(f"Saved optimized program to {results.saved_to}")
```

### Advanced: step-by-step control

If you prefer fine-grained control over each stage, you can call the individual
methods that `run()` orchestrates under the hood:

```python
prepared = auto.prepare()

baseline = auto.run_baseline(prepared=prepared)

optimized = auto.train(prepared=prepared)

final = auto.run_baseline(module=optimized, prepared=prepared)

# Or compare and promote
comparison = auto.compare(
    optimized_module=optimized,
    prepared=prepared,
)
auto.promote(
    optimized_module=optimized,
    destination=prepared.run_dir / "optimized_ticket_classifier.json",
)
```

## API

- `AutoGEPA(...)` — all configuration fields accepted directly in the constructor:
  - `input_fields: list[str]` — required
  - `output_fields: list[str]` — required
  - `rows: Any | None = None` — training data, accepts `list[dict]`, pandas DataFrame, polars DataFrame/LazyFrame, or any object with `.to_dicts()` or `.to_pandas()`
  - `module: dspy.Module | None = None` — the DSPy module to optimize
  - `name: str | None = None` — task name for artifact subdirectory
  - `metric: Path | str | None = None` — path to a custom metric `.py` file (skips generation)
  - `split: tuple[float, ...] = (0.7, 0.2, 0.1)`
  - `seed: int = 42`
  - `artifact_dir: Path | str = ".auto_gepa"`
  - `metric_lm: dspy.LM | None = None` — defaults to `dspy.LM("openrouter/openai/gpt-oss-120b")`
  - `reflection_lm: dspy.LM | None = None` — defaults to `dspy.LM("openrouter/moonshotai/kimi-k2.5")`
  - `gepa_auto: Literal["light", "medium", "heavy"] = "light"`
  - `num_threads: int = 16`
- `AutoGEPA.run(rows=None, module=None, name=None, metric=None, force=False)` → `RunResult`
  - Orchestrates the full pipeline: prepare → baseline → train → compare → promote.
  - Uses `rows`, `module`, `name`, `metric` from the constructor if not overridden.
  - If `force=False` and a saved model exists at `.auto_gepa/<name>/optimized_<name>.json`, loads it and skips training.
  - Returns a `RunResult` with `baseline`, `optimized`, `improvement`, `saved_to` (or `loaded_from` if cached).
- `AutoGEPA.prepare(rows=None, module=None, name=None, metric=None, force=False)` → `PreparedRun`
  - Uses `rows`, `module`, `name`, `metric` from the constructor if not overridden.
  - `name` sets the artifact subdirectory. Defaults to `module.__class__.__name__`.
  - `force=True` overwrites an existing metric file
- `AutoGEPA.run_baseline(module=None, prepared)` → baseline scores
  - Uses `module` from the constructor if not overridden.
- `AutoGEPA.train(module=None, prepared)` → optimized module
  - Uses `module` from the constructor if not overridden.
- `AutoGEPA.compare(optimized_module, prepared, baseline_module=None)` → side-by-side scores
  - Uses `module` from the constructor as `baseline_module` if not overridden.
- `AutoGEPA.promote(optimized_module, destination)` → save optimized program
- `PreparedRun.metric()` → lazily loads the generated metric
- `PreparedRun.train` / `PreparedRun.val` / `PreparedRun.test` → dataset splits
- `PreparedRun.run_dir` → artifact folder for this run

## License

MIT
