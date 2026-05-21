<a name="readme-top"></a>

<div align="center">
  <h3 align="center">AutoGEPA</h3>

  <p align="center">
    Thin orchestration package around <a href="https://dspy.ai">DSPy</a>'s <code>GEPA</code> optimizer.
    <br />
    <a href="#table-of-contents"><strong>Explore the Documentation »</strong></a>
    <br />
    <a href="https://github.com/thememium/dspy-auto-gepa/issues">Report Bug</a>
    <a href="https://github.com/thememium/dspy-auto-gepa/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->

<a name="table-of-contents"></a>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#api">API</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT -->

<a name="about"></a>

## About

AutoGEPA automates the tedious parts of setting up a DSPy optimization pipeline: converting raw data into `dspy.Example`s, generating a metric file with an LLM, splitting datasets, running baselines, and training with GEPA.

- **Automatic field inference** — Maps row columns to DSPy Signature fields automatically
- **LLM-generated metrics** — Drafts evaluation metrics automatically, saving them as reproducible `.py` files for human review
- **End-to-end pipeline** — Datasets → Baseline → GEPA optimization → Compare & promote
- **Zero-config defaults** — Sensible defaults for all hyperparameters and LMs

Requires **DSPy 3.1+**.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- QUICK START -->

<a name="quick-start"></a>

## Quick Start

### Install

Install AutoGEPA with uv (recommended):

```bash
uv add dspy-auto-gepa
```

Or with pip:

```bash
pip install dspy-auto-gepa
```

### Basic Usage

```python
import dspy
from dspy_auto_gepa import AutoGEPA

# Configure models
lm = dspy.LM("openrouter/openai/gpt-oss-120b")
large_lm = dspy.LM("openrouter/moonshotai/kimi-k2.5")
dspy.configure(lm=lm)

class TicketSignature(dspy.Signature):
    """Classify support tickets."""
    message: str = dspy.InputField()
    urgency: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()

program = dspy.ChainOfThought(TicketSignature)

rows = [
    {"message": "The server room AC is out and equipment is overheating.", "urgency": "high", "sentiment": "negative"},
    {"message": "Can someone clean conference room B next week?", "urgency": "low", "sentiment": "neutral"},
]

# Fields are automatically inferred from the module's signature
auto = AutoGEPA(
    rows=rows,
    module=program,
    name="TicketSignature",
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

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->

<a name="usage"></a>

## Usage

### Automatic field inference (recommended)

When your row columns match your module's signature fields, you don't need to specify any field mappings. AutoGEPA infers input and output fields directly from the `dspy.Signature` attached to your module.

If your row columns don't match the module's signature fields, AutoGEPA will raise a clear error telling you which fields are missing and suggesting you use dict mappings:

```
ValueError: Row columns do not match module signature fields. Missing from rows: ['message', 'sentiment', 'urgency']. Extra in rows: ['msg_text', 'sent', 'urg']. Pass input_fields/output_fields to map row columns to signature fields, or ensure row columns match exactly.
```

### With dict field mappings

When your row columns have different names than your module's signature fields:

```python
# Row columns: msg_text, urg, sent
# Signature fields: message, urgency, sentiment

auto = AutoGEPA(
    rows=rows,
    module=program,
    name="TicketSignature",
    input_fields={"msg_text": "message"},    # row_col → sig_field
    output_fields={"urg": "urgency", "sent": "sentiment"},
    metric_lm=large_lm,
    reflection_lm=large_lm,
)

results = auto.run()
```

### Advanced: step-by-step control

If you prefer fine-grained control over each stage, you can call the individual methods that `run()` orchestrates under the hood:

```python
# Optional: generate the metric file first for human inspection
metric_file = auto.build_metric()
print(f"Metric written to {metric_file}")
# After reviewing, proceed:

ds = auto.datasets()

baseline = auto.run_baseline(datasets=ds)

optimized = auto.train(datasets=ds)

final = auto.run_baseline(module=optimized, datasets=ds)

# Or compare and promote
comparison = auto.compare(
    optimized_module=optimized,
    datasets=ds,
)
auto.promote(
    optimized_module=optimized,
    destination=auto._run_dir / "optimized_ticket_classifier.json",
)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- API -->

<a name="api"></a>

## API

### Constructor

`AutoGEPA(...)` accepts all configuration fields directly:

| Parameter | Type | Description |
| --- | --- | --- |
| `rows` | `list[dict] \| DataFrame \| None` | Training data. Accepts `list[dict]`, pandas DataFrame, polars DataFrame/LazyFrame, or any object with `.to_dicts()` or `.to_pandas()` |
| `module` | `dspy.Module \| None` | The DSPy module to optimize |
| `name` | `str \| None` | Task name for artifact subdirectory |
| `input_fields` | `list[str] \| dict[str, str] \| None` | Input field names. List for exact match, dict for `{row_column: signature_field}` mapping. Inferred from signature if omitted |
| `output_fields` | `list[str] \| dict[str, str] \| None` | Output field names. Same format as `input_fields`. Inferred from signature if omitted |
| `metric` | `Path \| str \| None` | Path to a custom metric `.py` file (skips generation) |
| `split` | `tuple[float, ...]` | Train/val/test split ratios. Default `(0.7, 0.2, 0.1)` |
| `seed` | `int` | Random seed for reproducibility. Default `42` |
| `artifact_dir` | `Path \| str` | Root directory for artifacts. Default `".auto_gepa"` |
| `metric_lm` | `dspy.LM \| None` | LM used for metric generation. Defaults to `dspy.LM("openrouter/openai/gpt-oss-120b")` |
| `reflection_lm` | `dspy.LM \| None` | LM used for GEPA reflection. Defaults to `dspy.LM("openrouter/moonshotai/kimi-k2.5")` |
| `gepa_auto` | `Literal["light", "medium", "heavy"]` | GEPA optimization intensity. Default `"light"` |
| `num_threads` | `int` | Parallel threads for evaluation. Default `16` |
| `metric_generator_signature` | `Type[dspy.Signature] \| None` | Custom signature class for metric generation |
| `metric_generator_module` | `Type[dspy.Module] \| None` | Custom module class for metric generation |

### Methods

| Method | Signature | Description |
| --- | --- | --- |
| `build_metric` | `(rows, module, name, metric, out_path, force=False) → Path` | Generates the metric `.py` file explicitly. Skips if a custom `metric` path is provided. `out_path` overrides the default save location. Use `force=True` to overwrite an existing generated metric |
| `run` | `(rows, module, name, metric, force=False) → RunResult` | Orchestrates the full pipeline: datasets → baseline → train → compare → promote. If `force=False` and a saved model exists at `.auto_gepa/<name>/optimized_<name>.json`, loads it and skips training. Returns a `RunResult` with `baseline`, `optimized`, `improvement`, `saved_to` (or `loaded_from` if cached) |
| `datasets` | `(rows, module, name, metric, force=False) → Datasets` | Converts rows to `dspy.Example`s and splits into train/val/test. Uses constructor defaults if args omitted. `name` sets the artifact subdirectory. `force=True` overwrites an existing metric file |
| `run_baseline` | `(module=None, datasets) → float` | Evaluates the unoptimized module. Uses `module` from constructor if not overridden |
| `train` | `(module=None, datasets) → dspy.Module` | Runs GEPA optimization. Uses `module` from constructor if not overridden |
| `compare` | `(optimized_module, datasets, baseline_module=None) → dict` | Side-by-side score comparison. Uses constructor `module` as `baseline_module` if not overridden |
| `promote` | `(optimized_module, destination) → Path` | Saves the optimized program to the given destination |
| `load_metric` | `() → callable` | Lazily loads the generated metric function |

### Result Types

- `RunResult` — returned by `run()`:
  - `baseline: float` — score before optimization
  - `optimized: float` — score after optimization
  - `improvement: float` — absolute difference
  - `saved_to: Path \| None` — where the optimized program was saved
  - `loaded_from: Path \| None` — if a cached model was loaded instead of training
- `Datasets` — returned by `datasets()`:
  - `train: list[dspy.Example]`
  - `val: list[dspy.Example]`
  - `test: list[dspy.Example]`

### Field Resolution Behaviour

AutoGEPA resolves fields in this order:

1. **Both provided explicitly** (`list[str] | dict[str, str]`) — uses exactly what you gave it. Lists mean exact column names. Dicts mean `{row_column: signature_field}` mapping.
2. **Neither provided** — infers both from the module's DSPy Signature. Raises a clear error if row columns don't match signature fields, listing what's missing and what's extra.
3. **Only one provided** — if the other can be inferred from module signature or remaining row keys, great. If not, raises an error.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

<a name="contributing"></a>

## Contributing

Quick workflow:

1. Fork and branch: `git checkout -b feature/name`
2. Make changes
3. Commit and push
4. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

<a name="license"></a>

## License

MIT (as declared in `pyproject.toml`).

---

<div align="center">
  <p>
    <sub>Built by <a href="https://github.com/thememium">thememium</a></sub>
  </p>
</div>
