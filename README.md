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
metric_lm = dspy.LM("openrouter/openai/gpt-oss-120b")
reflection_lm = dspy.LM("openrouter/moonshotai/kimi-k2.5")
dspy.configure(lm=metric_lm)

auto = AutoGEPA(
    input_fields=["message"],
    output_fields=["urgency", "sentiment"],
    split=(0.7, 0.2, 0.1),
    gepa_auto="light",
    metric_lm=metric_lm,
    reflection_lm=reflection_lm,
)

prepared = auto.prepare(rows=rows, module=program, name="TicketSignature")

baseline = auto.run_baseline(module=program, prepared=prepared)

optimized = auto.train(module=program, prepared=prepared)

final = auto.run_baseline(module=optimized, prepared=prepared)

# Or compare and promote
comparison = auto.compare(
    baseline_module=program,
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
  - `split: tuple[float, ...] = (0.7, 0.2, 0.1)`
  - `seed: int = 42`
  - `artifact_dir: Path | str = ".auto_gepa"`
  - `metric_lm: dspy.LM | None = None` — defaults to `dspy.LM("openrouter/openai/gpt-oss-120b")`
  - `reflection_lm: dspy.LM | None = None` — defaults to `dspy.LM("openrouter/moonshotai/kimi-k2.5")`
  - `gepa_auto: Literal["light", "medium", "heavy"] = "light"`
  - `num_threads: int = 16`
- `AutoGEPA.prepare(rows, module, name=None, force=False)` → `PreparedRun`
  - `name` sets the artifact subdirectory. Defaults to `module.__class__.__name__`. Pass a meaningful name (e.g., `"TicketSignature"`) for readable folders.
  - `force=True` overwrites an existing metric file
- `AutoGEPA.run_baseline(module, prepared)` → baseline scores
- `AutoGEPA.train(module, prepared)` → optimized module
- `AutoGEPA.compare(baseline_module, optimized_module, prepared)` → side-by-side scores
- `AutoGEPA.promote(optimized_module, destination)` → save optimized program
- `PreparedRun.metric()` → lazily loads the generated metric
- `PreparedRun.train` / `PreparedRun.val` / `PreparedRun.test` → dataset splits
- `PreparedRun.run_dir` → artifact folder for this run

## License

MIT
