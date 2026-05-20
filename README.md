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
from dspy_auto_gepa import AutoGEPA, AutoGEPAConfig, load_metric

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

dspy.configure(
    lm=dspy.LM("openrouter/google/gemini-2.5-flash-lite")
)

auto = AutoGEPA(
    AutoGEPAConfig(
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
        split=(0.7, 0.2, 0.1),
        gepa_auto="light",
    )
)

prepared = auto.prepare(rows=rows, module=program)
metric = load_metric(prepared["metric_file"])

baseline = auto.run_baseline(
    module=program,
    testset=prepared["test"],
    metric=metric,
)

optimized = auto.train(
    module=program,
    trainset=prepared["train"],
    valset=prepared["val"],
    metric=metric,
)

final = auto.run_baseline(
    module=optimized,
    testset=prepared["test"],
    metric=metric,
)

# Or compare and promote
comparison = auto.compare(
    baseline_module=program,
    optimized_module=optimized,
    testset=prepared["test"],
    metric=metric,
)
auto.promote(optimized_module=optimized, destination="optimized_ticket_classifier.json")
```

## API

- `AutoGEPAConfig` — task settings, split, models, artifact directory
- `AutoGEPA.prepare(rows, module)` → train/val/test + generated metric file
- `AutoGEPA.run_baseline(module, testset, metric)` → baseline scores
- `AutoGEPA.train(module, trainset, valset, metric)` → optimized module
- `AutoGEPA.compare(baseline_module, optimized_module, testset, metric)` → side-by-side scores
- `AutoGEPA.promote(optimized_module, destination)` → save optimized program
- `load_metric(path)` → import a generated metric `.py` file dynamically

## License

MIT
