# Basic Usage: One-Shot

The simplest way to optimize a DSPy module with AutoGEPA. Everything happens in one call.

When your row columns match your module's signature fields, you don't need to configure anything.

```python
import dspy
from dspy_auto_gepa import AutoGEPA

class TicketSignature(dspy.Signature):
    """Classify support tickets."""
    message: str = dspy.InputField()
    urgency: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()

program = dspy.ChainOfThought(TicketSignature)

# Configure models
lm = dspy.LM("openrouter/openai/gpt-oss-120b")
large_lm = dspy.LM("openrouter/moonshotai/kimi-k2.5")
dspy.configure(lm=lm)

# Load your data — column names match signature fields
rows = [
    {"message": "Server room AC is out.", "urgency": "high", "sentiment": "negative"},
    {"message": "Clean conference room B?", "urgency": "low", "sentiment": "neutral"},
    # ... more rows
]

# One-shot: fields are inferred from the module's signature
auto = AutoGEPA(
    rows=rows,
    module=program,
    name="TicketSignature",
    metric_lm=large_lm,
    reflection_lm=large_lm,
)

results = auto.run()

if results.loaded_from:
    print(f"Loaded cached model from {results.loaded_from}")
else:
    print(f"Baseline: {results.baseline:.4f}")
    print(f"Optimized: {results.optimized:.4f}")
    print(f"Improvement: {results.improvement:.4f}")
    print(f"Saved to: {results.saved_to}")
```

## What happens under the hood

1. Fields are inferred from the module's signature
2. Row columns are checked against signature fields — exact match required
3. `datasets()` — converts rows to `dspy.Example`s, splits train/val/test, generates a metric `.py` file
4. `run_baseline()` — evaluates your unoptimized module
5. `train()` — runs GEPA optimization
6. `compare()` — scores baseline vs. optimized
7. `promote()` — saves the optimized module to `.auto_gepa/TicketSignature/optimized_TicketSignature.json`

## Cache behavior

If a saved model exists and `force=False`, AutoGEPA loads it and skips training:

```python
results = auto.run(force=False)  # Load cached model if available
results = auto.run(force=True)   # Always retrain from scratch
```

## Custom metric

Pass a pre-written metric file to skip LLM generation entirely:

```python
auto = AutoGEPA(
    rows=rows,
    module=program,
    name="TicketSignature",
    metric="/path/to/my_metric.py",  # skip generation, use this
)

results = auto.run()
```

## If columns don't match

If your row columns have different names, AutoGEPA raises a clear error:

```
ValueError: Row columns do not match module signature fields.
Missing from rows: ['message', 'sentiment', 'urgency'].
Extra in rows: ['msg_text', 'sent', 'urg'].
Pass input_fields/output_fields to map row columns to signature fields,
or ensure row columns match exactly.
```

See [medium.md](medium.md) for dict mapping.
