# Medium Usage: Generate, Review, Then Run

For when you want to inspect the LLM-generated metric before paying for a full GEPA optimization run.

```python
import dspy
from dspy_auto_gepa import AutoGEPA

class TicketSignature(dspy.Signature):
    """Classify support tickets."""
    message: str = dspy.InputField()
    urgency: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()

program = dspy.ChainOfThought(TicketSignature)

lm = dspy.LM("openrouter/openai/gpt-oss-120b")
large_lm = dspy.LM("openrouter/moonshotai/kimi-k2.5")
dspy.configure(lm=lm)

rows = [
    {"message": "Server room AC is out.", "urgency": "high", "sentiment": "negative"},
    {"message": "Clean conference room B?", "urgency": "low", "sentiment": "neutral"},
]

# Step 1: Set up AutoGEPA with your config
auto = AutoGEPA(
    input_fields=["message"],
    output_fields=["urgency", "sentiment"],
    metric_lm=large_lm,
    reflection_lm=large_lm,
)

# Step 2: Generate the metric file for human review
metric_path = auto.build_metric(
    rows=rows,
    module=program,
    name="TicketSignature",
)
print(f"Metric generated: {metric_path}")
```

## Review the metric

Open `metric_path` (e.g., `.auto_gepa/TicketSignature/metric.py`) and inspect the generated code. Edit as needed:

```python
# Example generated metric
import re

def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    score = 0.0
    feedback_parts = []

    if example.urgency == pred.urgency:
        score += 0.5
    else:
        feedback_parts.append(
            f"Expected urgency={example.urgency}, got {pred.urgency}."
        )

    if example.sentiment == pred.sentiment:
        score += 0.5
    else:
        feedback_parts.append(
            f"Expected sentiment={example.sentiment}, got {pred.sentiment}."
        )

    if pred_name is not None:
        return dspy.Prediction(score=score, feedback=" ".join(feedback_parts))
    return score
```

## Step 3: Run with the reviewed metric

Once you're satisfied with the metric, pass it back into `run()`:

```python
results = auto.run(
    rows=rows,
    module=program,
    name="TicketSignature",
    metric=metric_path,  # use reviewed metric, skip regeneration
)

print(f"Baseline: {results.baseline:.4f}")
print(f"Optimized: {results.optimized:.4f}")
print(f"Improvement: {results.improvement:.4f}")
```

## Custom save location

You can also save the metric to a specific path:

```python
metric_path = auto.build_metric(
    rows=rows,
    module=program,
    name="TicketSignature",
    out_path="/tmp/my_custom_metric.py",  # custom save location
)
```

## Custom metric generator

Swap the LLM generator that writes the metric:

```python
class MyMetricSignature(dspy.Signature):
    """Generate a Python metric function."""
    input_keys: list[str] = dspy.InputField()
    output_keys: list[str] = dspy.InputField()
    sample_rows_json: str = dspy.InputField()
    module_repr: str = dspy.InputField()
    metric_source: str = dspy.OutputField()

auto = AutoGEPA(
    input_fields=["message"],
    output_fields=["urgency", "sentiment"],
    metric_generator_signature=MyMetricSignature,     # custom prompt schema
    metric_generator_module=dspy.ChainOfThought,      # different generator
    metric_lm=large_lm,
)

metric_path = auto.build_metric(rows=rows, module=program, name="TicketSignature")
```
