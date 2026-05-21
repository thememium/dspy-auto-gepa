# Advanced Usage: Full Manual Control

For when you need to orchestrate every step yourself — custom splitting, iterative metric refinement, multiple baselines, or debugging a stubborn optimization.

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
    # ... more rows
]

# Step 1: Configure
auto = AutoGEPA(
    input_fields=["message"],
    output_fields=["urgency", "sentiment"],
    metric_lm=large_lm,
    reflection_lm=large_lm,
    gepa_auto="medium",  # heavier optimization
)
```

## Step 2: Generate and review metric

```python
metric_path = auto.build_metric(
    rows=rows,
    module=program,
    name="TicketSignature",
)
print(f"Metric: {metric_path}")
# ... human reviews and possibly edits the .py file ...
```

## Step 3: Build datasets

```python
ds = auto.datasets(
    rows=rows,
    module=program,
    name="TicketSignature",
    metric=metric_path,  # use reviewed metric
)

print(f"Train: {len(ds.train)}, Val: {len(ds.val)}, Test: {len(ds.test)}")
```

## Step 4: Baseline

```python
baseline = auto.run_baseline(datasets=ds)
print(f"Baseline score: {baseline['score']:.4f}")
```

## Step 5: Train

```python
optimized = auto.train(datasets=ds)
```

## Step 6: Compare

```python
comparison = auto.compare(
    baseline_module=program,
    optimized_module=optimized,
    datasets=ds,
)
print(f"Baseline: {comparison.baseline:.4f}")
print(f"Optimized: {comparison.optimized:.4f}")
print(f"Improvement: {comparison.improvement:.4f}")
```

## Step 7: Save

```python
model_path = auto._run_dir / "optimized_ticket.json"
auto.promote(optimized_module=optimized, destination=model_path)
print(f"Saved to: {model_path}")
```

## Full pipeline in one block

```python
auto = AutoGEPA(
    input_fields=["message"],
    output_fields=["urgency", "sentiment"],
    metric_lm=large_lm,
    reflection_lm=large_lm,
)

metric_path = auto.build_metric(rows=rows, module=program, name="TicketSignature")
# review metric_path ...

ds = auto.datasets(rows=rows, module=program, name="TicketSignature", metric=metric_path)
baseline = auto.run_baseline(datasets=ds)
optimized = auto.train(datasets=ds)
comparison = auto.compare(baseline_module=program, optimized_module=optimized, datasets=ds)
auto.promote(optimized_module=optimized, destination=auto._run_dir / "optimized.json")
```

## With custom metric generator

```python
class MyMetricSignature(dspy.Signature):
    input_keys: list[str] = dspy.InputField()
    output_keys: list[str] = dspy.InputField()
    sample_rows_json: str = dspy.InputField()
    module_repr: str = dspy.InputField()
    metric_source: str = dspy.OutputField()

auto = AutoGEPA(
    input_fields=["message"],
    output_fields=["urgency", "sentiment"],
    metric_generator_signature=MyMetricSignature,
    metric_generator_module=dspy.ChainOfThought,
    metric_lm=large_lm,
    reflection_lm=large_lm,
)

metric_path = auto.build_metric(rows=rows, module=program, name="TicketSignature")
ds = auto.datasets(rows=rows, module=program, name="TicketSignature", metric=metric_path)
baseline = auto.run_baseline(datasets=ds)
optimized = auto.train(datasets=ds)
```
