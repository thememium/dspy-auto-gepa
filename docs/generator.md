# Synthetic Data Generation

AutoGEPA includes `AutoData`, a synthetic data generator that uses LLMs to create training datasets from scratch or from seed examples.

## Basic Usage

```python
import dspy
from dspy_auto_gepa import AutoData

lm = dspy.LM("openrouter/openai/gpt-oss-120b")
dspy.configure(lm=lm)

class TicketSignature(dspy.Signature):
    """Classify support tickets."""
    message: str = dspy.InputField()
    urgency: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()

program = dspy.ChainOfThought(TicketSignature)

gen = AutoData(module=program, data_lm=lm)
result = gen.generate(n=50)
print(f"Generated {result.n_produced} rows")
```

## With Seed Examples

```python
gen = AutoData.from_csv("seeds.csv", module=program, data_lm=lm)
result = gen.generate(n=100)
```

## Custom Output Path

Output format is auto-detected from file extension (.jsonl, .csv, .parquet):

```python
result = gen.generate(n=100, output_path="data/train.parquet")
```

## Integration with AutoGEPA

```python
from dspy_auto_gepa import AutoGEPA

auto = AutoGEPA(module=program, metric_lm=lm)
rows = auto.generate(n=100, data_lm=lm)
results = auto.run(rows=rows)
```
