import ast
import json
from pathlib import Path
from typing import Any

import dspy


class MetricSpecGenerator(dspy.Signature):
    """Create a Python DSPy metric function for GEPA optimization.

    The metric must compare gold example fields against prediction fields and
    return either a scalar score or a dspy.Prediction with feedback.

    Requirements:
    - Define a top-level function named `metric`.
    - Signature: metric(example, pred, trace=None, pred_name=None, pred_trace=None)
    - When pred_name is None, return a float score in [0.0, 1.0].
    - When pred_name is provided, return dspy.Prediction(score=float, feedback=str).
    - The feedback string must explain concrete errors and suggest improvements.
    - Use exact string matching for enumerated fields; semantic similarity for free-text fields.
    - Include any necessary imports inside the function body or at module top.

    Example output for a classification task with fields urgency and sentiment:

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        import re
        score = 0.0
        feedback_parts = []

        if example.urgency == pred.urgency:
            score += 0.5
        else:
            feedback_parts.append(f"Expected urgency={example.urgency}, got {pred.urgency}.")

        if example.sentiment == pred.sentiment:
            score += 0.5
        else:
            feedback_parts.append(f"Expected sentiment={example.sentiment}, got {pred.sentiment}.")

        if pred_name is not None:
            return dspy.Prediction(score=score, feedback=" ".join(feedback_parts))
        return score
    """

    input_keys: list[str] = dspy.InputField()
    output_keys: list[str] = dspy.InputField()
    sample_rows_json: str = dspy.InputField()
    module_repr: str = dspy.InputField()
    metric_source: str = dspy.OutputField()


def _strip_markdown_fences(source: str) -> str:
    lines = source.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def generate_metric_file(
    *,
    input_fields: list[str],
    output_fields: list[str],
    sample_rows: list[dict[str, Any]],
    module: dspy.Module,
    out_path: Path,
    metric_lm: dspy.LM | None = None,
    metric_generator_signature: Any = MetricSpecGenerator,
    metric_generator_module: Any = dspy.RLM,
) -> Path:
    generator = metric_generator_module(metric_generator_signature)

    if metric_lm is not None:
        generator.set_lm(metric_lm)

    result = generator(
        input_keys=input_fields,
        output_keys=output_fields,
        sample_rows_json=json.dumps(sample_rows[:5], indent=2, default=str),
        module_repr=repr(module),
    )

    source = _strip_markdown_fences(result.metric_source)

    try:
        ast.parse(source)
    except SyntaxError:
        raise ValueError(
            f"Generated metric is not valid Python code:\n{source[:500]}..."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(source)
    return out_path
