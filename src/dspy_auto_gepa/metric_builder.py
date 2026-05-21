import ast
import json
from pathlib import Path
from typing import Any

import dspy


class MetricSpecGenerator(dspy.Signature):
    """Create a Python DSPy metric function for GEPA optimization.

    The metric must compare gold example fields against prediction fields and
    ALWAYS return a dspy.Prediction(score=float, feedback=str).

    CRITICAL RULES (violations break GEPA):
    1. ALWAYS return dspy.Prediction(score=..., feedback=...).
       dspy.Prediction supports __float__ and __add__, so it works with both
       dspy.Evaluate aggregation AND GEPA reflection. NEVER return a dict or
       bare float.
    2. Use MULTI-AXIS scoring. Compute independent sub-scores (e.g., correctness,
       completeness, format_adherence, reasoning_quality) each in [0.0, 1.0].
       Combine with EXPLICIT WEIGHTS declared as module-level constants.
    3. Feedback must TEACH the optimizer: explain WHY the prediction failed
       and WHAT GOOD LOOKS LIKE. Generic messages like "wrong answer" are
       useless. Target 2-5 lines of actionable, specific critique per issue.
    4. When pred_name is provided, write PER-PREDICTOR feedback that helps
       GEPA attribute errors to the specific predictor (credit assignment).
       Include the predictor name in the feedback text.
    5. Use exact string matching ONLY for enumerated/categorical fields.
    6. For free-text fields, prefer semantic similarity (e.g.,
       dspy.evaluate.answer_exact_match, fuzzy_match, or embedding cosine
       similarity) over exact string matching.
    7. For subjective quality assessment (writing style, reasoning soundness,
       helpfulness), you MAY use a lightweight LLM-as-judge call inside the
       metric, but ONLY when deterministic checks are impossible. Prefer
       fast/cheap checks first. If using an LLM judge, always provide a
       fallback on judge failure and use a cheaper model (e.g., gpt-4o-mini).
    8. Include all imports at the top of the generated code, not inside
       functions.

    Example 1: Classification task with fields urgency and sentiment
    (shows multi-axis scoring, per-predictor feedback, and normalization helpers):

    import dspy

    CORRECTNESS_WEIGHT = 0.6
    FORMAT_WEIGHT = 0.4

    def _normalize(val):
        if val is None:
            return None
        return str(val).strip().lower()

    def _safe_get(obj, key):
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        return None

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        score = 0.0
        feedback_parts = []

        gold_urgency = _normalize(_safe_get(example, "urgency"))
        pred_urgency = _normalize(_safe_get(pred, "urgency"))

        if gold_urgency == pred_urgency:
            score += CORRECTNESS_WEIGHT * 0.5
        else:
            feedback_parts.append(
                f"Predictor '{pred_name or 'main'}': Urgency mismatch. "
                f"Expected '{gold_urgency}', got '{pred_urgency}'. "
                f"Think about how you could have reasoned to get the correct urgency label."
            )

        gold_sentiment = _normalize(_safe_get(example, "sentiment"))
        pred_sentiment = _normalize(_safe_get(pred, "sentiment"))

        if gold_sentiment == pred_sentiment:
            score += CORRECTNESS_WEIGHT * 0.5
        else:
            feedback_parts.append(
                f"Predictor '{pred_name or 'main'}': Sentiment mismatch. "
                f"Expected '{gold_sentiment}', got '{pred_sentiment}'. "
                f"Consider the tone and emotional cues in the input message."
            )

        if pred_urgency and pred_sentiment:
            score += FORMAT_WEIGHT
        else:
            feedback_parts.append(
                "Format issue: predicted fields should not be empty or None."
            )

        if not feedback_parts:
            feedback_parts.append("Correct on all axes.")

        return dspy.Prediction(score=min(score, 1.0), feedback=" ".join(feedback_parts))

    Example 2: Free-text answer field (shows semantic similarity instead of exact match):

    import dspy

    CORRECTNESS_WEIGHT = 0.7
    FORMAT_WEIGHT = 0.3

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        score = 0.0
        feedback_parts = []

        gold_answer = str(getattr(example, "answer", "")).strip()
        pred_answer = str(getattr(pred, "answer", "")).strip()

        if not pred_answer:
            score += 0.0
            feedback_parts.append(
                f"Predictor '{pred_name or 'main'}': Empty answer. "
                "Provide a substantive response based on the input."
            )
        else:
            # Semantic similarity: exact_match returns 0 or 1
            similarity = dspy.evaluate.answer_exact_match(gold_answer, pred_answer)
            score += CORRECTNESS_WEIGHT * similarity
            if similarity < 1.0:
                feedback_parts.append(
                    f"Predictor '{pred_name or 'main'}': Answer differs semantically. "
                    f"Expected: '{gold_answer[:80]}...'. Got: '{pred_answer[:80]}...'. "
                    "Align your response more closely with the ground truth."
                )

        # Format check: reasonable length
        if 10 <= len(pred_answer) <= 500:
            score += FORMAT_WEIGHT
        else:
            feedback_parts.append(
                "Format issue: answer should be between 10 and 500 characters."
            )

        if not feedback_parts:
            feedback_parts.append("Correct and well-formatted.")

        return dspy.Prediction(score=min(score, 1.0), feedback=" ".join(feedback_parts))

    Example 3: Subjective quality assessment using an LLM judge
    (shows dspy.ChainOfThought judge setup with fallback on failure):

    import dspy
    from pydantic import Field

    # Use the user's configured LM, or fall back to a lightweight default
    _judge_lm = dspy.LM("openrouter/openai/gpt-oss-120b")
    dspy.configure(lm=_judge_lm)

    class JudgeSignature(dspy.Signature):
        gold_answer: str = dspy.InputField()
        predicted_answer: str = dspy.InputField()
        quality_score: float = dspy.OutputField(
            desc="Quality score from 0.0 to 1.0",
            json_schema_extra={"ge": 0.0, "le": 1.0},
        )
        critique: str = dspy.OutputField(desc="Specific critique explaining quality issues and what good looks like")

    _judge_program = dspy.ChainOfThought(JudgeSignature)

    CORRECTNESS_WEIGHT = 0.6
    REASONING_WEIGHT = 0.4

    def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        score = 0.0
        feedback_parts = []

        gold_answer = str(getattr(example, "answer", "")).strip()
        pred_answer = str(getattr(pred, "answer", "")).strip()

        if not pred_answer:
            return dspy.Prediction(
                score=0.0,
                feedback=f"Predictor '{pred_name or 'main'}': Empty answer. Provide a substantive response."
            )

        # Deterministic exact match first (fast, cheap)
        exact_match = dspy.evaluate.answer_exact_match(gold_answer, pred_answer)
        if exact_match == 1.0:
            score += CORRECTNESS_WEIGHT
            feedback_parts.append("Answer is exactly correct.")
        else:
            # Fallback to lightweight LLM judge for semantic quality
            try:
                judge_result = _judge_program(
                    gold_answer=gold_answer,
                    predicted_answer=pred_answer
                )
                quality_score = float(judge_result.quality_score)
                critique = str(judge_result.critique)
                score += CORRECTNESS_WEIGHT * quality_score
                if quality_score < 1.0:
                    feedback_parts.append(
                        f"Predictor '{pred_name or 'main'}': {critique}"
                    )
            except Exception:
                # Fallback on judge failure: use exact match score
                score += CORRECTNESS_WEIGHT * exact_match
                feedback_parts.append(
                    f"Predictor '{pred_name or 'main'}': Judge failed. "
                    f"Using exact-match score {exact_match}. "
                    "Ensure answers align closely with the ground truth."
                )

        # Check reasoning trace quality if available
        if trace and pred_trace:
            reasoning_steps = len(pred_trace)
            if reasoning_steps >= 2:
                score += REASONING_WEIGHT
            else:
                feedback_parts.append(
                    "Reasoning issue: provide at least 2 explicit reasoning steps."
                )
        else:
            score += REASONING_WEIGHT

        if not feedback_parts:
            feedback_parts.append("Correct with strong reasoning.")

        return dspy.Prediction(score=min(score, 1.0), feedback=" ".join(feedback_parts))
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


def _validate_metric_source(source: str) -> None:
    tree = ast.parse(source)

    class DictReturnVisitor(ast.NodeVisitor):
        def __init__(self):
            self.dict_returns = []

        def visit_Return(self, node: ast.Return) -> None:
            if isinstance(node.value, ast.Dict):
                self.dict_returns.append(node.lineno)
            self.generic_visit(node)

    visitor = DictReturnVisitor()
    visitor.visit(tree)
    if visitor.dict_returns:
        lines = ", ".join(str(l) for l in visitor.dict_returns)
        raise ValueError(
            f"Generated metric returns a dict on line(s) {lines}. "
            "GEPA metrics must return dspy.Prediction(score=..., feedback=...), "
            "not a dict. Dict returns crash dspy.Evaluate's parallel aggregator."
        )

    if "dspy.Prediction" not in source and "Prediction(" not in source:
        raise ValueError(
            "Generated metric does not appear to return dspy.Prediction. "
            "GEPA metrics must return dspy.Prediction(score=..., feedback=...)."
        )


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

    _validate_metric_source(source)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(source)
    return out_path
