import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import dspy


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    score: float
    feedback: str
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationResult:
    is_valid: bool
    failures: list[str] = field(default_factory=list)


@dataclass
class DiversityResult:
    is_diverse: bool
    avg_similarity: float
    max_similarity: float


@dataclass
class QualityResult:
    passed: bool
    score: float
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------

class _JudgeSignature(dspy.Signature):
    """Score a data row on multiple quality dimensions.

    Return a JSON object mapping each rubric dimension to a float score
    in [0.0, 1.0], and provide concise feedback explaining the scores.
    """

    row_data: str = dspy.InputField(desc="JSON representation of the row to judge")
    rubric: str = dspy.InputField(desc="Comma-separated list of scoring dimensions")
    task_description: str = dspy.InputField(desc="Optional task context", default="")
    scores_json: str = dspy.OutputField(
        desc='JSON dict mapping each rubric dimension to a score in [0.0, 1.0], e.g. {"correctness": 0.8, "relevance": 0.9}',
    )
    feedback: str = dspy.OutputField(desc="Concise feedback explaining the scores")


class LLMJudge:
    """Uses an LLM via a DSPy Signature to score data rows on a rubric."""

    def __init__(
        self,
        lm: dspy.LM,
        rubric: list[str] | None = None,
    ) -> None:
        self._lm = lm
        self._rubric = rubric or ["correctness", "relevance", "coherence"]
        self._predictor = dspy.Predict(_JudgeSignature)

    @property
    def rubric(self) -> list[str]:
        return list(self._rubric)

    def score(self, row: dict[str, Any], task_description: str = "") -> JudgeResult:
        """Score *row* against the configured rubric.

        Returns a :class:`JudgeResult` with an overall score (average of
        per-dimension scores) and per-dimension breakdown.
        """
        row_json = json.dumps(row, default=str)
        rubric_str = ", ".join(self._rubric)

        try:
            with dspy.context(lm=self._lm):
                result = self._predictor(
                    row_data=row_json,
                    rubric=rubric_str,
                    task_description=task_description,
                )
        except Exception:
            return JudgeResult(
                score=0.0,
                feedback="LLM judge call failed",
                scores={},
            )

        try:
            scores_raw: dict[str, Any] = json.loads(result.scores_json)
            scores = {k: float(v) for k, v in scores_raw.items()}
        except (json.JSONDecodeError, ValueError, TypeError):
            return JudgeResult(
                score=0.0,
                feedback="Failed to parse judge output",
                scores={},
            )

        avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        return JudgeResult(
            score=min(avg_score, 1.0),
            feedback=str(result.feedback),
            scores=scores,
        )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class Validator:
    """Chains a list of callable validators over row dicts.

    Each validator is ``Callable[[dict], tuple[bool, str]]`` where the bool
    indicates pass/fail and the str is the failure reason (ignored on pass).
    """

    def __init__(
        self,
        validators: list[Callable[[dict[str, Any]], tuple[bool, str]]],
    ) -> None:
        self._validators = list(validators)

    def validate(self, row: dict[str, Any]) -> ValidationResult:
        failures: list[str] = []
        for validator in self._validators:
            passed, reason = validator(row)
            if not passed:
                failures.append(reason)
        return ValidationResult(is_valid=len(failures) == 0, failures=failures)


# ---------------------------------------------------------------------------
# DiversityChecker
# ---------------------------------------------------------------------------

def _ngrams(text: str, n: int) -> set[str]:
    """Extract character-level n-grams from *text*."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


class DiversityChecker:
    """Measures diversity of a collection of texts via n-gram Jaccard similarity."""

    def __init__(
        self,
        diversity_threshold: float = 0.3,
        ngram_size: int = 3,
    ) -> None:
        self._threshold = diversity_threshold
        self._ngram_size = ngram_size

    def check(self, texts: list[str]) -> DiversityResult:
        """Compute pairwise n-gram Jaccard similarity across *texts*.

        Returns :class:`DiversityResult` where ``is_diverse`` is ``True`` when
        the average similarity is strictly below the configured threshold.
        """
        if len(texts) <= 1:
            return DiversityResult(
                is_diverse=True,
                avg_similarity=0.0,
                max_similarity=0.0,
            )

        ngram_sets = [_ngrams(t, self._ngram_size) for t in texts]
        similarities: list[float] = []
        max_sim = 0.0

        for i in range(len(ngram_sets)):
            for j in range(i + 1, len(ngram_sets)):
                sim = _jaccard(ngram_sets[i], ngram_sets[j])
                similarities.append(sim)
                if sim > max_sim:
                    max_sim = sim

        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        return DiversityResult(
            is_diverse=avg_sim < self._threshold,
            avg_similarity=avg_sim,
            max_similarity=max_sim,
        )


# ---------------------------------------------------------------------------
# RejectionSampler
# ---------------------------------------------------------------------------

class RejectionSampler:
    """Orchestrates LLMJudge, Validator, and DiversityChecker.

    Any component can be ``None`` to skip that check.  A row passes only if
    *all* enabled components pass.
    """

    def __init__(
        self,
        judge: LLMJudge | None = None,
        validator: Validator | None = None,
        diversity_checker: DiversityChecker | None = None,
        judge_threshold: float = 0.5,
    ) -> None:
        self._judge = judge
        self._validator = validator
        self._diversity_checker = diversity_checker
        self._judge_threshold = judge_threshold

    def sample(
        self,
        row: dict[str, Any],
        existing_texts: list[str] | None = None,
        task_description: str = "",
    ) -> QualityResult:
        """Evaluate *row* through all enabled quality gates.

        Returns :class:`QualityResult` with ``passed=True`` only when every
        enabled component passes.
        """
        failures: list[str] = []
        overall_score = 1.0

        # --- Judge ---
        if self._judge is not None:
            judge_result = self._judge.score(row, task_description=task_description)
            overall_score = judge_result.score
            if judge_result.score < self._judge_threshold:
                failures.append(
                    f"Judge score {judge_result.score:.3f} below threshold "
                    f"{self._judge_threshold:.3f}. {judge_result.feedback}"
                )

        # --- Validator ---
        if self._validator is not None:
            val_result = self._validator.validate(row)
            if not val_result.is_valid:
                failures.extend(val_result.failures)

        # --- Diversity ---
        if self._diversity_checker is not None and existing_texts is not None:
            div_result = self._diversity_checker.check(existing_texts)
            if not div_result.is_diverse:
                failures.append(
                    f"Diversity check failed: avg_similarity={div_result.avg_similarity:.3f} "
                    f"(max={div_result.max_similarity:.3f})"
                )

        return QualityResult(
            passed=len(failures) == 0,
            score=overall_score,
            failures=failures,
        )
