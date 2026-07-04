import json
import re
import unicodedata
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
# String sanitization & validation helpers
# ---------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"  # dingbats
    "\U000024c2-\U0001f251"  # enclosed characters
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001fa6f"  # chess symbols
    "\U0001fa70-\U0001faff"  # symbols extended-A
    "\U00002600-\U000026ff"  # misc symbols (includes ⚠️)
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"  # zero width joiner
    "\U00002b50"  # star
    "\U000023cf-\U000023fa"  # misc technical
    "]+",
    flags=re.UNICODE,
)


def sanitize_string(value: str) -> str:
    """Strip emoji, normalize Unicode, collapse whitespace, and trim.

    Returns the cleaned string.  Does **not** raise — safe to call on any input.
    """
    # Strip emoji characters
    cleaned = _EMOJI_RE.sub("", value)
    # Normalize Unicode to NFC (composed form) — collapses weird codepoints
    cleaned = unicodedata.normalize("NFC", cleaned)
    # Collapse runs of whitespace (including non-breaking spaces, narrow
    # no-break spaces, etc.) into a single ASCII space
    cleaned = re.sub(r"[\s\u00a0\u200b\u200c\u200d\u202f\u2007\u2060]+", " ", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Validator factory functions
# ---------------------------------------------------------------------------

ValidatorFn = Callable[[dict[str, Any]], tuple[bool, str]]


def non_empty_validator(*field_names: str) -> ValidatorFn:
    """Reject rows where any of *field_names* is empty or whitespace-only."""

    def _check(row: dict[str, Any]) -> tuple[bool, str]:
        for name in field_names:
            val = row.get(name)
            if val is None or (isinstance(val, str) and not val.strip()):
                return False, f"Field '{name}' must not be empty"
        return True, ""

    return _check


def no_emoji_validator(*field_names: str) -> ValidatorFn:
    """Reject rows where any of *field_names* contains emoji characters."""

    def _check(row: dict[str, Any]) -> tuple[bool, str]:
        for name in field_names:
            val = row.get(name)
            if isinstance(val, str) and _EMOJI_RE.search(val):
                return False, f"Field '{name}' must not contain emoji"
        return True, ""

    return _check


def enum_validator(field_name: str, allowed: list[str]) -> ValidatorFn:
    """Reject rows where *field_name* is not one of *allowed* (case-insensitive)."""
    allowed_lower = [a.lower() for a in allowed]

    def _check(row: dict[str, Any]) -> tuple[bool, str]:
        val = row.get(field_name)
        if val is None:
            return False, f"Field '{field_name}' must not be None"
        if isinstance(val, str) and val.strip().lower() not in allowed_lower:
            return False, (
                f"Field '{field_name}' value '{val}' not in allowed: {allowed}"
            )
        return True, ""

    return _check


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


class _BatchJudgeSignature(dspy.Signature):
    """Score multiple data rows on quality dimensions.

    For each row, return a JSON object with scores (each dimension in [0.0, 1.0])
    and brief feedback. Return a JSON array of result objects, one per input row,
    in the SAME ORDER.
    """

    rows_json: str = dspy.InputField(
        desc="JSON array of row objects to score"
    )
    rubric: str = dspy.InputField(desc="Comma-separated list of scoring dimensions")
    task_description: str = dspy.InputField(desc="Optional task context", default="")
    results_json: str = dspy.OutputField(
        desc=(
            'JSON array of {"scores": {dim: float}, "feedback": str} objects, '
            'one per input row, in the same order.'
        )
    )


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
        self._batch_predictor = dspy.Predict(_BatchJudgeSignature)

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

    def batch_score(
        self,
        rows: list[dict[str, Any]],
        task_description: str = "",
    ) -> list[JudgeResult]:
        """Score multiple rows in a single LLM call.

        Returns a list of :class:`JudgeResult` in the same order as *rows*.
        Falls back to individual scoring for rows that fail to parse.
        """
        if not rows:
            return []
        if len(rows) == 1:
            return [self.score(rows[0], task_description)]

        rows_json = json.dumps(rows, default=str)
        rubric_str = ", ".join(self._rubric)

        try:
            with dspy.context(lm=self._lm):
                result = self._batch_predictor(
                    rows_json=rows_json,
                    rubric=rubric_str,
                    task_description=task_description,
                )
        except Exception:
            return [
                JudgeResult(score=0.0, feedback="LLM judge call failed", scores={})
                for _ in rows
            ]

        try:
            parsed: list[dict[str, Any]] = json.loads(result.results_json)
            if not isinstance(parsed, list):
                parsed = [parsed]
        except (json.JSONDecodeError, ValueError, TypeError):
            return [
                JudgeResult(score=0.0, feedback="Failed to parse judge output", scores={})
                for _ in rows
            ]

        results: list[JudgeResult] = []
        for i in range(len(rows)):
            if i < len(parsed):
                entry = parsed[i]
                try:
                    scores = {k: float(v) for k, v in entry.get("scores", {}).items()}
                    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
                    results.append(
                        JudgeResult(
                            score=min(avg_score, 1.0),
                            feedback=str(entry.get("feedback", "")),
                            scores=scores,
                        )
                    )
                    continue
                except (ValueError, TypeError, AttributeError):
                    pass
            results.append(
                JudgeResult(score=0.0, feedback="Failed to parse judge entry", scores={})
            )
        return results


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
