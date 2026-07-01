from unittest.mock import MagicMock, patch

import pytest

from dspy_auto_gepa.quality import (
    DiversityChecker,
    JudgeResult,
    LLMJudge,
    RejectionSampler,
    Validator,
)

# ---------------------------------------------------------------------------
# DiversityChecker
# ---------------------------------------------------------------------------


def test_diversity_identical_texts():
    result = DiversityChecker(diversity_threshold=0.3).check(
        ["hello world", "hello world", "hello world"]
    )
    assert result.is_diverse is False
    assert result.avg_similarity == 1.0


def test_diversity_diverse_texts():
    result = DiversityChecker(diversity_threshold=0.3).check(
        ["server is down", "clean the room", "password reset needed"]
    )
    assert result.is_diverse is True
    assert result.avg_similarity < 0.3


def test_diversity_empty_list():
    result = DiversityChecker().check([])
    assert result.is_diverse is True
    assert result.avg_similarity == 0.0


def test_diversity_single_text():
    result = DiversityChecker().check(["hello"])
    assert result.is_diverse is True


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_passing():
    v = Validator([(lambda r: (len(r.get("text", "")) > 0, "text empty"))])
    result = v.validate({"text": "hello"})
    assert result.is_valid is True
    assert result.failures == []


def test_validator_failing():
    v = Validator([(lambda r: (len(r.get("text", "")) > 0, "text empty"))])
    result = v.validate({"text": ""})
    assert result.is_valid is False
    assert "text empty" in result.failures


def test_validator_multiple():
    v = Validator(
        [
            lambda r: (True, "never used"),
            lambda r: (len(r.get("name", "")) > 0, "name missing"),
        ]
    )
    result = v.validate({"name": ""})
    assert result.is_valid is False
    assert result.failures == ["name missing"]


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


@patch("dspy_auto_gepa.quality.dspy.Predict")
def test_llmjudge_success(mock_predict_cls):
    mock_predictor = MagicMock()
    mock_predictor.return_value = MagicMock(
        scores_json='{"correctness": 0.8, "relevance": 0.9}',
        feedback="Good quality",
    )
    mock_predict_cls.return_value = mock_predictor

    judge = LLMJudge(lm=MagicMock())
    result = judge.score({"text": "hello world"}, task_description="test")

    assert isinstance(result, JudgeResult)
    assert result.score == pytest.approx(0.85)
    assert result.scores == {"correctness": 0.8, "relevance": 0.9}
    assert result.feedback == "Good quality"


@patch("dspy_auto_gepa.quality.dspy.Predict")
def test_llmjudge_parse_failure(mock_predict_cls):
    mock_predictor = MagicMock()
    mock_predictor.return_value = MagicMock(
        scores_json="not valid json",
        feedback="",
    )
    mock_predict_cls.return_value = mock_predictor

    judge = LLMJudge(lm=MagicMock())
    result = judge.score({"text": "hello"})

    assert result.score == 0.0
    assert result.feedback == "Failed to parse judge output"
    assert result.scores == {}


@patch("dspy_auto_gepa.quality.dspy.Predict")
def test_llmjudge_llm_call_failure(mock_predict_cls):
    mock_predictor = MagicMock()
    mock_predictor.side_effect = RuntimeError("LLM unavailable")
    mock_predict_cls.return_value = mock_predictor

    judge = LLMJudge(lm=MagicMock())
    result = judge.score({"text": "hello"})

    assert result.score == 0.0
    assert result.feedback == "LLM judge call failed"


# ---------------------------------------------------------------------------
# RejectionSampler
# ---------------------------------------------------------------------------


def test_rejection_sampler_all_pass():
    passing_validator = Validator([lambda r: (True, "")])
    sampler = RejectionSampler(validator=passing_validator)
    result = sampler.sample({"text": "hello"})
    assert result.passed is True
    assert result.failures == []


def test_rejection_sampler_judge_rejects():
    mock_judge = MagicMock(spec=LLMJudge)
    mock_judge.score.return_value = JudgeResult(
        score=0.1, feedback="Low quality", scores={}
    )
    sampler = RejectionSampler(judge=mock_judge, judge_threshold=0.5)
    result = sampler.sample({"text": "hello"})
    assert result.passed is False
    assert any("below threshold" in f for f in result.failures)


def test_rejection_sampler_all_disabled():
    sampler = RejectionSampler()
    result = sampler.sample({"text": "hello"})
    assert result.passed is True
    assert result.failures == []
    assert result.score == 1.0


def test_rejection_sampler_diversity_check():
    checker = DiversityChecker(diversity_threshold=0.3)
    sampler = RejectionSampler(diversity_checker=checker)
    identical = ["hello world", "hello world"]
    result = sampler.sample({"text": "new row"}, existing_texts=identical)
    assert result.passed is False
    assert any("Diversity check failed" in f for f in result.failures)


def test_rejection_sampler_validator_fails():
    failing_validator = Validator([lambda r: (False, "bad row")])
    sampler = RejectionSampler(validator=failing_validator)
    result = sampler.sample({"text": "hello"})
    assert result.passed is False
    assert "bad row" in result.failures
