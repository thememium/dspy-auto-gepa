"""Coverage tests for artifacts, config, data, metric builder, quality, runner."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import dspy
import pandas as pd
import pytest

import dspy_auto_gepa
from dspy_auto_gepa.artifacts import load_metric, save_results
from dspy_auto_gepa.config import AutoDataConfig
from dspy_auto_gepa.data import (
    _extract_fields_from_signature,
    _to_dicts,
    apply_mapping,
    extract_signature_metadata,
    infer_fields_from_module,
    resolve_fields,
)
from dspy_auto_gepa.metric_builder import _validate_metric_source, generate_metric_file
from dspy_auto_gepa.quality import DiversityChecker, LLMJudge, enum_validator
from dspy_auto_gepa.runner import (
    AutoGEPA,
    Datasets,
    GenerationFailed,
    GenerationResult,
    RunResult,
)


class TicketSignature(dspy.Signature):
    """Classify support tickets."""

    message: str = dspy.InputField()
    urgency: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()


class DummyModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.signature = TicketSignature

    def forward(self, message: str) -> dspy.Prediction:
        return dspy.Prediction(urgency="low", sentiment="neutral")


# ---------------------------------------------------------------------------
# Package version fallback
# ---------------------------------------------------------------------------


def test_package_version_falls_back_when_import_fails() -> None:
    try:
        with patch("importlib.metadata.version", side_effect=ImportError("nope")):
            importlib.reload(dspy_auto_gepa)
        assert dspy_auto_gepa.__version__ == "unknown"
    finally:
        importlib.reload(dspy_auto_gepa)


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


def test_load_metric_success(tmp_path: Path) -> None:
    metric_file = tmp_path / "metric.py"
    metric_file.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")

    metric = load_metric(metric_file)

    assert metric(None, None) == 1.0


def test_load_metric_import_error_when_spec_missing(tmp_path: Path) -> None:
    with patch(
        "dspy_auto_gepa.artifacts.importlib.util.spec_from_file_location",
        return_value=None,
    ):
        with pytest.raises(ImportError, match="Cannot load metric"):
            load_metric(tmp_path / "metric.py")


def test_load_metric_rejects_file_without_metric(tmp_path: Path) -> None:
    metric_file = tmp_path / "metric.py"
    metric_file.write_text("VALUE = 1\n")

    with pytest.raises(ValueError, match="must define a top-level metric"):
        load_metric(metric_file)


def test_save_results_writes_all_paths(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    results = save_results(
        artifact_dir=artifact_dir,
        baseline_scores={"score": 0.4},
        optimized_scores={"score": 0.8},
        optimized_program_path=Path("program.json"),
        metric_path=Path("metric.py"),
    )

    payload = json.loads(results.read_text())
    assert payload == {
        "baseline": {"score": 0.4},
        "optimized": {"score": 0.8},
        "optimized_program": "program.json",
        "metric": "metric.py",
    }


def test_save_results_allows_optional_paths_to_be_omitted(tmp_path: Path) -> None:
    results = save_results(artifact_dir=tmp_path / "artifacts")

    payload = json.loads(results.read_text())
    assert payload == {
        "baseline": None,
        "optimized": None,
        "optimized_program": None,
        "metric": None,
    }


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": 0}, "n must be positive"),
        ({"max_retries": 0}, "max_retries must be positive"),
        ({"chunk_size": 0}, "chunk_size must be positive"),
        (
            {"max_inflight_requests": 0},
            "max_inflight_requests must be positive",
        ),
        (
            {"diversity_threshold": 1.5},
            "diversity_threshold must be between",
        ),
    ],
)
def test_autodata_config_rejects_invalid_values(
    kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        AutoDataConfig(**kwargs)


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------


def test_extract_signature_metadata_without_fields() -> None:
    meta = extract_signature_metadata(cast(Any, SimpleNamespace(fields=None)))

    assert meta.fields == []


def test_extract_signature_metadata_skips_non_io_field() -> None:
    fake_sig = SimpleNamespace(fields={"ignored": SimpleNamespace()})

    meta = extract_signature_metadata(cast(Any, fake_sig))

    assert meta.fields == []


def test_read_json_array_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('{"not_rows": 1}')

    with pytest.raises(ValueError, match="must contain an array"):
        _to_dicts(path)


def test_to_dicts_uses_to_dicts_method() -> None:
    obj = SimpleNamespace(to_dicts=lambda: [{"a": 1}])

    assert _to_dicts(obj) == [{"a": 1}]


def test_to_dicts_uses_collect_then_to_dicts() -> None:
    frame = SimpleNamespace(to_dicts=lambda: [{"a": 2}])
    obj = SimpleNamespace(collect=lambda: frame)

    assert _to_dicts(obj) == [{"a": 2}]


def test_to_dicts_ignores_to_dicts_returning_non_list() -> None:
    frame = SimpleNamespace(to_dict=lambda orient: [{"a": 3}])
    obj = SimpleNamespace(to_dicts=lambda: "nope", to_dict=frame.to_dict)

    assert _to_dicts(obj) == [{"a": 3}]


def test_to_dicts_uses_to_pandas() -> None:
    frame = pd.DataFrame({"a": [4]})
    obj = SimpleNamespace(to_pandas=lambda: frame)

    assert _to_dicts(obj) == [{"a": 4}]


def test_extract_fields_from_signature_without_fields() -> None:
    assert _extract_fields_from_signature(cast(Any, SimpleNamespace(fields=None))) == (
        [],
        [],
    )


def test_infer_fields_aggregates_multiple_predictors() -> None:
    class FirstSignature(dspy.Signature):
        """First."""

        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class SecondSignature(dspy.Signature):
        """Second."""

        topic: str = dspy.InputField()
        summary: str = dspy.OutputField()

    cot = dspy.ChainOfThought(FirstSignature)
    setattr(cot, "signature", cot.predict.signature)
    module = MagicMock()
    module.signature = None
    module.named_predictors.return_value = [
        ("cot", cot),
        ("predict", dspy.Predict(SecondSignature)),
        ("broken", SimpleNamespace(signature=None)),
    ]

    inputs, outputs = infer_fields_from_module(module)

    assert inputs == ["question", "topic"]
    assert outputs == ["answer", "summary"]


def test_infer_fields_strips_reasoning_from_chain_of_thought() -> None:
    module = dspy.ChainOfThought(TicketSignature)

    inputs, outputs = infer_fields_from_module(module)

    assert inputs == ["message"]
    assert outputs == ["urgency", "sentiment"]


def test_infer_fields_without_signature_or_predictors_raises() -> None:
    module = MagicMock()
    module.signature = None
    module.named_predictors.return_value = []

    with pytest.raises(ValueError, match="no .signature attribute"):
        infer_fields_from_module(module)


def test_infer_fields_with_signature_but_no_fields_raises() -> None:
    module = MagicMock()
    module.signature = SimpleNamespace(fields={})

    with pytest.raises(ValueError, match="has no .fields attribute"):
        infer_fields_from_module(module)


def test_resolve_fields_accepts_explicit_output_list() -> None:
    inputs, outputs, mapping = resolve_fields(
        DummyModule(),
        set(),
        {"msg": "message"},
        ["urgency"],
    )

    assert inputs == ["message"]
    assert outputs == ["urgency"]
    assert mapping == {"msg": "message"}


def test_resolve_fields_falls_back_to_signature_inputs() -> None:
    inputs, outputs, mapping = resolve_fields(
        DummyModule(),
        set(),
        None,
        ["urgency"],
    )

    assert inputs == ["message"]
    assert outputs == ["urgency"]
    assert mapping == {}


def test_resolve_fields_rejects_empty_resolved_fields() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        resolve_fields(DummyModule(), set(), [], [])


def test_apply_mapping_empty_mapping_returns_same_list() -> None:
    rows = [{"a": 1}]

    assert apply_mapping(rows, {}) is rows


# ---------------------------------------------------------------------------
# metric builder
# ---------------------------------------------------------------------------


def test_validate_metric_source_rejects_dict_return() -> None:
    source = "def metric(example, pred, trace=None):\n    return {'score': 1.0}\n"

    with pytest.raises(ValueError, match="returns a dict"):
        _validate_metric_source(source)


def test_validate_metric_source_rejects_missing_prediction() -> None:
    source = "def metric(example, pred, trace=None):\n    return 1.0\n"

    with pytest.raises(ValueError, match="does not appear to return dspy.Prediction"):
        _validate_metric_source(source)


def test_validate_metric_source_accepts_prediction() -> None:
    source = (
        "import dspy\n"
        "def metric(example, pred, trace=None):\n"
        '    return dspy.Prediction(score=1.0, feedback="ok")\n'
    )

    _validate_metric_source(source)


class _FakeMetricGenerator:
    source: str = ""

    def __init__(self, signature: object) -> None:
        self.signature = signature
        self.lm: object | None = None

    def set_lm(self, lm: object) -> None:
        self.lm = lm

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(metric_source=self.source)


def _generator_returning(source: str) -> type[_FakeMetricGenerator]:
    class Generator(_FakeMetricGenerator):
        pass

    Generator.source = source
    return Generator


def test_generate_metric_file_uses_custom_generator(tmp_path: Path) -> None:
    source = (
        "import dspy\n"
        "def metric(example, pred, trace=None):\n"
        '    return dspy.Prediction(score=1.0, feedback="ok")\n'
    )
    generator_cls = _generator_returning(source)
    out_path = tmp_path / "nested" / "metric.py"

    result = generate_metric_file(
        input_fields=["message"],
        output_fields=["urgency"],
        sample_rows=[{"message": "hello", "urgency": "low"}],
        module=DummyModule(),
        out_path=out_path,
        metric_lm=MagicMock(),
        metric_generator_module=generator_cls,
    )

    assert result == out_path
    assert out_path.read_text() == source.strip()


def test_generate_metric_file_rejects_syntax_error(tmp_path: Path) -> None:
    generator_cls = _generator_returning("def metric(:\n")

    with pytest.raises(ValueError, match="not valid Python code"):
        generate_metric_file(
            input_fields=["message"],
            output_fields=["urgency"],
            sample_rows=[],
            module=DummyModule(),
            out_path=tmp_path / "metric.py",
            metric_generator_module=generator_cls,
        )


# ---------------------------------------------------------------------------
# quality gaps
# ---------------------------------------------------------------------------


def test_enum_validator_rejects_none() -> None:
    validator = enum_validator("urgency", ["low", "high"])

    passed, reason = validator({"urgency": None})

    assert not passed
    assert "must not be None" in reason


def test_llmjudge_rubric_property_returns_copy() -> None:
    with patch("dspy_auto_gepa.quality.dspy.Predict"):
        judge = LLMJudge(lm=MagicMock(), rubric=["a", "b"])

    rubric = judge.rubric
    rubric.append("c")

    assert judge.rubric == ["a", "b"]


@patch("dspy_auto_gepa.quality.dspy.Predict")
def test_llmjudge_batch_score_success(mock_predict_cls: MagicMock) -> None:
    mock_predictor = MagicMock()
    mock_predictor.return_value = MagicMock(
        results_json=json.dumps(
            [
                {"scores": {"correctness": 0.5}, "feedback": "first"},
                {"scores": {"correctness": 0.9}, "feedback": "second"},
            ]
        )
    )
    mock_predict_cls.return_value = mock_predictor
    judge = LLMJudge(lm=MagicMock())

    results = judge.batch_score([{"a": 1}, {"a": 2}], task_description="task")

    assert [r.score for r in results] == [0.5, 0.9]
    assert [r.feedback for r in results] == ["first", "second"]


@patch("dspy_auto_gepa.quality.dspy.Predict")
def test_llmjudge_batch_score_empty(mock_predict_cls: MagicMock) -> None:
    judge = LLMJudge(lm=MagicMock())

    assert judge.batch_score([]) == []
    mock_predict_cls.return_value.assert_not_called()


@patch("dspy_auto_gepa.quality.dspy.Predict")
def test_llmjudge_batch_score_single_delegates(
    mock_predict_cls: MagicMock,
) -> None:
    mock_predictor = MagicMock()
    mock_predictor.return_value = MagicMock(
        scores_json='{"correctness": 0.75}', feedback="one"
    )
    mock_predict_cls.return_value = mock_predictor
    judge = LLMJudge(lm=MagicMock())

    results = judge.batch_score([{"a": 1}])

    assert len(results) == 1
    assert results[0].score == 0.75


@patch("dspy_auto_gepa.quality.dspy.Predict")
def test_llmjudge_batch_score_call_failure(mock_predict_cls: MagicMock) -> None:
    mock_predictor = MagicMock(side_effect=RuntimeError("down"))
    mock_predict_cls.return_value = mock_predictor
    judge = LLMJudge(lm=MagicMock())

    results = judge.batch_score([{"a": 1}, {"a": 2}])

    assert [r.feedback for r in results] == ["LLM judge call failed"] * 2


@patch("dspy_auto_gepa.quality.dspy.Predict")
def test_llmjudge_batch_score_invalid_json(mock_predict_cls: MagicMock) -> None:
    mock_predictor = MagicMock()
    mock_predictor.return_value = MagicMock(results_json="not json")
    mock_predict_cls.return_value = mock_predictor
    judge = LLMJudge(lm=MagicMock())

    results = judge.batch_score([{"a": 1}, {"a": 2}])

    assert [r.feedback for r in results] == ["Failed to parse judge output"] * 2


@patch("dspy_auto_gepa.quality.dspy.Predict")
def test_llmjudge_batch_score_non_list_and_missing_entries(
    mock_predict_cls: MagicMock,
) -> None:
    mock_predictor = MagicMock()
    mock_predictor.return_value = MagicMock(
        results_json=json.dumps({"scores": {"correctness": 0.6}, "feedback": "only"})
    )
    mock_predict_cls.return_value = mock_predictor
    judge = LLMJudge(lm=MagicMock())

    results = judge.batch_score([{"a": 1}, {"a": 2}])

    assert results[0].score == 0.6
    assert results[1].feedback == "Failed to parse judge entry"


@patch("dspy_auto_gepa.quality.dspy.Predict")
def test_llmjudge_batch_score_unparseable_entries(
    mock_predict_cls: MagicMock,
) -> None:
    mock_predictor = MagicMock()
    mock_predictor.return_value = MagicMock(
        results_json=json.dumps(
            [
                ["not", "a", "dict"],
                {"scores": {"correctness": "not-a-number"}},
            ]
        )
    )
    mock_predict_cls.return_value = mock_predictor
    judge = LLMJudge(lm=MagicMock())

    results = judge.batch_score([{"a": 1}, {"a": 2}])

    assert [r.feedback for r in results] == ["Failed to parse judge entry"] * 2


def test_diversity_check_with_empty_texts_hits_empty_ngrams() -> None:
    result = DiversityChecker(diversity_threshold=0.5, ngram_size=3).check(["", ""])

    assert result.avg_similarity == 0.0
    assert result.max_similarity == 0.0
    assert result.is_diverse is True


# ---------------------------------------------------------------------------
# runner gaps
# ---------------------------------------------------------------------------


def test_datasets_repr() -> None:
    example = dspy.Example(value=1)

    assert repr(Datasets([example], [example], [example])) == (
        "Datasets(train=1, val=1, test=1)"
    )


def test_run_result_repr_with_loaded_from() -> None:
    assert repr(RunResult(loaded_from="model.json")) == (
        "RunResult(loaded_from='model.json')"
    )


def test_run_result_repr_without_loaded_from() -> None:
    result = RunResult(baseline=0.1, optimized=0.2, improvement=0.1, saved_to="x")

    assert "baseline=0.1" in repr(result)
    assert "saved_to='x'" in repr(result)


def test_generation_result_repr() -> None:
    result = GenerationResult(
        rows=[],
        n_requested=2,
        n_produced=1,
        n_failed=1,
        seed_used=42,
        generation_time_seconds=1.234,
    )

    assert repr(result) == (
        "GenerationResult(n_produced=1, n_failed=1, generation_time_seconds=1.23)"
    )


def test_generation_failed_message() -> None:
    exc = GenerationFailed(n_requested=5, n_produced=2)

    assert exc.n_requested == 5
    assert exc.n_produced == 2
    assert str(exc) == "Generation failed: produced 2 of 5 requested rows"


def test_resolve_task_requires_rows() -> None:
    auto = AutoGEPA(module=DummyModule())

    with pytest.raises(ValueError, match="rows must be provided"):
        auto.datasets()


def test_resolve_task_requires_module() -> None:
    auto = AutoGEPA(rows=[{"message": "hi", "urgency": "low", "sentiment": "neutral"}])

    with pytest.raises(ValueError, match="module must be provided"):
        auto.datasets()


def test_load_metric_returns_callable(tmp_path: Path) -> None:
    metric_file = tmp_path / "metric.py"
    metric_file.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")
    auto = AutoGEPA(artifact_dir=tmp_path)
    auto._metric_file = metric_file

    metric = auto.load_metric()

    assert metric(None, None) == 1.0


def test_load_metric_requires_generated_metric_file() -> None:
    auto = AutoGEPA()

    with pytest.raises(RuntimeError, match="No metric file available"):
        auto.load_metric()


def test_datasets_reuses_existing_metric_file(tmp_path: Path) -> None:
    auto = AutoGEPA(
        rows=[{"message": "hi", "urgency": "low", "sentiment": "neutral"}],
        module=DummyModule(),
        name="ReuseMetric",
        artifact_dir=tmp_path,
    )
    run_dir = tmp_path / "ReuseMetric"
    run_dir.mkdir(parents=True)
    metric_file = run_dir / "metric.py"
    metric_file.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")

    with patch("dspy_auto_gepa.runner.generate_metric_file") as mock_generate:
        auto.datasets()

    mock_generate.assert_not_called()
    assert auto._metric_file == metric_file


def test_build_metric_reuses_existing_file(tmp_path: Path) -> None:
    auto = AutoGEPA(
        rows=[{"message": "hi", "urgency": "low", "sentiment": "neutral"}],
        module=DummyModule(),
        name="ReuseBuild",
        artifact_dir=tmp_path,
    )
    run_dir = tmp_path / "ReuseBuild"
    run_dir.mkdir(parents=True)
    metric_file = run_dir / "metric.py"
    metric_file.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")

    with patch("dspy_auto_gepa.runner.generate_metric_file") as mock_generate:
        result = auto.build_metric()

    mock_generate.assert_not_called()
    assert result == metric_file


def test_run_baseline_evaluates_module(tmp_path: Path) -> None:
    auto = AutoGEPA(
        module=DummyModule(),
        artifact_dir=tmp_path,
        num_threads=3,
    )
    auto._metric_file = tmp_path / "metric.py"
    datasets = Datasets(train=[], val=[], test=[])
    evaluator = MagicMock(return_value=SimpleNamespace(score=0.7))

    with (
        patch("dspy_auto_gepa.runner.dspy.Evaluate", return_value=evaluator),
        patch.object(auto, "load_metric", return_value="metric"),
    ):
        result = auto.run_baseline(datasets=datasets)

    assert result == {"score": 0.7}
    evaluator.assert_called_once()


def test_run_baseline_requires_module(tmp_path: Path) -> None:
    auto = AutoGEPA(artifact_dir=tmp_path)

    with pytest.raises(ValueError, match="module must be provided"):
        auto.run_baseline(datasets=Datasets(train=[], val=[], test=[]))


def test_train_compiles_with_run_dir(tmp_path: Path) -> None:
    auto = AutoGEPA(
        module=DummyModule(),
        artifact_dir=tmp_path,
        num_threads=2,
    )
    auto._run_dir = tmp_path / "run"
    datasets = Datasets(train=[], val=[], test=[])
    optimizer = MagicMock()
    optimizer.compile.return_value = "optimized"

    with (
        patch("dspy_auto_gepa.runner.dspy.GEPA", return_value=optimizer) as gepa_cls,
        patch.object(auto, "load_metric", return_value="metric"),
    ):
        result = auto.train(datasets=datasets)

    assert result == "optimized"
    assert gepa_cls.call_args.kwargs["log_dir"] == str(tmp_path / "run" / "gepa_logs")
    optimizer.compile.assert_called_once()


def test_train_uses_artifact_dir_when_no_run_dir(tmp_path: Path) -> None:
    auto = AutoGEPA(
        module=DummyModule(),
        artifact_dir=tmp_path,
    )
    optimizer = MagicMock()
    optimizer.compile.return_value = "optimized"

    with (
        patch("dspy_auto_gepa.runner.dspy.GEPA", return_value=optimizer) as gepa_cls,
        patch.object(auto, "load_metric", return_value="metric"),
    ):
        auto.train(datasets=Datasets(train=[], val=[], test=[]))

    assert gepa_cls.call_args.kwargs["log_dir"] == str(tmp_path / "gepa_logs")


def test_train_requires_module(tmp_path: Path) -> None:
    auto = AutoGEPA(artifact_dir=tmp_path)

    with pytest.raises(ValueError, match="module must be provided"):
        auto.train(datasets=Datasets(train=[], val=[], test=[]))


def test_compare_runs_both_modules(tmp_path: Path) -> None:
    auto = AutoGEPA(module=DummyModule(), artifact_dir=tmp_path)
    datasets = Datasets(train=[], val=[], test=[])
    scores = [{"score": 0.4}, {"score": 0.9}]

    with patch.object(auto, "run_baseline", side_effect=scores) as mock_baseline:
        result = auto.compare(
            optimized_module=DummyModule(),
            datasets=datasets,
        )

    assert result == RunResult(baseline=0.4, optimized=0.9, improvement=0.5)
    assert mock_baseline.call_count == 2


def test_promote_saves_module(tmp_path: Path) -> None:
    auto = AutoGEPA(artifact_dir=tmp_path)
    module = DummyModule()
    dest = tmp_path / "nested" / "model.json"

    with patch.object(module, "save") as mock_save:
        result = auto.promote(optimized_module=module, destination=dest)

    assert result == dest
    mock_save.assert_called_once_with(str(dest))


def test_generate_uses_fallbacks_and_returns_rows(tmp_path: Path) -> None:
    data_lm = MagicMock(spec=dspy.LM)
    auto = AutoGEPA(
        module=DummyModule(),
        artifact_dir=tmp_path,
        data_lm=data_lm,
        name="GenTask",
    )
    fake_gen = MagicMock()
    fake_gen.generate.return_value = GenerationResult(
        rows=[{"message": "generated"}],
        n_requested=1,
        n_produced=1,
        n_failed=0,
        seed_used=42,
        generation_time_seconds=0.1,
    )

    with patch("dspy_auto_gepa.generator.AutoData", return_value=fake_gen) as cls:
        rows = auto.generate(
            n=1,
            seed_examples=[{"message": "seed"}],
            schema="schema",
            output_path=tmp_path / "rows.jsonl",
        )

    assert rows == [{"message": "generated"}]
    kwargs = cls.call_args.kwargs
    assert kwargs["module"] is auto.module
    assert kwargs["data_lm"] is data_lm
    assert kwargs["schema"] == "schema"
    assert kwargs["name"] == "GenTask"
    assert kwargs["config"].seed_examples == [{"message": "seed"}]
    assert kwargs["config"].output_path == tmp_path / "rows.jsonl"


def test_generate_requires_module(tmp_path: Path) -> None:
    auto = AutoGEPA(artifact_dir=tmp_path)

    with pytest.raises(ValueError, match="module must be provided to generate data"):
        auto.generate()


def test_generate_uses_explicit_lm_over_constructor(tmp_path: Path) -> None:
    auto = AutoGEPA(module=DummyModule(), artifact_dir=tmp_path)
    fake_gen = MagicMock()
    fake_gen.generate.return_value = GenerationResult(
        rows=[],
        n_requested=0,
        n_produced=0,
        n_failed=0,
        seed_used=42,
        generation_time_seconds=0.0,
    )

    explicit_lm = MagicMock(spec=dspy.LM)

    with patch("dspy_auto_gepa.generator.AutoData", return_value=fake_gen) as cls:
        auto.generate(n=1, data_lm=explicit_lm)

    assert cls.call_args.kwargs["data_lm"] is explicit_lm
