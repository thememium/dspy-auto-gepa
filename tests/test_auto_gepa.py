from pathlib import Path
from typing import Any
from unittest.mock import patch

import dspy
import pytest

from dspy_auto_gepa import AutoGEPA, AutoGEPAConfig
from dspy_auto_gepa.data import _apply_mapping, _to_dicts, split_examples, to_examples
from dspy_auto_gepa.metric_builder import _strip_markdown_fences
from dspy_auto_gepa.runner import RunResult


class TicketSignature(dspy.Signature):
    """Classify support tickets."""

    message: str = dspy.InputField()
    urgency: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()


class DummyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = TicketSignature

    def forward(self, message: str) -> dspy.Prediction:
        return dspy.Prediction(urgency="low", sentiment="neutral")


def test_config_defaults():
    cfg = AutoGEPAConfig(
        input_fields=["a"],
        output_fields=["b"],
    )
    assert cfg.split == (0.7, 0.2, 0.1)
    assert cfg.seed == 42
    assert cfg.gepa_auto == "light"


def test_config_validation_bad_split():
    with pytest.raises(ValueError):
        AutoGEPAConfig(
            input_fields=["a"],
            output_fields=["b"],
            split=(0.5, 0.3, 0.3),
        )


def test_config_validation_bad_gepa_auto():
    with pytest.raises(ValueError):
        invalid: Any = "invalid"
        AutoGEPAConfig(
            input_fields=["a"],
            output_fields=["b"],
            gepa_auto=invalid,
        )


def test_flat_autogepa_constructor():
    auto = AutoGEPA(
        input_fields=["message"],
        output_fields=["label"],
        split=(0.8, 0.1, 0.1),
        seed=123,
        gepa_auto="medium",
        num_threads=4,
    )
    assert auto.config.input_fields is None  # stored on _raw_*, not config
    assert auto._raw_input_fields == ["message"]
    assert auto._raw_output_fields == ["label"]
    assert auto.config.split == (0.8, 0.1, 0.1)
    assert auto.config.seed == 123
    assert auto.config.gepa_auto == "medium"
    assert auto.config.num_threads == 4
    assert auto.config.artifact_dir.exists()
    assert auto.rows is None
    assert auto.module is None
    assert auto.name is None


def test_autogepa_stores_rows_module_name():
    module = DummyModule()
    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]
    auto = AutoGEPA(
        rows=rows,
        module=module,
        name="TestTask",
    )
    assert auto.rows is rows
    assert auto.module is module
    assert auto.name == "TestTask"


def test_run_loads_existing_model(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    task_name = "TestTask"
    run_dir = artifact_dir / task_name
    run_dir.mkdir(parents=True)

    metric_file = run_dir / "metric.py"
    metric_file.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")

    model_path = run_dir / f"optimized_{task_name}.json"
    model_path.write_text('{"version": "1.0"}')

    module = DummyModule()
    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name=task_name,
        artifact_dir=artifact_dir,
    )

    with patch.object(module, "load") as mock_load:
        results = auto.run(force=False)

        mock_load.assert_called_once_with(str(model_path))
    assert results.loaded_from == str(model_path)
    assert results.baseline is None
    assert results.optimized is None
    assert results.improvement is None


def test_run_force_retrains(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    task_name = "TestTask"
    run_dir = artifact_dir / task_name
    run_dir.mkdir(parents=True)

    metric_file = run_dir / "metric.py"
    metric_file.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")

    model_path = run_dir / f"optimized_{task_name}.json"
    model_path.write_text('{"version": "1.0"}')

    module = DummyModule()
    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name=task_name,
        artifact_dir=artifact_dir,
    )

    with patch.object(module, "load") as mock_load:
        with patch.object(auto, "train") as mock_train:
            with patch.object(auto, "compare") as mock_compare:
                with patch.object(auto, "promote") as mock_promote:
                    mock_train.return_value = module
                    mock_compare.return_value = RunResult(
                        baseline=0.5, optimized=0.8, improvement=0.3
                    )

                    result = auto.run(force=True)

                    mock_load.assert_not_called()
                    mock_train.assert_called_once()
                    mock_compare.assert_called_once()
                    mock_promote.assert_called_once()

    assert result.loaded_from is None
    assert result.baseline == 0.5
    assert result.optimized == 0.8
    assert result.improvement == 0.3


def test_to_examples():
    rows = [
        {"message": "hello", "label": "greeting"},
        {"message": "bye", "label": "farewell"},
    ]
    examples = to_examples(rows, input_fields=["message"], output_fields=["label"])
    assert len(examples) == 2
    assert examples[0].message == "hello"
    assert examples[0].label == "greeting"


def test_split_examples_two_way():
    rows = [{"x": i, "y": i} for i in range(10)]
    examples = to_examples(rows, input_fields=["x"], output_fields=["y"])
    train, val, test = split_examples(examples, split=(0.7, 0.3), seed=1)
    assert len(train) == 7
    assert len(val) == 0
    assert len(test) == 3


def test_split_examples_three_way():
    rows = [{"x": i, "y": i} for i in range(100)]
    examples = to_examples(rows, input_fields=["x"], output_fields=["y"])
    train, val, test = split_examples(examples, split=(0.7, 0.2, 0.1), seed=1)
    assert len(train) == 70
    assert len(val) == 20
    assert len(test) == 10


def test_strip_markdown_fences():
    source = "```python\ndef foo():\n    pass\n```"
    assert _strip_markdown_fences(source) == "def foo():\n    pass"


def test_strip_markdown_fences_no_fences():
    source = "def foo():\n    pass"
    assert _strip_markdown_fences(source) == source


def test_prepare_uses_custom_metric_path(tmp_path: Path) -> None:
    custom_metric = tmp_path / "custom_metric.py"
    custom_metric.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")

    module = DummyModule()
    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name="CustomMetricTask",
        artifact_dir=tmp_path,
    )

    auto.datasets(metric=custom_metric)
    assert auto._metric_file == custom_metric


def test_prepare_uses_constructor_metric(tmp_path: Path) -> None:
    custom_metric = tmp_path / "my_metric.py"
    custom_metric.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")

    module = DummyModule()
    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]

    auto = AutoGEPA(
        rows=rows,
        module=module,
        metric=custom_metric,
        name="CtorMetricTask",
        artifact_dir=tmp_path,
    )

    auto.datasets()
    assert auto._metric_file == custom_metric


def test_to_dicts_pandas_dataframe():
    import pandas as pd

    df = pd.DataFrame({"message": ["hello"], "label": ["greeting"]})
    result = _to_dicts(df)
    assert isinstance(result, list)
    assert result == [{"message": "hello", "label": "greeting"}]


def test_to_dicts_list_of_dicts():
    rows = [{"message": "hello", "label": "greeting"}]
    result = _to_dicts(rows)
    assert result is rows


def test_to_dicts_unsupported():
    with pytest.raises(TypeError):
        _to_dicts("not a dataframe")


def test_build_metric_generates_file(tmp_path: Path) -> None:
    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]
    module = DummyModule()

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name="TestBuildMetric",
        artifact_dir=tmp_path,
    )

    with patch("dspy_auto_gepa.runner.generate_metric_file") as mock_generate:
        metric_file = auto.build_metric()

        mock_generate.assert_called_once()
    assert metric_file == tmp_path / "TestBuildMetric" / "metric.py"


def test_build_metric_returns_custom_path(tmp_path: Path) -> None:
    custom_metric = tmp_path / "custom_metric.py"
    custom_metric.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")

    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]
    module = DummyModule()

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name="TestBuildMetricCustom",
        artifact_dir=tmp_path,
    )

    metric_file = auto.build_metric(metric=custom_metric)
    assert metric_file == custom_metric


def test_build_metric_with_out_path(tmp_path: Path) -> None:
    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]
    module = DummyModule()
    custom_out = tmp_path / "my_metric.py"

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name="TestBuildMetricOutPath",
        artifact_dir=tmp_path,
    )

    with patch("dspy_auto_gepa.runner.generate_metric_file") as mock_generate:
        metric_file = auto.build_metric(out_path=custom_out)

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["out_path"] == custom_out
    assert metric_file == custom_out


def test_build_metric_passes_generator_config(tmp_path: Path) -> None:
    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]
    module = DummyModule()

    class CustomMetricSig(dspy.Signature):
        input_keys: list[str] = dspy.InputField()
        output_keys: list[str] = dspy.InputField()
        sample_rows_json: str = dspy.InputField()
        module_repr: str = dspy.InputField()
        metric_source: str = dspy.OutputField()

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name="TestGeneratorConfig",
        artifact_dir=tmp_path,
        metric_generator_signature=CustomMetricSig,
        metric_generator_module=dspy.ChainOfThought,
    )

    with patch("dspy_auto_gepa.runner.generate_metric_file") as mock_generate:
        auto.build_metric()

        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["metric_generator_signature"] == CustomMetricSig
        assert call_kwargs["metric_generator_module"] == dspy.ChainOfThought


# ---- New tests for field inference and mapping ----


def test_infer_fields_from_module_signature(tmp_path: Path) -> None:
    """When no fields are provided, infer from module signature (exact match)."""
    module = DummyModule()
    rows = [
        {"message": "hello", "urgency": "low", "sentiment": "neutral"},
        {"message": "bye", "urgency": "high", "sentiment": "negative"},
        {"message": "help", "urgency": "high", "sentiment": "negative"},
        {"message": "thanks", "urgency": "low", "sentiment": "positive"},
        {"message": "check", "urgency": "medium", "sentiment": "neutral"},
    ]

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name="InferFields",
        artifact_dir=tmp_path,
    )

    with patch("dspy_auto_gepa.runner.generate_metric_file"):
        ds = auto.datasets()
    assert len(ds.train) >= 1


def test_infer_fields_with_dict_mapping(tmp_path: Path) -> None:
    module = DummyModule()
    rows = [
        {"msg": "hello", "urg": "high", "sent": "negative"},
        {"msg": "bye", "urg": "low", "sent": "positive"},
        {"msg": "help", "urg": "high", "sent": "negative"},
    ]

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name="MappedFields",
        artifact_dir=tmp_path,
        input_fields={"msg": "message"},
        output_fields={"urg": "urgency", "sent": "sentiment"},
    )

    with patch("dspy_auto_gepa.runner.generate_metric_file"):
        ds = auto.datasets()
    assert len(ds.train) >= 1
    all_msgs = [ex.message for ex in ds.train + ds.val + ds.test]
    assert "hello" in all_msgs
    assert "bye" in all_msgs


def test_infer_fields_mismatch_raises_error():
    module = DummyModule()
    rows = [{"msg": "hello", "urg": "high", "sent": "negative"}]

    auto = AutoGEPA(
        rows=rows,
        module=module,
    )

    with pytest.raises(ValueError) as exc:
        auto.datasets()

    assert "Row columns do not match module signature fields" in str(exc.value)
    assert "Missing from rows" in str(exc.value)


def test_apply_mapping():
    rows = [
        {"msg": "hello", "urg": "high"},
        {"msg": "bye", "urg": "low"},
    ]
    mapping = {"msg": "message", "urg": "urgency"}
    result = _apply_mapping(rows, mapping)
    assert result == [
        {"message": "hello", "urgency": "high"},
        {"message": "bye", "urgency": "low"},
    ]


def test_partial_explicit_fields_infer_rest(tmp_path: Path) -> None:
    module = DummyModule()
    rows = [
        {"message": "hello", "urgency": "low", "sentiment": "neutral"},
        {"message": "bye", "urgency": "high", "sentiment": "negative"},
        {"message": "help", "urgency": "high", "sentiment": "negative"},
        {"message": "thanks", "urgency": "low", "sentiment": "positive"},
        {"message": "check", "urgency": "medium", "sentiment": "neutral"},
    ]

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name="PartialFields",
        artifact_dir=tmp_path,
        input_fields=["message"],
    )

    ds = auto.datasets()
    assert len(ds.train) >= 1
    all_msgs = [ex.message for ex in ds.train + ds.val + ds.test]
    assert "hello" in all_msgs
    assert "bye" in all_msgs
