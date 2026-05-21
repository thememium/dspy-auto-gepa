from pathlib import Path
from typing import Any
from unittest.mock import patch

import dspy
import pytest

from dspy_auto_gepa import AutoGEPA, AutoGEPAConfig
from dspy_auto_gepa.data import _to_dicts, split_examples, to_examples
from dspy_auto_gepa.metric_builder import _strip_markdown_fences


class DummyModule(dspy.Module):
    def __init__(self):
        super().__init__()

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


def test_config_validation_empty_fields():
    with pytest.raises(ValueError):
        AutoGEPAConfig(input_fields=[], output_fields=["b"])


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
    assert auto.config.input_fields == ["message"]
    assert auto.config.output_fields == ["label"]
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
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
    )
    assert auto.rows is rows
    assert auto.module is module
    assert auto.name == "TestTask"


def test_run_loads_existing_model(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    task_name = "TestTask"
    run_dir = artifact_dir / task_name
    run_dir.mkdir(parents=True)

    # Create dummy metric file so datasets() skips generation
    metric_file = run_dir / "metric.py"
    metric_file.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")

    # Create dummy saved model
    model_path = run_dir / f"optimized_{task_name}.json"
    model_path.write_text('{"version": "1.0"}')

    module = DummyModule()
    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name=task_name,
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
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

    # Create dummy metric file so datasets() skips generation
    metric_file = run_dir / "metric.py"
    metric_file.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")

    # Create dummy saved model
    model_path = run_dir / f"optimized_{task_name}.json"
    model_path.write_text('{"version": "1.0"}')

    module = DummyModule()
    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name=task_name,
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
        artifact_dir=artifact_dir,
    )

    # force=True should bypass the load and attempt training
    # Since we don't have a real LLM, this will fail at train()
    # but it proves the force flag works to skip loading
    with pytest.raises(Exception):
        auto.run(force=True)


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
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
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
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
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
    from unittest.mock import patch

    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]
    module = DummyModule()

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name="TestBuildMetric",
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
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
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
        artifact_dir=tmp_path,
    )

    metric_file = auto.build_metric(metric=custom_metric)
    assert metric_file == custom_metric


def test_build_metric_with_out_path(tmp_path: Path) -> None:
    from unittest.mock import patch

    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]
    module = DummyModule()
    custom_out = tmp_path / "my_metric.py"

    auto = AutoGEPA(
        rows=rows,
        module=module,
        name="TestBuildMetricOutPath",
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
        artifact_dir=tmp_path,
    )

    with patch("dspy_auto_gepa.runner.generate_metric_file") as mock_generate:
        metric_file = auto.build_metric(out_path=custom_out)

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["out_path"] == custom_out
    assert metric_file == custom_out


def test_build_metric_passes_generator_config(tmp_path: Path) -> None:
    from unittest.mock import patch

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
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
        artifact_dir=tmp_path,
        metric_generator_signature=CustomMetricSig,
        metric_generator_module=dspy.ChainOfThought,
    )

    with patch("dspy_auto_gepa.runner.generate_metric_file") as mock_generate:
        auto.build_metric()

        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["metric_generator_signature"] == CustomMetricSig
        assert call_kwargs["metric_generator_module"] == dspy.ChainOfThought
