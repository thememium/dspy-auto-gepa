from pathlib import Path
from typing import Any
from unittest.mock import patch

import dspy
import pytest

from dspy_auto_gepa import AutoGEPA, AutoGEPAConfig
from dspy_auto_gepa.data import split_examples, to_examples
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


def test_run_loads_existing_model(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    task_name = "TestTask"
    run_dir = artifact_dir / task_name
    run_dir.mkdir(parents=True)

    # Create dummy metric file so prepare() skips generation
    metric_file = run_dir / "metric.py"
    metric_file.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")

    # Create dummy saved model
    model_path = run_dir / f"optimized_{task_name}.json"
    model_path.write_text('{"version": "1.0"}')

    auto = AutoGEPA(
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
        artifact_dir=artifact_dir,
    )

    module = DummyModule()
    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]

    with patch.object(module, "load") as mock_load:
        results = auto.run(rows=rows, module=module, name=task_name, force=False)

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

    # Create dummy metric file so prepare() skips generation
    metric_file = run_dir / "metric.py"
    metric_file.write_text("def metric(example, pred, trace=None):\n    return 1.0\n")

    # Create dummy saved model
    model_path = run_dir / f"optimized_{task_name}.json"
    model_path.write_text('{"version": "1.0"}')

    auto = AutoGEPA(
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
        artifact_dir=artifact_dir,
    )

    module = DummyModule()
    rows = [{"message": "hello", "urgency": "low", "sentiment": "neutral"}]

    # force=True should bypass the load and attempt training
    # Since we don't have a real LLM, this will fail at train()
    # but it proves the force flag works to skip loading
    with pytest.raises(Exception):
        auto.run(rows=rows, module=module, name=task_name, force=True)


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
