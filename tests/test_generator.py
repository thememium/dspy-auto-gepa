"""Comprehensive tests for AutoData and StreamingDatasetWriter.

All LLM calls are mocked — no real API traffic.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import dspy
import pandas as pd
import pytest

from dspy_auto_gepa.config import AutoDataConfig
from dspy_auto_gepa.generator import (
    AutoData,
    StreamingDatasetWriter,
    _compute_output_combos,
)
from dspy_auto_gepa.runner import GenerationResult

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


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


def _make_autodata_config(**overrides: Any) -> AutoDataConfig:
    """Return an AutoDataConfig with judge/diversity disabled for simpler tests."""
    defaults: dict[str, Any] = dict(
        n=5,
        seed=42,
        max_retries=3,
        num_threads=1,
        chunk_size=1,
        judge_enabled=False,
        diversity_enabled=False,
        validators_enabled=False,
        rejection_sampling_enabled=False,
    )
    defaults.update(overrides)
    return AutoDataConfig(**defaults)


def _make_sync_parallel_mock():
    """Create a mock for dspy.Parallel that executes tasks synchronously.

    Returns (mock_parallel_cls, executed_tasks) where executed_tasks is a list
    of (module, example) pairs that were executed.
    """
    executed_tasks = []

    class SyncParallel:
        def __init__(self, **kwargs):
            pass

        def __call__(self, tasks):
            results = []
            for module, example in tasks:
                executed_tasks.append((module, example))
                try:
                    result = module(**{k: example[k] for k in example.keys()})
                    results.append(result)
                except Exception as e:
                    results.append(e)
            return results

    mock_cls = MagicMock(side_effect=lambda **kwargs: SyncParallel(**kwargs))
    return mock_cls, executed_tasks


# ---------------------------------------------------------------------------
# StreamingDatasetWriter tests
# ---------------------------------------------------------------------------


class TestStreamingDatasetWriter:
    """Tests for StreamingDatasetWriter write / read / resume logic."""

    def test_jsonl_write_row(self, tmp_path: Path) -> None:
        """Write 2 rows to .jsonl — verify file has 2 valid-JSON lines."""
        path = tmp_path / "data.jsonl"
        writer = StreamingDatasetWriter(path)
        writer.write_row({"message": "hello", "urgency": "low"})
        writer.write_row({"message": "world", "urgency": "high"})

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        row0 = json.loads(lines[0])
        row1 = json.loads(lines[1])
        assert row0["message"] == "hello"
        assert row1["urgency"] == "high"

    def test_csv_write_row(self, tmp_path: Path) -> None:
        """Write 2 rows to .csv — verify header + 2 data rows."""
        path = tmp_path / "data.csv"
        writer = StreamingDatasetWriter(path)
        writer.write_row({"message": "hello", "urgency": "low"})
        writer.write_row({"message": "world", "urgency": "high"})

        df = pd.read_csv(path)
        assert len(df) == 2
        assert list(df.columns) == ["message", "urgency"]
        assert df.iloc[0]["message"] == "hello"
        assert df.iloc[1]["urgency"] == "high"

    def test_parquet_write_rows(self, tmp_path: Path) -> None:
        """Write rows to .parquet, read back with pandas, verify data."""
        path = tmp_path / "data.parquet"
        writer = StreamingDatasetWriter(path)
        writer.write_row({"message": "hello", "urgency": "low"})
        writer.write_row({"message": "world", "urgency": "high"})

        df = pd.read_parquet(path)
        assert len(df) == 2
        assert df.iloc[0]["message"] == "hello"
        assert df.iloc[1]["urgency"] == "high"

    def test_format_detection(self, tmp_path: Path) -> None:
        """Extension → format mapping. Unsupported extension raises ValueError."""
        assert StreamingDatasetWriter._detect_format(tmp_path / "a.jsonl") == "jsonl"
        assert StreamingDatasetWriter._detect_format(tmp_path / "a.json") == "jsonl"
        assert StreamingDatasetWriter._detect_format(tmp_path / "a.csv") == "csv"
        assert (
            StreamingDatasetWriter._detect_format(tmp_path / "a.parquet") == "parquet"
        )
        assert StreamingDatasetWriter._detect_format(tmp_path / "a.pq") == "parquet"

        with pytest.raises(ValueError, match="Unsupported file extension"):
            StreamingDatasetWriter._detect_format(tmp_path / "data.xyz")

    def test_resume_from_existing(self, tmp_path: Path) -> None:
        """Write 3 rows, re-open same file — row_count() == 3."""
        path = tmp_path / "data.jsonl"
        writer1 = StreamingDatasetWriter(path)
        for i in range(3):
            writer1.write_row({"idx": i, "val": f"row{i}"})
        assert writer1.row_count() == 3

        # New writer on same file should detect existing rows
        writer2 = StreamingDatasetWriter(path)
        assert writer2.row_count() == 3
        assert len(writer2.read_rows()) == 3


# ---------------------------------------------------------------------------
# AutoData constructor tests
# ---------------------------------------------------------------------------


class TestAutoDataConstructor:
    """Tests for AutoData __init__, field inference, and validation."""

    def test_field_inference(self) -> None:
        """Fields inferred from TicketSignature: input=['message'],
        output contains urgency+sentiment, reasoning excluded."""
        module = DummyModule()
        gen = AutoData(
            module=module,
            data_lm=MagicMock(),
            description="test task",
        )
        assert gen.input_fields == ["message"]
        assert "urgency" in gen.output_fields
        assert "sentiment" in gen.output_fields
        assert "reasoning" not in gen.output_fields

    def test_field_inference_strips_reasoning(self) -> None:
        """When signature includes reasoning (e.g. ChainOfThought), AutoData strips it."""

        class CoTSignature(dspy.Signature):
            """Classify support tickets."""

            message: str = dspy.InputField()
            reasoning: str = dspy.OutputField()
            urgency: str = dspy.OutputField()

        module = MagicMock()
        module.signature = CoTSignature
        module.named_predictors = lambda: []

        gen = AutoData(module=module, data_lm=MagicMock(), description="test")
        assert "reasoning" not in gen.output_fields
        assert "urgency" in gen.output_fields

    def test_data_lm_fallback(self) -> None:
        """When data_lm is not provided, falls back to dspy.settings.lm."""
        fake_lm = MagicMock()
        dspy.configure(lm=fake_lm)
        try:
            module = DummyModule()
            gen = AutoData(module=module, description="test")
            assert gen.data_lm is fake_lm
        finally:
            dspy.configure(lm=None)

    def test_data_lm_explicit(self) -> None:
        """When data_lm is provided, it is used directly."""
        explicit_lm = MagicMock()
        module = DummyModule()
        gen = AutoData(module=module, data_lm=explicit_lm, description="test")
        assert gen.data_lm is explicit_lm

    def test_description_required(self) -> None:
        """Module with empty docstring and no description → ValueError."""

        class NoDocSignature(dspy.Signature):
            """ """

            message: str = dspy.InputField()
            urgency: str = dspy.OutputField()

        NoDocSignature.__doc__ = ""
        module = MagicMock()
        module.signature = NoDocSignature
        module.named_predictors = lambda: []

        with pytest.raises(ValueError, match="no description was provided"):
            AutoData(module=module, data_lm=MagicMock())

    def test_description_override(self) -> None:
        """Module with empty docstring but description provided → no error."""

        class NoDocSig2(dspy.Signature):
            """ """

            message: str = dspy.InputField()
            urgency: str = dspy.OutputField()

        NoDocSig2.__doc__ = ""
        module = MagicMock()
        module.signature = NoDocSig2
        module.named_predictors = lambda: []

        gen = AutoData(module=module, data_lm=MagicMock(), description="my task")
        assert gen.description == "my task"

    def test_from_csv(self, tmp_path: Path) -> None:
        """from_csv loads seed_examples from a CSV file."""
        csv_path = tmp_path / "seeds.csv"
        csv_path.write_text(
            "message,urgency,sentiment\nhello,low,neutral\nworld,high,negative\n"
        )

        module = DummyModule()
        gen = AutoData.from_csv(
            csv_path, module=module, data_lm=MagicMock(), description="test"
        )

        assert gen.seed_examples is not None
        assert len(gen.seed_examples) == 2
        assert gen.seed_examples[0]["message"] == "hello"
        assert gen.seed_examples[1]["urgency"] == "high"

    def test_from_json(self, tmp_path: Path) -> None:
        """from_json loads seed_examples from a JSON file."""
        json_path = tmp_path / "seeds.json"
        json_path.write_text(
            json.dumps(
                [
                    {"message": "hi", "urgency": "low", "sentiment": "positive"},
                ]
            )
        )

        module = DummyModule()
        gen = AutoData.from_json(
            json_path, module=module, data_lm=MagicMock(), description="test"
        )

        assert gen.seed_examples is not None
        assert len(gen.seed_examples) == 1
        assert gen.seed_examples[0]["message"] == "hi"


# ---------------------------------------------------------------------------
# AutoData generation tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestAutoDataGeneration:
    """Tests for _generate_inputs, _generate_outputs, and generate() with mocked LLM."""

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_inputs_mocked(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
    ) -> None:
        """_generate_inputs returns correct count and field keys from mocked LLM."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = dspy.Prediction(
            generated_inputs='[{"message": "test1"}, {"message": "test2"}]'
        )
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, _ = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=2),
        )

        inputs = gen._generate_inputs(2, None, "Classify tickets")

        assert len(inputs) == 2
        assert inputs[0]["message"] == "test1"
        assert inputs[1]["message"] == "test2"
        for row in inputs:
            assert "message" in row

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_inputs_json_retry(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
    ) -> None:
        """First call returns invalid JSON, second succeeds → retry works."""
        mock_predictor = MagicMock()
        mock_predictor.side_effect = [
            dspy.Prediction(generated_inputs="this is not json"),
            dspy.Prediction(generated_inputs='[{"message": "recovered"}]'),
        ]
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, _ = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1),
        )

        inputs = gen._generate_inputs(1, None, "Classify tickets")

        assert len(inputs) == 1
        assert inputs[0]["message"] == "recovered"

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_outputs_mocked(self, mock_predict_cls: MagicMock) -> None:
        """_generate_outputs returns correct output fields from mocked LLM."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = dspy.Prediction(
            generated_outputs=[{"urgency": "high", "sentiment": "negative"}]
        )
        mock_predict_cls.return_value = mock_predictor

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(),
        )

        outputs, _ = gen._generate_outputs(
            [{"message": "server is down"}],
            "Classify tickets",
        )

        assert len(outputs) == 1
        assert outputs[0] is not None
        assert outputs[0]["urgency"] == "high"
        assert outputs[0]["sentiment"] == "negative"

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_outputs_json_retry(self, mock_predict_cls: MagicMock) -> None:
        """Output generation retries on invalid output."""
        mock_predictor = MagicMock()
        call_count = 0

        def predict_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return dspy.Prediction(generated_outputs="bad data")
            return dspy.Prediction(
                generated_output='{"urgency": "low", "sentiment": "neutral"}'
            )

        mock_predictor.side_effect = predict_side_effect
        mock_predict_cls.return_value = mock_predictor

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1),
        )

        outputs, _ = gen._generate_outputs(
            [{"message": "hello"}],
            "Classify tickets",
        )

        assert len(outputs) == 1
        assert outputs[0] is not None
        assert outputs[0]["urgency"] == "low"
        assert mock_predictor.call_count == 2

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_full_flow(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Mocked full generate() → returns GenerationResult with correct row count."""
        call_count = 0

        def predict_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return dspy.Prediction(
                    generated_inputs='[{"message": "ticket1"}, {"message": "ticket2"}]'
                )
            else:
                return dspy.Prediction(
                    generated_outputs=[
                        {"urgency": "high", "sentiment": "negative"},
                        {"urgency": "low", "sentiment": "positive"},
                    ]
                )

        mock_predictor = MagicMock()
        mock_predictor.side_effect = predict_side_effect
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, _ = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        output_path = tmp_path / "output.jsonl"
        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=2),
        )

        result = gen.generate(n=2, output_path=output_path, force=True)

        assert isinstance(result, GenerationResult)
        assert result.n_produced == 2
        assert result.n_requested == 2
        assert len(result.rows) == 2
        for row in result.rows:
            assert "message" in row
            assert "urgency" in row
            assert "sentiment" in row
        assert output_path.exists()

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_resume(self, mock_predict_cls: MagicMock, tmp_path: Path) -> None:
        """When file already has >= n rows and force=False, returns existing rows."""
        output_path = tmp_path / "output.jsonl"

        writer = StreamingDatasetWriter(output_path)
        writer.write_row(
            {"message": "existing1", "urgency": "low", "sentiment": "neutral"}
        )
        writer.write_row(
            {"message": "existing2", "urgency": "high", "sentiment": "negative"}
        )
        writer.write_row(
            {"message": "existing3", "urgency": "medium", "sentiment": "neutral"}
        )

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=3),
        )

        result = gen.generate(n=3, output_path=output_path, force=False)

        assert result.n_produced == 3
        assert len(result.rows) == 3
        assert result.rows[0]["message"] == "existing1"
        mock_predict_cls.assert_not_called()

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_force_overwrites(
        self, mock_predict_cls: MagicMock, tmp_path: Path
    ) -> None:
        """force=True ignores existing rows and generates fresh data."""
        output_path = tmp_path / "output.jsonl"

        writer = StreamingDatasetWriter(output_path)
        writer.write_row({"message": "old1", "urgency": "low", "sentiment": "neutral"})

        call_count = 0

        def predict_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return dspy.Prediction(generated_inputs='[{"message": "new1"}]')
            else:
                return dspy.Prediction(
                    generated_outputs=[{"urgency": "high", "sentiment": "positive"}]
                )

        mock_predictor = MagicMock()
        mock_predictor.side_effect = predict_side_effect
        mock_predict_cls.return_value = mock_predictor

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1),
        )

        result = gen.generate(n=1, output_path=output_path, force=True)

        assert result.n_produced == 1
        assert result.rows[0]["message"] == "new1"

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_outputs_all_retries_fail(
        self, mock_predict_cls: MagicMock
    ) -> None:
        """When all retries fail for output, None is returned."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = MagicMock(
            generated_outputs="not valid json at all"
        )
        mock_predict_cls.return_value = mock_predictor

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(max_retries=2),
        )

        outputs, _ = gen._generate_outputs(
            [{"message": "test"}],
            "Classify tickets",
        )

        assert len(outputs) == 1
        assert outputs[0] is None

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_inputs_respects_n(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
    ) -> None:
        """_generate_inputs returns exactly n rows even if LLM returns extra."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = dspy.Prediction(
            generated_inputs=json.dumps([{"message": f"msg{i}"} for i in range(5)])
        )
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, _ = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=3),
        )

        inputs = gen._generate_inputs(3, None, "Classify tickets")
        assert len(inputs) == 3

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_inputs_with_seed_examples(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
    ) -> None:
        """Seed examples are passed through for diversity reference."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = dspy.Prediction(
            generated_inputs='[{"message": "generated1"}]'
        )
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, executed_tasks = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1),
        )

        seeds = [{"message": "seed1", "urgency": "low", "sentiment": "neutral"}]
        inputs = gen._generate_inputs(1, seeds, "Classify tickets")

        assert len(inputs) == 1
        assert len(executed_tasks) > 0
        _, example = executed_tasks[0]
        assert "seed1" in example.recent_inputs_json

    def test_resolve_seeds_none_returns_stored(self) -> None:
        """_resolve_seeds(None) returns self.seed_examples."""
        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="test",
        )
        gen.seed_examples = [{"message": "stored"}]
        assert gen._resolve_seeds(None) == [{"message": "stored"}]

    def test_resolve_seeds_list_passthrough(self) -> None:
        """_resolve_seeds(list) returns the list as-is."""
        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="test",
        )
        seeds = [{"message": "direct"}]
        assert gen._resolve_seeds(seeds) is seeds

    def test_resolve_seeds_unsupported_extension(self, tmp_path: Path) -> None:
        """_resolve_seeds with unsupported file extension raises ValueError."""
        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="test",
        )
        bad_file = tmp_path / "seeds.xml"
        bad_file.write_text("<data/>")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            gen._resolve_seeds(str(bad_file))


# ---------------------------------------------------------------------------
# Validation & sanitization tests
# ---------------------------------------------------------------------------


class TestValidateAndSanitizeRow:
    """Tests for _validate_and_sanitize_row."""

    def test_strips_emoji(self) -> None:
        from dspy_auto_gepa.generator import _validate_and_sanitize_row

        row = {"message": "\u26a0\ufe0f Server is down", "urgency": "high"}
        cleaned, errors = _validate_and_sanitize_row(row, ["message", "urgency"])
        assert "\u26a0" not in cleaned["message"]
        assert "Server is down" in cleaned["message"]
        assert errors == []

    def test_rejects_empty_field(self) -> None:
        from dspy_auto_gepa.generator import _validate_and_sanitize_row

        row = {"message": "", "urgency": "high"}
        _, errors = _validate_and_sanitize_row(row, ["message", "urgency"])
        assert any("must not be empty" in e for e in errors)

    def test_rejects_whitespace_only(self) -> None:
        from dspy_auto_gepa.generator import _validate_and_sanitize_row

        row = {"message": "   ", "urgency": "high"}
        _, errors = _validate_and_sanitize_row(row, ["message", "urgency"])
        assert any("must not be empty" in e for e in errors)

    def test_rejects_enum_violation(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata
        from dspy_auto_gepa.generator import _validate_and_sanitize_row

        metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "medium", "high"],
                ),
            ],
            output_fields=["urgency"],
        )
        row = {"urgency": "critical"}
        _, errors = _validate_and_sanitize_row(row, ["urgency"], metadata)
        assert any("not in allowed" in e for e in errors)

    def test_accepts_valid_enum(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata
        from dspy_auto_gepa.generator import _validate_and_sanitize_row

        metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "medium", "high"],
                ),
            ],
            output_fields=["urgency"],
        )
        row = {"urgency": "high"}
        cleaned, errors = _validate_and_sanitize_row(row, ["urgency"], metadata)
        assert errors == []
        assert cleaned["urgency"] == "high"

    def test_normalizes_unicode_whitespace(self) -> None:
        from dspy_auto_gepa.generator import _validate_and_sanitize_row

        row = {"message": "Hello\u202fWorld\u200b"}
        cleaned, _ = _validate_and_sanitize_row(row, ["message"])
        assert "\u202f" not in cleaned["message"]
        assert "\u200b" not in cleaned["message"]


class TestBuildValidator:
    """Tests for _build_validator."""

    def test_rejects_empty(self) -> None:
        from dspy_auto_gepa.generator import _build_validator

        v = _build_validator(["message"])
        result = v.validate({"message": ""})
        assert not result.is_valid

    def test_rejects_emoji(self) -> None:
        from dspy_auto_gepa.generator import _build_validator

        v = _build_validator(["message"])
        result = v.validate({"message": "\U0001f600 hello"})
        assert not result.is_valid

    def test_rejects_bad_enum(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata
        from dspy_auto_gepa.generator import _build_validator

        metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "medium", "high"],
                ),
            ],
            output_fields=["urgency"],
        )
        v = _build_validator(["urgency"], metadata)
        result = v.validate({"urgency": "critical"})
        assert not result.is_valid

    def test_passes_valid_row(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata
        from dspy_auto_gepa.generator import _build_validator

        metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "medium", "high"],
                ),
            ],
            output_fields=["urgency"],
        )
        v = _build_validator(["urgency"], metadata)
        result = v.validate({"urgency": "high"})
        assert result.is_valid


# ---------------------------------------------------------------------------
# Signature metadata extraction tests
# ---------------------------------------------------------------------------


class TestExtractSignatureMetadata:
    """Tests for extract_signature_metadata."""

    def test_extracts_fields(self) -> None:
        from dspy_auto_gepa.data import extract_signature_metadata

        meta = extract_signature_metadata(TicketSignature)
        names = [f.name for f in meta.fields]
        assert "message" in names
        assert "urgency" in names
        assert "sentiment" in names

    def test_input_output_classification(self) -> None:
        from dspy_auto_gepa.data import extract_signature_metadata

        meta = extract_signature_metadata(TicketSignature)
        assert meta.input_fields == ["message"]
        assert "urgency" in meta.output_fields
        assert "sentiment" in meta.output_fields

    def test_infers_enum_from_seeds(self) -> None:
        from dspy_auto_gepa.data import extract_signature_metadata

        seeds = [
            {"urgency": "high"},
            {"urgency": "low"},
            {"urgency": "medium"},
            {"urgency": "high"},
        ]
        meta = extract_signature_metadata(TicketSignature, seed_examples=seeds)
        urg = meta.get("urgency")
        assert urg is not None
        assert urg.allowed_values is not None
        assert set(urg.allowed_values) == {"high", "low", "medium"}

    def test_no_enum_when_high_cardinality(self) -> None:
        from dspy_auto_gepa.data import extract_signature_metadata

        seeds = [{"message": f"msg{i}"} for i in range(20)]
        meta = extract_signature_metadata(TicketSignature, seed_examples=seeds)
        msg = meta.get("message")
        assert msg is not None
        assert msg.allowed_values is None

    def test_to_prompt_spec(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata

        meta = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="How urgent",
                    is_input=False,
                    allowed_values=["low", "high"],
                ),
            ],
        )
        spec = meta.to_prompt_spec()
        assert "urgency" in spec
        assert "low" in spec
        assert "high" in spec
        assert "How urgent" in spec


# ---------------------------------------------------------------------------
# Field spec in generation prompts
# ---------------------------------------------------------------------------


class TestFieldSpecInPrompts:
    """Tests that _field_spec_json produces correct spec for LLM prompts."""

    def test_field_spec_includes_allowed_values(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata
        from dspy_auto_gepa.generator import AutoData

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="test",
        )
        metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="Urgency level",
                    is_input=False,
                    allowed_values=["low", "medium", "high"],
                ),
            ],
        )
        spec = gen._field_spec_json(["urgency"], metadata)
        parsed = json.loads(spec)
        assert parsed["urgency"]["allowed"] == ["low", "medium", "high"]
        assert parsed["urgency"]["desc"] == "Urgency level"


# ---------------------------------------------------------------------------
# Generation rejects bad data (mocked LLM)
# ---------------------------------------------------------------------------


class TestGenerationRejectsBadData:
    """Tests that generation retries when LLM produces bad data."""

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_rejects_empty_input_and_retries(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
    ) -> None:
        mock_predictor = MagicMock()
        mock_predictor.side_effect = [
            dspy.Prediction(generated_inputs='[{"message": ""}]'),
            dspy.Prediction(generated_inputs='[{"message": "valid message"}]'),
        ]
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, _ = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1),
        )

        inputs = gen._generate_inputs(1, None, "Classify tickets")
        assert len(inputs) == 1
        assert inputs[0]["message"] == "valid message"

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_rejects_emoji_input_and_retries(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
    ) -> None:
        mock_predictor = MagicMock()
        mock_predictor.side_effect = [
            dspy.Prediction(
                generated_inputs='[{"message": "\u26a0\ufe0f server down"}]'
            ),
            dspy.Prediction(generated_inputs='[{"message": "server down"}]'),
        ]
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, _ = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1),
        )

        inputs = gen._generate_inputs(1, None, "Classify tickets")
        assert len(inputs) == 1
        assert "\u26a0" not in inputs[0]["message"]

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_accepts_non_seed_enum_output(self, mock_predict_cls: MagicMock) -> None:
        """Output fields are not constrained to seed-derived enum values."""
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata

        def predict_side_effect(*args, **kwargs):
            return dspy.Prediction(
                generated_outputs=[{"urgency": "critical", "sentiment": "negative"}]
            )

        mock_predictor = MagicMock()
        mock_predictor.side_effect = predict_side_effect
        mock_predict_cls.return_value = mock_predictor

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1, judge_enabled=False),
        )

        gen._sig_metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "medium", "high"],
                ),
                FieldMetadata(
                    name="sentiment",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["positive", "neutral", "negative"],
                ),
            ],
            output_fields=["urgency", "sentiment"],
        )

        outputs, _ = gen._generate_outputs(
            [{"message": "server down"}],
            "Classify tickets",
        )

        assert len(outputs) == 1
        assert outputs[0] is not None
        assert outputs[0]["urgency"] == "critical"


# ---------------------------------------------------------------------------
# Output balancing tests
# ---------------------------------------------------------------------------


class TestComputeOutputCombos:
    """Tests for _compute_output_combos."""

    def test_single_categorical_field(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata

        metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "medium", "high"],
                ),
            ],
            output_fields=["urgency"],
        )
        combos = _compute_output_combos(["urgency"], metadata)
        assert combos is not None
        assert len(combos) == 3
        values = [c["urgency"] for c in combos]
        assert set(values) == {"low", "medium", "high"}

    def test_two_categorical_fields(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata

        metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "high"],
                ),
                FieldMetadata(
                    name="sentiment",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["positive", "negative"],
                ),
            ],
            output_fields=["urgency", "sentiment"],
        )
        combos = _compute_output_combos(["urgency", "sentiment"], metadata)
        assert combos is not None
        assert len(combos) == 4
        combo_tuples = {tuple(sorted(c.items())) for c in combos}
        assert len(combo_tuples) == 4

    def test_no_categorical_fields_returns_none(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata

        metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="summary",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=None,
                ),
            ],
            output_fields=["summary"],
        )
        combos = _compute_output_combos(["summary"], metadata)
        assert combos is None

    def test_mixed_categorical_and_free(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata

        metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "high"],
                ),
                FieldMetadata(
                    name="summary",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=None,
                ),
            ],
            output_fields=["urgency", "summary"],
        )
        combos = _compute_output_combos(["urgency", "summary"], metadata)
        assert combos is not None
        assert len(combos) == 2
        for c in combos:
            assert "urgency" in c
            assert "summary" not in c

    def test_skips_reasoning_field(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata

        metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="reasoning",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["a", "b"],
                ),
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "high"],
                ),
            ],
            output_fields=["reasoning", "urgency"],
        )
        combos = _compute_output_combos(["reasoning", "urgency"], metadata)
        assert combos is not None
        assert len(combos) == 2
        for c in combos:
            assert "reasoning" not in c


class TestSubsampleBalanced:
    """Tests for _subsample_balanced greedy selection."""

    def test_picks_underrepresented_values_first(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=4),
        )

        gen._sig_metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "high"],
                ),
            ],
            output_fields=["urgency"],
        )

        rows = [{"message": f"msg{i}", "urgency": "low"} for i in range(5)] + [
            {"message": "msg5", "urgency": "high"},
        ]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.95]

        result_rows, result_scores = gen._subsample_balanced(rows, 4, scores)

        assert len(result_rows) == 4
        low_count = sum(1 for r in result_rows if r["urgency"] == "low")
        high_count = sum(1 for r in result_rows if r["urgency"] == "high")
        assert low_count == 3
        assert high_count == 1
        assert len(result_scores) == 4

    def test_balances_two_fields(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=6),
        )

        gen._sig_metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "high"],
                ),
                FieldMetadata(
                    name="sentiment",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["positive", "negative"],
                ),
            ],
            output_fields=["urgency", "sentiment"],
        )

        rows = [
            {"message": "m0", "urgency": "high", "sentiment": "negative"},
            {"message": "m1", "urgency": "high", "sentiment": "negative"},
            {"message": "m2", "urgency": "high", "sentiment": "negative"},
            {"message": "m3", "urgency": "high", "sentiment": "negative"},
            {"message": "m4", "urgency": "low", "sentiment": "positive"},
            {"message": "m5", "urgency": "low", "sentiment": "positive"},
        ]
        scores = [0.5] * 6

        result_rows, _ = gen._subsample_balanced(rows, 4, scores)

        assert len(result_rows) == 4
        urg_low = sum(1 for r in result_rows if r["urgency"] == "low")
        urg_high = sum(1 for r in result_rows if r["urgency"] == "high")
        sent_pos = sum(1 for r in result_rows if r["sentiment"] == "positive")
        sent_neg = sum(1 for r in result_rows if r["sentiment"] == "negative")
        assert urg_low >= 1
        assert urg_high >= 1
        assert sent_pos >= 1
        assert sent_neg >= 1

    def test_returns_all_when_pool_smaller_than_n(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=10),
        )

        gen._sig_metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "high"],
                ),
            ],
            output_fields=["urgency"],
        )

        rows = [
            {"message": "m0", "urgency": "low"},
            {"message": "m1", "urgency": "high"},
        ]
        scores = [0.5, 0.6]

        result_rows, result_scores = gen._subsample_balanced(rows, 10, scores)

        assert len(result_rows) == 2
        assert len(result_scores) == 2

    def test_no_categorical_fields_returns_first_n(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=3),
        )

        gen._sig_metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="summary",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=None,
                ),
            ],
            output_fields=["summary"],
        )

        rows = [{"message": f"m{i}", "summary": f"sum{i}"} for i in range(5)]
        scores = [0.1 * i for i in range(5)]

        result_rows, result_scores = gen._subsample_balanced(rows, 3, scores)

        assert len(result_rows) == 3
        assert len(result_scores) == 3

    def test_preserves_scores_alignment(self) -> None:
        from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=4),
        )

        gen._sig_metadata = SignatureMetadata(
            fields=[
                FieldMetadata(
                    name="urgency",
                    python_type=str,
                    description="",
                    is_input=False,
                    allowed_values=["low", "high"],
                ),
            ],
            output_fields=["urgency"],
        )

        rows = [
            {"message": "m0", "urgency": "low"},
            {"message": "m1", "urgency": "high"},
            {"message": "m2", "urgency": "low"},
            {"message": "m3", "urgency": "high"},
        ]
        scores = [0.9, 0.8, 0.7, 0.6]

        result_rows, result_scores = gen._subsample_balanced(rows, 4, scores)

        assert len(result_rows) == 4
        assert len(result_scores) == 4
        for row, sc in zip(result_rows, result_scores):
            idx = int(row["message"][1])
            assert sc == scores[idx]


# ---------------------------------------------------------------------------
# Signature generation mode tests
# ---------------------------------------------------------------------------


class TestSignatureGenerationMode:
    """Tests for generation_mode='signature'."""

    def test_config_generation_mode_validation(self) -> None:
        """AutoDataConfig rejects invalid generation_mode values."""
        with pytest.raises(ValueError, match="generation_mode must be one of"):
            AutoDataConfig(generation_mode="invalid")

    def test_config_generation_mode_default(self) -> None:
        """AutoDataConfig defaults to generation_mode='split'."""
        config = AutoDataConfig()
        assert config.generation_mode == "split"

    def test_config_generation_mode_signature(self) -> None:
        """AutoDataConfig accepts generation_mode='signature'."""
        config = AutoDataConfig(generation_mode="signature")
        assert config.generation_mode == "signature"

    def test_build_signature_generation_signature(self) -> None:
        """_build_signature_generation_signature returns a valid DSPy Signature class."""
        from dspy_auto_gepa.generator import _build_signature_generation_signature

        sig_cls = _build_signature_generation_signature()
        assert issubclass(sig_cls, dspy.Signature)

        # Check that the signature has the expected input/output fields
        fields = getattr(sig_cls, "fields", {})
        assert "task_description" in fields
        assert "field_spec" in fields
        assert "recent_rows_json" in fields
        assert "covered_themes" in fields
        assert "n_to_generate" in fields
        assert "generated_rows" in fields

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_signature_mode_mocked(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
    ) -> None:
        """_generate_signature_mode returns complete rows with all fields."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = dspy.Prediction(
            generated_rows=json.dumps(
                [
                    {
                        "message": "server is down",
                        "urgency": "high",
                        "sentiment": "negative",
                    },
                    {
                        "message": "thanks for help",
                        "urgency": "low",
                        "sentiment": "positive",
                    },
                ]
            )
        )
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, _ = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=2, generation_mode="signature"),
        )

        rows, scores = gen._generate_signature_mode(2, None, "Classify tickets")

        assert len(rows) == 2
        for row in rows:
            assert "message" in row
            assert "urgency" in row
            assert "sentiment" in row
        assert rows[0]["urgency"] == "high"
        assert rows[1]["urgency"] == "low"

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_signature_mode_respects_n(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
    ) -> None:
        """_generate_signature_mode returns exactly n rows."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = dspy.Prediction(
            generated_rows=json.dumps(
                [
                    {"message": f"msg{i}", "urgency": "low", "sentiment": "neutral"}
                    for i in range(5)
                ]
            )
        )
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, _ = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=3, generation_mode="signature"),
        )

        rows, _ = gen._generate_signature_mode(3, None, "Classify tickets")
        assert len(rows) == 3

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_signature_mode_retries_on_bad_json(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
    ) -> None:
        """_generate_signature_mode retries when LLM returns invalid JSON."""
        mock_predictor = MagicMock()
        mock_predictor.side_effect = [
            dspy.Prediction(generated_rows="not valid json"),
            dspy.Prediction(
                generated_rows=json.dumps(
                    [{"message": "recovered", "urgency": "low", "sentiment": "neutral"}]
                )
            ),
        ]
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, _ = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1, generation_mode="signature"),
        )

        rows, _ = gen._generate_signature_mode(1, None, "Classify tickets")
        assert len(rows) == 1
        assert rows[0]["message"] == "recovered"

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_signature_mode_with_seeds(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
    ) -> None:
        """_generate_signature_mode passes seed examples for diversity reference."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = dspy.Prediction(
            generated_rows=json.dumps(
                [{"message": "generated", "urgency": "low", "sentiment": "neutral"}]
            )
        )
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, executed_tasks = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1, generation_mode="signature"),
        )

        seeds = [{"message": "seed1", "urgency": "low", "sentiment": "neutral"}]
        rows, _ = gen._generate_signature_mode(1, seeds, "Classify tickets")

        assert len(rows) == 1
        assert len(executed_tasks) > 0
        _, example = executed_tasks[0]
        assert "seed1" in example.recent_rows_json

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_full_flow_signature_mode(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock, tmp_path: Path
    ) -> None:
        """generate() with signature mode returns correct GenerationResult."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = dspy.Prediction(
            generated_rows=json.dumps(
                [
                    {
                        "message": "server down",
                        "urgency": "high",
                        "sentiment": "negative",
                    },
                    {
                        "message": "thanks",
                        "urgency": "low",
                        "sentiment": "positive",
                    },
                ]
            )
        )
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, _ = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        output_path = tmp_path / "output.jsonl"
        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=2, generation_mode="signature"),
        )

        result = gen.generate(n=2, output_path=output_path, force=True)

        assert isinstance(result, GenerationResult)
        assert result.n_produced == 2
        assert result.n_requested == 2
        assert len(result.rows) == 2
        for row in result.rows:
            assert "message" in row
            assert "urgency" in row
            assert "sentiment" in row
        assert output_path.exists()

    def test_generate_signature_mode_with_react_style_signature(self) -> None:
        """Signature mode works with multi-output signatures like ReAct."""

        class ReActSignature(dspy.Signature):
            """Answer questions using step-by-step reasoning."""

            question: str = dspy.InputField()
            thought: str = dspy.OutputField()
            action: str = dspy.OutputField()
            observation: str = dspy.OutputField()
            answer: str = dspy.OutputField()

        module = MagicMock()
        module.signature = ReActSignature
        module.named_predictors = lambda: []

        gen = AutoData(
            module=module,
            data_lm=MagicMock(),
            description="Answer questions with reasoning",
            config=_make_autodata_config(generation_mode="signature"),
        )

        assert gen.input_fields == ["question"]
        assert "thought" in gen.output_fields
        assert "action" in gen.output_fields
        assert "observation" in gen.output_fields
        assert "answer" in gen.output_fields

    @patch("dspy_auto_gepa.generator.dspy.Parallel")
    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_signature_mode_rejects_empty_fields(
        self, mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
    ) -> None:
        """_generate_signature_mode rejects rows with empty required fields."""
        mock_predictor = MagicMock()
        mock_predictor.side_effect = [
            dspy.Prediction(
                generated_rows=json.dumps(
                    [{"message": "", "urgency": "high", "sentiment": "negative"}]
                )
            ),
            dspy.Prediction(
                generated_rows=json.dumps(
                    [
                        {
                            "message": "valid message",
                            "urgency": "low",
                            "sentiment": "neutral",
                        }
                    ]
                )
            ),
        ]
        mock_predict_cls.return_value = mock_predictor

        parallel_mock, _ = _make_sync_parallel_mock()
        mock_parallel_cls.side_effect = parallel_mock

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1, generation_mode="signature"),
        )

        rows, _ = gen._generate_signature_mode(1, None, "Classify tickets")
        assert len(rows) == 1
        assert rows[0]["message"] == "valid message"
