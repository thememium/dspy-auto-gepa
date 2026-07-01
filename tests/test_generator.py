"""Comprehensive tests for AutoData and StreamingDatasetWriter.

All LLM calls are mocked — no real API traffic.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import dspy
import pandas as pd
import pytest

from dspy_auto_gepa.config import AutoDataConfig
from dspy_auto_gepa.generator import AutoData, StreamingDatasetWriter
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


def _make_autodata_config(**overrides) -> AutoDataConfig:
    """Return an AutoDataConfig with judge/diversity disabled for simpler tests."""
    defaults = dict(
        n=5,
        seed=42,
        max_retries=3,
        judge_enabled=False,
        diversity_enabled=False,
        validators_enabled=False,
        rejection_sampling_enabled=False,
    )
    defaults.update(overrides)
    return AutoDataConfig(**defaults)


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
        pytest.importorskip("pyarrow")  # pyarrow needed for parquet engine
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
        assert StreamingDatasetWriter._detect_format(tmp_path / "a.parquet") == "parquet"
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
        csv_path.write_text("message,urgency,sentiment\nhello,low,neutral\nworld,high,negative\n")

        module = DummyModule()
        gen = AutoData.from_csv(csv_path, module=module, data_lm=MagicMock(), description="test")

        assert gen.seed_examples is not None
        assert len(gen.seed_examples) == 2
        assert gen.seed_examples[0]["message"] == "hello"
        assert gen.seed_examples[1]["urgency"] == "high"

    def test_from_json(self, tmp_path: Path) -> None:
        """from_json loads seed_examples from a JSON file."""
        json_path = tmp_path / "seeds.json"
        json_path.write_text(json.dumps([
            {"message": "hi", "urgency": "low", "sentiment": "positive"},
        ]))

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

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_inputs_mocked(self, mock_predict_cls: MagicMock) -> None:
        """_generate_inputs returns correct count and field keys from mocked LLM."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = MagicMock(
            generated_inputs='[{"message": "test1"}, {"message": "test2"}]'
        )
        mock_predict_cls.return_value = mock_predictor

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
        # Verify all rows have the required input field
        for row in inputs:
            assert "message" in row

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_inputs_json_retry(self, mock_predict_cls: MagicMock) -> None:
        """First call returns invalid JSON, second succeeds → retry works."""
        mock_predictor = MagicMock()
        mock_predictor.side_effect = [
            MagicMock(generated_inputs="this is not json"),
            MagicMock(generated_inputs='[{"message": "recovered"}]'),
        ]
        mock_predict_cls.return_value = mock_predictor

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1),
        )

        inputs = gen._generate_inputs(1, None, "Classify tickets")

        assert len(inputs) == 1
        assert inputs[0]["message"] == "recovered"
        assert mock_predictor.call_count == 2

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_outputs_mocked(self, mock_predict_cls: MagicMock) -> None:
        """_generate_outputs returns correct output fields from mocked LLM."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = MagicMock(
            generated_output='{"urgency": "high", "sentiment": "negative"}'
        )
        mock_predict_cls.return_value = mock_predictor

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(),
        )

        outputs = gen._generate_outputs(
            [{"message": "server is down"}],
            "Classify tickets",
        )

        assert len(outputs) == 1
        assert outputs[0]["urgency"] == "high"
        assert outputs[0]["sentiment"] == "negative"

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_outputs_json_retry(self, mock_predict_cls: MagicMock) -> None:
        """Output generation retries on invalid JSON."""
        mock_predictor = MagicMock()
        mock_predictor.side_effect = [
            MagicMock(generated_output="bad json"),
            MagicMock(generated_output='{"urgency": "low", "sentiment": "neutral"}'),
        ]
        mock_predict_cls.return_value = mock_predictor

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1),
        )

        outputs = gen._generate_outputs(
            [{"message": "hello"}],
            "Classify tickets",
        )

        assert len(outputs) == 1
        assert outputs[0]["urgency"] == "low"
        assert mock_predictor.call_count == 2

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_full_flow(self, mock_predict_cls: MagicMock, tmp_path: Path) -> None:
        """Mocked full generate() → returns GenerationResult with correct row count."""
        call_count = 0

        def predict_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call is _InputGenerationSignature, second is _OutputGenerationSignature
            if call_count == 1:
                return MagicMock(
                    generated_inputs='[{"message": "ticket1"}, {"message": "ticket2"}]'
                )
            else:
                return MagicMock(
                    generated_output='{"urgency": "high", "sentiment": "negative"}'
                )

        mock_predictor = MagicMock()
        mock_predictor.side_effect = predict_side_effect
        mock_predict_cls.return_value = mock_predictor

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
        # Verify rows have both input and output fields
        for row in result.rows:
            assert "message" in row
            assert "urgency" in row
            assert "sentiment" in row
        # Verify file was written
        assert output_path.exists()

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_resume(self, mock_predict_cls: MagicMock, tmp_path: Path) -> None:
        """When file already has rows and force=False, generate returns existing rows."""
        output_path = tmp_path / "output.jsonl"

        # Pre-write rows to the file
        writer = StreamingDatasetWriter(output_path)
        writer.write_row({"message": "existing1", "urgency": "low", "sentiment": "neutral"})
        writer.write_row({"message": "existing2", "urgency": "high", "sentiment": "negative"})
        writer.write_row({"message": "existing3", "urgency": "medium", "sentiment": "neutral"})

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=5),
        )

        result = gen.generate(n=5, output_path=output_path, force=False)

        # Should return existing rows without calling LLM
        assert result.n_produced == 3
        assert len(result.rows) == 3
        assert result.rows[0]["message"] == "existing1"
        # LLM should NOT have been called
        mock_predict_cls.assert_not_called()

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_force_overwrites(self, mock_predict_cls: MagicMock, tmp_path: Path) -> None:
        """force=True ignores existing rows and generates fresh data."""
        output_path = tmp_path / "output.jsonl"

        # Pre-write rows
        writer = StreamingDatasetWriter(output_path)
        writer.write_row({"message": "old1", "urgency": "low", "sentiment": "neutral"})

        call_count = 0

        def predict_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(generated_inputs='[{"message": "new1"}]')
            else:
                return MagicMock(generated_output='{"urgency": "high", "sentiment": "positive"}')

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
        mock_predictor.assert_called()

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_outputs_all_retries_fail(self, mock_predict_cls: MagicMock) -> None:
        """When all retries fail for output, empty values are used."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = MagicMock(generated_output="not valid json at all")
        mock_predict_cls.return_value = mock_predictor

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(max_retries=2),
        )

        outputs = gen._generate_outputs(
            [{"message": "test"}],
            "Classify tickets",
        )

        assert len(outputs) == 1
        # Should have empty strings for failed output fields
        assert outputs[0]["urgency"] == ""
        assert outputs[0]["sentiment"] == ""

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_inputs_respects_n(self, mock_predict_cls: MagicMock) -> None:
        """_generate_inputs returns exactly n rows even if LLM returns extra."""
        mock_predictor = MagicMock()
        # LLM returns 5 but we only asked for 3
        mock_predictor.return_value = MagicMock(
            generated_inputs=json.dumps([{"message": f"msg{i}"} for i in range(5)])
        )
        mock_predict_cls.return_value = mock_predictor

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=3),
        )

        inputs = gen._generate_inputs(3, None, "Classify tickets")
        assert len(inputs) == 3

    @patch("dspy_auto_gepa.generator.dspy.Predict")
    def test_generate_inputs_with_seed_examples(self, mock_predict_cls: MagicMock) -> None:
        """Seed examples are passed through for diversity reference."""
        mock_predictor = MagicMock()
        mock_predictor.return_value = MagicMock(
            generated_inputs='[{"message": "generated1"}]'
        )
        mock_predict_cls.return_value = mock_predictor

        gen = AutoData(
            module=DummyModule(),
            data_lm=MagicMock(),
            description="Classify tickets",
            config=_make_autodata_config(n=1),
        )

        seeds = [{"message": "seed1", "urgency": "low", "sentiment": "neutral"}]
        inputs = gen._generate_inputs(1, seeds, "Classify tickets")

        assert len(inputs) == 1
        # Verify the predictor was called with existing_inputs_json containing seed data
        call_kwargs = mock_predictor.call_args
        existing_json = call_kwargs.kwargs.get(
            "existing_inputs_json", call_kwargs[1].get("existing_inputs_json", "")
        )
        assert "seed1" in existing_json

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
        with pytest.raises(ValueError, match="Unsupported seed file format"):
            gen._resolve_seeds(str(tmp_path / "seeds.xml"))
