"""Coverage tests for generator helpers and edge paths."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import dspy
import pandas as pd
import pytest
from pydantic import BaseModel

from dspy_auto_gepa.config import AutoDataConfig
from dspy_auto_gepa.data import FieldMetadata, SignatureMetadata
from dspy_auto_gepa.generator import (
    AutoData,
    StreamingDatasetWriter,
    _canonicalize_for_fingerprint,
    _extract_json,
)
from dspy_auto_gepa.quality import ValidationResult


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


class PredictorOnlyModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(TicketSignature)

    def forward(self, message: str) -> dspy.Prediction:
        return dspy.Prediction(urgency="low", sentiment="neutral")


def _config(**overrides: Any) -> AutoDataConfig:
    defaults: dict[str, Any] = {
        "n": 1,
        "seed": 42,
        "max_retries": 1,
        "num_threads": 1,
        "chunk_size": 1,
        "max_inflight_requests": 1,
        "judge_enabled": False,
        "diversity_enabled": False,
        "validators_enabled": False,
        "rejection_sampling_enabled": False,
    }
    defaults.update(overrides)
    return AutoDataConfig(**defaults)


def _sync_parallel(script: list[list[Any]] | None = None):
    """Patch helper executing tasks synchronously or returning scripted results."""

    class SyncParallel:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __call__(self, tasks: list[Any]) -> list[Any]:
            if script is not None:
                return script.pop(0)
            results = []
            for module, example in tasks:
                try:
                    results.append(module(**{k: example[k] for k in example.keys()}))
                except Exception as exc:  # pragma: no cover - defensive in tests
                    results.append(exc)
            return results

    return MagicMock(side_effect=lambda **kwargs: SyncParallel(**kwargs))


def _always_parallel(result: Any):
    class AlwaysParallel:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __call__(self, tasks: list[Any]) -> Any:
            return result

    return MagicMock(side_effect=lambda **kwargs: AlwaysParallel(**kwargs))


def _alternating_parallel(first: list[Any], second: list[Any]):
    calls: list[int] = [0]

    class AlternatingParallel:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __call__(self, tasks: list[Any]) -> Any:
            calls[0] += 1
            return first if calls[0] % 2 == 1 else second

    return MagicMock(side_effect=lambda **kwargs: AlternatingParallel(**kwargs))


def _signature_metadata(**allowed: list[str]) -> SignatureMetadata:
    fields = [
        FieldMetadata(
            name=name,
            python_type=str,
            description="",
            is_input=False,
            allowed_values=values,
        )
        for name, values in allowed.items()
    ]
    return SignatureMetadata(fields=fields, output_fields=list(allowed))


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def test_extract_json_empty_string() -> None:
    assert _extract_json("") == ""


def test_extract_json_valid_fence() -> None:
    assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_invalid_fence_falls_through() -> None:
    text = "```json\nnot json\n```"

    assert _extract_json(text) == text


def test_extract_json_extracts_bracketed_object() -> None:
    assert _extract_json('prefix {"a": 1} suffix') == '{"a": 1}'


def test_extract_json_invalid_bracket_falls_through() -> None:
    text = "prefix {not json} suffix"

    assert _extract_json(text) == text


def test_extract_json_parses_python_literal() -> None:
    assert _extract_json("{'a': 1}") == '{"a": 1}'


# ---------------------------------------------------------------------------
# StreamingDatasetWriter gaps
# ---------------------------------------------------------------------------


def test_writer_detects_csv_header_on_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    pd.DataFrame([{"a": 1}]).to_csv(path, index=False)

    writer = StreamingDatasetWriter(path)
    writer.write_row({"a": 2})

    frame = pd.read_csv(path)
    assert list(frame["a"]) == [1, 2]


def test_writer_write_rows_empty_is_noop(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    writer = StreamingDatasetWriter(path)

    writer.write_rows([])

    assert not path.exists()


def test_writer_write_rows_csv(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    writer = StreamingDatasetWriter(path)

    writer.write_rows([{"a": 1}, {"a": 2}])

    assert pd.read_csv(path)["a"].tolist() == [1, 2]


def test_writer_write_rows_parquet(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    writer = StreamingDatasetWriter(path)

    writer.write_rows([{"a": 1}, {"a": 2}])

    assert pd.read_parquet(path)["a"].tolist() == [1, 2]


def test_writer_flush_and_close(tmp_path: Path) -> None:
    writer = StreamingDatasetWriter(tmp_path / "rows.jsonl")

    writer.flush()
    writer.close()


def test_writer_read_rows_missing_file(tmp_path: Path) -> None:
    writer = StreamingDatasetWriter(tmp_path / "rows.jsonl")

    assert writer.read_rows() == []


def test_writer_read_rows_unknown_format_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("")
    writer = StreamingDatasetWriter(path)
    writer._format = "unknown"

    assert writer.read_rows() == []


def test_writer_read_rows_csv_and_parquet(tmp_path: Path) -> None:
    csv_path = tmp_path / "rows.csv"
    pd.DataFrame([{"a": 1}]).to_csv(csv_path, index=False)
    parquet_path = tmp_path / "rows.parquet"
    pd.DataFrame([{"a": 2}]).to_parquet(parquet_path, index=False)

    assert StreamingDatasetWriter(csv_path).read_rows() == [{"a": 1}]
    assert StreamingDatasetWriter(parquet_path).read_rows() == [{"a": 2}]


# ---------------------------------------------------------------------------
# fingerprint canonicalization
# ---------------------------------------------------------------------------


def test_canonicalize_handles_all_value_types() -> None:
    assert _canonicalize_for_fingerprint(None) == "__none__"
    assert _canonicalize_for_fingerprint("  Text ") == "text"
    assert _canonicalize_for_fingerprint(3) == "3"
    assert _canonicalize_for_fingerprint(3.5) == "3.5"
    assert _canonicalize_for_fingerprint(True) == "True"
    assert _canonicalize_for_fingerprint({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
    assert _canonicalize_for_fingerprint([2, 1]) == "[2, 1]"
    assert _canonicalize_for_fingerprint((2, 1)) == "[2, 1]"
    assert json.loads(
        _canonicalize_for_fingerprint(SimpleNamespace(model_dump=lambda: {"a": 1}))
    ) == {"a": 1}
    assert json.loads(
        _canonicalize_for_fingerprint(SimpleNamespace(dict=lambda: {"a": 2}))
    ) == {"a": 2}
    value = object()
    assert _canonicalize_for_fingerprint(value) == str(value)


# ---------------------------------------------------------------------------
# constructor and metadata fallbacks
# ---------------------------------------------------------------------------


def test_constructor_uses_predictor_signature_and_schema_fields() -> None:
    class Schema(BaseModel):
        urgency: str

    gen = AutoData(
        module=PredictorOnlyModule(),
        data_lm=MagicMock(),
        description="tickets",
        schema=Schema,
        config=_config(),
    )

    assert gen.input_fields == ["message"]
    assert gen._schema_fields == {"urgency": "<class 'str'>"}


def test_ensure_metadata_uses_predictor_signature() -> None:
    gen = AutoData(
        module=PredictorOnlyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(),
    )
    gen._sig_metadata = None

    metadata = gen._ensure_metadata()

    assert metadata.get("message") is not None
    assert metadata.get("urgency") is not None


def test_ensure_metadata_without_signature_returns_empty() -> None:
    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(),
    )
    gen._sig_metadata = None
    gen.module = MagicMock()
    gen.module.signature = None
    gen.module.named_predictors.return_value = []

    metadata = gen._ensure_metadata()

    assert metadata.fields == []


def test_parallel_request_budget_respects_caps() -> None:
    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(
            chunk_size=10,
            num_threads=4,
            max_inflight_requests=None,
        ),
    )

    assert gen._parallel_request_budget() == 4

    gen.config.max_inflight_requests = 2
    assert gen._parallel_request_budget() == 2


# ---------------------------------------------------------------------------
# _generate_inputs edges
# ---------------------------------------------------------------------------


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_inputs_breaks_after_repeated_result_exceptions(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock()
    mock_parallel_cls.side_effect = _always_parallel([RuntimeError("down")]).side_effect

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1, max_retries=1),
    )

    assert gen._generate_inputs(1, None, "tickets") == []


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_inputs_accepts_single_object_response(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(generated_inputs='{"message": "single"}')
    )
    mock_parallel_cls.side_effect = _sync_parallel()

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1),
    )

    assert gen._generate_inputs(1, None, "tickets") == [{"message": "single"}]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_inputs_skips_non_dict_items_and_rows(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(
            generated_inputs='[42, {"message": "row"}, "junk"]'
        )
    )
    mock_parallel_cls.side_effect = _sync_parallel()

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1),
    )

    assert gen._generate_inputs(1, None, "tickets") == [{"message": "row"}]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_inputs_accepts_raw_dict_result(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock(return_value=MagicMock())
    mock_parallel_cls.side_effect = _sync_parallel(
        [[{"message": "from-list"}, {"message": "from-dict"}]]
    )

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=2),
    )

    assert gen._generate_inputs(2, None, "tickets") == [
        {"message": "from-list"},
        {"message": "from-dict"},
    ]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_inputs_skips_non_prediction_non_dict_items(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock()
    mock_parallel_cls.side_effect = _sync_parallel(
        [["junk", {"message": "from-dict-result"}]]
    )

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1),
    )

    assert gen._generate_inputs(1, None, "tickets") == [{"message": "from-dict-result"}]


@patch("dspy_auto_gepa.generator._build_validator")
@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_inputs_skips_validator_failures(
    mock_predict_cls: MagicMock,
    mock_parallel_cls: MagicMock,
    mock_build_validator: MagicMock,
) -> None:
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(generated_inputs='[{"message": "valid"}]')
    )
    mock_parallel_cls.side_effect = _sync_parallel()
    mock_build_validator.return_value.validate.return_value = ValidationResult(
        is_valid=False, failures=["bad"]
    )

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1, max_retries=1),
    )

    assert gen._generate_inputs(1, None, "tickets") == []


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_inputs_skips_duplicates(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(
            generated_inputs='[{"message": "duplicate"}, {"message": "fresh"}]'
        )
    )
    mock_parallel_cls.side_effect = _sync_parallel()

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1),
    )

    result = gen._generate_inputs(1, [{"message": "duplicate"}], "tickets")

    assert result == [{"message": "fresh"}]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_inputs_trims_recent_window(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    rows = [{"message": f"message {i}"} for i in range(25)]
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(generated_inputs=json.dumps(rows))
    )
    mock_parallel_cls.side_effect = _sync_parallel()

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=25),
    )

    result = gen._generate_inputs(25, None, "tickets", progress=MagicMock())

    assert len(result) == 25


# ---------------------------------------------------------------------------
# _generate_targeted_inputs
# ---------------------------------------------------------------------------


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_targeted_inputs_success(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(
            generated_inputs='[{"message": "first"}, {"message": "second"}]'
        )
    )
    mock_parallel_cls.side_effect = _sync_parallel()

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=2),
    )

    rows = gen._generate_targeted_inputs(
        2, [{"urgency": "high"}], None, "tickets", MagicMock()
    )

    assert rows == [
        {"message": "first", "urgency": "high"},
        {"message": "second", "urgency": "high"},
    ]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_targeted_inputs_unwraps_parallel_tuple(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(generated_inputs='[{"message": "wrapped"}]')
    )
    mock_parallel_cls.side_effect = _always_parallel(
        ([dspy.Prediction(generated_inputs='[{"message": "wrapped"}]')], [], [])
    )

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1),
    )

    rows = gen._generate_targeted_inputs(1, [{"urgency": "low"}], None, "tickets")

    assert rows == [{"message": "wrapped", "urgency": "low"}]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_targeted_inputs_handles_odd_items(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock()
    script = [
        [
            [
                Exception("down"),
                dspy.Prediction(generated_inputs="not json"),
                "junk",
                {"message": ""},
                {"message": "dict row"},
                dspy.Prediction(generated_inputs="[42]"),
                dspy.Prediction(generated_inputs='{"message": "single row"}'),
            ]
        ],
        [dspy.Prediction(generated_inputs='[{"message": "last row"}]')],
    ]
    mock_parallel_cls.side_effect = _sync_parallel(script)

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=3),
    )

    rows = gen._generate_targeted_inputs(
        3, [{"urgency": "high"}], None, "tickets", MagicMock()
    )

    assert [row["message"] for row in rows] == [
        "dict row",
        "single row",
        "last row",
    ]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_targeted_inputs_counts_empty_accepts(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock()
    script = [
        [Exception("down")],
        [dspy.Prediction(generated_inputs='[{"message": "recovered"}]')],
    ]
    mock_parallel_cls.side_effect = _sync_parallel(script)

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1),
    )

    rows = gen._generate_targeted_inputs(1, [{"urgency": "low"}], None, "tickets")

    assert rows == [{"message": "recovered", "urgency": "low"}]


@patch("dspy_auto_gepa.generator._build_validator")
@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_targeted_inputs_skips_validator_failures_and_counts_them(
    mock_predict_cls: MagicMock,
    mock_parallel_cls: MagicMock,
    mock_build_validator: MagicMock,
) -> None:
    mock_predict_cls.return_value = MagicMock()
    script = [
        [[{"message": "rejected"}]],
        [dspy.Prediction(generated_inputs='[{"message": "accepted"}]')],
    ]
    mock_parallel_cls.side_effect = _sync_parallel(script)
    mock_build_validator.return_value.validate.side_effect = [
        ValidationResult(is_valid=False, failures=["bad"]),
        ValidationResult(is_valid=True, failures=[]),
    ]

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1),
    )

    rows = gen._generate_targeted_inputs(1, [{"urgency": "low"}], None, "tickets")

    assert rows == [{"message": "accepted", "urgency": "low"}]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_targeted_inputs_stops_when_all_states_exhausted(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock()
    mock_parallel_cls.side_effect = _always_parallel([Exception("down")])

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1, max_retries=1),
    )

    assert gen._generate_targeted_inputs(1, [{"urgency": "low"}], None, "tickets") == []


# ---------------------------------------------------------------------------
# _generate_outputs edges
# ---------------------------------------------------------------------------


def _valid_output() -> dict[str, str]:
    return {"urgency": "low", "sentiment": "neutral"}


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_outputs_accepts_model_dump_and_dict_objects(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    class ModelDump:
        def model_dump(self) -> dict[str, str]:
            return {"urgency": "low", "sentiment": "neutral"}

    class DictLike:
        def dict(self) -> Any:
            return {"urgency": "high", "sentiment": "negative"}

    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(generated_outputs=[ModelDump(), DictLike()])
    )
    mock_parallel_cls.side_effect = _sync_parallel()

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=2, max_inflight_requests=None),
    )

    outputs, scores = gen._generate_outputs(
        [{"message": "a"}, {"message": "b"}], "tickets"
    )

    assert outputs == [
        {"urgency": "low", "sentiment": "neutral"},
        {"urgency": "high", "sentiment": "negative"},
    ]
    assert scores is None


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_outputs_recovers_exceptions_with_single_predictor(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    batch_result = dspy.Prediction(
        generated_outputs=[Exception("bad"), {"urgency": "low", "sentiment": "neutral"}]
    )
    fallback_result = dspy.Prediction(
        generated_output='{"urgency": "high", "sentiment": "negative"}'
    )
    mock_predict_cls.return_value = MagicMock()
    mock_parallel_cls.side_effect = _sync_parallel([[batch_result], [fallback_result]])

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=2),
    )

    outputs, _ = gen._generate_outputs([{"message": "a"}, {"message": "b"}], "tickets")

    assert outputs == [
        {"urgency": "high", "sentiment": "negative"},
        {"urgency": "low", "sentiment": "neutral"},
    ]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_outputs_skips_unparseable_predictions(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock()
    mock_parallel_cls.side_effect = _alternating_parallel(
        [
            dspy.Prediction(
                generated_outputs=[
                    dspy.Prediction(generated_output="not json"),
                    dspy.Prediction(generated_output='"plain string"'),
                ]
            )
        ],
        [Exception("bad"), Exception("bad")],
    ).side_effect

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=2, max_retries=1),
    )

    outputs, _ = gen._generate_outputs([{"message": "a"}, {"message": "b"}], "tickets")

    assert outputs == [None, None]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_outputs_breaks_on_extra_results_and_handles_list_results(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock()
    mock_parallel_cls.side_effect = _sync_parallel(
        [[[_valid_output()], "extra result"]]
    )

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1),
    )

    outputs, _ = gen._generate_outputs([{"message": "a"}], "tickets")

    assert outputs == [_valid_output()]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_outputs_breaks_when_single_results_exceed_failures(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock()
    mock_parallel_cls.side_effect = _sync_parallel(
        [
            [Exception("batch failed")],
            [
                dspy.Prediction(
                    generated_output='{"urgency": "low", "sentiment": "neutral"}'
                ),
                "extra result",
            ],
        ]
    )

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1),
    )

    outputs, _ = gen._generate_outputs([{"message": "a"}], "tickets")

    assert outputs == [_valid_output()]


@patch("dspy_auto_gepa.generator._build_validator")
@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_outputs_skips_validator_failures(
    mock_predict_cls: MagicMock,
    mock_parallel_cls: MagicMock,
    mock_build_validator: MagicMock,
) -> None:
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(generated_outputs=[_valid_output()])
    )
    mock_parallel_cls.side_effect = _alternating_parallel(
        [dspy.Prediction(generated_outputs=[_valid_output()])],
        [Exception("bad")],
    ).side_effect
    mock_build_validator.return_value.validate.return_value = ValidationResult(
        is_valid=False, failures=["bad"]
    )

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1, max_retries=1),
    )

    outputs, _ = gen._generate_outputs([{"message": "a"}], "tickets")

    assert outputs == [None]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_outputs_skips_rows_with_missing_fields(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(generated_outputs=[{"urgency": "low"}])
    )
    mock_parallel_cls.side_effect = _alternating_parallel(
        [dspy.Prediction(generated_outputs=[{"urgency": "low"}])],
        [Exception("bad")],
    ).side_effect

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1, max_retries=1),
    )

    outputs, _ = gen._generate_outputs([{"message": "a"}], "tickets")

    assert outputs == [None]


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_outputs_ignores_reasoning_field(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(
            generated_outputs=[{"urgency": "low", "sentiment": "neutral"}]
        )
    )
    mock_parallel_cls.side_effect = _alternating_parallel(
        [
            dspy.Prediction(
                generated_outputs=[{"urgency": "low", "sentiment": "neutral"}]
            )
        ],
        [Exception("bad")],
    ).side_effect

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1, max_retries=1),
    )
    gen.output_fields = ["reasoning", "urgency", "sentiment"]

    outputs, _ = gen._generate_outputs([{"message": "a"}], "tickets")

    assert outputs == [None]


@patch("dspy_auto_gepa.generator.LLMJudge")
@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_generate_outputs_scores_batches_and_keeps_unsored_rows(
    mock_predict_cls: MagicMock,
    mock_parallel_cls: MagicMock,
    mock_judge_cls: MagicMock,
) -> None:
    inputs = [{"message": f"message {i}"} for i in range(6)]
    inputs[-1] = {"message": "bad"}
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(
            generated_outputs=[_valid_output() for _ in inputs]
        )
    )
    mock_parallel_cls.side_effect = _sync_parallel()

    class FakeJudge:
        def __init__(self, lm: Any) -> None:
            pass

        def batch_score(
            self, rows: list[dict[str, Any]], task_description: str = ""
        ) -> list[SimpleNamespace]:
            if any(row.get("message") == "bad" for row in rows):
                raise RuntimeError("judge failed")
            return [SimpleNamespace(score=0.9) for _ in rows]

    mock_judge_cls.side_effect = FakeJudge

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=6, judge_enabled=True),
    )

    outputs, scores = gen._generate_outputs(inputs, "tickets")

    assert len(outputs) == 6
    assert all(output == _valid_output() for output in outputs)
    assert scores == [0.9] * 5


# ---------------------------------------------------------------------------
# _generate_signature_mode edges
# ---------------------------------------------------------------------------


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_signature_mode_breaks_after_repeated_failures(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock()
    mock_parallel_cls.side_effect = _always_parallel([RuntimeError("down")])

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1, generation_mode="signature", max_retries=1),
    )

    rows, scores = gen._generate_signature_mode(1, None, "tickets")

    assert rows == []
    assert scores is None


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_signature_mode_handles_mixed_result_shapes(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    mock_predict_cls.return_value = MagicMock()
    script = [
        [
            {
                "message": "dict row",
                "urgency": "low",
                "sentiment": "neutral",
            },
            "junk",
            42,
            dspy.Prediction(
                generated_rows='{"message": "single row", "urgency": "low", '
                '"sentiment": "neutral"}'
            ),
            dspy.Prediction(
                generated_rows='[7, {"message": "list row", "urgency": "high", '
                '"sentiment": "negative"}]'
            ),
        ],
        [
            dspy.Prediction(
                generated_rows='[{"message": "last row", "urgency": "low", '
                '"sentiment": "positive"}]'
            )
        ],
    ]
    mock_parallel_cls.side_effect = _sync_parallel(script)

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=4, generation_mode="signature"),
    )

    rows, scores = gen._generate_signature_mode(4, None, "tickets")

    assert [row["message"] for row in rows] == [
        "dict row",
        "single row",
        "list row",
        "last row",
    ]
    assert scores is None


@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_signature_mode_skips_duplicate_rows(
    mock_predict_cls: MagicMock, mock_parallel_cls: MagicMock
) -> None:
    duplicate = {
        "message": "duplicate",
        "urgency": "low",
        "sentiment": "neutral",
    }
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(
            generated_rows=json.dumps([duplicate, {**duplicate, "message": "fresh"}])
        )
    )
    mock_parallel_cls.side_effect = _sync_parallel()

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1, generation_mode="signature"),
    )

    rows, _ = gen._generate_signature_mode(1, [duplicate], "tickets")

    assert rows == [{**duplicate, "message": "fresh"}]


@patch("dspy_auto_gepa.generator._build_validator")
@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_signature_mode_skips_validator_failures(
    mock_predict_cls: MagicMock,
    mock_parallel_cls: MagicMock,
    mock_build_validator: MagicMock,
) -> None:
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(
            generated_rows=json.dumps(
                [
                    {
                        "message": "valid",
                        "urgency": "low",
                        "sentiment": "neutral",
                    }
                ]
            )
        )
    )
    mock_parallel_cls.side_effect = _sync_parallel()
    mock_build_validator.return_value.validate.return_value = ValidationResult(
        is_valid=False, failures=["bad"]
    )

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1, generation_mode="signature", max_retries=1),
    )

    rows, _ = gen._generate_signature_mode(1, None, "tickets")

    assert rows == []


@patch("dspy_auto_gepa.generator.LLMJudge")
@patch("dspy_auto_gepa.generator.dspy.Parallel")
@patch("dspy_auto_gepa.generator.dspy.Predict")
def test_signature_mode_scores_rows_and_trims_window(
    mock_predict_cls: MagicMock,
    mock_parallel_cls: MagicMock,
    mock_judge_cls: MagicMock,
) -> None:
    rows = [
        {
            "message": f"message {i}",
            "urgency": "low",
            "sentiment": "neutral",
        }
        for i in range(25)
    ]
    mock_predict_cls.return_value = MagicMock(
        return_value=dspy.Prediction(generated_rows=json.dumps(rows))
    )
    mock_parallel_cls.side_effect = _sync_parallel()

    class FakeJudge:
        def __init__(self, lm: Any) -> None:
            pass

        def batch_score(
            self, rows: list[dict[str, Any]], task_description: str = ""
        ) -> list[SimpleNamespace]:
            return [SimpleNamespace(score=0.5) for _ in rows]

    mock_judge_cls.side_effect = FakeJudge

    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(
            n=25,
            generation_mode="signature",
            judge_enabled=True,
        ),
    )

    result_rows, scores = gen._generate_signature_mode(25, None, "tickets")

    assert len(result_rows) == 25
    assert scores == [0.5] * 25


# ---------------------------------------------------------------------------
# from_json and generate() flows
# ---------------------------------------------------------------------------


def test_from_json_reads_rows_key(tmp_path: Path) -> None:
    path = tmp_path / "seeds.json"
    path.write_text(
        json.dumps(
            {"rows": [{"message": "seed", "urgency": "low", "sentiment": "neutral"}]}
        )
    )

    gen = AutoData.from_json(path, DummyModule(), data_lm=MagicMock())

    assert gen.seed_examples == [
        {"message": "seed", "urgency": "low", "sentiment": "neutral"}
    ]


def test_generate_default_output_path_and_quality_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=1),
    )
    with (
        patch.object(gen, "_generate_inputs", return_value=[{"message": "input"}]),
        patch.object(
            gen,
            "_generate_outputs",
            return_value=([{"urgency": "low", "sentiment": "neutral"}], [0.8]),
        ),
    ):
        result = gen.generate(force=True)

    assert result.rows == [
        {"message": "input", "urgency": "low", "sentiment": "neutral"}
    ]
    assert result.quality_scores == [0.8]


def test_generate_uses_targeted_when_balanced(
    tmp_path: Path,
) -> None:
    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(
            n=2,
            balance_outputs=True,
            output_path=tmp_path / "targeted.jsonl",
        ),
    )
    gen._sig_metadata = _signature_metadata(urgency=["low", "high"])
    targeted_rows = [
        {"message": "a", "urgency": "low", "sentiment": "neutral"},
        {"message": "b", "urgency": "high", "sentiment": "negative"},
    ]
    with patch.object(
        gen, "_generate_targeted_inputs", return_value=targeted_rows
    ) as mock_targeted:
        result = gen.generate(force=True)

    mock_targeted.assert_called_once()
    assert result.rows == targeted_rows
    assert result.n_produced == 2


def test_subsample_balanced_skips_reasoning_field() -> None:
    gen = AutoData(
        module=DummyModule(),
        data_lm=MagicMock(),
        description="tickets",
        config=_config(n=2),
    )
    gen.output_fields = ["reasoning", "urgency"]
    gen._sig_metadata = SignatureMetadata(
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
    rows = [
        {"message": "a", "reasoning": "a", "urgency": "low"},
        {"message": "b", "reasoning": "b", "urgency": "high"},
        {"message": "c", "reasoning": "a", "urgency": "low"},
    ]

    result_rows, result_scores = gen._subsample_balanced(rows, 2, [0.1, 0.2, 0.3])

    assert len(result_rows) == 2
    assert len(result_scores) == 2
    assert result_rows[0]["urgency"] != result_rows[1]["urgency"]
