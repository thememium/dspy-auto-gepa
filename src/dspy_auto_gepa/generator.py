import ast
import dataclasses
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, cast

import dspy
import pandas as pd
from pydantic import create_model
from tqdm import tqdm

from .config import AutoDataConfig
from .data import (
    SignatureMetadata,
    _to_dicts,
    extract_signature_metadata,
    infer_fields_from_module,
)
from .quality import (
    LLMJudge,
    Validator,
    enum_validator,
    no_emoji_validator,
    non_empty_validator,
    sanitize_string,
)
from .runner import GenerationResult


def _extract_json(text: str) -> str:
    """Extract JSON from text that may be wrapped in markdown code blocks.

    LLMs frequently wrap JSON output in ```json ... ``` fences.  This strips
    those wrappers so ``json.loads()`` succeeds.  Falls back to scanning for
    the outermost ``{``…``}`` or ``[``…``]`` pair, and finally tries
    ``ast.literal_eval`` for Python dict/list literals.
    """
    text = text.strip()
    if not text:
        return text

    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, default=str)
    except (ValueError, SyntaxError):
        pass

    return text


_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def _build_output_model(
    metadata: SignatureMetadata,
    output_fields: list[str],
    name: str = "GeneratedOutput",
) -> type:
    """Dynamically create a Pydantic model from signature output field metadata.

    Maps each output field's python_type to a model field.  Fields with
    ``allowed_values`` are annotated as ``Literal`` types so DSPy can emit
    constrained JSON schemas for the LLM.
    """
    field_defs: dict[str, Any] = {}
    for fname in output_fields:
        if fname == "reasoning":
            continue
        meta = metadata.get(fname)
        ftype: Any = str
        if meta and meta.python_type and isinstance(meta.python_type, type):
            ftype = meta.python_type
        if meta and meta.allowed_values:
            from typing import Literal

            ftype = Literal.__getitem__(tuple(meta.allowed_values))
        default = ...  # required
        field_defs[fname] = (ftype, default)
    return create_model(name, **field_defs)


class StreamingDatasetWriter:
    """Writes dataset rows to disk immediately (crash-safe).

    Supports JSONL (.jsonl/.json), CSV (.csv), and Parquet (.parquet/.pq).
    Format is auto-detected from file extension.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._format = self._detect_format(self._path)
        self._rows_written = 0
        self._all_rows: list[dict] = []
        self._header_written = False

        if self._path.exists():
            existing = self.read_rows()
            self._rows_written = len(existing)
            self._all_rows = existing
            if self._format == "csv" and self._rows_written > 0:
                self._header_written = True

    @staticmethod
    def _detect_format(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".json"):
            return "jsonl"
        if suffix == ".csv":
            return "csv"
        if suffix in (".parquet", ".pq"):
            return "parquet"
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            "Use .jsonl, .json, .csv, .parquet, or .pq"
        )

    def write_row(self, row: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._format == "jsonl":
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        elif self._format == "csv":
            df = pd.DataFrame([row])
            df.to_csv(
                self._path,
                mode="a",
                header=not self._header_written,
                index=False,
            )
            self._header_written = True
        elif self._format == "parquet":
            self._all_rows.append(row)
            pd.DataFrame(self._all_rows).to_parquet(self._path, index=False)

        self._rows_written += 1

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.write_row(row)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.flush()

    def truncate(self) -> None:
        if self._path.exists():
            self._path.unlink()
        self._rows_written = 0
        self._all_rows = []
        self._header_written = False

    def row_count(self) -> int:
        return self._rows_written

    def read_rows(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []

        if self._format == "jsonl":
            rows = []
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        elif self._format == "csv":
            return pd.read_csv(self._path).to_dict(orient="records")
        elif self._format == "parquet":
            return pd.read_parquet(self._path).to_dict(orient="records")
        return []


def _validate_and_sanitize_row(
    row: dict[str, Any],
    field_names: list[str],
    sig_metadata: SignatureMetadata | None = None,
    *,
    include_enum: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Validate and sanitize a generated row against field metadata.

    Returns (cleaned_row, errors). The cleaned row has emoji stripped and
    whitespace normalized.  Errors list is non-empty when validation fails.
    """
    errors: list[str] = []
    cleaned: dict[str, Any] = {}

    for name in field_names:
        val = row.get(name)

        if isinstance(val, str):
            val = sanitize_string(val)

        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(f"Field '{name}' must not be empty")
            cleaned[name] = val if val is not None else ""
            continue

        if include_enum:
            meta = sig_metadata.get(name) if sig_metadata else None
            if meta and meta.allowed_values and isinstance(val, str):
                allowed_lower = [a.lower() for a in meta.allowed_values]
                if val.strip().lower() not in allowed_lower:
                    errors.append(
                        f"Field '{name}' value '{val}' not in allowed: {meta.allowed_values}"
                    )

        cleaned[name] = val

    return cleaned, errors


def _build_validator(
    field_names: list[str],
    sig_metadata: SignatureMetadata | None = None,
    *,
    include_enum: bool = True,
) -> Validator:
    """Build a Validator with non-empty + no-emoji + enum checks for *field_names*."""
    fns: list[Any] = [
        non_empty_validator(*field_names),
        no_emoji_validator(*field_names),
    ]

    if include_enum and sig_metadata:
        for name in field_names:
            meta = sig_metadata.get(name)
            if meta and meta.allowed_values:
                fns.append(enum_validator(name, meta.allowed_values))

    return Validator(fns)


class _InputGenerationSignature(dspy.Signature):
    """Generate realistic input data for a task.

    CRITICAL: Every generated input MUST be completely unique. Do NOT repeat,
    closely paraphrase, or slightly modify any existing input. Each new input
    must cover a different scenario, use different wording, and represent a
    distinct real-world case.

    Input values can be strings, numbers, booleans, lists, or nested dicts.
    Match the structure defined in the field spec. For nested/complex types,
    vary the structure and content — not just top-level values.

    Return a JSON array of input objects. Each object must have exactly the
    specified input field names as keys. Do NOT include emoji or special
    Unicode characters in string values.
    """

    task_description: str = dspy.InputField(desc="Description of the task")
    input_field_spec: str = dspy.InputField(
        desc=(
            "JSON spec of input fields with types and constraints. "
            'Example: {"message": {"type": "str", "desc": "The support ticket text"}, '
            '"metadata": {"type": "dict", "desc": "Nested object with category and priority"}}'
        )
    )
    recent_inputs_json: str = dspy.InputField(
        desc=(
            "JSON array of the most recent inputs generated. Use these to ensure "
            "your new inputs are DIFFERENT in topic, wording, and scenario."
        ),
        default="[]",
    )
    covered_themes: str = dspy.InputField(
        desc=(
            "Comma-separated list of themes/values already covered. You MUST "
            "generate inputs that explore NEW themes not in this list."
        ),
        default="",
    )
    n_to_generate: int = dspy.InputField(desc="Number of input objects to generate")
    generated_inputs: str = dspy.OutputField(
        desc=(
            "JSON array of input objects. Each object must have exactly the "
            "specified field names. No emoji. No empty strings. Every entry "
            "MUST be completely different from all existing inputs in both "
            "structure and content."
        )
    )


class _OutputGenerationSignature(dspy.Signature):
    """Generate correct output values for a given input.

    Return a JSON object with exactly the specified output field names as keys.
    Values MUST match the allowed values listed in the field spec.
    Do NOT include emoji or special Unicode characters.
    """

    task_description: str = dspy.InputField(desc="Description of the task")
    input_data: str = dspy.InputField(desc="JSON object of the input row")
    output_field_spec: str = dspy.InputField(
        desc=(
            "JSON spec of output fields with types, allowed values, and descriptions. "
            'Example: {"urgency": {"type": "str", "allowed": ["low", "medium", "high"]}}'
        )
    )
    generated_output: str = dspy.OutputField(
        desc=(
            "JSON object with exactly the specified output field names. "
            "Values must be one of the allowed values if specified."
        )
    )


def _build_signature_generation_signature() -> type[dspy.Signature]:
    """Build a DSPy Signature that generates complete rows using the user's signature.

    The returned signature has:
    - Inputs: task_description, recent_rows_json, covered_themes, n_to_generate
    - Outputs: generated_rows (JSON array of complete rows)

    This is used in "signature" generation mode where inputs and outputs
    are generated together in one shot.
    """

    class _SignatureGenSig(dspy.Signature):
        """Generate realistic data rows for a task.

        Each row must contain ALL fields specified in the task signature —
        both input and output fields.  The values must be coherent: outputs
        must logically follow from the inputs.

        CRITICAL: Every generated row MUST be completely unique. Do NOT repeat,
        closely paraphrase, or slightly modify any existing row. Each new row
        must cover a different scenario, use different wording, and represent a
        distinct real-world case.

        Return a JSON array of row objects. Each object must have exactly the
        specified field names as keys. Do NOT include emoji or special
        Unicode characters in string values.
        """

        task_description: str = dspy.InputField(
            desc="Description of the task and what the fields mean"
        )
        field_spec: str = dspy.InputField(
            desc=(
                "JSON spec of ALL fields (inputs and outputs) with types, "
                "descriptions, and allowed values. "
                'Example: {"message": {"type": "str", "desc": "The support ticket text"}, '
                '"urgency": {"type": "str", "allowed": ["low", "medium", "high"]}}'
            )
        )
        recent_rows_json: str = dspy.InputField(
            desc=(
                "JSON array of the most recent rows generated. Use these to ensure "
                "your new rows are DIFFERENT in topic, wording, and scenario."
            ),
            default="[]",
        )
        covered_themes: str = dspy.InputField(
            desc=(
                "Comma-separated list of themes/values already covered. You MUST "
                "generate rows that explore NEW themes not in this list."
            ),
            default="",
        )
        n_to_generate: int = dspy.InputField(desc="Number of complete rows to generate")
        generated_rows: str = dspy.OutputField(
            desc=(
                "JSON array of complete row objects. Each object must have exactly the "
                "specified field names. No emoji. No empty strings. Every entry "
                "MUST be completely different from all existing rows in both "
                "structure and content. Outputs must logically follow from inputs."
            )
        )

    return _SignatureGenSig


def _build_batch_output_signature(output_model: type) -> type[dspy.Signature]:
    """Build a DSPy Signature with strongly-typed batch output.

    The returned signature has ``generated_outputs: list[output_model]`` so DSPy
    emits a JSON Schema for the LLM rather than a free-form string.  The type
    is set via ``with_updated_fields`` after class creation because static type
    checkers reject using a runtime variable (``output_model``) as a type
    argument in a class-body annotation.
    """

    class _BatchOutputSignature(dspy.Signature):
        """Generate correct output values for multiple inputs at once.

        Return one output object per input, in the SAME ORDER as the inputs
        array.  Each object must match the schema exactly.
        """

        task_description: str = dspy.InputField(desc="Description of the task")
        inputs_json: str = dspy.InputField(
            desc="JSON array of input objects to generate outputs for"
        )
        n_to_generate: int = dspy.InputField(
            desc="Number of output objects to generate (must match inputs_json length)"
        )
        generated_outputs: list[Any] = dspy.OutputField(
            desc="List of output objects, one per input, in the same order."
        )

    return _BatchOutputSignature.with_updated_fields(
        "generated_outputs",
        type_=cast("type | None", list.__class_getitem__((output_model,))),
    )


def _compute_output_combos(
    output_fields: list[str],
    metadata: SignatureMetadata,
) -> list[dict[str, str]] | None:
    """Compute the cartesian product of allowed_values for categorical output fields.

    Returns a list of target dicts (e.g. ``[{"urgency": "high", "sentiment": "positive"}, ...]``)
    or ``None`` when no output field has ``allowed_values`` (balancing not applicable).
    """
    import itertools

    categorical: dict[str, list[str]] = {}
    for fname in output_fields:
        if fname == "reasoning":
            continue
        meta = metadata.get(fname)
        if meta and meta.allowed_values:
            categorical[fname] = meta.allowed_values

    if not categorical:
        return None

    keys = list(categorical.keys())
    value_lists = [categorical[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


class AutoData:
    def __init__(
        self,
        module: dspy.Module,
        data_lm: dspy.LM | None = None,
        config: AutoDataConfig | None = None,
        schema: type[Any] | None = None,
        description: str | None = None,
        name: str | None = None,
    ) -> None:
        self.module = module
        self.config = config or AutoDataConfig()
        self.description = description
        self.schema = schema
        self._name = name

        self.data_lm = data_lm or self.config.data_lm or dspy.settings.lm

        self.input_fields, self.output_fields = infer_fields_from_module(module)
        self.output_fields = [f for f in self.output_fields if f != "reasoning"]

        sig = getattr(module, "signature", None)
        if sig is None:
            predictors = list(getattr(module, "named_predictors", lambda: [])())
            if predictors:
                sig = getattr(predictors[0][1], "signature", None)

        sig_docstring = getattr(sig, "__doc__", "") or ""
        if not sig_docstring.strip() and not self.description:
            raise ValueError(
                "Module signature has no docstring and no description was provided. "
                "Pass description='...' to describe the task for data generation."
            )

        self._schema_fields: dict[str, str] = {}
        if self.schema is not None:
            for name, field_info in self.schema.model_fields.items():
                self._schema_fields[name] = str(field_info.annotation)

        self.seed_examples: list[dict[str, Any]] | None = None
        self._sig_metadata: SignatureMetadata | None = None

    def _ensure_metadata(
        self, seed_examples: list[dict[str, Any]] | None = None
    ) -> SignatureMetadata:
        if self._sig_metadata is not None:
            return self._sig_metadata

        sig = getattr(self.module, "signature", None)
        if sig is None:
            predictors = list(getattr(self.module, "named_predictors", lambda: [])())
            if predictors:
                sig = getattr(predictors[0][1], "signature", None)

        if sig is not None:
            self._sig_metadata = extract_signature_metadata(
                sig, seed_examples=seed_examples or self.seed_examples
            )
        else:
            self._sig_metadata = SignatureMetadata(fields=[])

        return self._sig_metadata

    def _field_spec_json(
        self, field_names: list[str], metadata: SignatureMetadata
    ) -> str:
        spec: dict[str, Any] = {}
        for name in field_names:
            meta = metadata.get(name)
            entry: dict[str, Any] = {"type": "str"}
            if meta:
                if meta.description:
                    entry["desc"] = meta.description
                if meta.allowed_values:
                    entry["allowed"] = meta.allowed_values
            spec[name] = entry
        return json.dumps(spec)

    def _generate_inputs(
        self,
        n: int,
        seed_examples: list[dict[str, Any]] | None,
        description: str,
        progress: tqdm | None = None,
    ) -> list[dict[str, Any]]:
        predictor = dspy.Predict(_InputGenerationSignature)
        all_inputs: list[dict[str, Any]] = []
        max_retries = self.config.max_retries
        metadata = self._ensure_metadata(seed_examples)
        input_spec = self._field_spec_json(self.input_fields, metadata)
        validator = _build_validator(self.input_fields, metadata)

        all_existing: list[dict[str, Any]] = list(seed_examples or [])
        recent_window: list[dict[str, Any]] = list(seed_examples or [])[-20:]
        covered_values: dict[str, set[str]] = {f: set() for f in self.input_fields}
        for row in all_existing:
            for f in self.input_fields:
                val = row.get(f)
                if isinstance(val, str) and val.strip():
                    covered_values[f].add(val.strip().lower())

        def _canonicalize(value: Any) -> str:
            if value is None:
                return "__none__"
            if isinstance(value, str):
                return sanitize_string(value).lower()
            if isinstance(value, (int, float, bool)):
                return str(value)
            if isinstance(value, dict):
                return json.dumps(value, sort_keys=True, default=str)
            if isinstance(value, (list, tuple)):
                return json.dumps(list(value), sort_keys=True, default=str)
            if hasattr(value, "model_dump"):
                return json.dumps(value.model_dump(), sort_keys=True, default=str)
            if hasattr(value, "dict"):
                return json.dumps(value.dict(), sort_keys=True, default=str)
            return str(value)

        def _row_fingerprint(row: dict[str, Any]) -> str:
            parts = []
            for k in sorted(row.keys()):
                if k in self.input_fields:
                    parts.append(f"{k}={_canonicalize(row[k])}")
            return "|".join(parts)

        def _is_duplicate(row: dict[str, Any]) -> bool:
            fp = _row_fingerprint(row)
            return any(_row_fingerprint(e) == fp for e in all_existing)

        def _covered_themes_str() -> str:
            parts = []
            for f, vals in covered_values.items():
                if vals:
                    parts.append(f"{f}: {', '.join(sorted(vals))}")
            return "; ".join(parts)

        def _generate_one_batch(
            recent_json: str, themes_str: str, batch_size: int
        ) -> list[dict[str, Any]]:
            for _attempt in range(max_retries):
                try:
                    with dspy.settings.context(lm=self.data_lm):
                        result = predictor(
                            task_description=description,
                            input_field_spec=input_spec,
                            recent_inputs_json=recent_json,
                            covered_themes=themes_str,
                            n_to_generate=batch_size,
                        )
                    parsed = json.loads(_extract_json(result.generated_inputs))
                    if not isinstance(parsed, list):
                        parsed = [parsed]
                    return [r for r in parsed if isinstance(r, dict)]
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
            return []

        parallel = dspy.Parallel(
            num_threads=self.config.num_threads,
            return_failed_examples=True,
            max_errors=self.config.num_threads * 3,
        )

        max_total_attempts = max(max_retries * 5, n)
        consecutive_failures = 0

        with dspy.settings.context(lm=self.data_lm):
            while len(all_inputs) < n:
                if consecutive_failures >= max_total_attempts:
                    break

                chunk_batches = min(
                    self.config.chunk_size,
                    (n - len(all_inputs) + 9) // 10,
                )

                recent_json = json.dumps(recent_window, default=str)
                themes_str = _covered_themes_str()

                tasks = []
                for _ in range(chunk_batches):
                    remaining = n - len(all_inputs)
                    batch_size = min(10, remaining)
                    example = dspy.Example(
                        task_description=description,
                        input_field_spec=input_spec,
                        recent_inputs_json=recent_json,
                        covered_themes=themes_str,
                        n_to_generate=batch_size,
                    ).with_inputs(
                        "task_description",
                        "input_field_spec",
                        "recent_inputs_json",
                        "covered_themes",
                        "n_to_generate",
                    )
                    tasks.append((predictor, example))

                exec_results = parallel(tasks)

                accepted_this_chunk = 0
                for result in exec_results:
                    if isinstance(result, Exception):
                        continue
                    generated = result if isinstance(result, list) else [result]
                    for item in generated:
                        if isinstance(item, dspy.Prediction):
                            try:
                                parsed = json.loads(
                                    _extract_json(item.generated_inputs)
                                )
                                if not isinstance(parsed, list):
                                    parsed = [parsed]
                            except (json.JSONDecodeError, ValueError, TypeError):
                                continue
                        elif isinstance(item, dict):
                            parsed = [item]
                        else:
                            continue

                        for row in parsed:
                            if not isinstance(row, dict):
                                continue
                            clean_row, errs = _validate_and_sanitize_row(
                                row, self.input_fields, metadata
                            )
                            if errs:
                                continue
                            val_result = validator.validate(clean_row)
                            if not val_result.is_valid:
                                continue
                            if _is_duplicate(clean_row):
                                continue

                            all_inputs.append(clean_row)
                            all_existing.append(clean_row)
                            recent_window.append(clean_row)
                            if len(recent_window) > 20:
                                recent_window = recent_window[-20:]
                            for f in self.input_fields:
                                val = clean_row.get(f)
                                if isinstance(val, str) and val.strip():
                                    covered_values[f].add(val.strip().lower())

                            accepted_this_chunk += 1
                            consecutive_failures = 0
                            if progress is not None:
                                progress.update(1)
                            if len(all_inputs) >= n:
                                break
                        if len(all_inputs) >= n:
                            break
                    if len(all_inputs) >= n:
                        break

                if accepted_this_chunk == 0:
                    consecutive_failures += 1

        return all_inputs[:n]

    def _generate_outputs(
        self,
        inputs: list[dict[str, Any]],
        description: str,
        writer: StreamingDatasetWriter | None = None,
        progress: tqdm | None = None,
    ) -> tuple[list[dict[str, Any] | None], list[float] | None]:
        metadata = self._ensure_metadata()
        output_model = _build_output_model(metadata, self.output_fields)
        batch_sig = _build_batch_output_signature(output_model)
        batch_predictor = dspy.Predict(batch_sig)
        single_predictor = dspy.Predict(_OutputGenerationSignature)
        output_spec = self._field_spec_json(self.output_fields, metadata)
        validator = _build_validator(self.output_fields, metadata, include_enum=False)
        max_retries = self.config.max_retries
        batch_size = min(10, self.config.chunk_size * 2)

        judge = None
        if self.config.judge_enabled:
            judge_lm = self.config.judge_lm or self.data_lm
            judge = LLMJudge(lm=judge_lm)

        output_map: dict[int, tuple[dict[str, Any], float | None]] = {}
        written_indices: set[int] = set()
        pending_indices = list(range(len(inputs)))

        def _to_dict(raw: Any) -> dict[str, Any] | None:
            if isinstance(raw, dict):
                return raw
            if hasattr(raw, "model_dump"):
                result = raw.model_dump()
                if isinstance(result, dict):
                    return result
            if hasattr(raw, "dict"):
                result = raw.dict()
                if isinstance(result, dict):
                    return result
            return None

        def _validate_and_accept(
            inp: dict[str, Any], raw_output: Any, idx: int
        ) -> bool:
            if isinstance(raw_output, Exception):
                return False
            items = raw_output if isinstance(raw_output, list) else [raw_output]
            for item in items:
                parsed = _to_dict(item)
                if parsed is None:
                    if not isinstance(item, dspy.Prediction):
                        continue
                    try:
                        parsed = json.loads(_extract_json(item.generated_output))
                    except (json.JSONDecodeError, ValueError, TypeError):
                        continue
                    if not isinstance(parsed, dict):
                        continue
                clean_output = {}
                for k in self.output_fields:
                    if k == "reasoning":
                        continue
                    val = parsed.get(k)
                    if isinstance(val, str):
                        val = sanitize_string(val)
                    clean_output[k] = val if val is not None else ""
                clean_output, val_errors = _validate_and_sanitize_row(
                    clean_output, self.output_fields, metadata, include_enum=False
                )
                if val_errors:
                    continue
                val_result = validator.validate(clean_output)
                if not val_result.is_valid:
                    continue
                score = None
                if judge is not None:
                    full_row = {**inp, **clean_output}
                    score = judge.score(full_row, task_description=description).score
                output_map[idx] = (clean_output, score)
                if idx not in written_indices:
                    if writer is not None:
                        writer.write_row({**inp, **clean_output})
                    written_indices.add(idx)
                return True
            return False

        with dspy.settings.context(lm=self.data_lm):
            for attempt in range(max_retries * 3):
                if not pending_indices:
                    break

                batch_indices = pending_indices[:batch_size]
                batch_inputs = [inputs[i] for i in batch_indices]
                inputs_json = json.dumps(batch_inputs, default=str)

                try:
                    result = batch_predictor(
                        task_description=description,
                        inputs_json=inputs_json,
                        n_to_generate=len(batch_indices),
                    )
                    raw_outputs = result.generated_outputs
                    if not isinstance(raw_outputs, list):
                        raw_outputs = [raw_outputs]
                except Exception as exc:
                    tqdm.write(
                        f"  [warn] batch output generation failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    raw_outputs = []

                still_pending = []
                for i, idx in enumerate(batch_indices):
                    inp = inputs[idx]
                    if i < len(raw_outputs):
                        if _validate_and_accept(inp, raw_outputs[i], idx):
                            continue
                    still_pending.append(idx)

                for idx in still_pending:
                    inp = inputs[idx]
                    try:
                        single_result = single_predictor(
                            task_description=description,
                            input_data=json.dumps(inp, default=str),
                            output_field_spec=output_spec,
                        )
                        _validate_and_accept(inp, single_result, idx)
                    except Exception as exc:
                        tqdm.write(
                            f"  [warn] single output generation failed for "
                            f"row {idx}: {type(exc).__name__}: {exc}"
                        )

                accepted = (
                    len(batch_indices)
                    - len(still_pending)
                    + sum(1 for idx in still_pending if idx in output_map)
                )
                if progress is not None:
                    progress.update(accepted)

                pending_indices = [i for i in pending_indices if i not in output_map]

        all_outputs: list[dict[str, Any] | None] = []
        all_scores: list[float] = []
        for idx in range(len(inputs)):
            if idx in output_map:
                output, score = output_map[idx]
                all_outputs.append(output)
                if score is not None:
                    all_scores.append(score)
            else:
                all_outputs.append(None)

        return all_outputs, all_scores if judge is not None else None

    def _generate_signature_mode(
        self,
        n: int,
        seed_examples: list[dict[str, Any]] | None,
        description: str,
        writer: StreamingDatasetWriter | None = None,
        progress: tqdm | None = None,
    ) -> tuple[list[dict[str, Any]], list[float] | None]:
        """Generate complete rows (inputs + outputs) in one shot using the signature.

        This is used when ``config.generation_mode == "signature"``.
        """
        metadata = self._ensure_metadata(seed_examples)
        all_fields = self.input_fields + self.output_fields
        field_spec = self._field_spec_json(all_fields, metadata)
        validator = _build_validator(all_fields, metadata)
        max_retries = self.config.max_retries

        judge = None
        if self.config.judge_enabled:
            judge_lm = self.config.judge_lm or self.data_lm
            judge = LLMJudge(lm=judge_lm)

        sig_cls = _build_signature_generation_signature()
        predictor = dspy.Predict(sig_cls)

        all_rows: list[dict[str, Any]] = []
        all_scores: list[float] = []
        all_existing: list[dict[str, Any]] = list(seed_examples or [])
        recent_window: list[dict[str, Any]] = list(seed_examples or [])[-20:]
        covered_values: dict[str, set[str]] = {f: set() for f in all_fields}
        for row in all_existing:
            for f in all_fields:
                val = row.get(f)
                if isinstance(val, str) and val.strip():
                    covered_values[f].add(val.strip().lower())

        def _canonicalize(value: Any) -> str:
            if value is None:
                return "__none__"
            if isinstance(value, str):
                return sanitize_string(value).lower()
            if isinstance(value, (int, float, bool)):
                return str(value)
            if isinstance(value, dict):
                return json.dumps(value, sort_keys=True, default=str)
            if isinstance(value, (list, tuple)):
                return json.dumps(list(value), sort_keys=True, default=str)
            if hasattr(value, "model_dump"):
                return json.dumps(value.model_dump(), sort_keys=True, default=str)
            if hasattr(value, "dict"):
                return json.dumps(value.dict(), sort_keys=True, default=str)
            return str(value)

        def _row_fingerprint(row: dict[str, Any]) -> str:
            parts = []
            for k in sorted(row.keys()):
                if k in all_fields:
                    parts.append(f"{k}={_canonicalize(row[k])}")
            return "|".join(parts)

        def _is_duplicate(row: dict[str, Any]) -> bool:
            fp = _row_fingerprint(row)
            return any(_row_fingerprint(e) == fp for e in all_existing)

        def _covered_themes_str() -> str:
            parts = []
            for f, vals in covered_values.items():
                if vals:
                    parts.append(f"{f}: {', '.join(sorted(vals))}")
            return "; ".join(parts)

        parallel = dspy.Parallel(
            num_threads=self.config.num_threads,
            return_failed_examples=True,
            max_errors=self.config.num_threads * 3,
        )

        max_total_attempts = max(max_retries * 5, n)
        consecutive_failures = 0

        with dspy.settings.context(lm=self.data_lm):
            while len(all_rows) < n:
                if consecutive_failures >= max_total_attempts:
                    break

                chunk_batches = min(
                    self.config.chunk_size,
                    (n - len(all_rows) + 9) // 10,
                )

                recent_json = json.dumps(recent_window, default=str)
                themes_str = _covered_themes_str()

                tasks = []
                for _ in range(chunk_batches):
                    remaining = n - len(all_rows)
                    batch_size = min(10, remaining)
                    example = dspy.Example(
                        task_description=description,
                        field_spec=field_spec,
                        recent_rows_json=recent_json,
                        covered_themes=themes_str,
                        n_to_generate=batch_size,
                    ).with_inputs(
                        "task_description",
                        "field_spec",
                        "recent_rows_json",
                        "covered_themes",
                        "n_to_generate",
                    )
                    tasks.append((predictor, example))

                exec_results = parallel(tasks)

                accepted_this_chunk = 0
                for result in exec_results:
                    if isinstance(result, Exception):
                        continue
                    generated = result if isinstance(result, list) else [result]
                    for item in generated:
                        if isinstance(item, dspy.Prediction):
                            try:
                                parsed = json.loads(_extract_json(item.generated_rows))
                                if not isinstance(parsed, list):
                                    parsed = [parsed]
                            except (json.JSONDecodeError, ValueError, TypeError):
                                continue
                        elif isinstance(item, dict):
                            parsed = [item]
                        else:
                            continue

                        for row in parsed:
                            if not isinstance(row, dict):
                                continue
                            clean_row, errs = _validate_and_sanitize_row(
                                row, all_fields, metadata
                            )
                            if errs:
                                continue
                            val_result = validator.validate(clean_row)
                            if not val_result.is_valid:
                                continue
                            if _is_duplicate(clean_row):
                                continue

                            score = None
                            if judge is not None:
                                score = judge.score(
                                    clean_row, task_description=description
                                ).score

                            all_rows.append(clean_row)
                            all_scores.append(score or 0.0)
                            all_existing.append(clean_row)
                            recent_window.append(clean_row)
                            if len(recent_window) > 20:
                                recent_window = recent_window[-20:]
                            for f in all_fields:
                                val = clean_row.get(f)
                                if isinstance(val, str) and val.strip():
                                    covered_values[f].add(val.strip().lower())

                            if writer is not None:
                                writer.write_row(clean_row)

                            accepted_this_chunk += 1
                            consecutive_failures = 0
                            if progress is not None:
                                progress.update(1)
                            if len(all_rows) >= n:
                                break
                        if len(all_rows) >= n:
                            break
                    if len(all_rows) >= n:
                        break

                if accepted_this_chunk == 0:
                    consecutive_failures += 1

        return all_rows[:n], all_scores if judge else None

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        module: dspy.Module,
        **kwargs: Any,
    ) -> "AutoData":
        df = pd.read_csv(path)
        seed = df.to_dict(orient="records")
        gen = cls(module=module, **kwargs)
        gen.seed_examples = seed
        return gen

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        module: dspy.Module,
        **kwargs: Any,
    ) -> "AutoData":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "rows" in data:
            data = data["rows"]
        seed = _to_dicts(data)
        gen = cls(module=module, **kwargs)
        gen.seed_examples = seed
        return gen

    def generate(
        self,
        n: int | None = None,
        seed_examples: Any | None = None,
        force: bool = False,
        output_path: str | Path | None = None,
        *,
        seed: int | None = None,
        max_retries: int | None = None,
        num_threads: int | None = None,
        chunk_size: int | None = None,
        diversity_threshold: float | None = None,
        judge_enabled: bool | None = None,
        validators_enabled: bool | None = None,
        diversity_enabled: bool | None = None,
        rejection_sampling_enabled: bool | None = None,
        data_lm: dspy.LM | None = None,
        judge_lm: dspy.LM | None = None,
        balance_outputs: bool | None = None,
        balance_tolerance: float | None = None,
        oversample_factor: float | None = None,
    ) -> GenerationResult:
        overrides = {
            k: v
            for k, v in {
                "n": n,
                "seed": seed,
                "max_retries": max_retries,
                "num_threads": num_threads,
                "chunk_size": chunk_size,
                "diversity_threshold": diversity_threshold,
                "judge_enabled": judge_enabled,
                "validators_enabled": validators_enabled,
                "diversity_enabled": diversity_enabled,
                "rejection_sampling_enabled": rejection_sampling_enabled,
                "data_lm": data_lm,
                "judge_lm": judge_lm,
                "balance_outputs": balance_outputs,
                "balance_tolerance": balance_tolerance,
                "oversample_factor": oversample_factor,
            }.items()
            if v is not None
        }
        config = dataclasses.replace(self.config, **overrides)
        n = config.n
        self.config = config

        start_time = time.time()
        random.seed(config.seed)

        resolved_seeds = self._resolve_seeds(seed_examples)

        if output_path is None:
            name = self._name or "unnamed"
            output_path = Path(".auto_gepa") / name / "generated" / "rows.jsonl"
        output_path = Path(output_path)

        writer = StreamingDatasetWriter(output_path)

        if force:
            writer.truncate()

        if not force and writer.row_count() > 0:
            existing_rows = writer.read_rows()
            if len(existing_rows) >= n:
                elapsed = time.time() - start_time
                return GenerationResult(
                    rows=existing_rows[:n],
                    n_requested=n,
                    n_produced=len(existing_rows[:n]),
                    n_failed=0,
                    seed_used=self.config.seed,
                    generation_time_seconds=elapsed,
                )

        description = (
            self.description
            or getattr(getattr(self.module, "signature", None), "__doc__", "")
            or ""
        )

        self._ensure_metadata(resolved_seeds)

        if self.config.generation_mode == "signature":
            bar = tqdm(total=n, desc="Generating rows", unit="row", leave=True)
            complete_rows, quality_scores = self._generate_signature_mode(
                n, resolved_seeds, description, writer=writer, progress=bar
            )
            bar.close()
            row_scores = quality_scores or []
        else:
            output_combos: list[dict[str, str]] | None = None
            if self.config.balance_outputs and self._sig_metadata is not None:
                output_combos = _compute_output_combos(
                    self.output_fields, self._sig_metadata
                )
                if output_combos is not None:
                    tqdm.write(
                        f"Balancing outputs across {len(output_combos)} target combos"
                    )

            pool_n = n
            if output_combos is not None:
                pool_n = int(n * self.config.oversample_factor)
                tqdm.write(f"Oversampling: generating {pool_n} rows for balance pool")

            input_bar = tqdm(
                total=pool_n, desc="Generating inputs", unit="row", leave=True
            )
            inputs = self._generate_inputs(
                pool_n, resolved_seeds, description, input_bar
            )
            input_bar.close()

            output_bar = tqdm(
                total=len(inputs), desc="Generating outputs", unit="row", leave=True
            )
            outputs, quality_scores = self._generate_outputs(
                inputs, description, writer=writer, progress=output_bar
            )
            output_bar.close()

            complete_rows = []
            row_scores: list[float] = []
            for i, (inp, out) in enumerate(zip(inputs, outputs)):
                if out is not None:
                    complete_rows.append({**inp, **out})
                    if quality_scores is not None and i < len(quality_scores):
                        row_scores.append(quality_scores[i])

            if (
                self.config.balance_outputs
                and output_combos is not None
                and len(complete_rows) > n
            ):
                complete_rows, row_scores = self._subsample_balanced(
                    complete_rows, n, row_scores
                )
                tqdm.write(f"Rewriting {len(complete_rows)} balanced rows to disk")
                writer.truncate()
                for row in complete_rows:
                    writer.write_row(row)

        n_written = writer.row_count()
        elapsed = time.time() - start_time
        tqdm.write(
            f"Generated {n_written}/{n} rows in {elapsed:.1f}s "
            f"({n_written / elapsed:.1f} rows/s)"
        )
        return GenerationResult(
            rows=complete_rows,
            n_requested=n,
            n_produced=n_written,
            n_failed=n - n_written,
            seed_used=self.config.seed,
            generation_time_seconds=elapsed,
            quality_scores=row_scores if row_scores else None,
        )

    def _subsample_balanced(
        self,
        rows: list[dict[str, Any]],
        n: int,
        scores: list[float],
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Greedily select n rows from the pool to maximise per-column balance.

        For each categorical output field, tracks how many times each allowed
        value has been selected.  On each iteration, picks the row whose
        output values are most under-represented (highest cumulative deficit
        across all categorical fields).  Ties are broken randomly.
        """
        import random as _rand
        from collections import Counter

        metadata = self._sig_metadata
        assert metadata is not None

        categorical_fields: list[str] = []
        field_values: dict[str, list[str]] = {}
        for fname in self.output_fields:
            if fname == "reasoning":
                continue
            meta = metadata.get(fname)
            if meta and meta.allowed_values:
                categorical_fields.append(fname)
                field_values[fname] = meta.allowed_values

        if not categorical_fields:
            return rows[:n], scores[:n]

        target_per_value: dict[str, float] = {}
        for fname in categorical_fields:
            target_per_value[fname] = n / len(field_values[fname])

        counts: dict[str, Counter[str]] = {f: Counter() for f in categorical_fields}

        pool = (
            list(zip(rows, scores))
            if len(scores) == len(rows)
            else [(r, 0.0) for r in rows]
        )
        rng = _rand.Random(self.config.seed)
        rng.shuffle(pool)

        selected: list[tuple[dict[str, Any], float]] = []

        for _ in range(min(n, len(pool))):
            best_idx = -1
            best_score = -1.0
            for i, (row, _) in enumerate(pool):
                score = 0.0
                for fname in categorical_fields:
                    val = row.get(fname)
                    if isinstance(val, str):
                        val_lower = val.strip().lower()
                        deficit = target_per_value[fname] - counts[fname][val_lower]
                        score += max(0.0, deficit)
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx == -1:
                break
            row, sc = pool.pop(best_idx)
            selected.append((row, sc))
            for fname in categorical_fields:
                val = row.get(fname)
                if isinstance(val, str):
                    counts[fname][val.strip().lower()] += 1

        result_rows = [r for r, _ in selected]
        result_scores = [s for _, s in selected]

        tqdm.write("  [balance] final distribution:")
        for fname in categorical_fields:
            parts = [f"{v}: {counts[fname].get(v, 0)}" for v in field_values[fname]]
            tqdm.write(f"    {fname}: {', '.join(parts)}")

        return result_rows, result_scores

    def _resolve_seeds(self, seed_examples: Any | None) -> list[dict[str, Any]] | None:
        if seed_examples is None:
            return self.seed_examples
        return _to_dicts(seed_examples)
