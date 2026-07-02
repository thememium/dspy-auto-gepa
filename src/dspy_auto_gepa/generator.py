import json
import os
import random
import time
from pathlib import Path
from typing import Any

import dspy
import pandas as pd
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
) -> Validator:
    """Build a Validator with non-empty + no-emoji + enum checks for *field_names*."""
    fns: list[Any] = [
        non_empty_validator(*field_names),
        no_emoji_validator(*field_names),
    ]

    if sig_metadata:
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
                    parsed = json.loads(result.generated_inputs)
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

        with dspy.settings.context(lm=self.data_lm):
            while len(all_inputs) < n:
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
                                parsed = json.loads(item.generated_inputs)
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
                            if progress is not None:
                                progress.update(1)
                            if len(all_inputs) >= n:
                                break
                        if len(all_inputs) >= n:
                            break
                    if len(all_inputs) >= n:
                        break

        return all_inputs[:n]

    def _generate_outputs(
        self,
        inputs: list[dict[str, Any]],
        description: str,
        writer: StreamingDatasetWriter | None = None,
        progress: tqdm | None = None,
    ) -> list[dict[str, Any]]:
        predictor = dspy.Predict(_OutputGenerationSignature)
        all_outputs: list[dict[str, Any]] = []
        max_retries = self.config.max_retries

        judge = None
        if self.config.judge_enabled:
            judge_lm = self.config.judge_lm or self.data_lm
            judge = LLMJudge(lm=judge_lm)

        metadata = self._ensure_metadata()
        output_spec = self._field_spec_json(self.output_fields, metadata)
        validator = _build_validator(self.output_fields, metadata)

        for inp in inputs:
            row_output: dict[str, Any] | None = None
            last_errors: list[str] = []

            for attempt in range(max_retries):
                try:
                    with dspy.settings.context(lm=self.data_lm):
                        result = predictor(
                            task_description=description,
                            input_data=json.dumps(inp, default=str),
                            output_field_spec=output_spec,
                        )

                    parsed = json.loads(result.generated_output)
                    if not isinstance(parsed, dict):
                        last_errors.append("LLM did not return a JSON object")
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
                        clean_output, self.output_fields, metadata
                    )
                    if val_errors:
                        last_errors = val_errors
                        continue

                    val_result = validator.validate(clean_output)
                    if not val_result.is_valid:
                        last_errors = val_result.failures
                        continue

                    if judge is not None:
                        full_row = {**inp, **clean_output}
                        judge_result = judge.score(
                            full_row, task_description=description
                        )
                        if judge_result.score < 0.5 and attempt < max_retries - 1:
                            last_errors.append(
                                f"Judge score {judge_result.score:.2f} below threshold"
                            )
                            continue

                    row_output = clean_output
                    break

                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    last_errors.append(str(exc))
                    continue

            if row_output is not None:
                all_outputs.append(row_output)
                if writer is not None:
                    writer.write_row({**inp, **row_output})
            else:
                fallback = {k: "" for k in self.output_fields if k != "reasoning"}
                all_outputs.append(fallback)
                if writer is not None:
                    writer.write_row({**inp, **fallback})

            if progress is not None:
                progress.update(1)

        return all_outputs

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
    ) -> GenerationResult:
        n = n or self.config.n
        start_time = time.time()
        random.seed(self.config.seed)

        resolved_seeds = self._resolve_seeds(seed_examples)

        if output_path is None:
            name = self._name or "unnamed"
            output_path = Path(".auto_gepa") / name / "generated" / "rows.jsonl"
        output_path = Path(output_path)

        writer = StreamingDatasetWriter(output_path)

        if not force and writer.row_count() > 0:
            existing_rows = writer.read_rows()
            elapsed = time.time() - start_time
            return GenerationResult(
                rows=existing_rows,
                n_requested=n,
                n_produced=len(existing_rows),
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

        input_bar = tqdm(total=n, desc="Generating inputs", unit="row", leave=True)
        inputs = self._generate_inputs(n, resolved_seeds, description, input_bar)
        input_bar.close()

        output_bar = tqdm(
            total=len(inputs), desc="Generating outputs", unit="row", leave=True
        )
        outputs = self._generate_outputs(
            inputs, description, writer=writer, progress=output_bar
        )
        output_bar.close()

        complete_rows = []
        for inp, out in zip(inputs, outputs):
            complete_rows.append({**inp, **out})

        quality_scores = None
        if self.config.judge_enabled:
            judge = LLMJudge(lm=self.config.judge_lm or self.data_lm)
            quality_scores = []
            for row in complete_rows:
                qr = judge.score(row, task_description=description)
                quality_scores.append(qr.score)

        elapsed = time.time() - start_time
        tqdm.write(
            f"Generated {len(complete_rows)}/{n} rows in {elapsed:.1f}s "
            f"({len(complete_rows) / elapsed:.1f} rows/s)"
        )
        return GenerationResult(
            rows=complete_rows,
            n_requested=n,
            n_produced=len(complete_rows),
            n_failed=n - len(complete_rows),
            seed_used=self.config.seed,
            generation_time_seconds=elapsed,
            quality_scores=quality_scores,
        )

    def _resolve_seeds(self, seed_examples: Any | None) -> list[dict[str, Any]] | None:
        if seed_examples is None:
            return self.seed_examples
        return _to_dicts(seed_examples)
