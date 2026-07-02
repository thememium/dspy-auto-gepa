import json
import os
import random
import time
from pathlib import Path
from typing import Any

import dspy
import pandas as pd

from .config import AutoDataConfig
from .data import _to_dicts, infer_fields_from_module
from .quality import DiversityChecker, LLMJudge
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
        self._all_rows: list[dict] = []  # For parquet (needs full rewrite)
        self._header_written = False  # For CSV

        # On init, check for existing rows (resume support)
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
        """Append a single row to disk immediately."""
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
        """Append multiple rows."""
        for row in rows:
            self.write_row(row)

    def flush(self) -> None:
        """Ensure all data is on disk."""
        pass  # Each write_row already flushes

    def close(self) -> None:
        """Finalize (no-op for streaming writers)."""
        self.flush()

    def row_count(self) -> int:
        """Number of rows written so far."""
        return self._rows_written

    def read_rows(self) -> list[dict[str, Any]]:
        """Read back all rows from disk (for resume support)."""
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


class _InputGenerationSignature(dspy.Signature):
    """Generate realistic input data for a task.

    Return a JSON array of input objects. Each object must have exactly the
    specified input field names as keys. Values should be diverse, realistic,
    and cover edge cases.
    """

    task_description: str = dspy.InputField(desc="Description of the task")
    input_field_names: str = dspy.InputField(
        desc="Comma-separated list of input field names"
    )
    existing_inputs_json: str = dspy.InputField(
        desc="JSON array of already-generated inputs for diversity reference",
        default="[]",
    )
    n_to_generate: int = dspy.InputField(desc="Number of input objects to generate")
    generated_inputs: str = dspy.OutputField(
        desc="JSON array of input objects, each with exactly the specified field names"
    )


class _OutputGenerationSignature(dspy.Signature):
    """Generate correct output values for a given input.

    Return a JSON object with exactly the specified output field names as keys.
    Values should be realistic and correct for the given task.
    """

    task_description: str = dspy.InputField(desc="Description of the task")
    input_data: str = dspy.InputField(desc="JSON object of the input row")
    output_field_names: str = dspy.InputField(
        desc="Comma-separated list of output field names"
    )
    generated_output: str = dspy.OutputField(
        desc="JSON object with exactly the specified output field names"
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

        # data_lm: first-class param, falls back to dspy.settings.lm
        self.data_lm = data_lm or self.config.data_lm or dspy.settings.lm

        # Infer fields from module signature
        self.input_fields, self.output_fields = infer_fields_from_module(module)

        # Strip reasoning from output fields (ChainOfThought injects it)
        self.output_fields = [f for f in self.output_fields if f != "reasoning"]

        # Validate description
        sig = getattr(module, "signature", None)
        if sig is None:
            # Fallback for wrappers like ChainOfThought
            predictors = list(getattr(module, "named_predictors", lambda: [])())
            if predictors:
                sig = getattr(predictors[0][1], "signature", None)

        sig_docstring = getattr(sig, "__doc__", "") or ""
        if not sig_docstring.strip() and not self.description:
            raise ValueError(
                "Module signature has no docstring and no description was provided. "
                "Pass description='...' to describe the task for data generation."
            )

        # Parse schema if provided
        self._schema_fields: dict[str, str] = {}
        if self.schema is not None:
            for name, field_info in self.schema.model_fields.items():
                self._schema_fields[name] = str(field_info.annotation)

        # Seed examples storage
        self.seed_examples: list[dict[str, Any]] | None = None

    def _generate_inputs(
        self,
        n: int,
        seed_examples: list[dict[str, Any]] | None,
        description: str,
    ) -> list[dict[str, Any]]:
        """Pass 1: Generate n diverse input rows using the LLM."""
        predictor = dspy.Predict(_InputGenerationSignature)
        all_inputs: list[dict[str, Any]] = []
        batch_size = min(10, n)
        max_retries = self.config.max_retries
        diversity_checker = (
            DiversityChecker(diversity_threshold=self.config.diversity_threshold)
            if self.config.diversity_enabled
            else None
        )

        while len(all_inputs) < n:
            remaining = n - len(all_inputs)
            current_batch = min(batch_size, remaining)

            existing_json = json.dumps(
                [d for d in (seed_examples or [])[:5]] + all_inputs[-5:],
                default=str,
            )

            for attempt in range(max_retries):
                try:
                    with dspy.settings.context(lm=self.data_lm):
                        result = predictor(
                            task_description=description,
                            input_field_names=", ".join(self.input_fields),
                            existing_inputs_json=existing_json,
                            n_to_generate=current_batch,
                        )

                    parsed = json.loads(result.generated_inputs)
                    if not isinstance(parsed, list):
                        parsed = [parsed]

                    validated = []
                    for row in parsed:
                        if isinstance(row, dict):
                            clean_row = {k: row.get(k, "") for k in self.input_fields}
                            validated.append(clean_row)

                    if not validated:
                        continue

                    if diversity_checker and len(all_inputs) + len(validated) > 1:
                        all_texts = [
                            " ".join(str(row.get(k, "")) for k in self.input_fields)
                            for row in all_inputs + validated
                        ]
                        div_result = diversity_checker.check(all_texts)
                        if not div_result.is_diverse and attempt < max_retries - 1:
                            continue

                    all_inputs.extend(validated)
                    break

                except (json.JSONDecodeError, ValueError, TypeError):
                    if attempt == max_retries - 1:
                        pass
                    continue

        return all_inputs[:n]

    def _generate_outputs(
        self,
        inputs: list[dict[str, Any]],
        description: str,
    ) -> list[dict[str, Any]]:
        """Pass 2: Generate output values for each input row using the LLM."""
        predictor = dspy.Predict(_OutputGenerationSignature)
        all_outputs: list[dict[str, Any]] = []
        max_retries = self.config.max_retries

        # Setup quality pipeline components
        judge = None
        if self.config.judge_enabled:
            judge_lm = self.config.judge_lm or self.data_lm
            judge = LLMJudge(lm=judge_lm)

        for inp in inputs:
            for attempt in range(max_retries):
                try:
                    with dspy.settings.context(lm=self.data_lm):
                        result = predictor(
                            task_description=description,
                            input_data=json.dumps(inp, default=str),
                            output_field_names=", ".join(self.output_fields),
                        )

                    parsed = json.loads(result.generated_output)
                    if not isinstance(parsed, dict):
                        continue

                    # Keep only output fields, strip reasoning
                    clean_output = {}
                    for k in self.output_fields:
                        if k == "reasoning":
                            continue
                        clean_output[k] = parsed.get(k, "")

                    # Quality check via judge
                    if judge is not None:
                        full_row = {**inp, **clean_output}
                        judge_result = judge.score(
                            full_row, task_description=description
                        )
                        if judge_result.score < 0.5 and attempt < max_retries - 1:
                            continue  # Retry with better output

                    all_outputs.append(clean_output)
                    break

                except (json.JSONDecodeError, ValueError, TypeError):
                    if attempt == max_retries - 1:
                        # Last attempt failed — use empty output
                        all_outputs.append(
                            {k: "" for k in self.output_fields if k != "reasoning"}
                        )
                    continue

        return all_outputs

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        module: dspy.Module,
        **kwargs: Any,
    ) -> "AutoData":
        """Create AutoData with seed examples from a CSV file."""
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
        """Create AutoData with seed examples from a JSON file."""
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
        """Generate synthetic training data.

        Args:
            n: Number of rows to generate. Defaults to config.n.
            seed_examples: Seed data — list[dict], DataFrame, file path
                (.jsonl, .json, .csv, .parquet), or any object _to_dicts supports.
            force: If False, resume from partial save. If True, regenerate.
            output_path: Output file path. Format auto-detected from extension.
                Defaults to .auto_gepa/generated/rows.jsonl.

        Returns:
            GenerationResult with generated rows.
        """
        n = n or self.config.n
        start_time = time.time()
        random.seed(self.config.seed)

        # Resolve seed examples
        resolved_seeds = self._resolve_seeds(seed_examples)

        # Determine output path
        if output_path is None:
            name = self._name or "unnamed"
            output_path = Path(".auto_gepa") / name / "generated" / "rows.jsonl"
        output_path = Path(output_path)

        # Setup streaming writer
        writer = StreamingDatasetWriter(output_path)

        # Resume support
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
        inputs = self._generate_inputs(n, resolved_seeds, description)

        outputs = self._generate_outputs(inputs, description)

        complete_rows = []
        for inp, out in zip(inputs, outputs):
            complete_rows.append({**inp, **out})

        writer = StreamingDatasetWriter(output_path)
        for row in complete_rows:
            writer.write_row(row)

        quality_scores = None
        if self.config.judge_enabled:
            judge = LLMJudge(lm=self.config.judge_lm or self.data_lm)
            quality_scores = []
            for row in complete_rows:
                qr = judge.score(row, task_description=description)
                quality_scores.append(qr.score)

        elapsed = time.time() - start_time
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
