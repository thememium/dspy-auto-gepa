from pathlib import Path
from typing import Any, Literal

import dspy
from pydantic import BaseModel

from .artifacts import load_metric
from .config import AutoGEPAConfig
from .data import (
    _apply_mapping,
    _resolve_fields,
    _to_dicts,
    split_examples,
    to_examples,
)
from .metric_builder import generate_metric_file


class Datasets:
    def __init__(
        self,
        train: list[dspy.Example],
        val: list[dspy.Example],
        test: list[dspy.Example],
    ):
        self.train = train
        self.val = val
        self.test = test

    def __repr__(self) -> str:
        return (
            f"Datasets(train={len(self.train)}, val={len(self.val)}, "
            f"test={len(self.test)})"
        )


class RunResult(BaseModel):
    baseline: float | None = None
    optimized: float | None = None
    improvement: float | None = None
    saved_to: str | None = None
    loaded_from: str | None = None

    def __repr__(self) -> str:
        if self.loaded_from:
            return f"RunResult(loaded_from={self.loaded_from!r})"
        return (
            f"RunResult(baseline={self.baseline}, optimized={self.optimized}, "
            f"improvement={self.improvement}, saved_to={self.saved_to!r})"
        )


class AutoGEPA:
    def __init__(
        self,
        *,
        input_fields: list[str] | dict[str, str] | None = None,
        output_fields: list[str] | dict[str, str] | None = None,
        rows: Any | None = None,
        module: dspy.Module | None = None,
        name: str | None = None,
        metric: Path | str | None = None,
        split: tuple[float, ...] = (0.7, 0.2, 0.1),
        seed: int = 42,
        artifact_dir: Path | str = ".auto_gepa",
        metric_lm: dspy.LM | None = None,
        reflection_lm: dspy.LM | None = None,
        gepa_auto: Literal["light", "medium", "heavy"] = "light",
        num_threads: int = 16,
        metric_generator_signature: Any = None,
        metric_generator_module: Any = None,
    ):
        self.rows = rows
        self.module = module
        self.name = name
        self.metric = Path(metric) if metric is not None else None
        self.config = AutoGEPAConfig(
            split=split,
            seed=seed,
            artifact_dir=Path(artifact_dir),
            metric_lm=metric_lm,
            reflection_lm=reflection_lm,
            gepa_auto=gepa_auto,
            num_threads=num_threads,
            metric_generator_signature=metric_generator_signature,
            metric_generator_module=metric_generator_module,
        )
        self._raw_input_fields = input_fields
        self._raw_output_fields = output_fields
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir: Path | None = None
        self._metric_file: Path | None = None

    def _ensure_run_dir(self, name: str) -> Path:
        run_dir = self.config.artifact_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        return run_dir

    def _resolve_task(
        self,
        rows: Any | None = None,
        module: dspy.Module | None = None,
        name: str | None = None,
        metric: Path | str | None = None,
    ) -> tuple[list[dict[str, Any]], dspy.Module, str, Path | None]:
        resolved_rows_raw = rows if rows is not None else self.rows
        resolved_module = module if module is not None else self.module
        resolved_name = name or self.name
        if resolved_module is not None:
            resolved_name = resolved_name or resolved_module.__class__.__name__
        resolved_name = resolved_name or "UnknownTask"
        resolved_metric = None
        if metric is not None:
            resolved_metric = Path(metric)
        elif self.metric is not None:
            resolved_metric = self.metric

        if resolved_rows_raw is None:
            raise ValueError(
                "rows must be provided either to the constructor or to the method"
            )
        if resolved_module is None:
            raise ValueError(
                "module must be provided either to the constructor or to the method"
            )

        resolved_rows = _to_dicts(resolved_rows_raw)
        return resolved_rows, resolved_module, resolved_name, resolved_metric

    def _resolve_and_prepare(
        self,
        rows: Any | None = None,
        module: dspy.Module | None = None,
        name: str | None = None,
        metric: Path | str | None = None,
    ) -> tuple[
        list[dict[str, Any]], dspy.Module, str, Path | None, list[str], list[str]
    ]:
        task_rows, task_module, task_name, task_metric = self._resolve_task(
            rows, module, name, metric
        )

        resolved_in, resolved_out, mapping = _resolve_fields(
            task_module,
            set(task_rows[0].keys()) if task_rows else set(),
            self._raw_input_fields,
            self._raw_output_fields,
        )

        if mapping:
            task_rows = _apply_mapping(task_rows, mapping)

        return task_rows, task_module, task_name, task_metric, resolved_in, resolved_out

    def load_metric(self) -> Any:
        if self._metric_file is None:
            raise RuntimeError(
                "No metric file available. Call datasets() or build_metric() first."
            )
        return load_metric(self._metric_file)

    def datasets(
        self,
        *,
        rows: Any | None = None,
        module: dspy.Module | None = None,
        name: str | None = None,
        metric: Path | str | None = None,
        force: bool = False,
    ) -> Datasets:
        task_rows, task_module, task_name, task_metric, input_fields, output_fields = (
            self._resolve_and_prepare(rows, module, name, metric)
        )
        run_dir = self._ensure_run_dir(task_name)

        examples = to_examples(
            task_rows,
            input_fields,
            output_fields,
        )

        train, val, test = split_examples(
            examples,
            self.config.split,
            self.config.seed,
        )

        if task_metric is not None:
            metric_file = task_metric
        else:
            metric_file = run_dir / "metric.py"

            if metric_file.exists() and not force:
                pass
            else:
                generate_metric_file(
                    input_fields=input_fields,
                    output_fields=output_fields,
                    sample_rows=task_rows,
                    module=task_module,
                    out_path=metric_file,
                    metric_lm=self.config.metric_lm,
                    metric_generator_signature=self.config.metric_generator_signature,
                    metric_generator_module=self.config.metric_generator_module,
                )

        self._metric_file = metric_file

        return Datasets(
            train=train,
            val=val or test,
            test=test,
        )

    def build_metric(
        self,
        *,
        rows: Any | None = None,
        module: dspy.Module | None = None,
        name: str | None = None,
        metric: Path | str | None = None,
        out_path: Path | str | None = None,
        force: bool = False,
    ) -> Path:
        task_rows, task_module, task_name, task_metric, input_fields, output_fields = (
            self._resolve_and_prepare(rows, module, name, metric)
        )

        if task_metric is not None:
            self._metric_file = task_metric
            return task_metric

        if out_path is not None:
            metric_file = Path(out_path)
        else:
            run_dir = self._ensure_run_dir(task_name)
            metric_file = run_dir / "metric.py"

        if metric_file.exists() and not force:
            self._metric_file = metric_file
            return metric_file

        generate_metric_file(
            input_fields=input_fields,
            output_fields=output_fields,
            sample_rows=task_rows,
            module=task_module,
            out_path=metric_file,
            metric_lm=self.config.metric_lm,
            metric_generator_signature=self.config.metric_generator_signature,
            metric_generator_module=self.config.metric_generator_module,
        )

        self._metric_file = metric_file
        return metric_file

    def run_baseline(
        self,
        *,
        module: dspy.Module | None = None,
        datasets: Datasets,
    ) -> dict[str, Any]:
        task_module = module if module is not None else self.module
        if task_module is None:
            raise ValueError(
                "module must be provided either to the constructor or to run_baseline()"
            )

        evaluator = dspy.Evaluate(
            devset=datasets.test,
            metric=self.load_metric(),
            num_threads=self.config.num_threads,
            display_progress=True,
            display_table=True,
        )
        result = evaluator(task_module)
        return {"score": result.score}

    def train(
        self,
        *,
        module: dspy.Module | None = None,
        datasets: Datasets,
    ) -> dspy.Module:
        task_module = module if module is not None else self.module
        if task_module is None:
            raise ValueError(
                "module must be provided either to the constructor or to train()"
            )

        log_dir = (
            str(self._run_dir / "gepa_logs")
            if self._run_dir is not None
            else str(self.config.artifact_dir / "gepa_logs")
        )
        optimizer = dspy.GEPA(
            metric=self.load_metric(),
            auto=self.config.gepa_auto,
            reflection_lm=self.config.reflection_lm,
            num_threads=self.config.num_threads,
            track_stats=True,
            log_dir=log_dir,
        )

        optimized = optimizer.compile(
            task_module,
            trainset=datasets.train,
            valset=datasets.val,
        )

        return optimized

    def compare(
        self,
        *,
        baseline_module: dspy.Module | None = None,
        optimized_module: dspy.Module,
        datasets: Datasets,
    ) -> RunResult:
        baseline = self.run_baseline(
            module=baseline_module,
            datasets=datasets,
        )
        optimized = self.run_baseline(
            module=optimized_module,
            datasets=datasets,
        )
        return RunResult(
            baseline=baseline["score"],
            optimized=optimized["score"],
            improvement=optimized["score"] - baseline["score"],
        )

    def run(
        self,
        *,
        rows: Any | None = None,
        module: dspy.Module | None = None,
        name: str | None = None,
        metric: Path | str | None = None,
        force: bool = False,
    ) -> RunResult:
        task_rows, task_module, task_name, task_metric, input_fields, output_fields = (
            self._resolve_and_prepare(rows, module, name, metric)
        )
        model_path = (
            self.config.artifact_dir / task_name / f"optimized_{task_name}.json"
        )

        if model_path.exists() and not force:
            task_module.load(str(model_path))
            return RunResult(loaded_from=str(model_path))

        ds = self.datasets(
            rows=task_rows,
            module=task_module,
            name=task_name,
            metric=task_metric,
            force=force,
        )

        optimized = self.train(module=task_module, datasets=ds)
        comparison = self.compare(
            baseline_module=task_module,
            optimized_module=optimized,
            datasets=ds,
        )

        self.promote(
            optimized_module=optimized,
            destination=model_path,
        )

        return RunResult(
            baseline=comparison.baseline,
            optimized=comparison.optimized,
            improvement=comparison.improvement,
            saved_to=str(model_path),
        )

    def promote(
        self,
        *,
        optimized_module: dspy.Module,
        destination: Path | str,
    ) -> Path:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        optimized_module.save(str(dest))
        return dest
