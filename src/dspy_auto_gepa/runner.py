from pathlib import Path
from typing import Any, Literal

import dspy
from pydantic import BaseModel

from .artifacts import load_metric
from .config import AutoGEPAConfig
from .data import split_examples, to_examples
from .metric_builder import generate_metric_file


class PreparedRun:
    def __init__(
        self,
        train: list[dspy.Example],
        val: list[dspy.Example],
        test: list[dspy.Example],
        metric_file: Path,
        run_dir: Path,
    ):
        self.train = train
        self.val = val
        self.test = test
        self.metric_file = metric_file
        self.run_dir = run_dir

    def metric(self) -> Any:
        return load_metric(self.metric_file)

    def __repr__(self) -> str:
        return (
            f"PreparedRun(train={len(self.train)}, val={len(self.val)}, "
            f"test={len(self.test)}, run_dir={self.run_dir})"
        )


class RunResult(BaseModel):
    """Result returned by AutoGEPA.run()."""

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
        input_fields: list[str],
        output_fields: list[str],
        rows: list[dict[str, Any]] | None = None,
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
    ):
        self.rows = rows
        self.module = module
        self.name = name
        self.metric = Path(metric) if metric is not None else None
        self.config = AutoGEPAConfig(
            input_fields=input_fields,
            output_fields=output_fields,
            split=split,
            seed=seed,
            artifact_dir=Path(artifact_dir),
            metric_lm=metric_lm,
            reflection_lm=reflection_lm,
            gepa_auto=gepa_auto,
            num_threads=num_threads,
        )
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._current_run_dir: Path | None = None

    def _resolve_task(
        self,
        rows: list[dict[str, Any]] | None = None,
        module: dspy.Module | None = None,
        name: str | None = None,
        metric: Path | str | None = None,
    ) -> tuple[list[dict[str, Any]], dspy.Module, str, Path | None]:
        resolved_rows = rows if rows is not None else self.rows
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

        if resolved_rows is None:
            raise ValueError(
                "rows must be provided either to the constructor or to the method"
            )
        if resolved_module is None:
            raise ValueError(
                "module must be provided either to the constructor or to the method"
            )

        return resolved_rows, resolved_module, resolved_name, resolved_metric

    def _run_dir(self, name: str) -> Path:
        run_dir = self.config.artifact_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def prepare(
        self,
        *,
        rows: list[dict[str, Any]] | None = None,
        module: dspy.Module | None = None,
        name: str | None = None,
        metric: Path | str | None = None,
        force: bool = False,
    ) -> PreparedRun:
        task_rows, task_module, task_name, task_metric = self._resolve_task(
            rows, module, name, metric
        )
        run_dir = self._run_dir(task_name)
        self._current_run_dir = run_dir

        examples = to_examples(
            task_rows,
            self.config.input_fields,
            self.config.output_fields,
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
                    input_fields=self.config.input_fields,
                    output_fields=self.config.output_fields,
                    sample_rows=task_rows,
                    module=task_module,
                    out_path=metric_file,
                    metric_lm=self.config.metric_lm,
                )

        return PreparedRun(
            train=train,
            val=val or test,
            test=test,
            metric_file=metric_file,
            run_dir=run_dir,
        )

    def run_baseline(
        self,
        *,
        module: dspy.Module | None = None,
        prepared: PreparedRun,
    ) -> dict[str, Any]:
        task_module = module if module is not None else self.module
        if task_module is None:
            raise ValueError(
                "module must be provided either to the constructor or to run_baseline()"
            )

        evaluator = dspy.Evaluate(
            devset=prepared.test,
            metric=prepared.metric(),
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
        prepared: PreparedRun,
    ) -> dspy.Module:
        task_module = module if module is not None else self.module
        if task_module is None:
            raise ValueError(
                "module must be provided either to the constructor or to train()"
            )

        optimizer = dspy.GEPA(
            metric=prepared.metric(),
            auto=self.config.gepa_auto,
            reflection_lm=self.config.reflection_lm,
            num_threads=self.config.num_threads,
            track_stats=True,
            log_dir=str(self.config.artifact_dir / "gepa_logs"),
        )

        optimized = optimizer.compile(
            task_module,
            trainset=prepared.train,
            valset=prepared.val,
        )

        return optimized

    def compare(
        self,
        *,
        baseline_module: dspy.Module | None = None,
        optimized_module: dspy.Module,
        prepared: PreparedRun,
    ) -> RunResult:
        baseline = self.run_baseline(
            module=baseline_module,
            prepared=prepared,
        )
        optimized = self.run_baseline(
            module=optimized_module,
            prepared=prepared,
        )
        return RunResult(
            baseline=baseline["score"],
            optimized=optimized["score"],
            improvement=optimized["score"] - baseline["score"],
        )

    def run(
        self,
        *,
        rows: list[dict[str, Any]] | None = None,
        module: dspy.Module | None = None,
        name: str | None = None,
        metric: Path | str | None = None,
        force: bool = False,
    ) -> RunResult:
        task_rows, task_module, task_name, task_metric = self._resolve_task(
            rows, module, name, metric
        )
        model_path = (
            self.config.artifact_dir / task_name / f"optimized_{task_name}.json"
        )

        if model_path.exists() and not force:
            task_module.load(str(model_path))
            return RunResult(loaded_from=str(model_path))

        prepared = self.prepare(
            rows=task_rows,
            module=task_module,
            name=task_name,
            metric=task_metric,
            force=force,
        )

        optimized = self.train(module=task_module, prepared=prepared)
        comparison = self.compare(
            baseline_module=task_module,
            optimized_module=optimized,
            prepared=prepared,
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
