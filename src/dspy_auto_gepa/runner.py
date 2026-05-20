from pathlib import Path
from typing import Any, Literal

import dspy

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


class AutoGEPA:
    def __init__(
        self,
        *,
        input_fields: list[str],
        output_fields: list[str],
        name: str | None = None,
        split: tuple[float, ...] = (0.7, 0.2, 0.1),
        seed: int = 42,
        artifact_dir: Path | str = ".auto_gepa",
        metric_lm: dspy.LM | None = None,
        reflection_lm: dspy.LM | None = None,
        gepa_auto: Literal["light", "medium", "heavy"] = "light",
        num_threads: int = 16,
    ):
        self.name = name
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

    def _run_dir(self, name: str) -> Path:
        run_dir = self.config.artifact_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def prepare(
        self,
        *,
        rows: list[dict[str, Any]],
        module: dspy.Module,
        name: str | None = None,
        force: bool = False,
    ) -> PreparedRun:
        task_name = name or module.__class__.__name__
        run_dir = self._run_dir(task_name)
        self._current_run_dir = run_dir

        examples = to_examples(
            rows,
            self.config.input_fields,
            self.config.output_fields,
        )

        train, val, test = split_examples(
            examples,
            self.config.split,
            self.config.seed,
        )

        metric_file = run_dir / "metric.py"

        if metric_file.exists() and not force:
            pass
        else:
            generate_metric_file(
                input_fields=self.config.input_fields,
                output_fields=self.config.output_fields,
                sample_rows=rows,
                module=module,
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
        module: dspy.Module,
        prepared: PreparedRun,
    ) -> dict[str, Any]:
        evaluator = dspy.Evaluate(
            devset=prepared.test,
            metric=prepared.metric(),
            num_threads=self.config.num_threads,
            display_progress=True,
            display_table=True,
        )
        result = evaluator(module)
        return {"score": result.score}

    def train(
        self,
        *,
        module: dspy.Module,
        prepared: PreparedRun,
    ) -> dspy.Module:
        optimizer = dspy.GEPA(
            metric=prepared.metric(),
            auto=self.config.gepa_auto,
            reflection_lm=self.config.reflection_lm,
            num_threads=self.config.num_threads,
            track_stats=True,
            log_dir=str(self.config.artifact_dir / "gepa_logs"),
        )

        optimized = optimizer.compile(
            module,
            trainset=prepared.train,
            valset=prepared.val,
        )

        return optimized

    def compare(
        self,
        *,
        baseline_module: dspy.Module,
        optimized_module: dspy.Module,
        prepared: PreparedRun,
    ) -> dict[str, Any]:
        baseline = self.run_baseline(module=baseline_module, prepared=prepared)
        optimized = self.run_baseline(module=optimized_module, prepared=prepared)
        return {
            "baseline": baseline["score"],
            "optimized": optimized["score"],
            "improvement": optimized["score"] - baseline["score"],
        }

    def run(
        self,
        *,
        rows: list[dict[str, Any]],
        module: dspy.Module,
        name: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        task_name = name or self.name or module.__class__.__name__
        model_path = (
            self.config.artifact_dir / task_name / f"optimized_{task_name}.json"
        )

        if model_path.exists() and not force:
            module.load(str(model_path))
            return {
                "baseline": None,
                "optimized": None,
                "improvement": None,
                "loaded_from": str(model_path),
            }

        prepared = self.prepare(
            rows=rows,
            module=module,
            name=task_name,
            force=force,
        )

        optimized = self.train(module=module, prepared=prepared)
        comparison = self.compare(
            baseline_module=module,
            optimized_module=optimized,
            prepared=prepared,
        )

        self.promote(
            optimized_module=optimized,
            destination=model_path,
        )

        return {
            "baseline": comparison["baseline"],
            "optimized": comparison["optimized"],
            "improvement": comparison["improvement"],
            "saved_to": str(model_path),
        }

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
