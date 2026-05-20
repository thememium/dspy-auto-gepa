from pathlib import Path
from typing import Any

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
    def __init__(self, config: AutoGEPAConfig):
        self.config = config
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
            reflection_lm=dspy.LM(
                self.config.reflection_model,
                temperature=1.0,
                max_tokens=32000,
            ),
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
