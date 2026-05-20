from pathlib import Path
from typing import Any

import dspy

from .config import AutoGEPAConfig
from .data import split_examples, to_examples
from .metric_builder import generate_metric_file


class AutoGEPA:
    def __init__(self, config: AutoGEPAConfig):
        self.config = config
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)

    def scaffold(
        self,
        *,
        rows: list[dict[str, Any]],
        module: dspy.Module,
    ) -> dict[str, Any]:
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

        metric_path = self.config.artifact_dir / "metric.py"

        generate_metric_file(
            input_fields=self.config.input_fields,
            output_fields=self.config.output_fields,
            sample_rows=rows,
            module=module,
            out_path=metric_path,
        )

        return {
            "train": train,
            "val": val or test,
            "test": test,
            "metric_path": metric_path,
        }

    def run_baseline(
        self,
        *,
        module: dspy.Module,
        testset: list[dspy.Example],
        metric: Any,
    ) -> dict[str, Any]:
        evaluator = dspy.Evaluate(
            devset=testset,
            metric=metric,
            num_threads=self.config.num_threads,
            display_progress=True,
            display_table=True,
        )
        result = evaluator(module)
        return {"score": result}

    def train(
        self,
        *,
        module: dspy.Module,
        trainset: list[dspy.Example],
        valset: list[dspy.Example],
        metric: Any,
    ) -> dspy.Module:
        optimizer = dspy.GEPA(
            metric=metric,
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
            trainset=trainset,
            valset=valset,
        )

        optimized.save(str(self.config.artifact_dir / "optimized_program.json"))
        return optimized

    def compare(
        self,
        *,
        baseline_module: dspy.Module,
        optimized_module: dspy.Module,
        testset: list[dspy.Example],
        metric: Any,
    ) -> dict[str, Any]:
        baseline = self.run_baseline(
            module=baseline_module,
            testset=testset,
            metric=metric,
        )
        optimized = self.run_baseline(
            module=optimized_module,
            testset=testset,
            metric=metric,
        )
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
