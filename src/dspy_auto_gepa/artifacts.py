import importlib.util
from pathlib import Path
from typing import Any, Callable

import dspy


def load_metric(metric_path: Path) -> Callable[..., float | dspy.Prediction]:
    spec = importlib.util.spec_from_file_location("auto_gepa_metric", metric_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metric from {metric_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "metric"):
        raise ValueError("Metric file must define a top-level metric(...) function")

    return mod.metric


def save_results(
    *,
    artifact_dir: Path,
    baseline_scores: dict[str, Any] | None = None,
    optimized_scores: dict[str, Any] | None = None,
    optimized_program_path: Path | None = None,
    metric_path: Path | None = None,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_path = artifact_dir / "results.json"

    import json

    payload = {
        "baseline": baseline_scores,
        "optimized": optimized_scores,
        "optimized_program": str(optimized_program_path)
        if optimized_program_path
        else None,
        "metric": str(metric_path) if metric_path else None,
    }
    results_path.write_text(json.dumps(payload, indent=2, default=str))
    return results_path
