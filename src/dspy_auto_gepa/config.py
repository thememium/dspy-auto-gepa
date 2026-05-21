from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import dspy

from .metric_builder import MetricSpecGenerator


@dataclass
class AutoGEPAConfig:
    input_fields: list[str] | None = None
    output_fields: list[str] | None = None
    split: tuple[float, ...] = (0.7, 0.2, 0.1)
    seed: int = 42
    artifact_dir: Path = field(default_factory=lambda: Path(".auto_gepa"))
    metric_lm: dspy.LM | None = None
    reflection_lm: dspy.LM | None = None
    gepa_auto: Literal["light", "medium", "heavy"] = "light"
    num_threads: int = 16
    metric_generator_signature: Any = None
    metric_generator_module: Any = None

    def __post_init__(self) -> None:
        if sum(self.split) > 1.0:
            raise ValueError("split proportions must sum to <= 1.0")
        if self.gepa_auto not in ("light", "medium", "heavy"):
            raise ValueError("gepa_auto must be one of: light, medium, heavy")
        if self.metric_lm is None:
            self.metric_lm = dspy.LM("openrouter/openai/gpt-oss-120b")
        if self.reflection_lm is None:
            self.reflection_lm = dspy.LM("openrouter/moonshotai/kimi-k2.5")
        if self.metric_generator_signature is None:
            self.metric_generator_signature = MetricSpecGenerator
        if self.metric_generator_module is None:
            self.metric_generator_module = dspy.RLM
