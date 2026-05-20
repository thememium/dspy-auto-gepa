from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class AutoGEPAConfig:
    input_fields: list[str]
    output_fields: list[str]
    split: tuple[float, ...] = (0.7, 0.2, 0.1)
    seed: int = 42
    artifact_dir: Path = field(default_factory=lambda: Path(".auto_gepa"))
    metric_model: str = "openrouter/openai/gpt-oss-120b"
    reflection_model: str = "openrouter/moonshotai/kimi-k2.5"
    gepa_auto: Literal["light", "medium", "heavy"] = "light"
    num_threads: int = 16

    def __post_init__(self) -> None:
        if not self.input_fields or not self.output_fields:
            raise ValueError("input_fields and output_fields must be non-empty")
        if sum(self.split) > 1.0:
            raise ValueError("split proportions must sum to <= 1.0")
        if self.gepa_auto not in ("light", "medium", "heavy"):
            raise ValueError("gepa_auto must be one of: light, medium, heavy")
