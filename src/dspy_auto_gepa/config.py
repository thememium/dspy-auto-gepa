from __future__ import annotations

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
    metric_generator_verbose: bool = True

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


@dataclass
class AutoDataConfig:
    """Configuration for the automatic data generation pipeline.

    Attributes
    ----------
    generation_mode :
        How to generate data rows.

        * ``"split"`` — two-phase: generate inputs first, then outputs
          separately.  Works well for simple classification-style tasks.
        * ``"signature"`` — single-phase: generate complete rows
          (inputs + outputs) in one shot using the module's actual
          Signature.  Better for complex, tightly-coupled outputs such as
          ReAct reasoning traces or RLM explorations.
    """

    n: int = 100
    seed: int = 42
    max_retries: int = 8
    num_threads: int = 16
    chunk_size: int = 10
    diversity_threshold: float = 0.3
    judge_enabled: bool = True
    validators_enabled: bool = True
    diversity_enabled: bool = True
    rejection_sampling_enabled: bool = True
    data_lm: dspy.LM | None = None
    judge_lm: dspy.LM | None = None
    balance_outputs: bool = True
    balance_tolerance: float = 0.15
    oversample_factor: float = 2.0
    generation_mode: Literal["split", "signature"] = "split"
    diversity_categories: str = ""
    seed_examples: list[dict[str, Any]] | None = None
    output_path: str | Path | None = None
    force: bool = False

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not (0.0 <= self.diversity_threshold <= 1.0):
            raise ValueError("diversity_threshold must be between 0.0 and 1.0")
        if self.generation_mode not in ("split", "signature"):
            raise ValueError("generation_mode must be one of: 'split', 'signature'")
