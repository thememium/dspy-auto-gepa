from .artifacts import load_metric, save_results
from .config import AutoGEPAConfig
from .runner import AutoGEPA, Datasets

try:
    from importlib.metadata import version

    __version__ = version("dspy-auto-gepa")
except ImportError:
    __version__ = "unknown"

__all__ = [
    "AutoGEPA",
    "AutoGEPAConfig",
    "Datasets",
    "load_metric",
    "save_results",
]
