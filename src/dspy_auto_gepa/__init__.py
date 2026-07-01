from .artifacts import load_metric, save_results
from .config import AutoGEPAConfig
from .data import apply_mapping, infer_fields_from_module, resolve_fields
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
    "apply_mapping",
    "infer_fields_from_module",
    "load_metric",
    "resolve_fields",
    "save_results",
]
