from .artifacts import load_metric, save_results
from .config import AutoDataConfig, AutoGEPAConfig
from .data import (
    FieldMetadata,
    SignatureMetadata,
    apply_mapping,
    extract_signature_metadata,
    infer_fields_from_module,
    resolve_fields,
)
from .generator import AutoData
from .quality import (
    Validator,
    enum_validator,
    no_emoji_validator,
    non_empty_validator,
    sanitize_string,
)
from .runner import AutoGEPA, Datasets, GenerationFailed, GenerationResult

try:
    from importlib.metadata import version

    __version__ = version("dspy-auto-gepa")
except ImportError:
    __version__ = "unknown"

__all__ = [
    "AutoData",
    "AutoDataConfig",
    "AutoGEPA",
    "AutoGEPAConfig",
    "Datasets",
    "FieldMetadata",
    "GenerationFailed",
    "GenerationResult",
    "SignatureMetadata",
    "Validator",
    "apply_mapping",
    "enum_validator",
    "extract_signature_metadata",
    "infer_fields_from_module",
    "load_metric",
    "no_emoji_validator",
    "non_empty_validator",
    "resolve_fields",
    "sanitize_string",
    "save_results",
]
