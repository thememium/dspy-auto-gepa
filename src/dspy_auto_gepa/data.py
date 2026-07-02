import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dspy


@dataclass
class FieldMetadata:
    name: str
    python_type: type
    description: str
    is_input: bool
    allowed_values: list[str] | None = None


@dataclass
class SignatureMetadata:
    fields: list[FieldMetadata]
    input_fields: list[str] = field(default_factory=list)
    output_fields: list[str] = field(default_factory=list)

    def get(self, name: str) -> FieldMetadata | None:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def to_prompt_spec(self) -> str:
        parts: list[str] = []
        for f in self.fields:
            entry: dict[str, Any] = {"type": f.python_type.__name__}
            if f.description:
                entry["desc"] = f.description
            if f.allowed_values:
                entry["allowed"] = f.allowed_values
            parts.append(f'"{f.name}": {json.dumps(entry)}')
        return "{" + ", ".join(parts) + "}"


def _infer_allowed_values(
    field_name: str,
    seed_examples: list[dict[str, Any]] | None,
    max_cardinality: int = 10,
) -> list[str] | None:
    if not seed_examples:
        return None
    distinct: set[str] = set()
    for row in seed_examples:
        val = row.get(field_name)
        if isinstance(val, str) and val.strip():
            distinct.add(val.strip().lower())
    if 1 < len(distinct) <= max_cardinality:
        return sorted(distinct)
    return None


def extract_signature_metadata(
    sig: type[dspy.Signature],
    seed_examples: list[dict[str, Any]] | None = None,
) -> SignatureMetadata:
    fields_meta: list[FieldMetadata] = []
    input_names: list[str] = []
    output_names: list[str] = []

    raw_fields = getattr(sig, "fields", None)
    if raw_fields is None:
        return SignatureMetadata(fields=[])

    for name, field_info in raw_fields.items():
        json_extra = getattr(field_info, "json_schema_extra", {}) or {}
        field_type_str = json_extra.get("__dspy_field_type")
        is_input = field_type_str == "input" or getattr(field_info, "is_input", False)
        is_output = field_type_str == "output" or getattr(
            field_info, "is_output", False
        )

        if not is_input and not is_output:
            continue

        annotation = getattr(field_info, "annotation", str) or str
        desc = getattr(field_info, "description", "") or ""

        allowed = _infer_allowed_values(name, seed_examples) if not is_input else None

        meta = FieldMetadata(
            name=name,
            python_type=annotation if isinstance(annotation, type) else str,
            description=desc,
            is_input=is_input,
            allowed_values=allowed,
        )
        fields_meta.append(meta)

        if is_input:
            input_names.append(name)
        else:
            output_names.append(name)

    return SignatureMetadata(
        fields=fields_meta,
        input_fields=input_names,
        output_fields=output_names,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    if not isinstance(data, list):
        raise ValueError(
            f"JSON file must contain an array of objects, got {type(data).__name__}"
        )
    return data


def _read_csv(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_csv(path).to_dict(orient="records")


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_parquet(path).to_dict(orient="records")


def _to_dicts(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, (str, Path)):
        path = Path(obj)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return _read_jsonl(path)
        if suffix == ".json":
            return _read_json_array(path)
        if suffix == ".csv":
            return _read_csv(path)
        if suffix in (".parquet", ".pq"):
            return _read_parquet(path)
        raise ValueError(
            f"Unsupported file extension '{suffix}' for dataset file. "
            "Use .jsonl, .json, .csv, .parquet, or .pq"
        )

    if isinstance(obj, list):
        return obj

    if hasattr(obj, "to_dicts"):
        result = obj.to_dicts()
        if isinstance(result, list):
            return result

    if hasattr(obj, "collect"):
        df = obj.collect()
        if hasattr(df, "to_dicts"):
            return df.to_dicts()

    if hasattr(obj, "to_dict"):
        result = obj.to_dict(orient="records")
        if isinstance(result, list):
            return result

    if hasattr(obj, "to_pandas"):
        pd_df = obj.to_pandas()
        result = pd_df.to_dict(orient="records")
        if isinstance(result, list):
            return result

    raise TypeError(
        f"Cannot convert {type(obj).__name__} to list of dicts. "
        "Expected: list[dict], pandas DataFrame, polars DataFrame/LazyFrame, "
        "file path (str/Path to .jsonl, .json, .csv, .parquet), "
        "or any object with .to_dicts() or .to_pandas()."
    )


def _extract_fields_from_signature(
    sig: dspy.Signature,
) -> tuple[list[str], list[str]]:
    fields = getattr(sig, "fields", None)
    if fields is None:
        return [], []

    input_fields = []
    output_fields = []

    for name, field_info in fields.items():
        field_type = getattr(field_info, "json_schema_extra", {}).get(
            "__dspy_field_type"
        )
        is_input = field_type == "input" or getattr(field_info, "is_input", False)
        is_output = field_type == "output" or getattr(field_info, "is_output", False)
        if is_input:
            input_fields.append(name)
        elif is_output:
            output_fields.append(name)

    return input_fields, output_fields


def infer_fields_from_module(module: dspy.Module) -> tuple[list[str], list[str]]:
    sig = getattr(module, "signature", None)

    # Fallback for wrappers like ChainOfThought that store the signature on an
    # inner predictor rather than directly on the module.
    if sig is None:
        predictors = list(getattr(module, "named_predictors", lambda: [])())

        if len(predictors) > 1:
            # Multi-predictor module (e.g. a custom dspy.Module subclass
            # wrapping multiple ChainOfThought predictors). Aggregate
            # inputs/outputs across ALL predictors so that no fields are
            # missed, and strip the auto-generated "reasoning" field that
            # ChainOfThought injects into each sub-predictor's signature.
            all_inputs: set[str] = set()
            all_outputs: set[str] = set()
            for _, predictor in predictors:
                pred_sig = getattr(predictor, "signature", None)
                if pred_sig is None:
                    continue
                pred_in, pred_out = _extract_fields_from_signature(pred_sig)
                if isinstance(predictor, dspy.ChainOfThought):
                    pred_out = [f for f in pred_out if f != "reasoning"]
                all_inputs.update(pred_in)
                all_outputs.update(pred_out)
            if all_inputs or all_outputs:
                return sorted(all_inputs), sorted(all_outputs)

        if predictors:
            sig = getattr(predictors[0][1], "signature", None)

    if sig is None:
        raise ValueError(
            "Cannot infer fields: module has no .signature attribute. "
            "Pass input_fields and output_fields explicitly, "
            "or provide a module with a DSPy Signature."
        )

    input_fields, output_fields = _extract_fields_from_signature(sig)

    if not input_fields and not output_fields:
        raise ValueError(
            "Cannot infer fields: module.signature has no .fields attribute. "
            "Pass input_fields and output_fields explicitly."
        )

    # ChainOfThought prepends an auto-generated "reasoning" output field to the
    # predictor's signature. It is never present in training rows, so strip it
    # so that row validation and example creation work without explicit mappings.
    if (
        getattr(module, "signature", None) is None
        and isinstance(module, dspy.ChainOfThought)
        and "reasoning" in output_fields
    ):
        output_fields = [f for f in output_fields if f != "reasoning"]

    return input_fields, output_fields


def resolve_fields(
    module: dspy.Module,
    row_keys: set[str],
    input_fields: list[str] | dict[str, str] | None,
    output_fields: list[str] | dict[str, str] | None,
) -> tuple[list[str], list[str], dict[str, str]]:
    sig_in, sig_out = infer_fields_from_module(module)
    all_sig = set(sig_in + sig_out)

    mapping: dict[str, str] = {}
    resolved_input: list[str] | None = None
    resolved_output: list[str] | None = None

    if input_fields is None and output_fields is None:
        if not all_sig <= row_keys:
            missing = all_sig - row_keys
            extra = row_keys - all_sig
            msg = (
                f"Row columns do not match module signature fields. "
                f"Missing from rows: {sorted(missing)}. "
            )
            if extra:
                msg += f"Extra in rows: {sorted(extra)}. "
            msg += (
                "Pass input_fields/output_fields to map row columns to "
                "signature fields, or ensure row columns match exactly."
            )
            raise ValueError(msg)
        return sig_in, sig_out, {}

    if isinstance(input_fields, list):
        resolved_input = input_fields
    elif isinstance(input_fields, dict):
        resolved_input = list(input_fields.values())
        mapping.update(input_fields)

    if isinstance(output_fields, list):
        resolved_output = output_fields
    elif isinstance(output_fields, dict):
        resolved_output = list(output_fields.values())
        mapping.update(output_fields)

    if resolved_input is None:
        resolved_input = sig_in
    if resolved_output is None:
        resolved_output = sig_out

    if not resolved_input or not resolved_output:
        raise ValueError("input_fields and output_fields must be non-empty")

    return resolved_input, resolved_output, mapping


def apply_mapping(
    rows: list[dict[str, Any]], mapping: dict[str, str]
) -> list[dict[str, Any]]:
    if not mapping:
        return rows
    result = []
    for row in rows:
        new_row = {}
        for k, v in row.items():
            new_key = mapping.get(k, k)
            new_row[new_key] = v
        result.append(new_row)
    return result


def to_examples(
    rows: list[dict[str, Any]],
    input_fields: list[str],
    output_fields: list[str],
) -> list[dspy.Example]:
    examples = []
    for row in rows:
        payload = {k: row[k] for k in input_fields + output_fields}
        examples.append(dspy.Example(**payload).with_inputs(*input_fields))
    return examples


def split_examples(
    examples: list[dspy.Example],
    split: tuple[float, ...] = (0.7, 0.2, 0.1),
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    items = list(examples)
    random.Random(seed).shuffle(items)

    if len(split) == 2:
        train_pct, test_pct = split
        n_train = int(len(items) * train_pct)
        return items[:n_train], [], items[n_train:]

    train_pct, val_pct, test_pct = split
    n_train = int(len(items) * train_pct)
    n_val = int(len(items) * val_pct)

    return (
        items[:n_train],
        items[n_train : n_train + n_val],
        items[n_train + n_val :],
    )
