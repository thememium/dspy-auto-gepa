import random
from typing import Any

import dspy


def _to_dicts(obj: Any) -> list[dict[str, Any]]:
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
        "or any object with .to_dicts() or .to_pandas()."
    )


def _infer_fields_from_module(module: dspy.Module) -> tuple[list[str], list[str]]:
    sig = getattr(module, "signature", None)
    if sig is None:
        raise ValueError(
            "Cannot infer fields: module has no .signature attribute. "
            "Pass input_fields and output_fields explicitly, "
            "or provide a module with a DSPy Signature."
        )

    fields = getattr(sig, "fields", None)
    if fields is None:
        raise ValueError(
            "Cannot infer fields: module.signature has no .fields attribute. "
            "Pass input_fields and output_fields explicitly."
        )

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


def _resolve_fields(
    module: dspy.Module,
    row_keys: set[str],
    input_fields: list[str] | dict[str, str] | None,
    output_fields: list[str] | dict[str, str] | None,
) -> tuple[list[str], list[str], dict[str, str]]:
    sig_in, sig_out = _infer_fields_from_module(module)
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


def _apply_mapping(
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
