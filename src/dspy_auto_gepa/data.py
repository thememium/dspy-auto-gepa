import random
from typing import Any

import dspy


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
