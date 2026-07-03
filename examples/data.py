"""Example usage of AutoData for synthetic dataset generation."""

import dspy

from dspy_auto_gepa import AutoData, AutoDataConfig

# ---------------------------------------------------------------------------
# Configure models
# ---------------------------------------------------------------------------

data_lm = dspy.LM(
    model="openrouter/openai/gpt-oss-120b:nitro",
    cache=False,
)

dspy.configure(lm=data_lm)


# ---------------------------------------------------------------------------
# Example 1: Classification (split mode — default)
# ---------------------------------------------------------------------------


class TicketSignature(dspy.Signature):
    """Classify support tickets by urgency and sentiment."""

    message: str = dspy.InputField()
    urgency: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()


program = dspy.ChainOfThought(TicketSignature)


# Seed examples (optional — omit for generation from scratch)

# fmt: off
seed_rows = [
    {"message": "The server room AC is out and equipment is overheating.", "urgency": "high", "sentiment": "negative"},
    {"message": "Can someone clean conference room B next week?", "urgency": "low", "sentiment": "neutral"},
    {"message": "Thanks for fixing the VPN, works perfectly now!", "urgency": "medium", "sentiment": "positive"},
]
# fmt: on


def example_split_mode() -> None:
    """Generate data using the default split mode (inputs first, then outputs)."""
    name = "TicketSignature-split"

    config = AutoDataConfig(
        n=100,
        generation_mode="split",
        seed_examples=seed_rows,
        output_path=f".auto_gepa/{name}/generated/rows.jsonl",
        force=True,
    )

    gen = AutoData(
        module=program,
        data_lm=data_lm,
        config=config,
        name=name,
    )

    print(f"Input fields:  {gen.input_fields}")
    print(f"Output fields: {gen.output_fields}")

    result = gen.generate()

    print(f"\nGenerated {result.n_produced} of {result.n_requested} rows")
    print(f"Failed:    {result.n_failed}")
    print(f"Time:      {result.generation_time_seconds:.1f}s")

    for i, row in enumerate(result.rows[:3]):
        print(f"\n  Row {i}: {row}")


# ---------------------------------------------------------------------------
# Example 2: ReAct-style (signature mode)
# ---------------------------------------------------------------------------


class ReActSignature(dspy.Signature):
    """Answer the users question"""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


react_program = dspy.ReAct(ReActSignature, tools=[])


def example_signature_mode() -> None:
    """Generate data using signature mode (complete rows in one shot)."""
    name = "ReActSignature-signature"

    config = AutoDataConfig(
        n=50,
        generation_mode="signature",
        diversity_categories=(
            "science, history, geography, mathematics, literature, technology, "
            "sports, music, cooking, travel, nature, philosophy, economics, "
            "medicine, art, politics, psychology, astronomy, biology, physics"
        ),
        seed_examples=[],
        output_path=f".auto_gepa/{name}/generated/rows.jsonl",
        force=True,
    )

    gen = AutoData(
        module=react_program,
        data_lm=data_lm,
        config=config,
        name=name,
    )

    print(f"Input fields:  {gen.input_fields}")
    print(f"Output fields: {gen.output_fields}")

    result = gen.generate()

    print(f"\nGenerated {result.n_produced} of {result.n_requested} rows")
    print(f"Failed:    {result.n_failed}")
    print(f"Time:      {result.generation_time_seconds:.1f}s")

    if result.quality_scores:
        avg = sum(result.quality_scores) / len(result.quality_scores)
        print(f"Avg quality score: {avg:.3f}")

    for i, row in enumerate(result.rows[:3]):
        print(f"\n  Row {i}:")
        print(f"    Q: {row.get('question', '')}")
        print(f"    Answer: {row.get('answer', '')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Example 1: Split mode (classification)")
    print("=" * 60)
    example_split_mode()

    print("\n" + "=" * 60)
    print("Example 2: Signature mode (ReAct-style)")
    print("=" * 60)
    example_signature_mode()


if __name__ == "__main__":
    main()
