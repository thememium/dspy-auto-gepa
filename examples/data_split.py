"""Example usage of AutoData for synthetic dataset generation — Split Mode."""

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
# Classification (split mode — default)
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Split mode (classification)")
    print("=" * 60)
    example_split_mode()


if __name__ == "__main__":
    main()
