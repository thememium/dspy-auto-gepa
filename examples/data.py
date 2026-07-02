"""Example usage of AutoData for synthetic dataset generation."""

import dspy

from dspy_auto_gepa import AutoData

# ---------------------------------------------------------------------------
# Configure models
# ---------------------------------------------------------------------------

data_lm = dspy.LM(
    model="openrouter/openai/gpt-oss-120b:nitro",
    cache=False,
)

dspy.configure(lm=data_lm)


# ---------------------------------------------------------------------------
# Define the task signature
# ---------------------------------------------------------------------------


class TicketSignature(dspy.Signature):
    """Classify support tickets by urgency and sentiment."""

    message: str = dspy.InputField()
    urgency: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()


program = dspy.ChainOfThought(TicketSignature)


# ---------------------------------------------------------------------------
# Seed examples (optional — omit for generation from scratch)
# ---------------------------------------------------------------------------

# fmt: off
seed_rows = [
    {"message": "The server room AC is out and equipment is overheating.", "urgency": "high", "sentiment": "negative"},
    {"message": "Can someone clean conference room B next week?", "urgency": "low", "sentiment": "neutral"},
    {"message": "Thanks for fixing the VPN, works perfectly now!", "urgency": "medium", "sentiment": "positive"},
]
# fmt: on


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    name = "TicketSignature-data-gen"

    # --- Option A: Generate from seed examples ---
    gen = AutoData(
        module=program,
        data_lm=data_lm,
        name=name,
    )

    print(f"Input fields:  {gen.input_fields}")
    print(f"Output fields: {gen.output_fields}")

    output_path = f".auto_gepa/{name}/generated/rows.jsonl"

    result = gen.generate(
        n=100,
        seed_examples=seed_rows,
        output_path=output_path,
        force=True,
    )

    print(f"\nGenerated {result.n_produced} of {result.n_requested} rows")
    print(f"Failed:    {result.n_failed}")
    print(f"Seed used: {result.seed_used}")
    print(f"Time:      {result.generation_time_seconds:.1f}s")
    print(f"Output:    {output_path}")

    if result.quality_scores:
        avg = sum(result.quality_scores) / len(result.quality_scores)
        print(f"Avg quality score: {avg:.3f}")

    # Print first 3 rows
    for i, row in enumerate(result.rows[:3]):
        print(f"\n  Row {i}: {row}")


if __name__ == "__main__":
    main()
