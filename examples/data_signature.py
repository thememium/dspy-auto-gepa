"""Example usage of AutoData for synthetic dataset generation — Signature Mode."""

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
# ReAct-style (signature mode)
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
    print("Signature mode (ReAct-style)")
    print("=" * 60)
    example_signature_mode()


if __name__ == "__main__":
    main()
