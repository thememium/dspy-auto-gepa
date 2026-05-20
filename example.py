"""Example usage of dspy-auto-gepa with a ticket classification task."""

import warnings

import dspy

from dspy_auto_gepa import AutoGEPA

warnings.filterwarnings("ignore", module="litellm")
warnings.filterwarnings("ignore", module="dspy")


class TicketSignature(dspy.Signature):
    """Classify support tickets."""

    message: str = dspy.InputField()
    urgency: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()


program = dspy.ChainOfThought(TicketSignature)

# fmt: off
rows = [
    {"message": "The server room AC is out and equipment is overheating.", "urgency": "high", "sentiment": "negative"},
    {"message": "Can someone clean conference room B next week?", "urgency": "low", "sentiment": "neutral"},
]
# fmt: on


def main() -> None:
    metric_lm = dspy.LM(
        "openrouter/openai/gpt-oss-120b",
        extra_body={"provider": {"order": ["groq"], "allow_fallbacks": False}},
        cache=False,
    )

    teacher_lm = dspy.LM("openrouter/moonshotai/kimi-k2.5", cache=False)
    dspy.configure(lm=metric_lm)

    auto = AutoGEPA(
        input_fields=["message"],
        output_fields=["urgency", "sentiment"],
        metric_lm=teacher_lm,
        reflection_lm=teacher_lm,
    )

    prepared = auto.prepare(rows=rows, module=program, name="TicketSignature")

    baseline = auto.run_baseline(module=program, prepared=prepared)
    print(f"Baseline score: {baseline['score']}")

    optimized = auto.train(module=program, prepared=prepared)

    final = auto.run_baseline(module=optimized, prepared=prepared)
    print(f"Optimized score: {final['score']}")

    comparison = auto.compare(
        baseline_module=program,
        optimized_module=optimized,
        prepared=prepared,
    )
    print(f"Improvement: {comparison['improvement']:.4f}")

    auto.promote(
        optimized_module=optimized,
        destination=prepared.run_dir / "optimized_program.json",
    )
    print(f"Saved optimized program to {prepared.run_dir / 'optimized_program.json'}")


if __name__ == "__main__":
    main()
