"""Example usage of dspy-auto-gepa with a ticket classification task."""

import logging
import os

# litellm warns about missing botocore on import even when Bedrock/SageMaker are unused.
os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("litellm").setLevel(logging.ERROR)

import dspy  # noqa: E402

from dspy_auto_gepa import AutoGEPA, AutoGEPAConfig, load_metric  # noqa: E402


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
    lm = dspy.LM("openrouter/google/gemini-2.5-flash-lite", cache=False)
    dspy.configure(lm=lm)

    auto = AutoGEPA(
        AutoGEPAConfig(
            input_fields=["message"],
            output_fields=["urgency", "sentiment"],
            split=(0.7, 0.2, 0.1),
            gepa_auto="light",
        )
    )

    prepared = auto.prepare(rows=rows, module=program, name="TicketSignature")
    metric = load_metric(prepared["metric_file"])

    baseline = auto.run_baseline(
        module=program,
        testset=prepared["test"],
        metric=metric,
    )
    print(f"Baseline score: {baseline['score']}")

    optimized = auto.train(
        module=program,
        trainset=prepared["train"],
        valset=prepared["val"],
        metric=metric,
    )

    final = auto.run_baseline(
        module=optimized,
        testset=prepared["test"],
        metric=metric,
    )
    print(f"Optimized score: {final['score']}")

    comparison = auto.compare(
        baseline_module=program,
        optimized_module=optimized,
        testset=prepared["test"],
        metric=metric,
    )
    print(f"Improvement: {comparison['improvement']}")

    auto.promote(
        optimized_module=optimized,
        destination=prepared["run_dir"] / "optimized_ticket_classifier.json",
    )
    print(
        f"Saved optimized program to {prepared['run_dir'] / 'optimized_ticket_classifier.json'}"
    )


if __name__ == "__main__":
    main()
