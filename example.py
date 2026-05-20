"""Example usage of dspy-auto-gepa with a ticket classification task."""

import dspy

from dspy_auto_gepa import AutoGEPA


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
    {"message": "Thanks for fixing the VPN, works perfectly now!", "urgency": "low", "sentiment": "positive"},
    {"message": "All login credentials expired overnight, nobody can access the staging environment.", "urgency": "high", "sentiment": "negative"},
    {"message": "Would it be possible to get another monitor for my desk?", "urgency": "low", "sentiment": "neutral"},
    {"message": "Build pipeline broken since 3am, deployments are failing across all projects.", "urgency": "high", "sentiment": "negative"},
    {"message": "The new onboarding checklist is really helpful, appreciate the update!", "urgency": "low", "sentiment": "positive"},
    {"message": "Conference room camera autofocus keeps glitching during client calls.", "urgency": "medium", "sentiment": "negative"},
    {"message": "Reminder to renew SSL certs for *.internal.company.com before March 15.", "urgency": "medium", "sentiment": "neutral"},
    {"message": "Shout out to IT for the quick turnaround on my laptop swap last Friday!", "urgency": "low", "sentiment": "positive"},
]
# fmt: on


def main(force: bool = False) -> None:
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

    results = auto.run(rows=rows, module=program, name="TicketSignature", force=force)

    if results.get("loaded_from"):
        print(f"Loaded existing model from {results['loaded_from']}")
        return

    print(f"Baseline score: {results['baseline']}")
    print(f"Optimized score: {results['optimized']}")
    print(f"Improvement: {results['improvement']:.4f}")
    print(f"Saved optimized program to {results['saved_to']}")


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    main(force=force)
