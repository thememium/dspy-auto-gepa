"""Combined AutoData + AutoGEPA pipeline: generate synthetic data, then optimize."""

from pathlib import Path

import dspy

from dspy_auto_gepa import AutoData, AutoDataConfig, AutoGEPA

# ---------------------------------------------------------------------------
# Configure models
# ---------------------------------------------------------------------------

# Data generation + metric LM (fast, cheap)
data_lm = dspy.LM(
    model="openrouter/openai/gpt-oss-20b",
    # extra_body={"provider": {"order": ["groq"], "allow_fallbacks": False}},
    cache=False,
)

# Reflection LM (capable, for GEPA optimization)
reflection_lm = dspy.LM(model="openrouter/xiaomi/mimo-v2.5-pro", cache=False)

dspy.configure(lm=data_lm)


# ---------------------------------------------------------------------------
# Program to optimize
# ---------------------------------------------------------------------------


class TicketSignature(dspy.Signature):
    """Classify support tickets by urgency and sentiment."""

    message: str = dspy.InputField()
    urgency: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()


program = dspy.ChainOfThought(TicketSignature)


# Seed examples for data generation (optional — omit for generation from scratch)

# fmt: off
seed_rows = [
    {"message": "The server room AC is out and equipment is overheating.", "urgency": "high", "sentiment": "negative"},
    {"message": "Can someone clean conference room B next week?", "urgency": "low", "sentiment": "neutral"},
    {"message": "Thanks for fixing the VPN, works perfectly now!", "urgency": "medium", "sentiment": "positive"},
]
# fmt: on


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    name = "TicketSignature-auto"

    # --- Step 1: Generate synthetic data with standalone AutoData ---

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

    rows = result.rows

    review = input("\nReview the generated data? (y/N): ")
    if review.strip().lower() == "y":
        print(f"\n--- {config.output_path} ---")
        print(Path(config.output_path).read_text())
        print("--- End of data ---\n")

    proceed = input("Continue with AutoGEPA optimization? (Y/n): ")
    if proceed.strip().lower() == "n":
        print("Aborted.")
        return

    # --- Step 2: AutoGEPA optimization ---

    auto = AutoGEPA(
        name=name,
        rows=rows,
        module=program,
        metric_lm=data_lm,
        reflection_lm=reflection_lm,
    )

    model_path = auto.config.artifact_dir / name / f"optimized_{name}.json"
    metric_path = auto.config.artifact_dir / name / "metric.py"

    force = False
    if model_path.exists():
        response = input(f"Model found at {model_path}. Run GEPA again? (y/N): ")
        force = response.strip().lower() == "y"
        if not force:
            results = auto.run(force=False)
            print(f"Loaded existing model from {results.loaded_from}")
            return

    if not metric_path.exists() or force:
        metric_path = auto.build_metric(
            name=name,
            rows=rows,
            module=program,
            force=force,
        )
        print(f"Metric generated: {metric_path}")
    else:
        print(f"Metric already exists at {metric_path}")

    review = input(f"Review the generated metric at {metric_path}? (y/N): ")
    if review.strip().lower() == "y":
        print(f"\n--- {metric_path} ---")
        print(metric_path.read_text())
        print("--- End of metric ---\n")

    proceed = input("Continue with GEPA training? (Y/n): ")
    if proceed.strip().lower() == "n":
        print("Aborted.")
        return

    results = auto.run(force=force)

    if results.loaded_from:
        print(f"Loaded existing model from {results.loaded_from}")
        return

    print(f"Baseline score: {results.baseline}")
    print(f"Optimized score: {results.optimized}")
    print(f"Improvement: {results.improvement:.4f}")
    print(f"Saved optimized program to {results.saved_to}")


if __name__ == "__main__":
    main()
