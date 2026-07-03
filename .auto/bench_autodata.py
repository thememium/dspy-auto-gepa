from __future__ import annotations

import json
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch

import dspy

from dspy_auto_gepa.config import AutoDataConfig
from dspy_auto_gepa.generator import AutoData
from dspy_auto_gepa.quality import DiversityChecker


class TicketSignature(dspy.Signature):
    """Classify support tickets."""

    message: str = dspy.InputField()
    urgency: str = dspy.OutputField()
    sentiment: str = dspy.OutputField()


class DummyModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.signature = TicketSignature

    def forward(self, message: str) -> dspy.Prediction:
        return dspy.Prediction(urgency="low", sentiment="neutral")


SAFE_CONCURRENCY = 4
REQUEST_DELAY_S = 0.002


class BenchState:
    def __init__(self) -> None:
        self.input_counter = 0
        self.signature_counter = 0
        self.parallel_calls = 0
        self.total_requests = 0
        self.rate_limit_failures = 0


class FakePredict:
    def __init__(self, signature: Any, state: BenchState) -> None:
        self.signature = signature
        self.state = state
        self.output_field_names = {"generated_inputs", "generated_output", "generated_outputs", "generated_rows"}

    def __call__(self, **kwargs: Any) -> dspy.Prediction:
        time.sleep(REQUEST_DELAY_S)
        fields = set(getattr(self.signature, "fields", {}).keys())
        if "generated_inputs" in fields:
            n = int(kwargs["n_to_generate"])
            rows = []
            for _ in range(n):
                idx = self.state.input_counter
                self.state.input_counter += 1
                rows.append({"message": f"ticket-{idx:04d} topic-{idx % 23} severity-{idx % 5}"})
            return dspy.Prediction(generated_inputs=json.dumps(rows))

        if "generated_rows" in fields:
            n = int(kwargs["n_to_generate"])
            rows = []
            for _ in range(n):
                idx = self.state.signature_counter
                self.state.signature_counter += 1
                rows.append(
                    {
                        "message": f"sig-ticket-{idx:04d} topic-{idx % 29} user-{idx % 7}",
                        "urgency": ["low", "medium", "high"][idx % 3],
                        "sentiment": ["neutral", "negative"][idx % 2],
                    }
                )
            return dspy.Prediction(generated_rows=json.dumps(rows))

        if "generated_outputs" in fields:
            inputs = json.loads(kwargs["inputs_json"])
            outputs = []
            for row in inputs:
                msg = row["message"]
                idx = int(msg.split("ticket-")[-1].split()[0])
                outputs.append(
                    {
                        "urgency": ["low", "medium", "high"][idx % 3],
                        "sentiment": ["neutral", "negative"][idx % 2],
                    }
                )
            return dspy.Prediction(generated_outputs=outputs)

        if "generated_output" in fields:
            row = json.loads(kwargs["input_data"])
            idx = int(row["message"].split("ticket-")[-1].split()[0])
            return dspy.Prediction(
                generated_output=json.dumps(
                    {
                        "urgency": ["low", "medium", "high"][idx % 3],
                        "sentiment": ["neutral", "negative"][idx % 2],
                    }
                )
            )

        if "scores_json" in fields:
            return dspy.Prediction(
                scores_json='{"correctness": 0.9, "relevance": 0.88, "coherence": 0.91}',
                feedback="good",
            )

        raise RuntimeError(f"Unexpected signature fields: {fields}")


class FakeParallel:
    def __init__(self, state: BenchState, **_: Any) -> None:
        self.state = state

    def __call__(self, tasks: list[tuple[Any, Any]]) -> list[Any]:
        self.state.parallel_calls += 1
        self.state.total_requests += len(tasks)
        results: list[Any] = []
        for i, (module, example) in enumerate(tasks):
            if i >= SAFE_CONCURRENCY:
                self.state.rate_limit_failures += 1
                results.append(RuntimeError("429 rate limit"))
                continue
            try:
                payload = {k: example[k] for k in example.keys()}
                results.append(module(**payload))
            except Exception as exc:  # pragma: no cover
                results.append(exc)
        return results


def _balance_error_pct(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 100.0
    counts = Counter((r["urgency"], r["sentiment"]) for r in rows)
    target = len(rows) / max(len(counts), 1)
    error = sum(abs(v - target) for v in counts.values()) / len(rows)
    return error * 100.0


def _diversity_score(rows: list[dict[str, Any]]) -> float:
    texts = [str(r.get("message", "")) for r in rows]
    result = DiversityChecker(diversity_threshold=0.3).check(texts)
    return max(0.0, 1.0 - result.avg_similarity)


def _run_split(tmp: Path) -> tuple[float, dict[str, float]]:
    state = BenchState()
    with (
        patch("dspy_auto_gepa.generator.dspy.Predict", side_effect=lambda sig: FakePredict(sig, state)),
        patch("dspy_auto_gepa.generator.dspy.Parallel", side_effect=lambda **kw: FakeParallel(state, **kw)),
    ):
        gen = AutoData(
            module=DummyModule(),
            data_lm=object(),
            description="Classify support tickets",
            config=AutoDataConfig(
                n=72,
                max_retries=4,
                num_threads=16,
                chunk_size=8,
                judge_enabled=False,
                output_path=tmp / "split.jsonl",
                force=True,
                oversample_factor=2.0,
                balance_outputs=True,
                generation_mode="split",
            ),
        )
        t0 = time.perf_counter()
        result = gen.generate()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    metrics = {
        "produced_pct": 100.0 * result.n_produced / result.n_requested,
        "diversity_score": _diversity_score(result.rows),
        "balance_error_pct": _balance_error_pct(result.rows),
        "rate_limit_failures": float(state.rate_limit_failures),
    }
    return elapsed_ms, metrics


def _run_signature(tmp: Path) -> tuple[float, dict[str, float]]:
    state = BenchState()
    with (
        patch("dspy_auto_gepa.generator.dspy.Predict", side_effect=lambda sig: FakePredict(sig, state)),
        patch("dspy_auto_gepa.generator.dspy.Parallel", side_effect=lambda **kw: FakeParallel(state, **kw)),
    ):
        gen = AutoData(
            module=DummyModule(),
            data_lm=object(),
            description="Classify support tickets",
            config=AutoDataConfig(
                n=48,
                max_retries=4,
                num_threads=16,
                chunk_size=8,
                judge_enabled=False,
                output_path=tmp / "signature.jsonl",
                force=True,
                generation_mode="signature",
                diversity_categories="outage, billing, account, bug, feature request, scheduling",
            ),
        )
        t0 = time.perf_counter()
        result = gen.generate()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    metrics = {
        "produced_pct": 100.0 * result.n_produced / result.n_requested,
        "diversity_score": _diversity_score(result.rows),
        "balance_error_pct": _balance_error_pct(result.rows),
        "rate_limit_failures": float(state.rate_limit_failures),
    }
    return elapsed_ms, metrics


def main() -> None:
    split_runs: list[float] = []
    sig_runs: list[float] = []
    produced: list[float] = []
    diversity: list[float] = []
    balance_error: list[float] = []
    rate_limit_failures: list[float] = []

    for _ in range(3):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            split_ms, split_metrics = _run_split(tmp)
            sig_ms, sig_metrics = _run_signature(tmp)

        split_runs.append(split_ms)
        sig_runs.append(sig_ms)
        produced.append((split_metrics["produced_pct"] + sig_metrics["produced_pct"]) / 2.0)
        diversity.append((split_metrics["diversity_score"] + sig_metrics["diversity_score"]) / 2.0)
        balance_error.append((split_metrics["balance_error_pct"] + sig_metrics["balance_error_pct"]) / 2.0)
        rate_limit_failures.append(split_metrics["rate_limit_failures"] + sig_metrics["rate_limit_failures"])

    split_ms = statistics.median(split_runs)
    sig_ms = statistics.median(sig_runs)
    total_ms = split_ms + sig_ms

    print(f"METRIC total_ms={total_ms:.3f}")
    print(f"METRIC split_ms={split_ms:.3f}")
    print(f"METRIC signature_ms={sig_ms:.3f}")
    print(f"METRIC produced_pct={statistics.median(produced):.3f}")
    print(f"METRIC diversity_score={statistics.median(diversity):.6f}")
    print(f"METRIC balance_error_pct={statistics.median(balance_error):.3f}")
    print(f"METRIC rate_limit_failures={statistics.median(rate_limit_failures):.3f}")


if __name__ == "__main__":
    main()
