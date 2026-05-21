from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys


def coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        output = coerce_output(exc.stdout) + coerce_output(exc.stderr)
        raise AssertionError("Command timed out. Output:\n" + output) from exc


def assert_python_import() -> None:
    result = run_command(
        [
            sys.executable,
            "-c",
            "import dspy_auto_gepa, importlib.metadata as m; "
            "print(dspy_auto_gepa.__version__); "
            "\ntry:\n print(m.version('dspy-auto-gepa'))\n"
            "except m.PackageNotFoundError:\n print('missing')",
        ]
    )
    output = coerce_output(result.stdout) + coerce_output(result.stderr)
    if result.returncode != 0:
        raise AssertionError(
            "Python import failed with exit code "
            f"{result.returncode}. Output:\n{output}"
        )

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 1:
        raise AssertionError("Expected version output lines. Output:\n" + output)

    version_pattern = re.compile(r"\d+\.\d+\.\d+")
    if version_pattern.search(lines[0]) is None:
        raise AssertionError("Invalid __version__. Output:\n" + output)
    if len(lines) > 1 and lines[1] != "missing":
        if version_pattern.search(lines[1]) is None:
            raise AssertionError("Invalid metadata version. Output:\n" + output)


def assert_public_api() -> None:
    import dspy_auto_gepa

    for name in [
        "AutoGEPA",
        "AutoGEPAConfig",
        "Datasets",
        "load_metric",
        "save_results",
    ]:
        if not hasattr(dspy_auto_gepa, name):
            raise AssertionError(f"Missing public API: {name}")

    try:
        importlib.metadata.version("dspy-auto-gepa")
    except importlib.metadata.PackageNotFoundError:
        pass


def test_smoke() -> None:
    assert_public_api()
    assert_python_import()
