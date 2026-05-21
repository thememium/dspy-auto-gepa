# AutoGEPA Contributing Guide

Hi! I'm really excited that you are interested in contributing to AutoGEPA. Before submitting your contribution, please make sure to take a moment and read through the following guidelines:

- [Code of Conduct](#code-of-conduct)
- [Issue Reporting Guidelines](#issue-reporting-guidelines)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Development Setup](#development-setup)
- [Scripts](#scripts)
- [Project Structure](#project-structure)

## Code of Conduct

This project and everyone participating in it is governed by a code of conduct. By participating, you are expected to uphold this code.

## Issue Reporting Guidelines

- Use the GitHub issue tracker to create new issues.
- Check if the issue has already been reported before creating a new one.
- Use issue templates when available.
- Provide a clear description of the problem with steps to reproduce.
- Include relevant environment information (Python version, OS, DSPy version, etc.).

## Pull Request Guidelines

### What kinds of Pull Requests are accepted?

- **Bug fixes** that address a clearly identified bug. **"Clearly identified bug"** means the bug has a proper reproduction either from a related open issue, or is included in the PR itself. Avoid submitting PRs that claim to fix something but do not sufficiently explain what is being fixed.

- **New features** that address a clearly explained and widely applicable use case. **"Widely applicable"** means the new feature should provide non-trivial improvements to the majority of the user base. We are cautious about adding new features - if the use case is niche and can be addressed via external implementations, it likely isn't suitable to go into core.

  The feature implementation should also consider the trade-off between the added complexity vs. the benefits gained.

- **Chore**: typos, comment improvements, build config, CI config, etc. For typos and comment changes, try to combine multiple of them into a single PR.

- **Documentation improvements** that clarify existing features or add missing documentation.

- **It should be noted that we discourage contributors from submitting code refactors that are largely stylistic.** Code refactors are only accepted if they improve performance, or come with sufficient explanations on why they objectively improve the code quality.

  The reason is that code readability is subjective. Contributors should respect the established conventions when contributing code.

### Pull Request Checklist

- [Make sure to tick the "Allow edits from maintainers" box](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork). This allows us to directly make minor edits / refactors and saves a lot of time.

- **All PRs must reference an existing issue.** Before opening a PR, open an issue describing the bug or feature. Use `Fixes #123` or `Closes #123` in your PR description to link the issue.

- If adding a new feature:
  - Add accompanying test case.
  - Provide a convincing reason to add this feature. Ideally, you should open a suggestion issue first and have it approved before working on it.

- If fixing a bug:
  - If you are resolving a special issue, add `(fix #xxxx[,#xxxx])` (#xxxx is the issue id) in your PR title for a better release log, e.g. `fix: resolve prefix matching edge case (fix #123)`.
  - Provide a detailed description of the bug in the PR.
  - Add appropriate test coverage if applicable.

- It's OK to have multiple small commits as you work on the PR - GitHub can automatically squash them before merging.

- Make sure tests pass and code quality checks pass (`uv run poe clean-full`)!

- PR titles should follow conventional commit standards:
  - `feat:` new feature or functionality
  - `fix:` bug fix
  - `docs:` documentation or README changes
  - `chore:` maintenance tasks, dependency updates, etc.
  - `refactor:` code refactoring without changing behavior
  - `test:` adding or updating tests

### Advanced Pull Request Tips

- The PR should fix the intended bug **only** and not introduce unrelated changes. This includes unnecessary refactors - a PR should focus on the fix and not code style, this makes it easier to trace changes in the future.

- Keep pull requests small and focused.

- No AI-generated walls of text in PR descriptions. Write short, focused descriptions in your own words.

## Development Setup

You will need:

- [Python](https://python.org) 3.12 or higher
- [uv](https://github.com/astral-sh/uv) for dependency management and running the project

After cloning the repo, run:

```bash
$ uv sync  # install the dependencies of the project
```

A high level overview of tools used:

- [Python](https://python.org/) as the development language
- [pytest](https://docs.pytest.org/) for testing
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) for async test support
- [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [ty](https://github.com/astral-sh/ty) for type checking
- [deptry](https://deptry.com/) for dependency analysis
- [usechange](https://github.com/thememium/usechange) for release management

## Scripts

This project uses `poe` (poethepoet) for task management via `uv run poe <task>`.

- [`uv run poe dev`](#uv-run-poe-dev)
- [`uv run poe test`](#uv-run-poe-test)
- [`uv run poe test-e2e`](#uv-run-poe-test-e2e)
- [`uv run poe clean`](#uv-run-poe-clean)
- [`uv run poe clean-full`](#uv-run-poe-clean-full)
- [`uv run poe sort`](#uv-run-poe-sort)
- [`uv run poe lint`](#uv-run-poe-lint)
- [`uv run poe format`](#uv-run-poe-format)
- [`uv run poe typecheck`](#uv-run-poe-typecheck)
- [`uv run poe deptry`](#uv-run-poe-deptry)
- [`uv run poe release`](#uv-run-poe-release)

### `uv run poe dev`

Run the package entrypoint in development mode:

```bash
$ uv run poe dev
```

This is equivalent to `uv run src/dspy_auto_gepa/__init__.py`.

### `uv run poe test`

Run the test suite:

```bash
$ uv run poe test
```

This runs pytest with verbose output. You can also run specific tests:

```bash
# Run a specific test file
$ uv run pytest tests/test_runner.py -v

# Run a specific test
$ uv run pytest tests/test_runner.py::TestAutoGEPA -v
```

### `uv run poe test-e2e`

Run the full test suite including end-to-end tests:

```bash
$ uv run poe test-e2e
```

### `uv run poe clean`

Quick clean - sorts imports and formats code:

```bash
$ uv run poe clean
```

This runs `isort` and `ruff format`.

### `uv run poe clean-full`

Full code quality check - runs all quality tools:

```bash
$ uv run poe clean-full
```

This runs:
- `isort` - Sort imports
- `ruff check` - Lint and auto-fix
- `ruff format` - Format code
- `deptry` - Check for dependency issues
- `ty check` - Type check

**Run this before submitting a PR!**

### `uv run poe sort`

Sort imports only:

```bash
$ uv run poe sort
```

### `uv run poe lint`

Lint code only:

```bash
$ uv run poe lint
```

### `uv run poe format`

Format code only:

```bash
$ uv run poe format
```

### `uv run poe typecheck`

Type check the codebase:

```bash
$ uv run poe typecheck
```

### `uv run poe deptry`

Check for dependency issues:

```bash
$ uv run poe deptry
```

### `uv run poe release`

Create a new release using usechange:

```bash
$ uv run poe release
```

## Project Structure

```
├── src/dspy_auto_gepa/    # Source code
│   ├── __init__.py        # Main entry point and public API
│   ├── runner.py          # AutoGEPA orchestration logic
│   ├── metric_builder.py  # LLM-generated metric creation
│   ├── data.py            # Dataset conversion and field resolution
│   ├── config.py          # Configuration and defaults
│   └── artifacts.py       # Metric loading and result saving
├── tests/                 # Test files
├── docs/                  # Documentation assets (if any)
└── dist/                  # Build output
```

### Importing Code

When working within the codebase:

- Use absolute imports from the `dspy_auto_gepa` package for local modules
- Follow the import order: `__future__`, stdlib, third-party, local
- Use `from __future__ import annotations` for forward references

Example:

```python
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import dspy

from dspy_auto_gepa.runner import AutoGEPA

if TYPE_CHECKING:
    from dspy.clients.base_lm import BaseLM
```

## Style Guidelines

These are general guidelines for contributing code:

- **Functions:** Keep logic within a single function unless breaking it out adds clear reuse or composition benefits.
- **Type hints:** All functions must have type hints. Use `from __future__ import annotations` for forward references.
- **Imports:** Follow the import order: `__future__`, stdlib, third-party, local.
- **Error handling:** Prefer specific exceptions over generic `Exception`.
- **Variables:** Stick to immutable patterns when possible.
- **Naming:** Choose concise identifiers. Classes are PascalCase, functions/variables are snake_case, constants are UPPER_CASE.
- **Docstrings:** Use Google-style docstrings for public methods and classes.

## Credits

Thank you to all the people who have already contributed to AutoGEPA!

<a href="https://github.com/thememium/dspy-auto-gepa/graphs/contributors"><img src="https://contrib.rocks/image?repo=thememium/dspy-auto-gepa" /></a>