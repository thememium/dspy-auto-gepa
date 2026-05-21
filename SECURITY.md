# Reporting a Vulnerability

To report a security vulnerability, please email boswell.labs@gmail.com.

We take security seriously and will respond to security reports within 48 hours. Please include as much detail as possible about the vulnerability, including:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)

While the discovery of new vulnerabilities is rare, we also recommend always using the latest version of dspy-auto-gepa to ensure your application remains as secure as possible.

## Security Considerations for dspy-auto-gepa

As dspy-auto-gepa automates DSPy optimization pipelines by generating evaluation metrics with LLMs and orchestrating GEPA training, please be aware of the following security practices:

- **LLM-Generated Code Execution**: This package generates Python metric files using a language model and dynamically imports and executes them via `importlib`. Generated code is validated with `ast.parse()` for syntax, but semantic correctness and safety are not guaranteed. Always review generated `.py` metric files in your artifact directory before using them in production, and provide a custom `metric` path if you require fully vetted evaluation logic.
- **Data Exposure**: Training data (`rows`) is sent to LLMs during both metric generation and GEPA optimization reflection steps. This includes the content of your examples and their field names. Do not use AutoGEPA with sensitive, personally identifiable, or proprietary data unless you fully trust and have contractual agreements with your LLM provider.
- **DSPy Dependency**: AutoGEPA relies on DSPy's `GEPA` optimizer, which internally executes LLM-generated prompts and may perform reflective reasoning over program execution traces. Security properties of the optimization process are bounded by DSPy's own guarantees. Always pin to a known-compatible DSPy version.
- **Artifact Storage**: By default, generated metrics, optimized programs, and training artifacts are written to the `.auto_gepa/` directory. Ensure this directory has appropriate filesystem permissions and is excluded from version control to prevent accidental leakage of training data or model weights.

## Security Hall of Fame

We would like to thank the following security researchers for responsibly disclosing security issues to us.

*No security researchers have been added to the hall of fame yet. Will you be the first?*
