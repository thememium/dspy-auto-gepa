# Ideas Backlog

- **Pre-compute output combos before input gen**: If we know the target output combos (from allowed_values), generate inputs in a way that's already balanced, avoiding the oversampling + subsample step entirely.
- **Stream outputs as inputs arrive**: Instead of generating all inputs first, then all outputs, start output generation as soon as a batch of inputs is ready (pipeline parallelism).
- **Skip judge for high-confidence rows**: If the output model's JSON schema strongly constrains values (Literal types), skip the LLM judge for rows that pass schema validation. Only judge ambiguous cases.
- **Async judge scoring**: Run judge scoring in background while generating the next batch of outputs. Don't block on judge results.
