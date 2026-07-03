# Ideas Backlog

- **Batch judge calls**: Instead of calling LLM judge per-row, send multiple rows in a single judge call with a rubric. This would reduce LLM calls from N to ceil(N/batch_size).
- **Pre-compute output combos before input gen**: If we know the target output combos (from allowed_values), generate inputs in a way that's already balanced, avoiding the 2x oversampling + subsample step.
- **Stream outputs as inputs arrive**: Instead of generating all inputs first, then all outputs, start output generation as soon as a batch of inputs is ready (pipeline parallelism).
