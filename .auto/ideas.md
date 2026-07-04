# Ideas Backlog

- **Parallelize combo generation**: Currently combos are processed sequentially in `_generate_targeted_inputs`. Process all combos' batches in a single `dspy.Parallel` call for further speedup.
- **Skip judge for schema-constrained outputs**: If the output model's JSON schema strongly constrains values (Literal types), skip the LLM judge for rows that pass schema validation.
- **Stream outputs as inputs arrive**: For non-balanced mode, start output generation as soon as a batch of inputs is ready (pipeline parallelism).
- **Pre-compute output combos before input gen**: For signature mode, use the same targeted approach — generate complete rows backwards from target outputs.
