# Repo Collaboration Rules

This repository uses a split execution model:

- Codex may explore and modify repo-tracked files in this repository.
- The user runs environment-dependent commands, tests, container commands, and runtime validation.
- The user pastes back command output, logs, tracebacks, or screenshots for the next iteration.

## Default Working Style

- Prefer small, focused code changes over broad speculative refactors.
- After making changes, always provide:
  - a short change summary
  - the exact next command the user should run
  - what output the user should paste back if the run fails
- Do not claim validation for external environments that were not actually executed here.
- If the root cause is outside this repository, say so explicitly and point to the exact external path or dependency.

## Request Format

The most effective user request format is:

```text
Goal:
Environment:
Observed issue:
Constraint:
```

Example:

```text
Goal: make run_all_tests.sh work for dense profile by default
Environment: wrapper repo is /workspace/aloysha/test_sglang, actual sglang source is /sgl-workspace/sglang
Observed issue: bench_serving fails with tokenizer.bos_token related traceback
Constraint: you may edit this repo; I will run commands and paste back the output
```

## Iteration Loop

Use this loop by default:

1. Inspect the repository and identify the exact ownership boundary.
2. Modify the relevant files in this repository.
3. Tell the user the next command to run.
4. Wait for runtime results from the user.
5. Refine based on the returned logs.

## Boundary Rules

- Treat `/sgl-workspace/sglang` and similar external runtime trees as separate from this repository unless explicitly mounted into the current workspace.
- When an external dependency needs a change, provide either:
  - a precise manual edit snippet, or
  - a patch file in this repository that the user can apply in the external tree.
- Keep README changes concise and behavior-aligned; do not let docs drift from script defaults.
