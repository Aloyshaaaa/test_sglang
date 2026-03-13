# Codex Workflow

## Goal

Use Codex as a coding partner that edits this repository, while you handle runtime execution and feed the results back.

## Default Loop

1. Give Codex the goal, environment, issue, and constraints.
2. Let Codex inspect the repository and make the code changes here.
3. Run the command Codex gives you.
4. Paste back the new output.
5. Repeat until the issue is closed.

## Best Request Template

```text
Goal:
Environment:
Observed issue:
Constraint: you can modify this repo; I will run commands and return logs
```

## Best Follow-up Template

```text
Continue from the previous change set. Do not restate background.
Here is the new output:
[paste output]

Please:
1. explain what this means
2. modify the repo if needed
3. give me the next command to run
```

## What Codex Should Return

For each coding iteration, expect:

- what changed
- the next command to run
- what output to paste back if the run fails

## High-Value Tasks

Codex is especially useful here for:

- tracing script entrypoints and environment-variable flow
- hardening shell wrappers and default configs
- shrinking long known-good commands into built-in defaults
- identifying whether a bug belongs to this repo or to external runtime code
- rewriting documentation to match actual script behavior
