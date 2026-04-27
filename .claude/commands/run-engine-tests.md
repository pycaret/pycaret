---
description: Fast local test loop for the engine package
---

Run the engine test suite. If the user provided a pattern (test name
or module), scope to that; otherwise run everything fast.

## Default — everything fast

```bash
uv run pytest packages/engine/tests/ -q
```

## Scoped — pattern matching

If the user said something like "run the regression tests" or "run
test_compare_models":

```bash
uv run pytest packages/engine/tests/ -q -k "<pattern>"
```

## Slow tests too

If the user explicitly asks for slow / nightly tests:

```bash
uv run pytest packages/engine/tests/ -q -m "slow"
```

(default suite excludes `@pytest.mark.slow`)

## After tests pass — always lint

```bash
uv run ruff check packages/engine/
```

If lint fails, run the formatter and re-check:

```bash
uv run ruff format packages/engine/
uv run ruff check packages/engine/
```

## Report

Tell the user:
- Pass count / fail count
- For failures: the first failing test name + the assertion line
- For lint issues: a summary count, not the full output

Don't paste the full pytest stdout into chat — it's noise. Show what
matters.
