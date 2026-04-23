# PyCaret 4.0 — agent-facing documentation

This folder is the deep-dive companion to `/AGENTS.md`. Each file answers *one* question an AI agent or tooling integrator will ask before writing code.

| File | Question it answers |
|---|---|
| [`ENGINE_WALKTHROUGH.md`](ENGINE_WALKTHROUGH.md) | What happens at every step of `fit → compare_models → predict_model`? |
| [`TYPED_RESULTS.md`](TYPED_RESULTS.md) | What does every Experiment verb return, and what are the fields? |
| [`EVENT_STREAM.md`](EVENT_STREAM.md) | What events can the engine emit, and how do I subscribe? |
| [`INTROSPECTION_API.md`](INTROSPECTION_API.md) | How do I enumerate models / metrics / setup params for a UI or agent prompt? |
| [`TASK_CHEATSHEET.md`](TASK_CHEATSHEET.md) | One-page reference — which verbs exist on which task, at a glance. |

Read `/AGENTS.md` first (60-second briefing, rules, repo map). Use these files when you need structured field-level detail.
