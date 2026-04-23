# Issue-triage playbook

How to work through the buckets in `triage.md`. Designed so a maintainer or an AI agent can execute it without per-issue judgment calls for ~58% of the backlog.

## 1. `fixed_in_4_0` (8 issues) — close

Paste this reply template, then close:

```
Thanks for the report. PyCaret 4.0 is a ground-up revamp that resolves the
underlying cause — see the migration notes: ../docs/revamp/release_notes_pycaret4.md
and the 4.0 README. Closing as fixed.

If you still see this in 4.0 on a supported Python (3.11 / 3.12 / 3.13),
please reopen with a minimal repro against the 4.0 `pycaret.tasks.*Experiment`
API.
```

Use `gh issue close <n> --comment "<template>"`.

## 2. `out_of_scope` (92 issues) — close

Paste this reply template:

```
The {feature} integration was removed in PyCaret 4.0 as part of the revamp.
Rationale and the full removal list: ../docs/revamp/KILL_LIST.md.

If there's community interest we'd welcome a standalone `pycaret-{feature}`
package that ships the removed adapter separately. Closing this issue as
out of scope for the 4.0 engine.
```

Use `gh issue close <n> --comment "..."`. Tag `wontfix-in-core` if you want to revisit later.

## 3. `stale` (123 issues) — auto-ping, then close

Post this reply, wait 30 days, close if no reply:

```
This issue hasn't been updated in 2+ years and predates the PyCaret 4.0
revamp. Does it still reproduce against 4.0?

If you can, paste a minimal repro using:

    from pycaret.tasks import ClassificationExperiment   # or your task
    exp = ClassificationExperiment(...).fit(data)
    ...

Auto-closing after 30 days of silence — reopen anytime.
```

A GitHub Action can auto-close `stale` after 30 days — see `.github/workflows/` (future).

## 4. `still_relevant_bug` (58 issues) — label `4.0-candidate`

For each:

1. Reproduce against 4.0 (`uv run pytest` plus a manual notebook if needed).
2. **If reproducible** → label `4.0-candidate` + `bug`. It goes into the Phase 5 repair queue. Assign to a milestone.
3. **If NOT reproducible** → close with "Cannot reproduce on 4.0 — see release notes. Reopen with a minimal repro if still encountering this."

## 5. `still_relevant_enhancement` (107 issues) — per-item decision

For each, the call is: does it fit the 4.0 engine vision (sklearn-composable, agent-friendly, lean)?

| Verdict | Action |
|---|---|
| Fits 4.0 core | Label `4.0-candidate` + `enhancement`. Add to a milestone. |
| Good idea but out of core scope | Label `wontfix-in-core` + `extras-candidate`. Propose as community package. |
| Outdated / not useful | Close with explanation. |

## Running at scale

Two helpers live in `scripts/`:

```bash
# Re-run classification after cleaning a bucket:
uv run python scripts/triage_issues.py

# Bulk-close a bucket (TBD):
# uv run python scripts/close_bucket.py --bucket fixed_in_4_0 --template path/to/template.md
```

(The bulk-close script is next on the issue-cleanup TODO — it reads `triage.json` and calls `gh issue close` for each item in the target bucket with the appropriate template.)
