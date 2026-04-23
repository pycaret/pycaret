# Open-issue triage — PyCaret 4.0

This folder holds the snapshot and triage of the 388 open issues that existed on `pycaret/pycaret` at the start of the 4.0 revamp.

## Files

| File | Purpose |
|---|---|
| [`open_issues_raw.json`](open_issues_raw.json) | Raw snapshot from `gh issue list --state open --json ...`. Date-stamped; never modified. |
| [`triage.json`](triage.json) | Machine-readable bucketing produced by `scripts/triage_issues.py`. |
| [`triage.md`](triage.md) | Human-readable triage report — same data, browsable. |
| [`PLAYBOOK.md`](PLAYBOOK.md) | Step-by-step instructions for working through each bucket. |

## Headline

**Of 388 open issues, 224 (58%) can be closed or auto-pinged immediately without human triage:**

| Bucket | Count | % | Action |
|---|---:|---:|---|
| `fixed_in_4_0` | 8 | 2% | Close with `release_notes_pycaret4.md` link |
| `out_of_scope` | 92 | 24% | Close with `KILL_LIST.md` link |
| `stale` | 123 | 32% | Auto-ping the reporter, close after 30d of silence |
| `still_relevant_bug` | 58 | 15% | Label `4.0-candidate`; triage into Phase 5 |
| `still_relevant_enhancement` | 107 | 28% | Label `4.0-candidate`; accept / defer / reject per item |

**224 can be closed/pinged, 165 need real human (or agent) triage.** For a project that's been "unmaintained with 300+ open issues" for 3 years, this cuts the backlog to ~165 meaningful items in one pass.

## How it was classified

`scripts/triage_issues.py` runs these heuristics in order:

1. **Title mentions a killed feature** (mlflow/comet/fugue/yellowbrick/...) → `out_of_scope`.
2. **Body mentions a killed feature (and isn't a pip-freeze dump)** → `out_of_scope`.
3. **Title/body matches a 4.0-revamp-fix pattern** (Python 3.12+, NumPy 2, pandas 2.2, sklearn 1.5+, distutils, `np.NaN`, `np.product`, bloat) → `fixed_in_4_0`.
4. **Not updated since 2023-01-01** → `stale`.
5. **Has the `bug` label** → `still_relevant_bug`.
6. **Otherwise** → `still_relevant_enhancement`.

Heuristics catch the obvious cases. Run the script (`uv run python scripts/triage_issues.py`) to regenerate after cleaning any buckets.

## Re-running

```bash
# Re-snapshot issues from GitHub (needs `gh` auth):
gh issue list --repo pycaret/pycaret --state open --limit 1000 \
    --json number,title,labels,createdAt,updatedAt,author,comments,body,state \
    > docs/revamp/github_issues/open_issues_raw.json

# Re-classify:
uv run python scripts/triage_issues.py
```
