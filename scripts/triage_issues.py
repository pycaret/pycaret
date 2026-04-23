"""Categorize the 388 open pycaret/pycaret issues against the 4.0 kill list
and the current architecture.

Buckets:
- fixed_in_4_0: issues about features we removed OR bugs the revamp resolves.
  Reply with the 4.0 release-notes link and close.
- out_of_scope: issues requesting features we deliberately dropped per
  KILL_LIST.md. Reply pointing at KILL_LIST and close (or leave open with
  label "wontfix-in-core").
- stale: issues from 2020-2022 with no updates in 2+ years. Ask reporter if
  still reproducible on 4.0.
- still_relevant_bug / still_relevant_enhancement: genuine 4.0 candidates.

Run:
    uv run python scripts/triage_issues.py

Writes:
    docs/revamp/github_issues/triage.md     — human-readable summary
    docs/revamp/github_issues/triage.json   — machine-readable buckets
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "docs" / "revamp" / "github_issues" / "open_issues_raw.json"
OUT_MD = ROOT / "docs" / "revamp" / "github_issues" / "triage.md"
OUT_JSON = ROOT / "docs" / "revamp" / "github_issues" / "triage.json"


# -----------------------------------------------------------------------------
# Classification keywords
# -----------------------------------------------------------------------------


# Any issue mentioning these features is "out_of_scope" — we deliberately
# removed the feature in 4.0, so the issue is no longer actionable.
OUT_OF_SCOPE_KEYWORDS = [
    r"\bmlflow\b", r"\bcomet\b", r"\bwandb\b", r"\bdagshub\b",
    r"\bfugue\b", r"\bdask\b", r"\bray[\s_/-]+tune\b", r"\bray\s*\[tune\]", r"\btune[_-]sklearn\b",
    r"\byellowbrick\b", r"\bmljar\b", r"\bscikit[_-]plot\b", r"\bschemdraw\b",
    r"\bplotly[_-]resampler\b",
    r"\bevidently\b",
    r"\bfairlearn\b", r"\bcheck_fairness\b", r"\bcheck_drift\b",
    r"\bydata[_-]profiling\b", r"\beda\b.*profile", r"pandas[_-]profiling",
    r"\bexplainerdashboard\b", r"\bdashboard\(\)", r"\.dashboard\b",
    r"\bgradio\b", r"\bcreate_app\b", r"\bcreate_api\b", r"\bcreate_docker\b",
    r"\bboto3\b", r"\baws[_\s]s3\b", r"deploy_model.*s3",
    r"\bm2cgen\b", r"\bconvert_model\b",
    r"scikit[_-]learn[_-]intelex\b", r"\bsklearnex\b", r"\bdaal4py\b",
]

# Features that are KEPT in 4.0 — but the area name correlates with "still relevant"
KEPT_AREAS = [
    r"\bsetup\b", r"\bcompare_models\b", r"\bcreate_model\b", r"\btune_model\b",
    r"\bensemble_model\b", r"\bblend_models\b", r"\bstack_models\b",
    r"\bcalibrate_model\b", r"\bpredict_model\b", r"\bsave_model\b",
    r"\bload_model\b", r"\bfinalize_model\b",
    r"\bpreprocess\b", r"\bencoding\b", r"\bimputation\b",
    r"\bpipeline\b", r"\bmodel registry\b",
    r"\bclassification\b", r"\bregression\b", r"\bclustering\b", r"\banomaly\b", r"\btime[_-]series\b",
]

FIXED_IN_4_HINTS = [
    # Upstream-compat bugs fixed in 4.0
    r"scikit[-_ ]learn\s*(?:>=|=|version)\s*(?:1\.[5-9]|1\.[0-9]{2})",
    r"sklearn\s*1\.[5-9]",
    r"numpy\s*2",
    r"pandas\s*2\.[2-9]",
    r"python\s*3\.1[2-9]",
    r"distutils",
    r"np\.NaN", r"np\.product",
    r"ValueError:.*_check_reg_targets",
    # Tech-debt problems the revamp resolved
    r"bulky", r"too many dependencies", r"installation size",
    r"takes forever to install",
]

# Older than this and no update since this date → stale
STALE_DATE_CUTOFF = datetime(2023, 1, 1, tzinfo=timezone.utc)


# -----------------------------------------------------------------------------


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _labels(issue: Dict[str, Any]) -> List[str]:
    return [l["name"] for l in issue.get("labels", [])]


def _title(issue: Dict[str, Any]) -> str:
    return (issue.get("title") or "").lower()


def _body(issue: Dict[str, Any]) -> str:
    return (issue.get("body") or "").lower()


def _regex_any(patterns: List[str], text: str) -> List[str]:
    hits = []
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            hits.append(p)
    return hits


def _looks_like_pip_dump(body: str) -> bool:
    """Body contains a pip/conda environment listing? (many `==` lines)"""
    # A pip freeze / conda list output has many `<name>==<version>` lines.
    return len(re.findall(r"^\s*[a-zA-Z0-9_\-\.]+\s*==\s*[0-9]", body, re.M)) >= 5


def classify(issue: Dict[str, Any]) -> Dict[str, Any]:
    labels = _labels(issue)
    title = _title(issue)
    body = _body(issue)
    text = title + "\n" + body
    updated = _parse_iso(issue["updatedAt"])

    title_oos_hits = _regex_any(OUT_OF_SCOPE_KEYWORDS, title)
    body_oos_hits = _regex_any(OUT_OF_SCOPE_KEYWORDS, body)
    fixed_hints = _regex_any(FIXED_IN_4_HINTS, text)

    is_pip_dump = _looks_like_pip_dump(body)

    # Strong OOS signal: title mentions a killed feature.
    # Weak OOS signal: body mentions it but looks like a pip dump.
    if title_oos_hits:
        bucket = "out_of_scope"
        reason = f"Title mentions killed feature(s): {', '.join(sorted(set(title_oos_hits)))}"
    elif body_oos_hits and not is_pip_dump:
        bucket = "out_of_scope"
        reason = f"Body mentions killed feature(s): {', '.join(sorted(set(body_oos_hits)))}"
    elif fixed_hints:
        bucket = "fixed_in_4_0"
        reason = f"Matches 4.0 revamp fixes: {', '.join(sorted(set(fixed_hints)))}"
    elif updated < STALE_DATE_CUTOFF:
        bucket = "stale"
        reason = f"No update since {updated.date()} (> 2 years)"
    elif "bug" in labels:
        bucket = "still_relevant_bug"
        reason = "Labeled bug; not kill-listed; recent"
    else:
        bucket = "still_relevant_enhancement"
        reason = "Not kill-listed; recent"

    return {
        "number": issue["number"],
        "title": issue["title"],
        "labels": labels,
        "created": issue["createdAt"][:10],
        "updated": issue["updatedAt"][:10],
        "bucket": bucket,
        "reason": reason,
    }


# -----------------------------------------------------------------------------


def main() -> None:
    with IN_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)

    triaged = [classify(i) for i in raw]
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for t in triaged:
        buckets.setdefault(t["bucket"], []).append(t)

    # Sort each bucket by issue number descending (newest first)
    for b in buckets.values():
        b.sort(key=lambda t: t["number"], reverse=True)

    # Write JSON (full)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({"counts": {k: len(v) for k, v in buckets.items()}, "buckets": buckets}, f, indent=2)

    # Write human-readable MD summary
    order = [
        "fixed_in_4_0",
        "out_of_scope",
        "stale",
        "still_relevant_bug",
        "still_relevant_enhancement",
    ]
    total = len(triaged)
    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write(f"# PyCaret 4.0 — Open-issue triage\n\n")
        f.write(f"*Auto-generated from `open_issues_raw.json` by `scripts/triage_issues.py`.*\n\n")
        f.write(f"Total open issues: **{total}**. Triage buckets:\n\n")
        f.write("| Bucket | Count | % | Suggested action |\n")
        f.write("|---|---:|---:|---|\n")
        actions = {
            "fixed_in_4_0": "Close with pointer to `release_notes_pycaret4.md` and the 4.0 release announcement.",
            "out_of_scope": "Close with pointer to `KILL_LIST.md`. Optionally tag `wontfix-in-core` and leave for a community-maintained `pycaret-extras` repo.",
            "stale": "Reopener ping: \"Does this still reproduce on PyCaret 4.0?\" Auto-close after 30 days of silence.",
            "still_relevant_bug": "Label `4.0-candidate`. Triage into Phase 5 repair queue.",
            "still_relevant_enhancement": "Label `4.0-candidate`. Decide per-issue whether to accept, defer to 4.1+, or close.",
        }
        for b in order:
            c = len(buckets.get(b, []))
            f.write(f"| `{b}` | {c} | {c/total:.0%} | {actions[b]} |\n")
        f.write("\n---\n\n")

        for b in order:
            items = buckets.get(b, [])
            f.write(f"## `{b}` — {len(items)} issue(s)\n\n")
            f.write(f"**Action:** {actions[b]}\n\n")
            if not items:
                f.write("_(none)_\n\n")
                continue
            f.write("| # | Title | Labels | Updated | Reason |\n")
            f.write("|---:|---|---|---|---|\n")
            for t in items:
                title = t["title"].replace("|", "\\|")
                lbl = ", ".join(t["labels"]) if t["labels"] else ""
                f.write(f"| [#{t['number']}](https://github.com/pycaret/pycaret/issues/{t['number']}) "
                        f"| {title} | {lbl} | {t['updated']} | {t['reason']} |\n")
            f.write("\n")

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print()
    for b in order:
        c = len(buckets.get(b, []))
        print(f"  {b:32s} {c:4d}  ({c/total:.0%})")


if __name__ == "__main__":
    main()
