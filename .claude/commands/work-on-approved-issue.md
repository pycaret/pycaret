---
description: Pick an Approved GitHub issue and resolve it end-to-end
---

You're going to help fix one PyCaret GitHub issue that the maintainer
has reviewed and labelled `Approved`. Follow this protocol:

## 1. List the available issues

```bash
gh issue list --repo pycaret/pycaret --label Approved --state open --limit 20
```

If there are no Approved issues, tell the user and stop. Don't go
hunting for un-Approved issues to work on — the label is the gate.

## 2. Pick one

If the user has already named an issue number, use it. Otherwise,
print the top 5 with title + 1-line summary and ask them to pick one
by number.

## 3. Read it carefully

```bash
gh issue view <N> --comments
```

Read every comment, not just the body. Often clarification arrives
in comments. Note any reproductions, environment details, or
maintainer notes.

## 4. Sanity check against the kill list

Before writing any code, delegate to the **kill-list-checker**
sub-agent with the feature/symptom in question. If it comes back
"on the kill list" or "adjacent feature on the kill list", **stop
and comment on the issue** asking the maintainer to confirm scope.

## 5. Branch off `main`

```bash
git checkout main && git pull --ff-only
git checkout -b fix-<N>-<short-slug>
```

The slug should be 2-4 words, kebab-case, descriptive. E.g.
`fix-1234-tune-model-cv-fold-mismatch`.

## 6. Delegate the actual fix

Hand off to the **issue-fixer** sub-agent with the issue number.
That agent owns: reproduce → write failing test → fix → run tests →
lint → commit → push → open PR → comment on issue.

When it returns, verify the PR exists by URL and that:
- The PR description includes "Fixes #<N>"
- There's a test that demonstrates the fix
- CI is queued or running

## 7. Stop

Do **not** push to main. Do **not** auto-merge. The maintainer
reviews and merges. Tell the user:

```
✅ PR opened: <url>
The maintainer will review and merge.
```

If you got blocked at any step, leave a comment on the issue with
the blocker and stop. Don't open a half-baked PR.
