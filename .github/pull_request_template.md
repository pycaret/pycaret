<!-- PyCaret 4.0 pull request template. Fill in the sections below. -->

## Summary

<!-- One-paragraph description of what this PR does and why.
     The diff shows *what*; this section should explain *why*. -->

## Related issue

Closes #<issue-number>

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / tech-debt cleanup
- [ ] Docs / contributor-facing
- [ ] CI / build
- [ ] Verb migration (legacy → native — see `docs/for_developers/DRAINING_THE_GODCLASS.md`)

## PR checklist

- [ ] Tests added or updated (`tests/test_core_architecture.py` and/or `tests/test_e2e_oop.py`)
- [ ] `uv run ruff check pycaret tests` passes
- [ ] `uv run ruff format --check pycaret tests` passes
- [ ] `uv run pytest tests/test_core_architecture.py tests/test_datasets.py -q` passes locally
- [ ] Release-notes entry appended to `docs/revamp/release_notes_pycaret4.md` under the current session block, tagged `BREAKING` / `REMOVED` / `ADDED` / `CHANGED` / `FIXED` / `DEPRECATED` / `SECURITY` / `DOCS` / `BUILD` / `TESTS` / `DEPS` / `INTERNAL`
- [ ] For new runtime deps: ADR added in `docs/revamp/DECISIONS.md`
- [ ] For API changes: notebooks / README / docs updated where relevant
- [ ] If a verb was migrated off `_legacy`: the legacy method is deleted and the public API signature is unchanged

## Notes for reviewers

<!-- Anything the reviewer should know that isn't obvious from the diff.
     e.g. "I moved X into Y but did not rename; next PR renames it." -->
