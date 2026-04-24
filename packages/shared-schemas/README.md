# `packages/shared-schemas` — canonical schema definitions

**Status:** stub. Intended as the single source of truth for types that cross
the Python ↔ TypeScript boundary.

Today `services/api/pycaret_server/api/schemas.py` (Pydantic) and
`apps/web/src/api/types.ts` (hand-written TS) maintain parallel definitions
of ~12 types (Workspace, Project, Experiment, Run, ...). Drift is already
possible.

When work starts here, this directory will hold:

```
packages/shared-schemas/
├── schemas/              # canonical JSON schemas
│   ├── workspace.json
│   ├── project.json
│   ├── run_config.json   # the RunConfig Pydantic model (§ 6.1 of spec)
│   └── ...
└── generators/
    ├── to_pydantic.py    # emit Python pydantic classes
    └── to_typescript.ts  # emit TS types
```

Both `services/api` and `apps/web` generate their types at build time. No
drift possible — if a schema change breaks compatibility, CI fails on both
sides.

Particularly important for `RunConfig` (§ 6 of CONTROL_PLANE_SPEC.md), which
needs to be deeply validated server-side but also drives a dynamic form in
the UI.
