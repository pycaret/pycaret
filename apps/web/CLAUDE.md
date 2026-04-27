# CLAUDE.md — `apps/web/` (control plane UI)

Path-scoped pointer for Claude Code working in this directory.
Authoritative conventions live in the repo-root
[`AGENTS.md`](../../AGENTS.md).

## Stack

- Vite 5 + React 18, TypeScript strict (`verbatimModuleSyntax`)
- Tailwind CSS with `ink-*` / `accent-*` design tokens
- React Router for routing
- Zustand for client state, TanStack Query for server state
- WebSocket connection to backend for live event streaming

## Local dev

The Vite dev server proxies `/api` and `/ws` to FastAPI on `:8000`.
Both must be running:

```bash
# Terminal 1
uv run --package pycaret-server pycaret-server serve --reload

# Terminal 2
cd apps/web && npm run dev    # http://localhost:3000
```

## Quick checks

```bash
cd apps/web
npm run typecheck && npm run lint && npm test && npm run build
```

## Engine-of-the-UI principle

The form-rendering and model-list code does **not** hardcode any
parameter name, model name, or metric. It introspects the backend's
`describe_setup_params(...)`, `list_models(...)`, `list_metrics(...)`
endpoints. **Don't** reach for hardcoded copies. If something looks
wrong, fix it on the backend or in the schema, not in the UI.
