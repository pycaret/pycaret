# `@pycaret/ui`

React frontend for the PyCaret 4.0 application platform. Talks to
[`pycaret-server`](../pycaret-server/) via `/api/v1` + WebSocket.

**Status:** alpha. Session 12 (Phase 10 start) — bootstrap + auth + workspace/project CRUD.

## Stack

- Vite 5 + React 18 + TypeScript 5 (strict, `verbatimModuleSyntax`).
- Tailwind CSS 3 (dark-mode first, class-based toggle).
- TanStack Query for server state, Zustand for UI state.
- React Router 6.
- axios with a single-flight refresh interceptor.
- Vitest + Testing Library for tests.

## Dev loop

```bash
cd pycaret-ui
npm install
npm run dev            # http://localhost:3020 (proxies /api → :8020)
npm run typecheck
npm run lint
npm test
npm run build          # production bundle -> dist/
```

To regenerate typed API types from a running backend:

```bash
# backend running on :8020
npm run gen:api
```

That writes `src/api/schema.ts` from `/openapi.json`. For now the UI uses
hand-written mirrors in `src/api/types.ts` — the generated `schema.ts` is
wired but not imported.

## Screens (session 12)

- `/setup` — first-run bootstrap wizard (admin + default workspace).
- `/login` — sign in.
- `/` — workspace list + create.
- `/workspaces/:id` — project list + create.

Screens 5–8 (project detail, experiment setup-form, run view with live event stream,
admin) land in session 13+.

## Design principles

- **Minimalistic.** No chrome, no noise. Single-column forms, generous whitespace, keyboard-first.
- **Dark-mode first**, light opt-in (deferred).
- **Desktop-first** (this is an analyst tool).
- **No icons without labels.**
