# `apps/desktop` — Electron desktop app (V2)

**Status:** roadmapped. Not yet implemented.

Per `docs/revamp/DECISIONS.md § 2026-04-24 · electron-defer`, the desktop
distribution is a V2 deliverable. It builds on top of the `apps/web` bundle
and the `services/api` backend, both of which ship inside the Electron
process, using bundled Python + SQLite for zero-config local use.

When work starts here, this directory should contain:

```
apps/desktop/
├── package.json         # electron + electron-builder
├── electron/
│   ├── main.ts          # spawns uvicorn subprocess, loads packaged web UI
│   └── preload.ts
├── build/               # per-OS icons, installer config
└── scripts/             # signing, notarization
```

Python runtime is packaged via pyoxidizer or a standalone CPython bundle;
the frontend is the exact same Vite bundle served by `apps/web`.

See `docs/revamp/ROADMAP.md § V2 — distribution` for the scope.
