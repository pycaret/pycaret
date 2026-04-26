# `apps/site` — agent maintenance guide

This file tells AI agents (Claude, Cursor, Copilot, …) how to maintain
the public PyCaret site. Every artifact on `pycaret.org` is generated
from this directory or auto-imported from elsewhere in the monorepo —
so an agent with read/write access to the repo can keep the site
current without human intervention.

## What lives where

```
apps/site/
├── app/                 # Next.js 15 App Router routes
│   ├── page.tsx         # Landing page (hero / features / CTAs)
│   ├── docs/            # Hand-written guides + tutorials (MDX)
│   ├── reference/       # Auto-generated API reference (from griffe)
│   ├── blog/            # Posts (hand-written + auto-imported)
│   └── changelog/       # Mirror of repo-root CHANGELOG.md
├── components/          # React components (MDX renderer, header, etc.)
├── content/             # Page content (the only place agents edit for content)
│   ├── docs/<section>/<page>.mdx
│   ├── blog/<slug>.mdx
│   ├── changelog.md     # Auto-imported, do not edit by hand.
│   └── api-tree.json    # Auto-generated, do not edit by hand.
├── lib/                 # Content + API-tree loaders (TypeScript)
├── public/              # Static assets (favicon, og:image, etc.)
├── scripts/
│   ├── gen_api_tree.py  # Python: griffe → content/api-tree.json
│   └── sync-content.mjs # Node: imports release notes + changelog
└── package.json
```

## How to add a new docs page

Drop an MDX file at `content/docs/<section>/<slug>.mdx` with this
front-matter:

```mdx
---
title: "Tuning hyperparameters"
description: "How tune_model works in PyCaret 4.0."
section: "Guides"
order: 3
---

# body in markdown / MDX
```

- `section` decides the sidebar group. Use `Getting started`,
  `Concepts`, or `Guides` — those have stable orderings. Anything else
  goes under "Other".
- `order` orders pages within a section (lower first).
- The route is automatic: `/docs/<section-folder>/<slug>/`.

## How to add a blog post

Two paths:

**Auto-imported from release notes.** The
`docs/revamp/release_notes_pycaret4.md` file is canonical. Each
`# Session NN — DATE — TITLE` block becomes one blog post. Just append
sessions there (per the existing format) and the next CI build will
publish them.

**Hand-written.** Drop an MDX file at `content/blog/<slug>.mdx` with
front-matter that includes a `date: "YYYY-MM-DD"` field. The blog
index sorts by date desc.

## How to update the API reference

Don't — it's auto-generated. The `scripts/gen_api_tree.py` script
walks the `pycaret` package via griffe and emits
`content/api-tree.json`. CI runs it on every push to `v4`/`main`.
To regenerate locally:

```bash
cd apps/site
npm run gen:api
```

If a public symbol shows up in the wrong section, update the
`PUBLIC_ROOTS` list in `scripts/gen_api_tree.py`.

## How to update the changelog

The `/changelog` route reads `apps/site/content/changelog.md`, which
is a mirror of the repo-root `CHANGELOG.md` (synced by
`scripts/sync-content.mjs`). Edit `CHANGELOG.md` at the repo root and
the next CI build picks it up.

## Local development

```bash
cd apps/site
npm install
npm run sync          # generate api-tree.json + import release notes
npm run dev           # http://localhost:3001
```

## Build

```bash
npm run build         # static export to `out/`
```

`npm run build` is what CI runs. It calls `sync-content.mjs` first so
the API tree + release notes are fresh.

## Deployment

CI (`.github/workflows/site.yml`) builds on every push to `v4`/`main`
and deploys to GitHub Pages. To switch to Vercel / Cloudflare Pages,
point the deploy step at the `apps/site/out/` directory — the build
itself is portable.

## Conventions

- **Tone**: concise, technical, no hype. Match the rest of the docs.
- **Code samples**: must run as-is (the agent should test them
  mentally if not literally). Always import the public API
  (`from pycaret.classification import ClassificationExperiment`),
  never internal modules.
- **Diff with 3.x**: when documenting a feature, mention what's
  different from 3.x if the migration would surprise a returning
  user.
- **Plot examples**: always use `pycaret.plots.<task>.<kind>` —
  never call the deleted `plot_model`.
- **No emojis** unless the user explicitly asks for them.
- **Inter / JetBrains Mono** are the only fonts. The Tailwind config
  ships them via Google Fonts.

## Adding a new section to the site

If you're adding something bigger than a page (e.g. a `/showcase`
gallery or a `/community` page):

1. Add a route file under `app/<section>/page.tsx`.
2. Add the link to `components/SiteHeader.tsx` (the `NAV` constant).
3. Add the link to `components/SiteFooter.tsx` (the `COLUMNS`
   constant).
4. If the section has its own sidebar nav, follow the pattern in
   `app/docs/layout.tsx`.

## Don't

- Don't write or edit anything inside `content/api-tree.json` —
  regenerated on every build.
- Don't write or edit `content/changelog.md` — synced from the repo
  root.
- Don't edit auto-imported blog posts (`content/blog/session-*.mdx`)
  — they're regenerated from `docs/revamp/release_notes_pycaret4.md`.
- Don't import the engine directly into the site. The site is a
  pure-static export; the engine only runs at build time inside the
  griffe extractor.
