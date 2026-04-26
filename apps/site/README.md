# `apps/site` — pycaret.org

The public marketing site, documentation, API reference, and blog for
PyCaret. Single Next.js 15 app deployed at `pycaret.org`.

## Stack

- **Next.js 15** with the App Router
- **TypeScript**, **Tailwind CSS**
- **MDX** content via `next-mdx-remote-client`
- **Shiki** for syntax highlighting (server-rendered, zero-runtime)
- **griffe** for API auto-generation (Python → JSON tree)

## Run locally

```bash
cd apps/site
npm install
npm run sync   # generate API tree + import release notes / changelog
npm run dev    # http://localhost:3001
```

## Build

```bash
npm run build  # static export to ./out
```

CI builds on every push to `v4` / `main` and deploys to GitHub Pages
(see `.github/workflows/site.yml`). Switching to Vercel / Cloudflare
Pages is a one-line change in the workflow.

## Adding content

See [`AGENTS.md`](./AGENTS.md). Short version:

- New docs page → drop an MDX in `content/docs/<section>/`.
- New blog post → MDX in `content/blog/` with a `date:` front-matter
  field.
- API reference is auto-generated; don't edit it.
- Changelog auto-syncs from repo-root `CHANGELOG.md`.

## License

MIT, same as the rest of the repo.
