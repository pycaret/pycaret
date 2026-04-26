/**
 * Sync auto-generated content from the rest of the monorepo into the
 * Next.js content tree.
 *
 * What this script does (in order):
 *
 * 1. Generates ``content/api-tree.json`` by running the Python griffe
 *    extractor (``scripts/gen_api_tree.py``).
 * 2. Imports release notes from ``../../docs/revamp/release_notes_pycaret4.md``
 *    into ``content/blog/`` as individual MDX posts (one per session
 *    block).
 * 3. Imports the user-facing changelog (``../../CHANGELOG.md``) into
 *    ``content/changelog.md`` so the ``/changelog`` route can render it.
 *
 * Idempotent — safe to re-run; rebuilds output files from scratch each
 * time. Run via ``npm run sync`` or implicitly before ``npm run build``.
 */

import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SITE_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(SITE_ROOT, '..', '..');

function log(msg) {
  console.log(`[sync] ${msg}`);
}

function generateApiTree() {
  log('generating API tree via griffe…');
  try {
    execSync('uv run --with griffe python scripts/gen_api_tree.py', {
      cwd: SITE_ROOT,
      stdio: 'inherit',
    });
  } catch (err) {
    console.warn(
      '[sync] WARN: griffe extraction failed — the /reference pages will use a stale or empty tree.',
    );
    console.warn(`[sync]       ${err.message ?? err}`);
    // Write an empty tree so the build still succeeds in environments
    // without the engine source available (e.g. site-only deploy preview).
    const out = path.join(SITE_ROOT, 'content', 'api-tree.json');
    if (!fs.existsSync(out)) {
      fs.mkdirSync(path.dirname(out), { recursive: true });
      fs.writeFileSync(out, '{}', 'utf8');
    }
  }
}

function importReleaseNotes() {
  const sourceFile = path.join(REPO_ROOT, 'docs', 'revamp', 'release_notes_pycaret4.md');
  if (!fs.existsSync(sourceFile)) {
    log(`skipping release notes import: ${sourceFile} not found`);
    return;
  }
  const blogDir = path.join(SITE_ROOT, 'content', 'blog');
  fs.mkdirSync(blogDir, { recursive: true });

  // Wipe previously-generated posts so we don't accumulate stale ones.
  for (const file of fs.readdirSync(blogDir)) {
    if (file.startsWith('session-') && file.endsWith('.mdx')) {
      fs.rmSync(path.join(blogDir, file));
    }
  }

  const raw = fs.readFileSync(sourceFile, 'utf8');
  // Sessions are delimited by `# Session NN — DATE — TITLE` headings.
  const sessionRegex = /^# Session (\d+) — (\d{4}-\d{2}-\d{2}) — (.+)$/gm;
  const blocks = [];
  const matches = [...raw.matchAll(sessionRegex)];
  for (let i = 0; i < matches.length; i++) {
    const m = matches[i];
    const start = m.index ?? 0;
    const end = i + 1 < matches.length ? matches[i + 1].index : raw.length;
    blocks.push({
      number: m[1],
      date: m[2],
      title: m[3].trim(),
      body: raw.slice(start + m[0].length, end).trim(),
    });
  }

  let written = 0;
  for (const b of blocks) {
    const slug = `session-${b.number.padStart(2, '0')}`;
    const frontmatter = [
      '---',
      `title: "Session ${b.number}: ${b.title.replace(/"/g, '\\"')}"`,
      `date: "${b.date}"`,
      `description: "Engineering log for session ${b.number}."`,
      `tags: ["release-notes", "engineering"]`,
      '---',
    ].join('\n');
    const out = `${frontmatter}\n\n${b.body}\n`;
    fs.writeFileSync(path.join(blogDir, `${slug}.mdx`), out, 'utf8');
    written += 1;
  }
  log(`imported ${written} release-note posts into content/blog/`);
}

function importChangelog() {
  const sourceFile = path.join(REPO_ROOT, 'CHANGELOG.md');
  if (!fs.existsSync(sourceFile)) {
    log('skipping changelog import: CHANGELOG.md not found');
    return;
  }
  const target = path.join(SITE_ROOT, 'content', 'changelog.md');
  fs.copyFileSync(sourceFile, target);
  log('imported CHANGELOG.md into content/changelog.md');
}

generateApiTree();
importReleaseNotes();
importChangelog();
log('done.');
