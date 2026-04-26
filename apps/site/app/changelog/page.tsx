/**
 * `/changelog` — auto-rendered from `content/changelog.md`.
 *
 * The sync script (`scripts/sync-content.mjs`) copies the repo-root
 * `CHANGELOG.md` into `content/changelog.md`, so this page always
 * reflects the latest release notes a maintainer has written.
 */
import fs from 'node:fs';
import path from 'node:path';

import { MdxRenderer } from '@/components/MdxRenderer';

export const metadata = {
  title: 'Changelog',
  description: 'User-facing changes across PyCaret releases.',
};

function loadChangelog(): string {
  const target = path.join(process.cwd(), 'content', 'changelog.md');
  if (!fs.existsSync(target)) {
    return [
      '# Changelog',
      '',
      "_The repo's `CHANGELOG.md` hasn't been imported yet._",
      '',
      'Run `npm run sync` from `apps/site` (or trigger a CI build) to populate this page.',
    ].join('\n');
  }
  return fs.readFileSync(target, 'utf8');
}

export default function ChangelogPage() {
  const source = loadChangelog();
  return (
    <div className="mx-auto max-w-3xl px-6 py-16">
      <MdxRenderer source={source} />
    </div>
  );
}
