/**
 * Superuser-only backup + restore controls.
 *
 * - "Download backup" → GET /api/v1/admin/backup as a tarball.
 * - "Restore from file" → POST /api/v1/admin/restore (multipart) with
 *   confirm=true. Wipes the existing DB + artifact dir.
 *
 * Lives on AdminUsers since it's already gated on superuser.
 */

import { useRef, useState } from 'react';
import { useAuthStore } from '@/state/auth';

const API_BASE = '/api/v1';

export function BackupRestoreCard() {
  const accessToken = useAuthStore((s) => s.accessToken);
  const [busy, setBusy] = useState<'idle' | 'backing-up' | 'restoring'>('idle');
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function downloadBackup() {
    setBusy('backing-up');
    setError(null);
    setMessage(null);
    try {
      const res = await fetch(`${API_BASE}/admin/backup`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `pycaret-backup-${new Date().toISOString().replace(/[:.]/g, '-')}.tar.gz`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setMessage('Backup downloaded.');
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy('idle');
    }
  }

  async function restoreBackup(file: File) {
    if (
      !confirm(
        `Restore from "${file.name}"?\n\nThis WIPES the current database + artifact dir and replaces them.`,
      )
    ) {
      return;
    }
    setBusy('restoring');
    setError(null);
    setMessage(null);
    try {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('confirm', 'true');
      const res = await fetch(`${API_BASE}/admin/restore`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${accessToken}` },
        body: fd,
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      }
      const result = await res.json();
      const total = Object.values(result.restored_rows ?? {}).reduce<number>(
        (acc, n) => acc + (n as number),
        0,
      );
      setMessage(
        `Restored ${total} rows across ${Object.keys(result.restored_rows ?? {}).length} tables. Reload to refresh the UI state.`,
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy('idle');
      if (fileRef.current) fileRef.current.value = '';
    }
  }

  return (
    <section className="card">
      <h2 className="h-section mb-2">Backup &amp; restore</h2>
      <p className="text-sm text-ink-500 mb-4">
        Tarball includes the entire database + every artifact under the
        configured artifact directory. Sessions are not backed up.
      </p>
      <div className="flex flex-wrap gap-2">
        <button
          className="btn-secondary"
          disabled={busy !== 'idle'}
          onClick={downloadBackup}
        >
          {busy === 'backing-up' ? 'Building…' : 'Download backup (.tar.gz)'}
        </button>
        <label className="btn-secondary cursor-pointer">
          {busy === 'restoring' ? 'Restoring…' : 'Restore from file…'}
          <input
            ref={fileRef}
            type="file"
            accept=".tar.gz,application/gzip"
            className="hidden"
            disabled={busy !== 'idle'}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) restoreBackup(f);
            }}
          />
        </label>
      </div>
      {message && (
        <p className="mt-3 text-sm text-success-600 dark:text-success-500">
          {message}
        </p>
      )}
      {error && <p className="mt-3 text-sm text-danger-600">{error}</p>}
    </section>
  );
}
