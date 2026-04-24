/**
 * Workspace-scoped data-sources card. Lives in the sidebar of
 * WorkspaceDetail.tsx.
 *
 * Two responsibilities:
 *  1. List existing CSV uploads in the workspace (most recent first).
 *  2. Offer a tiny CSV upload form — file picker + name, hits the existing
 *     `POST /workspaces/:id/data-sources/upload` multipart endpoint.
 *
 * Registering S3 / Postgres connectors is deferred — the UI for those is a
 * separate "integrations" concern the spec tracks in its own section.
 */

import { useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { dataSourcesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { AnalyzeDatasetModal } from './AnalyzeDatasetModal';

export interface DataSourcesCardProps {
  workspaceId: string;
}

function fmtBytes(n: number | null | undefined): string {
  if (n == null) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function DataSourcesCard({ workspaceId }: DataSourcesCardProps) {
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement | null>(null);

  const list = useQuery({
    queryKey: ['data-sources', workspaceId],
    queryFn: () => dataSourcesApi.list(workspaceId),
    enabled: !!workspaceId,
  });

  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  // When set, renders the AnalyzeDatasetModal for this data source.
  const [analyzing, setAnalyzing] = useState<{ id: string; name: string } | null>(null);

  const upload = useMutation({
    mutationFn: () => {
      if (!file) throw new Error('Pick a CSV file first.');
      return dataSourcesApi.uploadCsv(workspaceId, file, name.trim() || file.name);
    },
    onSuccess: () => {
      setFile(null);
      setName('');
      if (fileRef.current) fileRef.current.value = '';
      qc.invalidateQueries({ queryKey: ['data-sources', workspaceId] });
    },
  });

  const remove = useMutation({
    mutationFn: (id: string) => dataSourcesApi.remove(id),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['data-sources', workspaceId] }),
  });

  const csvs = (list.data ?? []).filter((d) => d.kind === 'csv_upload');

  return (
    <div className="card space-y-4">
      <header className="flex items-baseline justify-between">
        <h2 className="text-sm font-medium text-ink-100">Data sources</h2>
        <span className="hint">{csvs.length} CSV</span>
      </header>

      {csvs.length === 0 ? (
        <p className="hint">
          No CSV uploads yet. Pick a file below; experiments can then reference it
          from the data-source picker.
        </p>
      ) : (
        <ul className="divide-y divide-ink-800 -mx-5">
          {csvs.map((d) => {
            const cfg = d.config as {
              rows?: number;
              size_bytes?: number;
              columns?: string[];
            };
            return (
              <li key={d.id} className="px-5 py-3 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="font-medium text-ink-100 truncate">{d.name}</p>
                  <p className="text-xs text-ink-200/60 mt-0.5">
                    {cfg.rows != null && <>{cfg.rows.toLocaleString()} rows · </>}
                    {fmtBytes(cfg.size_bytes)}
                    {cfg.columns?.length != null && <> · {cfg.columns.length} cols</>}
                  </p>
                </div>
                <div className="flex items-center gap-1 shrink-0">
                  <button
                    className="btn-ghost text-xs"
                    onClick={() => setAnalyzing({ id: d.id, name: d.name })}
                    title="Analyze with AI"
                  >
                    ✨ AI
                  </button>
                  <button
                    className="btn-ghost text-xs"
                    onClick={() => {
                      if (window.confirm(`Delete "${d.name}"? The uploaded file is removed from disk.`)) {
                        remove.mutate(d.id);
                      }
                    }}
                    disabled={remove.isPending}
                    title="Delete"
                  >
                    ✕
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      {/* ────────── upload */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (file) upload.mutate();
        }}
        className="space-y-3 border-t border-ink-800 pt-4"
      >
        <div>
          <label className="field" htmlFor="csv-file">
            Upload CSV
          </label>
          <input
            id="csv-file"
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => {
              const f = e.target.files?.[0] ?? null;
              setFile(f);
              if (f && !name) setName(f.name);
            }}
            className="text-sm text-ink-200/80 file:mr-3 file:py-1 file:px-3 file:rounded file:border-0 file:bg-ink-800 file:text-ink-100 file:cursor-pointer hover:file:bg-ink-700"
          />
        </div>
        <div>
          <label className="field" htmlFor="ds-name">
            Name
          </label>
          <input
            id="ds-name"
            className="input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={file?.name ?? 'my-dataset'}
          />
        </div>
        {upload.error && <p className="error">{errorMessage(upload.error)}</p>}
        <button
          type="submit"
          className="btn-primary w-full"
          disabled={!file || upload.isPending}
        >
          {upload.isPending ? 'Uploading…' : 'Upload'}
        </button>
        <p className="hint">64 MB cap. CSV is parsed + SHA-256-checksummed on upload.</p>
      </form>

      {analyzing && (
        <AnalyzeDatasetModal
          workspaceId={workspaceId}
          dataSourceId={analyzing.id}
          dataSourceName={analyzing.name}
          open
          onClose={() => setAnalyzing(null)}
        />
      )}
    </div>
  );
}
