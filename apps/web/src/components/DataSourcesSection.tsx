/**
 * Workspace-scoped data-sources section. Replaces the legacy
 * "DataSourcesCard" sidebar widget — now a full-width section with
 * a primary CTA that opens an upload modal, plus a clean list of
 * existing CSVs.
 */
import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { dataSourcesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { Dialog } from './Dialog';
import { AnalyzeDatasetModal } from './AnalyzeDatasetModal';
import { DatasetExploreModal } from './DatasetExploreModal';
import { SampleDatasetsBrowserModal } from './SampleDatasetsBrowserModal';

export interface DataSourcesSectionProps {
  workspaceId: string;
}

function fmtBytes(n: number | null | undefined): string {
  if (n == null) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function DataSourcesSection({ workspaceId }: DataSourcesSectionProps) {
  const qc = useQueryClient();

  const list = useQuery({
    queryKey: ['data-sources', workspaceId],
    queryFn: () => dataSourcesApi.list(workspaceId),
    enabled: !!workspaceId,
  });

  const remove = useMutation({
    mutationFn: (id: string) => dataSourcesApi.remove(id),
    onSuccess: () =>
      qc.invalidateQueries({ queryKey: ['data-sources', workspaceId] }),
  });

  const [uploadOpen, setUploadOpen] = useState(false);
  const [sampleBrowserOpen, setSampleBrowserOpen] = useState(false);
  const [analyzing, setAnalyzing] = useState<{ id: string; name: string } | null>(
    null,
  );
  const [exploring, setExploring] = useState<{ id: string; name: string } | null>(
    null,
  );

  const csvs = (list.data ?? []).filter((d) => d.kind === 'csv_upload');

  return (
    <section>
      <header className="mb-3 flex items-end justify-between gap-4">
        <div>
          <h2 className="h-section">Data sources</h2>
          <p className="text-sm text-ink-500 mt-0.5">
            CSV uploads available to experiments in this workspace.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setSampleBrowserOpen(true)}
            className="btn-secondary"
            title="Browse bundled sample datasets"
          >
            <SparkIcon />
            Browse samples
          </button>
          {csvs.length > 0 && (
            <button
              type="button"
              onClick={() => setUploadOpen(true)}
              className="btn-secondary"
            >
              <PlusIcon />
              Upload CSV
            </button>
          )}
        </div>
      </header>

      {list.isLoading && (
        <div className="card text-sm text-ink-500">Loading…</div>
      )}
      {list.error && (
        <div className="card text-sm text-danger-600">
          {errorMessage(list.error)}
        </div>
      )}

      {list.data && csvs.length === 0 && (
        <EmptyState
          icon={<DataIcon />}
          title="No data sources yet"
          description="Upload a CSV — experiments can then reference it from the data-source picker. 64 MB cap; parsed + SHA-256-checksummed on upload."
          cta={{ label: 'Upload CSV', onClick: () => setUploadOpen(true) }}
        />
      )}

      {csvs.length > 0 && (
        <ul className="rounded-xl bg-white dark:bg-ink-900 border border-ink-200 dark:border-ink-800 shadow-soft-1 divide-y divide-ink-100">
          {csvs.map((d) => {
            const cfg = d.config as {
              rows?: number;
              size_bytes?: number;
              columns?: string[];
            };
            return (
              <li
                key={d.id}
                className="px-4 py-3 flex items-center justify-between gap-3"
              >
                <button
                  type="button"
                  className="min-w-0 flex items-center gap-3 text-left flex-1 rounded-md -mx-2 px-2 py-1 hover:bg-ink-50 dark:hover:bg-ink-800 transition-colors"
                  onClick={() => setExploring({ id: d.id, name: d.name })}
                  title="Open interactive EDA modal"
                >
                  <span className="h-8 w-8 rounded-md bg-ink-100 dark:bg-ink-800 text-ink-600 dark:text-ink-400 dark:text-ink-500 flex items-center justify-center shrink-0">
                    <DataIcon />
                  </span>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-ink-900 dark:text-ink-50 truncate">
                      {d.name}
                    </p>
                    <p className="text-xs text-ink-500 mt-0.5">
                      {cfg.rows != null && (
                        <>{cfg.rows.toLocaleString()} rows · </>
                      )}
                      {fmtBytes(cfg.size_bytes)}
                      {cfg.columns?.length != null && (
                        <> · {cfg.columns.length} columns</>
                      )}
                    </p>
                  </div>
                </button>
                <div className="flex items-center gap-1 shrink-0">
                  <Link
                    to={`/workspaces/${workspaceId}/datasets/${d.id}/profile`}
                    className="btn-ghost text-xs"
                    title="Open EDA dashboard"
                  >
                    <ChartIcon />
                    Explore
                  </Link>
                  <Link
                    to={`/workspaces/${workspaceId}/datasets/${d.id}`}
                    className="btn-ghost text-xs"
                    title="Dataset versions + schema (Phase 4)"
                  >
                    Versions
                  </Link>
                  <button
                    className="btn-ghost text-xs"
                    onClick={() => setAnalyzing({ id: d.id, name: d.name })}
                    title="Analyze with AI"
                  >
                    <SparkIcon />
                    AI
                  </button>
                  <button
                    className="btn-ghost text-xs text-ink-500 hover:text-danger-600"
                    onClick={() => {
                      if (
                        window.confirm(
                          `Delete "${d.name}"? The uploaded file is removed from disk.`,
                        )
                      ) {
                        remove.mutate(d.id);
                      }
                    }}
                    disabled={remove.isPending}
                    title="Delete"
                  >
                    <TrashIcon />
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <UploadDialog
        workspaceId={workspaceId}
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
      />

      {analyzing && (
        <AnalyzeDatasetModal
          workspaceId={workspaceId}
          dataSourceId={analyzing.id}
          dataSourceName={analyzing.name}
          open
          onClose={() => setAnalyzing(null)}
        />
      )}

      {exploring && (
        <DatasetExploreModal
          dataSourceId={exploring.id}
          dataSourceName={exploring.name}
          open
          onClose={() => setExploring(null)}
        />
      )}

      <SampleDatasetsBrowserModal
        workspaceId={workspaceId}
        open={sampleBrowserOpen}
        onClose={() => setSampleBrowserOpen(false)}
      />
    </section>
  );
}

// ─── Upload modal ──────────────────────────────────────────────────

function UploadDialog({
  workspaceId,
  open,
  onClose,
}: {
  workspaceId: string;
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');

  // Reset state on close.
  useEffect(() => {
    if (!open) {
      setFile(null);
      setName('');
      if (fileRef.current) fileRef.current.value = '';
    }
  }, [open]);

  const upload = useMutation({
    mutationFn: () => {
      if (!file) throw new Error('Pick a CSV file first.');
      return dataSourcesApi.uploadCsv(
        workspaceId,
        file,
        name.trim() || file.name,
      );
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['data-sources', workspaceId] });
      onClose();
    },
  });

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Upload CSV"
      description="Pick a CSV file from disk. 64 MB cap; the upload is parsed + SHA-256-checksummed."
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (file) upload.mutate();
        }}
        className="space-y-4"
      >
        <div>
          <label className="field" htmlFor="csv-file">
            File
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
            className="text-sm text-ink-700 dark:text-ink-300 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border file:border-ink-200 file:bg-white dark:bg-ink-900 file:text-ink-800 dark:text-ink-100 file:cursor-pointer file:font-medium hover:file:bg-ink-50 dark:bg-ink-950"
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
        <div className="flex items-center justify-end gap-2 pt-2">
          <button
            type="button"
            onClick={onClose}
            className="btn-ghost"
          >
            Cancel
          </button>
          <button
            type="submit"
            className="btn-primary"
            disabled={!file || upload.isPending}
          >
            {upload.isPending ? 'Uploading…' : 'Upload'}
          </button>
        </div>
      </form>
    </Dialog>
  );
}

// ─── Empty-state primitive ─────────────────────────────────────────

function EmptyState({
  icon,
  title,
  description,
  cta,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  cta: { label: string; onClick: () => void };
}) {
  return (
    <div className="rounded-xl bg-white dark:bg-ink-900 border border-dashed border-ink-300 dark:border-ink-700 px-6 py-10 text-center">
      <div className="mx-auto h-10 w-10 rounded-lg bg-ink-100 dark:bg-ink-800 text-ink-500 flex items-center justify-center mb-3">
        {icon}
      </div>
      <h3 className="text-sm font-semibold text-ink-900 dark:text-ink-50">{title}</h3>
      <p className="mt-1 text-sm text-ink-500 max-w-md mx-auto">{description}</p>
      <button
        type="button"
        onClick={cta.onClick}
        className="btn-primary mt-5"
      >
        <PlusIcon />
        {cta.label}
      </button>
    </div>
  );
}

// ─── Icons ─────────────────────────────────────────────────────────

const sx = {
  width: '14',
  height: '14',
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: '2',
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
};

function PlusIcon() {
  return (
    <svg {...sx}>
      <path d="M12 5v14 M5 12h14" />
    </svg>
  );
}
function DataIcon() {
  return (
    <svg {...sx}>
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14a9 3 0 0 0 18 0V5" />
      <path d="M3 12a9 3 0 0 0 18 0" />
    </svg>
  );
}
function SparkIcon() {
  return (
    <svg {...sx}>
      <path d="M12 3v18 M3 12h18" />
    </svg>
  );
}
function TrashIcon() {
  return (
    <svg {...sx}>
      <path d="M3 6h18 M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6 m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  );
}
function ChartIcon() {
  return (
    <svg {...sx}>
      <path d="M3 3v18h18 M7 14l3-3 4 4 5-6" />
    </svg>
  );
}
