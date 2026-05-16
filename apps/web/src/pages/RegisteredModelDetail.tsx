/**
 * /workspaces/:wsId/models/:modelId — Phase 7 version history.
 *
 * Lists every RegisteredModelVersion sorted by version desc.
 * Per-row: status pill (staging / production / archived), metrics
 * digest, promote-to-production button, rollback button. The
 * production version is highlighted; only one production version
 * exists at a time (the API enforces it).
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { BackButton } from '@/components/BackButton';
import { Dialog } from '@/components/Dialog';
import { errorMessage } from '@/api/client';
import { deploymentsApi, governanceApi, registryApi } from '@/api/endpoints';
import type {
  Deployment,
  RegisteredModelVersion,
  RegisteredModelVersionStatus,
} from '@/api/types';

// Mirrors services/api/pycaret_server/api/deployments.py:_SLUG_RE.
const SLUG_RE = /^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$/;

const STATUS_PILL: Record<RegisteredModelVersionStatus, string> = {
  staging: 'pill-accent',
  production: 'pill-success',
  archived: 'pill-neutral',
};

export function RegisteredModelDetail() {
  const { wsId = '', modelId = '' } = useParams<{
    wsId: string;
    modelId: string;
  }>();
  const qc = useQueryClient();

  const model = useQuery({
    queryKey: ['registered-models', 'detail', modelId],
    queryFn: () => registryApi.get(modelId),
    enabled: !!modelId,
  });
  const versions = useQuery({
    queryKey: ['registered-models', 'versions', modelId],
    queryFn: () => registryApi.versions(modelId),
    enabled: !!modelId,
  });

  // All workspace deployments — we filter by registered_model_id below so
  // the user can see which versions are live without leaving the page.
  // (Mirrors the inline "Deployments" panel the old PipelineDetail had,
  // now folded into the Model Registry as part of the session-56 UI merge.)
  const deployments = useQuery({
    queryKey: ['deployments', 'by-workspace', wsId],
    queryFn: () => deploymentsApi.list(wsId),
    enabled: !!wsId,
  });
  const modelDeployments: Deployment[] = (deployments.data ?? []).filter(
    (d) => d.registered_model_id === modelId,
  );
  const deploymentsByVersion = new Map<string, Deployment[]>();
  for (const d of modelDeployments) {
    if (!d.registered_model_version_id) continue;
    const arr = deploymentsByVersion.get(d.registered_model_version_id) ?? [];
    arr.push(d);
    deploymentsByVersion.set(d.registered_model_version_id, arr);
  }

  const [deployingVersion, setDeployingVersion] =
    useState<RegisteredModelVersion | null>(null);

  const setStatus = useMutation({
    mutationFn: ({
      versionId,
      status,
    }: {
      versionId: string;
      status: RegisteredModelVersionStatus;
    }) =>
      registryApi.setStatus(modelId, versionId, {
        status,
        set_current: status === 'production',
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['registered-models', 'versions', modelId] });
      qc.invalidateQueries({ queryKey: ['registered-models', 'detail', modelId] });
    },
  });

  const rollback = useMutation({
    mutationFn: (versionId: string) => registryApi.rollback(modelId, versionId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['registered-models', 'versions', modelId] });
      qc.invalidateQueries({ queryKey: ['registered-models', 'detail', modelId] });
    },
  });

  // Phase 12: gated promote-to-production via an ApprovalWorkflow.
  // The requester auto-signs as the first approver; with the default
  // ``required_approvals=1`` it lands in "approved" immediately, so the
  // next user step is to hit Execute from the Approvals inbox.
  const nav = useNavigate();
  const [requestedFor, setRequestedFor] = useState<{
    versionId: string;
    approvalId: string;
    status: string;
    required: number;
    signatures: number;
  } | null>(null);
  const requestPromotion = useMutation({
    mutationFn: (versionId: string) =>
      registryApi.requestPromotion(modelId, versionId).then((res) => ({
        versionId,
        ...res,
      })),
    onSuccess: (res) =>
      setRequestedFor({
        versionId: res.versionId,
        approvalId: res.id,
        status: res.status,
        required: res.required_approvals,
        signatures: (res.approvals ?? []).length,
      }),
  });
  const executeApproval = useMutation({
    mutationFn: (approvalId: string) => governanceApi.execute(approvalId),
    onSuccess: () => {
      setRequestedFor(null);
      qc.invalidateQueries({ queryKey: ['registered-models', 'versions', modelId] });
      qc.invalidateQueries({ queryKey: ['registered-models', 'detail', modelId] });
    },
  });

  return (
    <div className="space-y-6">
      <header>
        <BackButton />
        <h1 className="h-page mt-2">{model.data?.name ?? '…'}</h1>
        {model.data?.description && (
          <p className="muted small mt-1">{model.data.description}</p>
        )}
        <p className="text-xs text-ink-500 mt-2">
          Current version:{' '}
          {model.data?.current_version_id ? (
            <code>{model.data.current_version_id.slice(0, 8)}…</code>
          ) : (
            'none'
          )}{' '}
          ·{' '}
          <Link to={`/workspaces/${wsId}/models`} className="hover:underline">
            back to list
          </Link>
        </p>
      </header>

      {/* Phase 12 approval banner — shows after the user has opened
          a "request promotion" workflow on a version. Either auto-
          approved (1 sig) ⇒ Execute, or pending more signers. */}
      {requestedFor && (
        <div
          className={`card text-sm ${
            requestedFor.status === 'approved'
              ? 'border-success-400'
              : 'border-accent-400'
          }`}
        >
          <p className="font-medium">
            Promotion {requestedFor.status === 'approved' ? 'approved' : 'pending approval'}
          </p>
          <p className="muted small mt-1">
            {requestedFor.signatures} of {requestedFor.required} signatures
            {requestedFor.status === 'approved'
              ? ' · ready to execute.'
              : ' · awaiting more approvers in the inbox.'}
          </p>
          <div className="flex gap-2 mt-3">
            {requestedFor.status === 'approved' && (
              <button
                type="button"
                className="btn-primary text-xs"
                onClick={() => executeApproval.mutate(requestedFor.approvalId)}
                disabled={executeApproval.isPending}
              >
                {executeApproval.isPending ? 'Executing…' : 'Execute promotion'}
              </button>
            )}
            <button
              type="button"
              className="btn-secondary text-xs"
              onClick={() => nav(`/workspaces/${wsId}/approvals`)}
            >
              Open approvals inbox
            </button>
            <button
              type="button"
              className="btn-secondary text-xs"
              onClick={() => setRequestedFor(null)}
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      <section>
        <h2 className="h-section mb-3">
          Versions ({versions.data?.length ?? 0})
        </h2>
        {versions.isLoading && (
          <div className="card text-sm text-ink-500">Loading…</div>
        )}
        {versions.data && versions.data.length === 0 && (
          <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-8 text-center">
            <p className="text-sm text-ink-500">
              No versions yet. Promote a Trial from a Run's detail page
              to create the first version.
            </p>
          </div>
        )}
        {versions.data && versions.data.length > 0 && (
          <ul className="space-y-2">
            {versions.data.map((v) => (
              <VersionRow
                key={v.id}
                version={v}
                isCurrent={v.id === model.data?.current_version_id}
                deployments={deploymentsByVersion.get(v.id) ?? []}
                onPromote={() => requestPromotion.mutate(v.id)}
                onArchive={() =>
                  setStatus.mutate({
                    versionId: v.id,
                    // Archive bypasses the approval gate — it's not
                    // adding production exposure.
                    status: 'archived',
                  })
                }
                onRollback={() => rollback.mutate(v.id)}
                onDeploy={() => setDeployingVersion(v)}
                disabled={
                  setStatus.isPending ||
                  rollback.isPending ||
                  requestPromotion.isPending
                }
              />
            ))}
          </ul>
        )}
      </section>

      {deployingVersion && (
        <DeployDialog
          modelId={modelId}
          version={deployingVersion}
          onClose={() => setDeployingVersion(null)}
          onDeployed={() => {
            setDeployingVersion(null);
            qc.invalidateQueries({
              queryKey: ['deployments', 'by-workspace', wsId],
            });
          }}
        />
      )}
    </div>
  );
}

function VersionRow({
  version,
  isCurrent,
  deployments,
  onPromote,
  onArchive,
  onRollback,
  onDeploy,
  disabled,
}: {
  version: RegisteredModelVersion;
  isCurrent: boolean;
  deployments: Deployment[];
  onPromote: () => void;
  onArchive: () => void;
  onRollback: () => void;
  onDeploy: () => void;
  disabled: boolean;
}) {
  const metrics = version.metrics ?? {};
  const primary =
    metrics.Accuracy ?? metrics.AUC ?? metrics.F1 ?? metrics.R2 ?? null;

  return (
    <li
      className={`card flex items-start gap-4 ${
        isCurrent ? 'ring-1 ring-success-400 dark:ring-success-500' : ''
      }`}
    >
      <div className="shrink-0">
        <p className="text-xs text-ink-500">v</p>
        <p className="text-2xl font-bold tabular-nums">{version.version}</p>
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={STATUS_PILL[version.status]}>{version.status}</span>
          {isCurrent && (
            <span className="pill-success">current</span>
          )}
          <code className="text-xs text-ink-500">
            {version.id.slice(0, 8)}…
          </code>
        </div>
        <p className="text-xs text-ink-500 mt-1">
          {version.promoted_at
            ? `promoted ${new Date(version.promoted_at).toLocaleString()}`
            : 'not promoted'}
          {version.promoted_by ? ` · by ${version.promoted_by.slice(0, 8)}…` : ''}
        </p>
        {primary !== null && (
          <p className="text-xs text-ink-700 dark:text-ink-300 mt-1 tabular-nums">
            primary metric: <strong>{Number(primary).toFixed(4)}</strong>
          </p>
        )}
        {version.notes && (
          <p className="text-xs text-ink-600 dark:text-ink-300 mt-1">
            {version.notes}
          </p>
        )}
        {deployments.length > 0 && (
          <div className="mt-2 flex items-center gap-1.5 flex-wrap">
            <span className="text-[10px] uppercase tracking-wider text-ink-500 mr-1">
              Live
            </span>
            {deployments.map((d) => (
              <Link
                key={d.id}
                to={`/deployments/${d.id}`}
                className="text-[11px] px-2 py-0.5 rounded bg-success-500/15 text-success-700 dark:text-success-300 hover:opacity-80 font-mono"
                title={`${d.status} · ${d.endpoint_slug}`}
              >
                /{d.endpoint_slug}
              </Link>
            ))}
          </div>
        )}
      </div>
      <div className="shrink-0 flex flex-col gap-1">
        <button
          type="button"
          className="btn-secondary text-xs"
          onClick={onDeploy}
          disabled={disabled}
          title="Create a new serving endpoint pointing at this version"
        >
          Deploy
        </button>
        {version.status !== 'production' && (
          <button
            type="button"
            className="btn-primary text-xs"
            onClick={onPromote}
            disabled={disabled}
            title="Opens an approval workflow. Single-signer setups land immediately at 'approved'; multi-signer workflows wait for inbox sign-off."
          >
            Request promote
          </button>
        )}
        {version.status === 'archived' && (
          <button
            type="button"
            className="btn-secondary text-xs"
            onClick={onRollback}
            disabled={disabled}
          >
            Roll back to this
          </button>
        )}
        {version.status === 'production' && (
          <button
            type="button"
            className="btn-secondary text-xs"
            onClick={onArchive}
            disabled={disabled}
          >
            Archive
          </button>
        )}
      </div>
    </li>
  );
}

// ─── Deploy dialog (per-version) ─────────────────────────────────────

function DeployDialog({
  modelId,
  version,
  onClose,
  onDeployed,
}: {
  modelId: string;
  version: RegisteredModelVersion;
  onClose: () => void;
  onDeployed: () => void;
}) {
  const [slug, setSlug] = useState('');
  const [authMode, setAuthMode] = useState<'workspace' | 'api-key' | 'public'>(
    'workspace',
  );

  const deploy = useMutation({
    mutationFn: () =>
      registryApi.deploy(modelId, version.id, {
        endpoint_slug: slug,
        auth_mode: authMode,
      }),
    onSuccess: () => onDeployed(),
  });

  const slugValid = SLUG_RE.test(slug);

  return (
    <Dialog
      open
      onClose={onClose}
      size="md"
      title={`Deploy v${version.version}`}
      description="Create a new serving endpoint pointing at this version. Endpoint slug must be globally unique."
    >
      <div className="space-y-4">
        <div>
          <label className="field" htmlFor="dep-slug">
            Endpoint slug <span className="text-danger-500">*</span>
          </label>
          <input
            id="dep-slug"
            className="input"
            value={slug}
            onChange={(e) => setSlug(e.target.value.toLowerCase())}
            placeholder="iris-v1"
            required
            autoFocus
          />
          <p className="hint mt-1">
            Lowercase letters, digits, dashes. 3–64 chars.
          </p>
          {slug && !slugValid && (
            <p className="error mt-1">
              Invalid — must match{' '}
              <code className="font-mono">[a-z0-9][a-z0-9-]{'{1,62}'}[a-z0-9]</code>.
            </p>
          )}
        </div>

        <div>
          <label className="field" htmlFor="dep-auth">
            Auth mode
          </label>
          <select
            id="dep-auth"
            className="input"
            value={authMode}
            onChange={(e) =>
              setAuthMode(e.target.value as 'workspace' | 'api-key' | 'public')
            }
          >
            <option value="workspace">workspace — JWT of a workspace member</option>
            <option value="api-key" disabled>
              api-key — (V2)
            </option>
            <option value="public" disabled>
              public — (V2)
            </option>
          </select>
        </div>

        {deploy.error && <p className="error">{errorMessage(deploy.error)}</p>}

        <div className="flex items-center justify-end gap-2 pt-2">
          <button type="button" onClick={onClose} className="btn-secondary">
            Cancel
          </button>
          <button
            type="button"
            onClick={() => deploy.mutate()}
            disabled={!slugValid || deploy.isPending}
            className="btn-primary"
          >
            {deploy.isPending ? 'Deploying…' : 'Deploy'}
          </button>
        </div>
      </div>
    </Dialog>
  );
}
