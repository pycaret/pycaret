/**
 * Deploy a Pipeline behind a slug — invokable from the trial detail and
 * run detail pages so users don't have to bounce through the Pipelines /
 * Deployments tabs after promoting.
 *
 * Same shape as the inline form on PipelineDetail.tsx, lifted into a
 * dialog so it can be triggered from anywhere a Pipeline id is known.
 */

import { useEffect, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { deploymentsApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { Dialog } from './Dialog';

const SLUG_RE = /^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$/;

export function DeployFromPipelineDialog({
  open,
  onClose,
  pipelineId,
  pipelineName,
}: {
  open: boolean;
  onClose: () => void;
  pipelineId: string;
  pipelineName?: string | null;
}) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [slug, setSlug] = useState('');
  const [authMode, setAuthMode] = useState<'workspace' | 'api-key' | 'public'>(
    'workspace',
  );

  // Seed a sensible default slug from the pipeline name on open.
  useEffect(() => {
    if (open && pipelineName && !slug) {
      const seeded = pipelineName
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 60);
      if (SLUG_RE.test(seeded)) {
        setSlug(seeded);
      }
    }
    if (!open) {
      setSlug('');
      setAuthMode('workspace');
    }
    // intentionally only on open transition
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const slugValid = SLUG_RE.test(slug);
  const deploy = useMutation({
    mutationFn: () =>
      deploymentsApi.create(pipelineId, {
        endpoint_slug: slug,
        auth_mode: authMode,
      }),
    onSuccess: (d) => {
      qc.invalidateQueries({ queryKey: ['deployments'] });
      onClose();
      navigate(`/deployments/${d.id}`);
    },
  });

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={`Deploy ${pipelineName ?? 'pipeline'}`}
      description="Spin up a slug-addressable serving endpoint backed by this pipeline."
    >
      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (slugValid) deploy.mutate();
        }}
      >
        <div>
          <label className="block text-xs text-ink-500 mb-1">Endpoint slug</label>
          <input
            className="input"
            value={slug}
            onChange={(e) =>
              setSlug(e.target.value.toLowerCase().replace(/\s+/g, '-'))
            }
            placeholder="iris-prod"
            required
          />
          <p className="text-[11px] text-ink-500 mt-1">
            Lowercase letters, digits, and dashes (3–64 chars). Reachable at{' '}
            <code className="font-mono">
              POST /api/v1/deployments/{slug || '<slug>'}/predict
            </code>
            .
          </p>
        </div>
        <div>
          <label className="block text-xs text-ink-500 mb-1">Auth mode</label>
          <select
            className="input"
            value={authMode}
            onChange={(e) =>
              setAuthMode(e.target.value as 'workspace' | 'api-key' | 'public')
            }
          >
            <option value="workspace">Workspace (member token)</option>
            <option value="api-key">API key</option>
            <option value="public">Public (no auth)</option>
          </select>
        </div>
        {deploy.error && (
          <p className="text-xs text-danger-600">{errorMessage(deploy.error)}</p>
        )}
        <div className="flex items-center justify-end gap-2 pt-2">
          <button
            type="button"
            className="btn-secondary"
            onClick={onClose}
            data-dialog-close
          >
            Cancel
          </button>
          <button
            type="submit"
            className="btn-primary"
            disabled={!slugValid || deploy.isPending}
          >
            {deploy.isPending ? 'Deploying…' : 'Deploy'}
          </button>
        </div>
      </form>
    </Dialog>
  );
}
