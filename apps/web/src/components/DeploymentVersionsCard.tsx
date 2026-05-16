/**
 * Pipeline-versions panel + rollback control for a deployment.
 *
 * Lists every Pipeline in the same family as the deployment's current
 * pipeline. Lets the user repoint the deployment at a different version.
 * The in-memory registry is evicted server-side; the next /predict
 * reloads the new artifact.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { deploymentsApi, pipelinesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import type { Pipeline } from '@/api/types';

export interface DeploymentVersionsCardProps {
  deploymentId: string;
  currentPipelineId: string | null;
}

export function DeploymentVersionsCard({
  deploymentId,
  currentPipelineId,
}: DeploymentVersionsCardProps) {
  const qc = useQueryClient();
  const versions = useQuery({
    queryKey: ['pipelines', currentPipelineId, 'versions'],
    queryFn: () => pipelinesApi.versions(currentPipelineId!),
    enabled: !!currentPipelineId,
  });

  const rollback = useMutation({
    mutationFn: (pipeline_id: string) =>
      deploymentsApi.rollback(deploymentId, { pipeline_id }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['deployments', deploymentId] });
      qc.invalidateQueries({
        queryKey: ['pipelines', currentPipelineId, 'versions'],
      });
    },
  });

  if (versions.isPending) {
    return (
      <section>
        <h2 className="h-section mb-3">Versions</h2>
        <div className="card text-sm text-ink-500">Loading versions…</div>
      </section>
    );
  }
  const items: Pipeline[] = versions.data?.items ?? [];
  if (items.length <= 1) {
    return (
      <section>
        <h2 className="h-section mb-3">Versions</h2>
        <p className="text-sm text-ink-500">
          Only one version of this pipeline exists. Promote a newer run with
          the same name to enable rollback.
        </p>
      </section>
    );
  }

  return (
    <section>
      <h2 className="h-section mb-3">Versions</h2>
      <div className="card overflow-hidden p-0">
        <table className="w-full text-sm">
          <thead className="bg-white text-ink-500 dark:bg-ink-900">
            <tr>
              <th className="px-4 py-2 text-left font-medium w-12">v</th>
              <th className="px-4 py-2 text-left font-medium">Model</th>
              <th className="px-4 py-2 text-left font-medium">Created</th>
              <th className="px-4 py-2 text-left font-medium">SHA-256</th>
              <th className="px-4 py-2 text-right font-medium">Action</th>
            </tr>
          </thead>
          <tbody>
            {items.map((p) => {
              const isCurrent = p.id === currentPipelineId;
              return (
                <tr
                  key={p.id}
                  className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
                >
                  <td className="px-4 py-2 font-mono text-ink-900 dark:text-ink-50">
                    v{p.version}
                  </td>
                  <td className="px-4 py-2 text-ink-700 dark:text-ink-300">
                    {p.model_id ?? '—'}
                  </td>
                  <td className="px-4 py-2 text-xs text-ink-500 whitespace-nowrap">
                    {new Date(p.created_at).toLocaleString()}
                  </td>
                  <td
                    className="px-4 py-2 font-mono text-xs text-ink-500 max-w-[12rem] truncate"
                    title={p.sha256 ?? ''}
                  >
                    {p.sha256 ? `${p.sha256.slice(0, 12)}…` : '—'}
                  </td>
                  <td className="px-4 py-2 text-right">
                    {isCurrent ? (
                      <span className="pill-success">active</span>
                    ) : (
                      <button
                        className="text-xs text-accent-600 hover:underline"
                        disabled={rollback.isPending}
                        onClick={() => {
                          if (confirm(`Rollback to v${p.version}?`)) {
                            rollback.mutate(p.id);
                          }
                        }}
                      >
                        Rollback to v{p.version}
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rollback.error && (
        <p className="text-sm text-danger-600 mt-2">{errorMessage(rollback.error)}</p>
      )}
    </section>
  );
}
