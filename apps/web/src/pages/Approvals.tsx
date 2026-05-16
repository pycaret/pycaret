/**
 * /workspaces/:wsId/approvals — Phase 12 governance inbox.
 *
 * Two-column layout: pending list on the left, detail pane on the
 * right (signature progress bar, request payload, action buttons).
 * Approving / rejecting / executing fires the matching governanceApi
 * call and refreshes the list.
 */

import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { governanceApi } from '@/api/endpoints';
import { BackButton } from '@/components/BackButton';
import type { ApprovalStatus, ApprovalWorkflow } from '@/api/types';

const STATUS_PILL: Record<ApprovalStatus, string> = {
  pending: 'pill-accent',
  approved: 'pill-success',
  rejected: 'pill bg-rose-50 text-rose-700 dark:bg-rose-500/15 dark:text-rose-300',
  executed: 'pill-success',
  cancelled: 'pill-neutral',
};

export function Approvals() {
  const { wsId = '' } = useParams<{ wsId: string }>();
  const qc = useQueryClient();
  const [filter, setFilter] = useState<ApprovalStatus | ''>('pending');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [comment, setComment] = useState('');

  const list = useQuery({
    queryKey: ['approvals', wsId, filter || 'all'],
    queryFn: () => governanceApi.list(wsId, filter || undefined),
    enabled: !!wsId,
  });

  const selected = useMemo(
    () => list.data?.find((a) => a.id === selectedId) ?? null,
    [list.data, selectedId],
  );

  const onSettled = () => {
    qc.invalidateQueries({ queryKey: ['approvals', wsId] });
    setComment('');
  };

  const approve = useMutation({
    mutationFn: (id: string) => governanceApi.approve(id, { comment: comment || undefined }),
    onSettled,
  });
  const reject = useMutation({
    mutationFn: (id: string) => governanceApi.reject(id, { comment: comment || undefined }),
    onSettled,
  });
  const exec = useMutation({
    mutationFn: (id: string) => governanceApi.execute(id),
    onSettled,
  });

  return (
    <div className="space-y-6">
      <header>
        <BackButton />
        <h1 className="h-page mt-2">Approvals</h1>
        <p className="muted small">
          Governance gates for sensitive actions (Phase 12). Approve,
          reject, or execute pending requests.
        </p>
      </header>

      <div className="flex gap-2">
        {(['pending', 'approved', 'executed', 'rejected', 'cancelled', ''] as const).map(
          (f) => (
            <button
              key={f || 'all'}
              type="button"
              onClick={() => setFilter(f)}
              className={`pill ${
                filter === f
                  ? 'bg-accent-500 text-white'
                  : 'bg-ink-100 text-ink-700 dark:bg-ink-800 dark:text-ink-300'
              }`}
            >
              {f || 'all'}
            </button>
          ),
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <section>
          <h2 className="h-section mb-3">
            {list.data?.length ?? 0} request{(list.data?.length ?? 0) === 1 ? '' : 's'}
          </h2>
          {list.isLoading && <div className="card text-sm text-ink-500">Loading…</div>}
          {list.data && list.data.length === 0 && (
            <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-8 text-center">
              <p className="text-sm text-ink-500">No approvals in this state.</p>
            </div>
          )}
          {list.data && list.data.length > 0 && (
            <ul className="space-y-2">
              {list.data.map((a) => (
                <ApprovalRow
                  key={a.id}
                  approval={a}
                  active={a.id === selectedId}
                  onClick={() => setSelectedId(a.id)}
                />
              ))}
            </ul>
          )}
        </section>

        <section>
          {selected ? (
            <ApprovalDetail
              approval={selected}
              comment={comment}
              setComment={setComment}
              onApprove={() => approve.mutate(selected.id)}
              onReject={() => reject.mutate(selected.id)}
              onExecute={() => exec.mutate(selected.id)}
              isPending={approve.isPending || reject.isPending || exec.isPending}
            />
          ) : (
            <div className="rounded-xl border border-dashed border-ink-300 dark:border-ink-700 p-8 text-center">
              <p className="text-sm text-ink-500">Pick a request to inspect.</p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function ApprovalRow({
  approval,
  active,
  onClick,
}: {
  approval: ApprovalWorkflow;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        className={`w-full text-left card p-4 transition-colors ${
          active
            ? 'ring-2 ring-accent-500'
            : 'hover:bg-ink-50 dark:hover:bg-ink-950/40'
        }`}
      >
        <div className="flex items-center justify-between gap-2">
          <span className="font-medium">{approval.action}</span>
          <span className={STATUS_PILL[approval.status]}>{approval.status}</span>
        </div>
        <p className="text-xs text-ink-500 mt-1">
          {approval.target_kind}
          {approval.target_id ? ` · ${approval.target_id.slice(0, 8)}…` : ''}
        </p>
        <p className="text-xs text-ink-500 mt-1">
          {(approval.approvals ?? []).length}/{approval.required_approvals}{' '}
          signatures
        </p>
      </button>
    </li>
  );
}

function ApprovalDetail({
  approval,
  comment,
  setComment,
  onApprove,
  onReject,
  onExecute,
  isPending,
}: {
  approval: ApprovalWorkflow;
  comment: string;
  setComment: (v: string) => void;
  onApprove: () => void;
  onReject: () => void;
  onExecute: () => void;
  isPending: boolean;
}) {
  const sigs = approval.approvals ?? [];
  return (
    <div className="card space-y-4">
      <header>
        <h3 className="text-base font-semibold">{approval.action}</h3>
        <p className="text-xs text-ink-500 mt-1">
          Target: <code>{approval.target_kind}</code>{' '}
          {approval.target_id ? <code>{approval.target_id}</code> : null}
        </p>
      </header>

      <div>
        <p className="text-xs text-ink-500 mb-1">Signatures</p>
        <div className="space-y-1">
          {sigs.length === 0 && (
            <p className="text-sm text-ink-400">No signatures yet.</p>
          )}
          {sigs.map((s, i) => (
            <div
              key={i}
              className="text-xs text-ink-600 dark:text-ink-300 border-l-2 border-accent-500 pl-2"
            >
              {s.user_id} · {s.approved_at ? 'approved' : 'rejected'}
              {s.comment ? ` — ${s.comment}` : ''}
            </div>
          ))}
        </div>
      </div>

      {approval.request_payload && (
        <details>
          <summary className="text-xs text-ink-500 cursor-pointer">
            request payload
          </summary>
          <pre className="mt-2 text-xs bg-ink-50 dark:bg-ink-950 p-2 rounded">
            {JSON.stringify(approval.request_payload, null, 2)}
          </pre>
        </details>
      )}

      {approval.status === 'pending' && (
        <div className="space-y-2">
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Comment (optional)…"
            className="input w-full"
            rows={2}
          />
          <div className="flex gap-2">
            <button
              type="button"
              onClick={onApprove}
              disabled={isPending}
              className="btn-primary"
            >
              Approve
            </button>
            <button
              type="button"
              onClick={onReject}
              disabled={isPending}
              className="btn-secondary"
            >
              Reject
            </button>
          </div>
        </div>
      )}
      {approval.status === 'approved' && (
        <button
          type="button"
          onClick={onExecute}
          disabled={isPending}
          className="btn-primary"
        >
          Execute action
        </button>
      )}
    </div>
  );
}
