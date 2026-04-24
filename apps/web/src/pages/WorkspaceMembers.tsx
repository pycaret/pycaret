/**
 * /workspaces/:wsId/members — list / invite / change-role / remove members.
 *
 * Admin-gated actions degrade gracefully: non-admins see a read-only member
 * list. Admins see inline role selects + remove buttons + an invite form.
 * The last-admin guard on the server is mirrored in the UI — the dropdown
 * for the sole admin is disabled.
 */

import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { membersApi, workspacesApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { useAuthStore } from '@/state/auth';
import type { MemberRead, WorkspaceRole } from '@/api/types';

export function WorkspaceMembers() {
  const { wsId = '' } = useParams<{ wsId: string }>();
  const qc = useQueryClient();
  const me = useAuthStore((s) => s.user);

  const ws = useQuery({
    queryKey: ['workspaces', wsId],
    queryFn: () => workspacesApi.get(wsId),
    enabled: !!wsId,
  });
  const members = useQuery({
    queryKey: ['members', wsId],
    queryFn: () => membersApi.list(wsId),
    enabled: !!wsId,
  });

  // My own role — determines whether to show admin actions. Compute from
  // the members list we already fetched; no extra query required.
  const myRole: WorkspaceRole | null = useMemo(() => {
    if (!members.data || !me) return null;
    const mine = members.data.find((m) => m.user_id === me.id);
    return mine?.role ?? null;
  }, [members.data, me]);

  const isAdmin = myRole === 'admin' || me?.is_superuser === true;
  // Drive the last-admin guard in the UI. Mirrors the server check.
  const adminCount = useMemo(
    () => (members.data ?? []).filter((m) => m.role === 'admin').length,
    [members.data],
  );

  // ────────── invite form
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState<WorkspaceRole>('member');

  const invite = useMutation({
    mutationFn: () =>
      membersApi.invite(wsId, { email: inviteEmail.trim(), role: inviteRole }),
    onSuccess: () => {
      setInviteEmail('');
      setInviteRole('member');
      qc.invalidateQueries({ queryKey: ['members', wsId] });
    },
  });

  const changeRole = useMutation({
    mutationFn: ({
      userId,
      role,
    }: {
      userId: string;
      role: WorkspaceRole;
    }) => membersApi.changeRole(wsId, userId, { role }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['members', wsId] }),
  });

  const remove = useMutation({
    mutationFn: (userId: string) => membersApi.remove(wsId, userId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['members', wsId] }),
  });

  return (
    <div className="space-y-8">
      <header>
        <nav className="text-xs text-ink-200/60 mb-2">
          <Link to="/" className="hover:text-ink-100">
            Workspaces
          </Link>
          <span className="mx-1">/</span>
          <Link to={`/workspaces/${wsId}`} className="hover:text-ink-100">
            {ws.data?.name ?? '…'}
          </Link>
          <span className="mx-1">/</span>
          <span>Members</span>
        </nav>
        <h1 className="text-xl font-semibold">Members</h1>
        <p className="mt-1 text-sm text-ink-200/70">
          {isAdmin
            ? 'Invite existing users + manage roles. Email invites (with account creation) land in V2.'
            : 'Read-only view — only workspace admins can change membership.'}
        </p>
      </header>

      {/* ────────── invite form (admins only) */}
      {isAdmin && (
        <div className="card">
          <h2 className="text-sm font-medium text-ink-100 mb-4">Invite an existing user</h2>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (inviteEmail.trim()) invite.mutate();
            }}
            className="flex flex-wrap items-end gap-3"
          >
            <div className="flex-1 min-w-60">
              <label className="field" htmlFor="inv-email">
                Email
              </label>
              <input
                id="inv-email"
                type="email"
                className="input"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                placeholder="alice@example.com"
                required
              />
            </div>
            <div>
              <label className="field" htmlFor="inv-role">
                Role
              </label>
              <select
                id="inv-role"
                className="input"
                value={inviteRole}
                onChange={(e) => setInviteRole(e.target.value as WorkspaceRole)}
              >
                <option value="member">member</option>
                <option value="admin">admin</option>
              </select>
            </div>
            <button
              type="submit"
              className="btn-primary"
              disabled={invite.isPending || !inviteEmail.trim()}
            >
              {invite.isPending ? 'Inviting…' : 'Invite'}
            </button>
          </form>
          {invite.error && <p className="error mt-3">{errorMessage(invite.error)}</p>}
        </div>
      )}

      {/* ────────── list */}
      <section>
        <header className="mb-3 flex items-baseline justify-between">
          <h2 className="text-sm font-medium text-ink-100">All members</h2>
          <span className="hint">{members.data?.length ?? 0} total</span>
        </header>
        {members.isLoading && <p className="hint">Loading…</p>}
        {members.error && <p className="error">{errorMessage(members.error)}</p>}

        {members.data && members.data.length > 0 && (
          <div className="card p-0 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-ink-800 text-ink-200/70">
                <tr>
                  <th className="px-4 py-2 text-left font-medium">User</th>
                  <th className="px-4 py-2 text-left font-medium">Role</th>
                  <th className="px-4 py-2 text-left font-medium">Status</th>
                  <th className="px-4 py-2 text-left font-medium">Joined</th>
                  {isAdmin && (
                    <th className="px-4 py-2 text-right font-medium">Action</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {members.data.map((m) => {
                  const isMe = m.user_id === me?.id;
                  const isLastAdmin = m.role === 'admin' && adminCount <= 1;
                  return (
                    <MemberRow
                      key={m.user_id}
                      m={m}
                      isMe={isMe}
                      isLastAdmin={isLastAdmin}
                      isAdmin={isAdmin}
                      onChangeRole={(role) =>
                        changeRole.mutate({ userId: m.user_id, role })
                      }
                      onRemove={() => {
                        if (
                          window.confirm(
                            `Remove ${m.email} from this workspace?`,
                          )
                        ) {
                          remove.mutate(m.user_id);
                        }
                      }}
                      disabled={changeRole.isPending || remove.isPending}
                    />
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        {changeRole.error && (
          <p className="error mt-3">{errorMessage(changeRole.error)}</p>
        )}
        {remove.error && <p className="error mt-3">{errorMessage(remove.error)}</p>}
      </section>
    </div>
  );
}

interface MemberRowProps {
  m: MemberRead;
  isMe: boolean;
  isLastAdmin: boolean;
  isAdmin: boolean;
  onChangeRole: (role: WorkspaceRole) => void;
  onRemove: () => void;
  disabled: boolean;
}

function MemberRow({
  m,
  isMe,
  isLastAdmin,
  isAdmin,
  onChangeRole,
  onRemove,
  disabled,
}: MemberRowProps) {
  return (
    <tr className="border-t border-ink-800 hover:bg-ink-800/50">
      <td className="px-4 py-2">
        <div className="font-medium text-ink-100">
          {m.display_name ?? m.email}
          {isMe && <span className="ml-2 text-xs text-ink-200/60">(you)</span>}
        </div>
        <div className="text-xs text-ink-200/60 font-mono">{m.email}</div>
      </td>
      <td className="px-4 py-2">
        {isAdmin ? (
          <select
            className="input py-1 px-2 text-sm"
            value={m.role}
            disabled={disabled || isLastAdmin}
            title={isLastAdmin ? "Can't demote the last admin" : undefined}
            onChange={(e) => onChangeRole(e.target.value as WorkspaceRole)}
          >
            <option value="member">member</option>
            <option value="admin">admin</option>
          </select>
        ) : (
          <span className="font-mono text-xs">{m.role}</span>
        )}
      </td>
      <td className="px-4 py-2">
        <span className={m.is_active ? 'text-success-500' : 'text-ink-200/60'}>
          {m.is_active ? 'active' : 'deactivated'}
        </span>
      </td>
      <td className="px-4 py-2 text-xs text-ink-200/60">
        {new Date(m.created_at).toLocaleDateString()}
      </td>
      {isAdmin && (
        <td className="px-4 py-2 text-right">
          <button
            className="btn-ghost text-xs"
            onClick={onRemove}
            disabled={disabled || isLastAdmin}
            title={isLastAdmin ? "Can't remove the last admin" : undefined}
          >
            Remove
          </button>
        </td>
      )}
    </tr>
  );
}
