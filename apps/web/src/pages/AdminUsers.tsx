/**
 * /admin/users — superuser-only platform user management.
 *
 * Lists every user with workspace count + active/superuser flags.
 * Toggle is_superuser / is_active inline.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { adminApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { useAuthStore } from '@/state/auth';
import { BackupRestoreCard } from '@/components/BackupRestoreCard';

export function AdminUsers() {
  const me = useAuthStore((s) => s.user);
  const qc = useQueryClient();
  const { data, isPending, error } = useQuery({
    queryKey: ['admin', 'users'],
    queryFn: () => adminApi.listUsers({ limit: 200 }),
  });

  const patch = useMutation({
    mutationFn: (vars: {
      user_id: string;
      patch: { is_superuser?: boolean; is_active?: boolean };
    }) => adminApi.patchUser(vars.user_id, vars.patch),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['admin', 'users'] }),
  });

  if (!me?.is_superuser) {
    return (
      <div className="card">
        <p className="error">Superuser access required.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <header>
        <h1 className="h-page">Users</h1>
        <p className="mt-2 text-sm text-ink-500">
          Platform-level user management. Superuser-only.
        </p>
      </header>

      {isPending ? (
        <div className="card text-sm text-ink-500">Loading…</div>
      ) : error ? (
        <div className="card text-sm text-danger-600">Could not load users.</div>
      ) : (
        <div className="card overflow-hidden p-0">
          <table className="w-full text-sm">
            <thead className="bg-white text-ink-500 dark:bg-ink-900">
              <tr>
                <th className="px-4 py-2 text-left font-medium">User</th>
                <th className="px-4 py-2 text-right font-medium">Workspaces</th>
                <th className="px-4 py-2 text-left font-medium">Joined</th>
                <th className="px-4 py-2 text-center font-medium">Active</th>
                <th className="px-4 py-2 text-center font-medium">Superuser</th>
              </tr>
            </thead>
            <tbody>
              {data?.items.map((u) => {
                const isMe = u.id === me?.id;
                return (
                  <tr
                    key={u.id}
                    className="border-t border-ink-200 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-950/40"
                  >
                    <td className="px-4 py-2">
                      <div className="font-medium text-ink-900 dark:text-ink-50">
                        {u.display_name ?? u.email}
                        {isMe && (
                          <span className="ml-2 text-xs text-ink-500">(you)</span>
                        )}
                      </div>
                      <div className="text-xs text-ink-500 font-mono">{u.email}</div>
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums text-ink-700 dark:text-ink-300">
                      {u.workspace_count}
                    </td>
                    <td className="px-4 py-2 text-xs text-ink-500 whitespace-nowrap">
                      {u.created_at
                        ? new Date(u.created_at).toLocaleDateString()
                        : '—'}
                    </td>
                    <td className="px-4 py-2 text-center">
                      <input
                        type="checkbox"
                        checked={u.is_active}
                        disabled={isMe || patch.isPending}
                        onChange={(e) =>
                          patch.mutate({
                            user_id: u.id,
                            patch: { is_active: e.target.checked },
                          })
                        }
                        title={isMe ? "You can't deactivate yourself" : 'Toggle active'}
                      />
                    </td>
                    <td className="px-4 py-2 text-center">
                      <input
                        type="checkbox"
                        checked={u.is_superuser}
                        disabled={patch.isPending}
                        onChange={(e) =>
                          patch.mutate({
                            user_id: u.id,
                            patch: { is_superuser: e.target.checked },
                          })
                        }
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <BackupRestoreCard />

      {patch.error && (
        <p className="text-sm text-danger-600">{errorMessage(patch.error)}</p>
      )}
    </div>
  );
}
