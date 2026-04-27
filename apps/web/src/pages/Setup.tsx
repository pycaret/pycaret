import { useEffect, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { setupApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { useAuthStore } from '@/state/auth';

/**
 * First-run wizard. Creates the admin user + the default workspace in one
 * shot. After success, the token pair lands in the auth store and the UI
 * jumps to /.
 *
 * If the server is already bootstrapped, /setup redirects to /login so users
 * can't accidentally re-bootstrap (the API rejects with 409 anyway).
 */
export function Setup() {
  const nav = useNavigate();
  const setTokens = useAuthStore((s) => s.setTokens);

  const status = useQuery({ queryKey: ['setup', 'status'], queryFn: setupApi.status });

  useEffect(() => {
    if (status.data?.is_bootstrapped) nav('/login', { replace: true });
  }, [status.data, nav]);

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [workspace, setWorkspace] = useState('My workspace');

  const boot = useMutation({
    mutationFn: () =>
      setupApi.bootstrap({
        email,
        password,
        display_name: displayName || null,
        workspace_name: workspace,
      }),
    onSuccess: (pair) => {
      setTokens(pair);
      nav('/', { replace: true });
    },
  });

  const canSubmit =
    email.trim().length > 0 &&
    password.length >= 8 &&
    workspace.trim().length > 0 &&
    !boot.isPending;

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-form">
        <header className="mb-8">
          <h1 className="text-2xl font-semibold text-ink-900">Welcome to PyCaret</h1>
          <p className="mt-2 text-sm text-ink-500">
            Create the admin account and your first workspace.
          </p>
        </header>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (canSubmit) boot.mutate();
          }}
          className="card space-y-5"
        >
          <div>
            <label className="field" htmlFor="email">
              Admin email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              className="input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div>
            <label className="field" htmlFor="password">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="new-password"
              className="input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              minLength={8}
              required
            />
            <p className="hint mt-1">Minimum 8 characters. Store it safely — no recovery.</p>
          </div>

          <div>
            <label className="field" htmlFor="display">
              Display name (optional)
            </label>
            <input
              id="display"
              type="text"
              className="input"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
            />
          </div>

          <div>
            <label className="field" htmlFor="workspace">
              Workspace name
            </label>
            <input
              id="workspace"
              type="text"
              className="input"
              value={workspace}
              onChange={(e) => setWorkspace(e.target.value)}
              required
            />
          </div>

          {boot.error && <p className="error">{errorMessage(boot.error)}</p>}

          <button type="submit" className="btn-primary w-full" disabled={!canSubmit}>
            {boot.isPending ? 'Creating…' : 'Create workspace'}
          </button>
        </form>
      </div>
    </div>
  );
}
