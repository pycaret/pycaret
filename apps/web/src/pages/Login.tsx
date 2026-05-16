import { useEffect, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { authApi, setupApi } from '@/api/endpoints';
import { errorMessage } from '@/api/client';
import { useAuthStore } from '@/state/auth';

export function Login() {
  const nav = useNavigate();
  const loc = useLocation();
  const setTokens = useAuthStore((s) => s.setTokens);

  // If the server hasn't been bootstrapped yet, push the user into setup.
  const status = useQuery({ queryKey: ['setup', 'status'], queryFn: setupApi.status });
  useEffect(() => {
    if (status.data && !status.data.is_bootstrapped) nav('/setup', { replace: true });
  }, [status.data, nav]);

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const login = useMutation({
    mutationFn: () => authApi.login({ email, password }),
    onSuccess: (pair) => {
      setTokens(pair);
      const from = (loc.state as { from?: string } | null)?.from ?? '/';
      nav(from, { replace: true });
    },
  });

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-form">
        <header className="mb-8">
          <h1 className="h-page">Sign in</h1>
          <p className="mt-2 text-sm muted">
            Welcome back.{' '}
            {status.data?.is_bootstrapped === false && (
              <Link to="/setup" className="text-accent-600 hover:underline">
                First time? Run setup.
              </Link>
            )}
          </p>
        </header>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            login.mutate();
          }}
          className="card space-y-5"
        >
          <div>
            <label className="field" htmlFor="email">
              Email
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
              autoComplete="current-password"
              className="input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          {login.error && <p className="error">{errorMessage(login.error)}</p>}
          <button
            type="submit"
            className="btn-primary w-full"
            disabled={login.isPending || !email || !password}
          >
            {login.isPending ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
      </div>
    </div>
  );
}
