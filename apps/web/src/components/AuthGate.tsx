import { useEffect, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '@/state/auth';

/**
 * Guards authenticated routes. If there's an access token, render children.
 * If there's only a refresh token (e.g. after a reload), try to refresh
 * once before bouncing to /login.
 */
export function AuthGate({ children }: { children: React.ReactNode }) {
  const loc = useLocation();
  const accessToken = useAuthStore((s) => s.accessToken);
  const refreshToken = useAuthStore((s) => s.refreshToken);
  const refresh = useAuthStore((s) => s.refresh);
  const [resolving, setResolving] = useState(!accessToken && !!refreshToken);

  useEffect(() => {
    if (!accessToken && refreshToken) {
      refresh().finally(() => setResolving(false));
    }
  }, [accessToken, refreshToken, refresh]);

  if (resolving) {
    return (
      <div className="min-h-screen flex items-center justify-center text-ink-200/60">
        Restoring session…
      </div>
    );
  }
  if (!useAuthStore.getState().accessToken) {
    return <Navigate to="/login" replace state={{ from: loc.pathname }} />;
  }
  return <>{children}</>;
}
