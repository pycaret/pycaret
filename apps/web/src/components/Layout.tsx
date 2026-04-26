import { Link, NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/state/auth';
import { authApi } from '@/api/endpoints';
import { useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { CommandPalette } from '@/components/CommandPalette';

/** The root authenticated-shell. Top nav + content. */
export function Layout() {
  const nav = useNavigate();
  const location = useLocation();
  const clear = useAuthStore((s) => s.clear);
  const setUser = useAuthStore((s) => s.setUser);
  const user = useAuthStore((s) => s.user);

  // Derive the active workspace id from the URL so the palette + nav
  // can scope their actions properly.
  const activeWsId = useMemo<string | undefined>(() => {
    const m = location.pathname.match(/\/workspaces\/([^/]+)/);
    return m ? m[1] : undefined;
  }, [location.pathname]);

  // Lazy hydrate `user` on first render after login / reload.
  const me = useQuery({
    queryKey: ['auth', 'me'],
    queryFn: authApi.me,
    staleTime: 60_000,
    enabled: useAuthStore.getState().accessToken !== null,
  });

  useEffect(() => {
    if (me.data) setUser(me.data);
  }, [me.data, setUser]);

  const logout = async () => {
    try {
      await authApi.logout();
    } finally {
      clear();
      nav('/login', { replace: true });
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-ink-800 bg-ink-900">
        <div className="mx-auto max-w-6xl px-6 h-14 flex items-center justify-between">
          <Link
            to="/"
            className="flex items-center gap-2 font-semibold text-ink-100 hover:text-accent-400"
          >
            <span className="inline-block h-6 w-6 rounded bg-accent-500" aria-hidden />
            PyCaret
          </Link>
          <nav className="flex items-center gap-6 text-sm">
            <NavLink
              to="/"
              className={({ isActive }) =>
                isActive ? 'text-accent-400' : 'text-ink-200 hover:text-ink-100'
              }
              end
            >
              Workspaces
            </NavLink>
            {activeWsId && (
              <>
                <NavLink
                  to={`/workspaces/${activeWsId}/home`}
                  className={({ isActive }) =>
                    isActive ? 'text-accent-400' : 'text-ink-200 hover:text-ink-100'
                  }
                >
                  Dashboard
                </NavLink>
                <NavLink
                  to={`/workspaces/${activeWsId}/compare`}
                  className={({ isActive }) =>
                    isActive ? 'text-accent-400' : 'text-ink-200 hover:text-ink-100'
                  }
                >
                  Compare
                </NavLink>
                <NavLink
                  to={`/workspaces/${activeWsId}/drift`}
                  className={({ isActive }) =>
                    isActive ? 'text-accent-400' : 'text-ink-200 hover:text-ink-100'
                  }
                >
                  Drift
                </NavLink>
                <NavLink
                  to={`/workspaces/${activeWsId}/predictions`}
                  className={({ isActive }) =>
                    isActive ? 'text-accent-400' : 'text-ink-200 hover:text-ink-100'
                  }
                >
                  Predict
                </NavLink>
              </>
            )}
            <NavLink
              to="/account/api-keys"
              className={({ isActive }) =>
                isActive ? 'text-accent-400' : 'text-ink-200 hover:text-ink-100'
              }
            >
              API keys
            </NavLink>
            {user?.is_superuser && (
              <NavLink
                to="/admin/audit"
                className={({ isActive }) =>
                  isActive
                    ? 'text-accent-400'
                    : 'text-ink-200 hover:text-ink-100'
                }
              >
                Audit log
              </NavLink>
            )}
            <kbd
              title="Open command palette"
              className="text-ink-200/60 text-xs px-2 py-0.5 border border-ink-700 rounded"
              style={{ fontFamily: 'ui-monospace, monospace' }}
            >
              ⌘K
            </kbd>
            {user && (
              <span className="text-ink-200/70" title={user.email}>
                {user.display_name ?? user.email}
              </span>
            )}
            <button onClick={logout} className="btn-ghost">
              Sign out
            </button>
          </nav>
        </div>
      </header>
      <main className="flex-1 mx-auto w-full max-w-6xl px-6 py-8">
        <Outlet />
      </main>
      <CommandPalette wsId={activeWsId} />
    </div>
  );
}
