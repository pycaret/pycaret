import { Link, NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/state/auth';
import { authApi } from '@/api/endpoints';
import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { CommandPalette } from '@/components/CommandPalette';

/**
 * Authenticated app shell — persistent left sidebar + content pane.
 *
 * Modeled after Linear / Vercel / Resend: 248px sidebar with grouped
 * nav, a workspace-scoped section that only shows when you're inside
 * one, and a quiet user menu at the bottom. Top of the content pane
 * is a thin header with the search trigger.
 */
export function Layout() {
  const nav = useNavigate();
  const location = useLocation();
  const clear = useAuthStore((s) => s.clear);
  const setUser = useAuthStore((s) => s.setUser);
  const user = useAuthStore((s) => s.user);

  const activeWsId = useMemo<string | undefined>(() => {
    const m = location.pathname.match(/\/workspaces\/([^/]+)/);
    return m ? m[1] : undefined;
  }, [location.pathname]);

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

  const [userMenuOpen, setUserMenuOpen] = useState(false);

  return (
    <div className="min-h-screen flex bg-ink-50">
      {/* ─── Sidebar ─────────────────────────────────────────────── */}
      <aside className="hidden md:flex md:w-60 lg:w-64 flex-col border-r border-ink-200 bg-white">
        {/* Brand */}
        <div className="flex h-14 items-center px-5 border-b border-ink-200">
          <Link
            to="/"
            className="flex items-center gap-2.5 text-ink-900 font-semibold tracking-tight"
          >
            <span
              aria-hidden
              className="block h-6 w-6 rounded-md bg-gradient-to-br from-accent-400 to-accent-600 shadow-soft-1"
            />
            PyCaret
          </Link>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-6 text-sm">
          <NavGroup label="General">
            <SidebarLink to="/" exact icon={<HomeIcon />}>
              Workspaces
            </SidebarLink>
          </NavGroup>

          {activeWsId && (
            <NavGroup label="Workspace">
              <SidebarLink
                to={`/workspaces/${activeWsId}`}
                exact
                icon={<FolderIcon />}
              >
                Projects
              </SidebarLink>
              <SidebarLink
                to={`/workspaces/${activeWsId}/home`}
                icon={<ActivityIcon />}
              >
                Dashboard
              </SidebarLink>
              <SidebarLink
                to={`/workspaces/${activeWsId}/compare`}
                icon={<CompareIcon />}
              >
                Compare runs
              </SidebarLink>
              <SidebarLink
                to={`/workspaces/${activeWsId}/predictions`}
                icon={<PredictIcon />}
              >
                Predict
              </SidebarLink>
              <SidebarLink
                to={`/workspaces/${activeWsId}/pipelines`}
                icon={<PipelineIcon />}
              >
                Pipelines
              </SidebarLink>
              <SidebarLink
                to={`/workspaces/${activeWsId}/deployments`}
                icon={<DeployIcon />}
              >
                Deployments
              </SidebarLink>
              <SidebarLink
                to={`/workspaces/${activeWsId}/drift`}
                icon={<DriftIcon />}
              >
                Drift
              </SidebarLink>
              <SidebarLink
                to={`/workspaces/${activeWsId}/llm`}
                icon={<SparkIcon />}
              >
                LLM
              </SidebarLink>
              <SidebarLink
                to={`/workspaces/${activeWsId}/members`}
                icon={<UsersIcon />}
              >
                Members
              </SidebarLink>
            </NavGroup>
          )}

          <NavGroup label="Account">
            <SidebarLink to="/account/api-keys" icon={<KeyIcon />}>
              API keys
            </SidebarLink>
            {user?.is_superuser && (
              <SidebarLink to="/admin/audit" icon={<ShieldIcon />}>
                Audit log
              </SidebarLink>
            )}
          </NavGroup>
        </nav>

        {/* User menu — bottom anchor */}
        <div className="border-t border-ink-200 p-3 relative">
          <button
            type="button"
            onClick={() => setUserMenuOpen((v) => !v)}
            className="w-full flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm
                       text-ink-700 hover:bg-ink-100 transition-colors"
          >
            <span className="h-7 w-7 rounded-full bg-accent-500 text-white text-xs
                             font-semibold flex items-center justify-center">
              {(user?.display_name ?? user?.email ?? '?').slice(0, 1).toUpperCase()}
            </span>
            <span className="flex-1 text-left truncate">
              {user?.display_name ?? user?.email ?? 'You'}
            </span>
            <DotsIcon />
          </button>
          {userMenuOpen && (
            <div
              className="absolute bottom-14 left-3 right-3 rounded-md border border-ink-200
                         bg-white shadow-soft-3 py-1 text-sm"
            >
              {user?.email && (
                <div className="px-3 py-2 border-b border-ink-200">
                  <div className="font-medium text-ink-800 truncate">
                    {user.display_name ?? user.email}
                  </div>
                  <div className="text-xs text-ink-500 truncate">{user.email}</div>
                </div>
              )}
              <button
                onClick={logout}
                className="w-full text-left px-3 py-2 text-ink-700 hover:bg-ink-50"
              >
                Sign out
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* ─── Main column ─────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Slim top bar */}
        <header className="h-14 flex items-center justify-between px-6 border-b border-ink-200 bg-white">
          <div className="md:hidden flex items-center gap-2 text-ink-900 font-semibold">
            <span
              aria-hidden
              className="block h-5 w-5 rounded bg-gradient-to-br from-accent-400 to-accent-600"
            />
            PyCaret
          </div>
          <div className="hidden md:block" />

          <div className="flex items-center gap-3">
            <button
              type="button"
              className="hidden md:inline-flex items-center gap-2 rounded-md border border-ink-200
                         bg-white px-2.5 py-1 text-xs text-ink-600 hover:bg-ink-50 hover:text-ink-900
                         transition-colors"
              title="Open command palette"
              onClick={() => {
                window.dispatchEvent(
                  new KeyboardEvent('keydown', { key: 'k', metaKey: true }),
                );
              }}
            >
              <SearchIcon />
              <span>Search</span>
              <span className="kbd ml-2">⌘K</span>
            </button>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto px-6 py-8">
          <div className="mx-auto w-full max-w-6xl">
            <Outlet />
          </div>
        </main>
      </div>

      <CommandPalette wsId={activeWsId} />
    </div>
  );
}

// ─── Sidebar primitives ────────────────────────────────────────────

function NavGroup({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="px-2.5 mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-ink-400">
        {label}
      </div>
      <div className="space-y-0.5">{children}</div>
    </div>
  );
}

function SidebarLink({
  to,
  exact = false,
  icon,
  children,
}: {
  to: string;
  exact?: boolean;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <NavLink
      to={to}
      end={exact}
      className={({ isActive }) =>
        isActive ? 'nav-link-active' : 'nav-link'
      }
    >
      {icon && <span className="text-ink-500 [.nav-link-active_&]:text-ink-900">{icon}</span>}
      <span className="truncate">{children}</span>
    </NavLink>
  );
}

// ─── Icons (inline SVG, 16px, currentColor) ────────────────────────

const stroke = {
  width: '16',
  height: '16',
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: '2',
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
};

const HomeIcon = () => (
  <svg {...stroke}>
    <path d="M3 9.5 12 3l9 6.5V21H3z" />
    <path d="M9 21V12h6v9" />
  </svg>
);
const FolderIcon = () => (
  <svg {...stroke}>
    <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
  </svg>
);
const ActivityIcon = () => (
  <svg {...stroke}>
    <path d="M22 12h-4l-3 9-6-18-3 9H2" />
  </svg>
);
const CompareIcon = () => (
  <svg {...stroke}>
    <path d="M21 16V8a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v8" />
    <path d="M3 12h18" />
  </svg>
);
const PredictIcon = () => (
  <svg {...stroke}>
    <path d="M9.5 7 14 12l-4.5 5" />
    <path d="M5 12h14" />
  </svg>
);
const PipelineIcon = () => (
  <svg {...stroke}>
    <path d="M5 6h6" />
    <path d="M13 6h6" />
    <path d="M5 12h6" />
    <path d="M13 12h6" />
    <path d="M5 18h6" />
    <path d="M13 18h6" />
  </svg>
);
const DeployIcon = () => (
  <svg {...stroke}>
    <path d="M12 2v6" />
    <path d="M5 9l7-7 7 7" />
    <path d="M5 13v6a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-6" />
  </svg>
);
const DriftIcon = () => (
  <svg {...stroke}>
    <path d="M3 17l4-7 5 4 5-9 4 7" />
  </svg>
);
const SparkIcon = () => (
  <svg {...stroke}>
    <path d="M12 3v3" />
    <path d="M12 18v3" />
    <path d="M3 12h3" />
    <path d="M18 12h3" />
    <path d="M5.6 5.6l2.1 2.1" />
    <path d="M16.3 16.3l2.1 2.1" />
    <path d="M5.6 18.4l2.1-2.1" />
    <path d="M16.3 7.7l2.1-2.1" />
  </svg>
);
const UsersIcon = () => (
  <svg {...stroke}>
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>
);
const KeyIcon = () => (
  <svg {...stroke}>
    <circle cx="7.5" cy="15.5" r="3.5" />
    <path d="M21 2l-9.6 9.6" />
    <path d="M14 6l4 4" />
  </svg>
);
const ShieldIcon = () => (
  <svg {...stroke}>
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </svg>
);
const SearchIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <circle cx="11" cy="11" r="8" />
    <path d="m21 21-4.35-4.35" />
  </svg>
);
const DotsIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <circle cx="12" cy="12" r="1" />
    <circle cx="19" cy="12" r="1" />
    <circle cx="5" cy="12" r="1" />
  </svg>
);
