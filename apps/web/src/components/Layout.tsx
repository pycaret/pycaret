import { Link, NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/state/auth';
import {
  authApi,
  deploymentsApi,
  runsApi,
  workspacesApi,
} from '@/api/endpoints';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { CommandPalette } from '@/components/CommandPalette';
import { AIAdvisorWidget } from '@/components/AIAdvisorWidget';
import { useUIPrefs, type Theme } from '@/state/uiPrefs';

/**
 * Authenticated app shell — persistent left sidebar + content pane.
 *
 * Linear/Vercel-style: 240px sidebar that collapses to 56px (icons
 * only) when the user toggles. State persisted in localStorage.
 * Light/dark/system theme toggle in the user menu. Slim top bar with
 * search trigger.
 */
export function Layout() {
  const nav = useNavigate();
  const location = useLocation();
  const clear = useAuthStore((s) => s.clear);
  const setUser = useAuthStore((s) => s.setUser);
  const user = useAuthStore((s) => s.user);

  const sidebarCollapsed = useUIPrefs((s) => s.sidebarCollapsed);
  const toggleSidebar = useUIPrefs((s) => s.toggleSidebar);

  // Most routes embed the workspace ID directly in the URL. `/runs/:id` is
  // a flat path — fall back to the workspace_id the backend serialises onto
  // the Run response so the sidebar keeps its context.
  const wsFromUrl = useMemo<string | undefined>(() => {
    const m = location.pathname.match(/\/workspaces\/([^/]+)/);
    return m ? m[1] : undefined;
  }, [location.pathname]);

  const runIdFromUrl = useMemo<string | undefined>(() => {
    const m = location.pathname.match(/^\/runs\/([^/]+)/);
    return m ? m[1] : undefined;
  }, [location.pathname]);

  const deploymentIdFromUrl = useMemo<string | undefined>(() => {
    const m = location.pathname.match(/^\/deployments\/([^/]+)/);
    return m ? m[1] : undefined;
  }, [location.pathname]);

  const runForCtx = useQuery({
    queryKey: ['runs', runIdFromUrl],
    queryFn: () => runsApi.get(runIdFromUrl!),
    enabled: !!runIdFromUrl && !wsFromUrl,
    staleTime: 60_000,
  });

  const deploymentForCtx = useQuery({
    queryKey: ['deployments', deploymentIdFromUrl],
    queryFn: () => deploymentsApi.get(deploymentIdFromUrl!),
    enabled: !!deploymentIdFromUrl && !wsFromUrl,
    staleTime: 60_000,
  });

  const activeWsId =
    wsFromUrl ??
    runForCtx.data?.workspace_id ??
    deploymentForCtx.data?.workspace_id ??
    undefined;

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
    <div className="min-h-screen flex bg-ink-50 dark:bg-ink-950">
      {/* ─── Sidebar ─────────────────────────────────────────────── */}
      {/* h-screen sticky top-0 → sidebar always exactly matches the
          viewport height and stays put when main content scrolls.
          Without these, a tall main content lets the aside grow + scroll
          off-screen, which is what burned us with the bottom-pinned
          profile + collapse controls disappearing on scroll. */}
      <aside
        className={`hidden md:flex md:h-screen md:sticky md:top-0 flex-col border-r border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 transition-[width] duration-200 ease-out ${
          sidebarCollapsed ? 'md:w-14' : 'md:w-60 lg:w-64'
        }`}
      >
        {/* Brand + collapse toggle (top-positioned, OpenAI/Linear style) */}
        <div className="flex h-14 items-center gap-1 px-2 border-b border-ink-200 dark:border-ink-800 shrink-0">
          {!sidebarCollapsed && (
            <Link
              to="/"
              className="flex items-center gap-2.5 px-1 text-ink-900 dark:text-ink-50 font-semibold tracking-tight flex-1 min-w-0"
              title="Home"
            >
              <span
                aria-hidden
                className="block h-6 w-6 shrink-0 rounded-md bg-gradient-to-br from-accent-400 to-accent-600 shadow-soft-1"
              />
              <span className="truncate">PyCaret</span>
            </Link>
          )}
          <button
            type="button"
            onClick={toggleSidebar}
            className={`p-1.5 rounded-md text-ink-500 hover:text-ink-900 dark:hover:text-ink-50 hover:bg-ink-100 dark:hover:bg-ink-800 transition-colors ${
              sidebarCollapsed ? 'mx-auto' : ''
            }`}
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <ChevronIcon direction={sidebarCollapsed ? 'right' : 'left'} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto scrollbar-hover-only px-2 py-4 space-y-6 text-sm">
          {!activeWsId && (
            <div className="px-3 py-6 text-center">
              <div className="mx-auto h-10 w-10 rounded-lg bg-ink-100 dark:bg-ink-800 text-ink-500 flex items-center justify-center mb-2">
                <HomeIcon />
              </div>
              {!sidebarCollapsed && (
                <>
                  <p className="text-sm font-medium text-ink-700 dark:text-ink-300">
                    Pick a workspace
                  </p>
                  <p className="mt-1 text-xs text-ink-500">
                    Use the switcher in the top-right.
                  </p>
                </>
              )}
            </div>
          )}

          {activeWsId && (
            <>
              {/* Build — the create / ingest surface */}
              <NavGroup label="Build" collapsed={sidebarCollapsed}>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/home`}
                  icon={<ActivityIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Dashboard
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}`}
                  exact
                  icon={<FolderIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Projects
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/datasets`}
                  icon={<DatasetsNavIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Datasets
                </SidebarLink>
              </NavGroup>

              {/* Models — the full model lifecycle in one block:
                  registry (named + versioned + governed) → deployment
                  (serving) → monitoring / drift / lineage (post-deploy ops).
                  Session-56 collapsed the old Pipelines page into Model
                  Registry: a "version" on the registry IS the artifact. */}
              <NavGroup label="Models" collapsed={sidebarCollapsed}>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/models`}
                  icon={<ModelRegistryIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Model registry
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/deployments`}
                  icon={<DeployIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Deployments
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/monitoring`}
                  icon={<MonitoringIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Monitoring
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/drift`}
                  icon={<DriftIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Drift
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/lineage`}
                  icon={<LineageIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Lineage
                </SidebarLink>
              </NavGroup>

              {/* Automation — recurring jobs, gates, outbound notifications,
                  saved experiment recipes. */}
              <NavGroup label="Automation" collapsed={sidebarCollapsed}>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/approvals`}
                  icon={<ApprovalIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Approvals
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/schedules`}
                  icon={<ClockIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Schedules
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/templates`}
                  icon={<TemplateIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Templates
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/webhooks`}
                  icon={<HookIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Webhooks
                </SidebarLink>
              </NavGroup>

              {/* Settings — workspace configuration: integrations, secrets,
                  people, admin. */}
              <NavGroup label="Settings" collapsed={sidebarCollapsed}>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/connections`}
                  icon={<ConnectionIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Connections
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/secrets`}
                  icon={<KeyIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Secrets
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/git`}
                  icon={<GitIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Git
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/llm`}
                  icon={<SparkIcon />}
                  collapsed={sidebarCollapsed}
                >
                  LLM
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/members`}
                  icon={<UsersIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Members
                </SidebarLink>
                <SidebarLink
                  to={`/workspaces/${activeWsId}/admin`}
                  icon={<ShieldIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Admin
                </SidebarLink>
              </NavGroup>
            </>
          )}

          <NavGroup label="Account" collapsed={sidebarCollapsed}>
            <SidebarLink
              to="/account/api-keys"
              icon={<KeyIcon />}
              collapsed={sidebarCollapsed}
            >
              API keys
            </SidebarLink>
            {user?.is_superuser && (
              <>
                <SidebarLink
                  to="/admin/users"
                  icon={<UsersIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Platform users
                </SidebarLink>
                <SidebarLink
                  to="/admin/audit"
                  icon={<ShieldIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Audit log
                </SidebarLink>
                <SidebarLink
                  to="/admin/queues"
                  icon={<QueueIcon />}
                  collapsed={sidebarCollapsed}
                >
                  Queues & workers
                </SidebarLink>
              </>
            )}
          </NavGroup>
        </nav>

        {/* User menu — bottom anchor */}
        <div className="border-t border-ink-200 dark:border-ink-800 p-2 relative shrink-0">
          <button
            type="button"
            onClick={() => setUserMenuOpen((v) => !v)}
            className={`w-full flex items-center gap-2.5 rounded-md py-1.5 text-sm
                       text-ink-700 dark:text-ink-300
                       hover:bg-ink-100 dark:hover:bg-ink-800 transition-colors ${
                         sidebarCollapsed ? 'px-1.5 justify-center' : 'px-2'
                       }`}
            title={user?.email}
          >
            <span className="h-7 w-7 rounded-full bg-accent-500 text-white text-xs
                             font-semibold flex items-center justify-center shrink-0">
              {(user?.display_name ?? user?.email ?? '?').slice(0, 1).toUpperCase()}
            </span>
            {!sidebarCollapsed && (
              <>
                <span className="flex-1 text-left truncate">
                  {user?.display_name ?? user?.email ?? 'You'}
                </span>
                <DotsIcon />
              </>
            )}
          </button>

          {userMenuOpen && (
            <UserMenu
              user={user}
              collapsed={sidebarCollapsed}
              onClose={() => setUserMenuOpen(false)}
              onLogout={logout}
            />
          )}
        </div>
      </aside>

      {/* ─── Main column ─────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Slim top bar */}
        <header className="h-14 flex items-center justify-between px-6 border-b border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 shrink-0">
          <div className="md:hidden flex items-center gap-2 text-ink-900 dark:text-ink-50 font-semibold">
            <span
              aria-hidden
              className="block h-5 w-5 rounded bg-gradient-to-br from-accent-400 to-accent-600"
            />
            PyCaret
          </div>
          <div className="hidden md:block" />

          <div className="flex items-center gap-2">
            <WorkspaceSwitcher activeWsId={activeWsId} />
            <button
              type="button"
              className="hidden md:inline-flex items-center gap-2 rounded-md border border-ink-200 dark:border-ink-800
                         bg-white dark:bg-ink-900 px-2.5 py-1.5 text-xs text-ink-600 dark:text-ink-400
                         hover:bg-ink-50 dark:hover:bg-ink-800 hover:text-ink-900 dark:hover:text-ink-50
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
      <AIAdvisorWidget />
    </div>
  );
}

// ─── Workspace switcher ───────────────────────────────────────────

function WorkspaceSwitcher({ activeWsId }: { activeWsId: string | undefined }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const nav = useNavigate();

  const list = useQuery({
    queryKey: ['workspaces'],
    queryFn: workspacesApi.list,
    staleTime: 30_000,
  });

  // Close on outside click + Escape.
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    setTimeout(() => document.addEventListener('click', onDoc), 0);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('click', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const active = list.data?.find((w) => w.id === activeWsId);
  const label = active?.name ?? 'Pick a workspace';

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-2 rounded-md border border-ink-200 dark:border-ink-800
                   bg-white dark:bg-ink-900 px-3 py-1.5 text-sm font-medium
                   text-ink-800 dark:text-ink-100
                   hover:bg-ink-50 dark:hover:bg-ink-800
                   transition-colors max-w-[200px]"
      >
        <span
          aria-hidden
          className="h-5 w-5 rounded bg-accent-50 dark:bg-accent-500/15 text-accent-600 dark:text-accent-400 flex items-center justify-center shrink-0"
        >
          <HomeIcon />
        </span>
        <span className="truncate">{label}</span>
        <ChevronDownIcon />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1.5 w-72 z-50 rounded-lg border border-ink-200 dark:border-ink-800 bg-white dark:bg-ink-900 shadow-soft-3 py-1.5 text-sm">
          <div className="px-3 pb-1.5 pt-1 text-[11px] font-semibold uppercase tracking-wider text-ink-400">
            Workspaces
          </div>
          <ul className="max-h-72 overflow-y-auto">
            {list.isLoading && (
              <li className="px-3 py-2 text-ink-500 text-xs">Loading…</li>
            )}
            {list.data && list.data.length === 0 && (
              <li className="px-3 py-2 text-ink-500 text-xs">No workspaces yet.</li>
            )}
            {list.data?.map((w) => {
              const isActive = w.id === activeWsId;
              return (
                <li key={w.id}>
                  <button
                    type="button"
                    onClick={() => {
                      setOpen(false);
                      nav(`/workspaces/${w.id}`);
                    }}
                    className={`w-full text-left flex items-center gap-2.5 px-3 py-2 transition-colors ${
                      isActive
                        ? 'bg-ink-100 dark:bg-ink-800 text-ink-900 dark:text-ink-50'
                        : 'text-ink-700 dark:text-ink-300 hover:bg-ink-50 dark:hover:bg-ink-800'
                    }`}
                  >
                    <span
                      aria-hidden
                      className={`h-5 w-5 rounded flex items-center justify-center shrink-0 ${
                        isActive
                          ? 'bg-accent-500 text-white'
                          : 'bg-ink-100 dark:bg-ink-800 text-ink-500'
                      }`}
                    >
                      <HomeIcon />
                    </span>
                    <span className="truncate flex-1">{w.name}</span>
                    {isActive && (
                      <span className="text-accent-600 dark:text-accent-400 shrink-0">
                        <CheckIcon />
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
          <div className="border-t border-ink-100 dark:border-ink-800 mt-1 pt-1">
            <button
              type="button"
              onClick={() => {
                setOpen(false);
                nav('/');
              }}
              className="w-full text-left flex items-center gap-2.5 px-3 py-2 text-ink-700 dark:text-ink-300 hover:bg-ink-50 dark:hover:bg-ink-800 transition-colors"
            >
              <span className="text-ink-500"><GridIcon /></span>
              <span>Manage all workspaces</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── User menu popover ────────────────────────────────────────────

function UserMenu({
  user,
  collapsed,
  onClose,
  onLogout,
}: {
  user: { email?: string; display_name?: string | null } | null | undefined;
  collapsed: boolean;
  onClose: () => void;
  onLogout: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const theme = useUIPrefs((s) => s.theme);
  const setTheme = useUIPrefs((s) => s.setTheme);

  // Close on outside click.
  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    setTimeout(() => document.addEventListener('click', onDoc), 0);
    return () => document.removeEventListener('click', onDoc);
  }, [onClose]);

  return (
    <div
      ref={ref}
      className={`absolute bottom-12 ${
        collapsed ? 'left-14' : 'left-2 right-2'
      } z-50 rounded-lg border border-ink-200 dark:border-ink-800
         bg-white dark:bg-ink-900 shadow-soft-3 py-1 text-sm
         ${collapsed ? 'w-56' : ''}`}
    >
      {user?.email && (
        <div className="px-3 py-2 border-b border-ink-100 dark:border-ink-800">
          <div className="font-medium text-ink-800 dark:text-ink-100 truncate">
            {user.display_name ?? user.email}
          </div>
          <div className="text-xs text-ink-500 truncate">{user.email}</div>
        </div>
      )}

      {/* Theme picker */}
      <div className="px-3 py-2 border-b border-ink-100 dark:border-ink-800">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-400 mb-1.5">
          Theme
        </div>
        <div className="grid grid-cols-3 gap-1 rounded-md bg-ink-100 dark:bg-ink-800 p-0.5">
          {(['light', 'dark', 'system'] as Theme[]).map((t) => (
            <button
              key={t}
              onClick={() => setTheme(t)}
              className={`text-xs font-medium py-1 rounded transition-colors capitalize ${
                theme === t
                  ? 'bg-white dark:bg-ink-900 text-ink-900 dark:text-ink-50 shadow-soft-1'
                  : 'text-ink-600 dark:text-ink-400 hover:text-ink-900 dark:hover:text-ink-50'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={onLogout}
        className="w-full text-left px-3 py-2 text-ink-700 dark:text-ink-300
                   hover:bg-ink-50 dark:hover:bg-ink-800"
      >
        Sign out
      </button>
    </div>
  );
}

// ─── Sidebar primitives ────────────────────────────────────────────

function NavGroup({
  label,
  collapsed,
  children,
}: {
  label: string;
  collapsed: boolean;
  children: React.ReactNode;
}) {
  return (
    <div>
      {!collapsed && (
        <div className="px-2.5 mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-ink-400 dark:text-ink-500">
          {label}
        </div>
      )}
      <div className="space-y-0.5">{children}</div>
    </div>
  );
}

function SidebarLink({
  to,
  exact = false,
  icon,
  collapsed,
  children,
}: {
  to: string;
  exact?: boolean;
  icon?: React.ReactNode;
  collapsed: boolean;
  children: React.ReactNode;
}) {
  return (
    <NavLink
      to={to}
      end={exact}
      className={({ isActive }) =>
        `${isActive ? 'nav-link-active' : 'nav-link'} ${
          collapsed ? 'justify-center px-1.5' : ''
        }`
      }
      title={collapsed ? String(children) : undefined}
    >
      {icon && (
        <span className="text-ink-500 dark:text-ink-400 [.nav-link-active_&]:text-ink-900 [.nav-link-active_&]:dark:text-ink-50 shrink-0">
          {icon}
        </span>
      )}
      {!collapsed && <span className="truncate">{children}</span>}
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
// PipelineIcon retired in session 56 with the Pipelines nav entry —
// the icon was only used by that link.
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
const ChevronIcon = ({ direction }: { direction: 'left' | 'right' }) => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    {direction === 'left' ? (
      <path d="m15 18-6-6 6-6" />
    ) : (
      <path d="m9 18 6-6-6-6" />
    )}
  </svg>
);
const ChevronDownIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="m6 9 6 6 6-6" />
  </svg>
);
const CheckIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d="M20 6L9 17l-5-5" />
  </svg>
);
const GridIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <rect x="3" y="3" width="7" height="7" rx="1" />
    <rect x="14" y="3" width="7" height="7" rx="1" />
    <rect x="3" y="14" width="7" height="7" rx="1" />
    <rect x="14" y="14" width="7" height="7" rx="1" />
  </svg>
);
const ClockIcon = () => (
  <svg {...stroke}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </svg>
);
const TemplateIcon = () => (
  <svg {...stroke}>
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <path d="M3 9h18" />
    <path d="M9 21V9" />
  </svg>
);
const HookIcon = () => (
  <svg {...stroke}>
    <path d="M18 6V3" />
    <path d="M18 6a4 4 0 0 0-4 4v6a3 3 0 1 1-6 0v-2" />
    <circle cx="18" cy="3" r="1" />
  </svg>
);
const ModelRegistryIcon = () => (
  <svg {...stroke}>
    <path d="M20 7L12 3 4 7l8 4 8-4z" />
    <path d="M4 12l8 4 8-4" />
    <path d="M4 17l8 4 8-4" />
  </svg>
);
const MonitoringIcon = () => (
  <svg {...stroke}>
    <path d="M3 12h4l3 8 4-16 3 8h4" />
  </svg>
);
const ApprovalIcon = () => (
  <svg {...stroke}>
    <path d="M9 12l2 2 4-4" />
    <circle cx="12" cy="12" r="10" />
  </svg>
);
const LineageIcon = () => (
  <svg {...stroke}>
    <circle cx="6" cy="6" r="2" />
    <circle cx="18" cy="6" r="2" />
    <circle cx="6" cy="18" r="2" />
    <circle cx="18" cy="18" r="2" />
    <path d="M6 8v8" />
    <path d="M8 6h8" />
    <path d="M8 18h8" />
    <path d="M18 8v8" />
  </svg>
);
const QueueIcon = () => (
  <svg {...stroke}>
    <rect x="3" y="4" width="18" height="4" rx="1" />
    <rect x="3" y="10" width="18" height="4" rx="1" />
    <rect x="3" y="16" width="18" height="4" rx="1" />
  </svg>
);
const ConnectionIcon = () => (
  <svg {...stroke}>
    <rect x="2" y="6" width="6" height="12" rx="1" />
    <rect x="16" y="6" width="6" height="12" rx="1" />
    <path d="M8 12h8" />
  </svg>
);
const GitIcon = () => (
  <svg {...stroke}>
    <circle cx="6" cy="6" r="2" />
    <circle cx="18" cy="6" r="2" />
    <circle cx="12" cy="18" r="2" />
    <path d="M6 8v3a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3V8" />
    <path d="M12 14v2" />
  </svg>
);
const DatasetsNavIcon = () => (
  <svg {...stroke}>
    <ellipse cx="12" cy="5" rx="9" ry="3" />
    <path d="M3 5v14a9 3 0 0 0 18 0V5" />
    <path d="M3 12a9 3 0 0 0 18 0" />
  </svg>
);
