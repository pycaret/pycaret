/**
 * UI preferences store: sidebar collapse + theme.
 *
 * Both are persisted to localStorage so the user's choice survives
 * reload. The theme has three values:
 *   - 'light': always light
 *   - 'dark':  always dark
 *   - 'system': follow `prefers-color-scheme`
 *
 * `effectiveTheme` is the resolved 'light' | 'dark' that should
 * actually be applied to the DOM. Components rarely need it — the
 * `applyEffectiveTheme()` side effect toggles the `dark` class on
 * <html> for Tailwind to pick up automatically.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Theme = 'light' | 'dark' | 'system';

interface UIPrefsState {
  sidebarCollapsed: boolean;
  theme: Theme;
  toggleSidebar: () => void;
  setSidebar: (collapsed: boolean) => void;
  setTheme: (theme: Theme) => void;
}

export const useUIPrefs = create<UIPrefsState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      // Default to light. OS dark-mode preference was bleeding into
      // pages that aren't designed for dark surfaces yet (Setup,
      // Login). Users can flip to dark / system from the user menu
      // once those surfaces are reviewed.
      theme: 'light',
      toggleSidebar: () =>
        set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setSidebar: (collapsed) => set({ sidebarCollapsed: collapsed }),
      setTheme: (theme) => set({ theme }),
    }),
    {
      name: 'pycaret-ui-prefs',
      // Bump on default-value changes so the next page load discards
      // any stale persisted state and re-reads the in-code default.
      version: 2,
    },
  ),
);

/** Resolves a stored theme value to 'light' | 'dark' using the system setting. */
export function resolveTheme(theme: Theme): 'light' | 'dark' {
  if (theme === 'system') {
    if (typeof window === 'undefined') return 'light';
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  }
  return theme;
}

/** Applies the effective theme to <html> by toggling the `dark` class. */
export function applyEffectiveTheme(theme: Theme): void {
  if (typeof document === 'undefined') return;
  const effective = resolveTheme(theme);
  document.documentElement.classList.toggle('dark', effective === 'dark');
}
