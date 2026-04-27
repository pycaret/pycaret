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
      theme: 'system',
      toggleSidebar: () =>
        set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setSidebar: (collapsed) => set({ sidebarCollapsed: collapsed }),
      setTheme: (theme) => set({ theme }),
    }),
    {
      name: 'pycaret-ui-prefs',
      version: 1,
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
