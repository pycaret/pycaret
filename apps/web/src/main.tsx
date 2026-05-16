import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';
import './index.css';
import { applyEffectiveTheme, useUIPrefs } from '@/state/uiPrefs';

// Always boot in light mode regardless of persisted state or OS
// preference. The dark-variant CSS still lives in the codebase for
// surfaces that opt in, but the unauthenticated pages (Setup / Login)
// + a handful of older components weren't fully reviewed for dark
// mode and were rendering with low-contrast text. Once those are
// audited, swap this back to ``applyEffectiveTheme(useUIPrefs.getState().theme)``.
useUIPrefs.setState({ theme: 'light' });
applyEffectiveTheme('light');

// React to OS theme changes when the user has theme === 'system'.
if (typeof window !== 'undefined') {
  window
    .matchMedia('(prefers-color-scheme: dark)')
    .addEventListener('change', () => {
      if (useUIPrefs.getState().theme === 'system') applyEffectiveTheme('system');
    });
}

// Re-apply when the user toggles theme via the UI.
useUIPrefs.subscribe((state, prev) => {
  if (state.theme !== prev.theme) applyEffectiveTheme(state.theme);
});

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Don't refetch on every focus — this is a tool, not a social feed.
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30_000,
    },
  },
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
