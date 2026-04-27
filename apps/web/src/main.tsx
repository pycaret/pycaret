import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';
import './index.css';
import { applyEffectiveTheme, useUIPrefs } from '@/state/uiPrefs';

// Apply saved theme as early as possible — the persist middleware
// has already hydrated from localStorage by the time this runs, so
// no flash of wrong theme.
applyEffectiveTheme(useUIPrefs.getState().theme);

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
