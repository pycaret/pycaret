import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Setup } from './Setup';

// Mock the endpoints module — we test the UI, not axios.
vi.mock('@/api/endpoints', () => ({
  setupApi: {
    status: vi.fn().mockResolvedValue({
      is_bootstrapped: false,
      user_count: 0,
      workspace_count: 0,
    }),
    bootstrap: vi.fn().mockResolvedValue({
      access_token: 'A',
      refresh_token: 'R',
      token_type: 'bearer',
      expires_in: 3600,
    }),
  },
}));

function wrap(children: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={['/setup']}>
        <Routes>
          <Route path="/setup" element={children} />
          <Route path="/" element={<div>HOME</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('<Setup>', () => {
  beforeEach(() => vi.clearAllMocks());

  it('renders the first-run form', () => {
    render(wrap(<Setup />));
    expect(screen.getByRole('heading', { name: /welcome to pycaret/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/admin email/i)).toBeInTheDocument();
  });

  it('disables submit until password is long enough', async () => {
    const user = userEvent.setup();
    render(wrap(<Setup />));

    const btn = screen.getByRole('button', { name: /create workspace/i });
    expect(btn).toBeDisabled();

    await user.type(screen.getByLabelText(/admin email/i), 'me@example.com');
    await user.type(screen.getByLabelText(/password/i), 'short');
    expect(btn).toBeDisabled();

    await user.clear(screen.getByLabelText(/password/i));
    await user.type(screen.getByLabelText(/password/i), 'longenough');
    expect(btn).toBeEnabled();
  });
});
