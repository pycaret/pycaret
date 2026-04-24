import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuditLogViewer } from './AuditLogViewer';
import { useAuthStore } from '@/state/auth';

const listAdminMock = vi.fn();

vi.mock('@/api/endpoints', () => ({
  auditApi: {
    listAdmin: (f: unknown) => listAdminMock(f),
    listForWorkspace: vi.fn(),
  },
}));

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={qc}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  // Default: signed-in as a superuser.
  useAuthStore.setState({
    accessToken: 'tok',
    refreshToken: null,
    user: {
      id: 'u-admin',
      email: 'admin@example.com',
      display_name: 'Admin',
      is_active: true,
      is_superuser: true,
      created_at: new Date().toISOString(),
    },
  });
});

describe('<AuditLogViewer>', () => {
  it('lists rows + expands a row on click to show the scrubbed payload', async () => {
    listAdminMock.mockResolvedValue([
      {
        id: 'a-1',
        workspace_id: 'ws-1',
        user_id: 'u-admin',
        action: 'workspaces.create',
        method: 'POST',
        path: '/api/v1/workspaces',
        target_type: 'workspace',
        target_id: null,
        status_code: 201,
        payload: { name: 'Demo', password: '***REDACTED***' },
        ip_address: '127.0.0.1',
        user_agent: 'vitest',
        created_at: new Date().toISOString(),
      },
    ]);

    render(wrap(<AuditLogViewer />));
    await waitFor(() => expect(listAdminMock).toHaveBeenCalled());
    expect(await screen.findByText('workspaces.create')).toBeInTheDocument();

    // Click the row → expands to show payload.
    await userEvent.click(screen.getByText('workspaces.create'));
    // Payload panel rendered — the scrubbed password marker is there.
    expect(await screen.findByText(/REDACTED/)).toBeInTheDocument();
  });

  it('renders the forbidden message for a non-superuser and skips the API call', () => {
    useAuthStore.setState({
      accessToken: 'tok',
      refreshToken: null,
      user: {
        id: 'u-1',
        email: 'alice@example.com',
        display_name: 'Alice',
        is_active: true,
        is_superuser: false,
        created_at: new Date().toISOString(),
      },
    });
    render(wrap(<AuditLogViewer />));
    expect(
      screen.getByText(/superuser privilege required/i),
    ).toBeInTheDocument();
    expect(listAdminMock).not.toHaveBeenCalled();
  });

  it('applies the filter form', async () => {
    listAdminMock.mockResolvedValue([]);
    const user = userEvent.setup();
    render(wrap(<AuditLogViewer />));
    await waitFor(() => expect(listAdminMock).toHaveBeenCalled());
    await user.type(screen.getByLabelText(/^action/i), 'runs.create');
    await user.click(screen.getByRole('button', { name: /apply/i }));
    await waitFor(() => {
      const last = listAdminMock.mock.calls.at(-1)?.[0];
      expect(last.action).toBe('runs.create');
    });
  });
});
