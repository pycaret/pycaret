import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { WorkspaceMembers } from './WorkspaceMembers';
import { useAuthStore } from '@/state/auth';
import type { MemberRead } from '@/api/types';

const listMock = vi.fn();
const inviteMock = vi.fn();
const changeRoleMock = vi.fn();
const removeMock = vi.fn();
const getWsMock = vi.fn();

vi.mock('@/api/endpoints', () => ({
  membersApi: {
    list: (ws: string) => listMock(ws),
    invite: (ws: string, body: unknown) => inviteMock(ws, body),
    changeRole: (ws: string, uid: string, body: unknown) =>
      changeRoleMock(ws, uid, body),
    remove: (ws: string, uid: string) => removeMock(ws, uid),
  },
  workspacesApi: {
    get: (id: string) => getWsMock(id),
  },
}));

function wrap(initialPath = '/workspaces/ws-1/members') {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[initialPath]}>
        <Routes>
          <Route path="/workspaces/:wsId/members" element={<WorkspaceMembers />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

const BASE_MEMBER: Omit<MemberRead, 'user_id' | 'email' | 'role'> = {
  display_name: null,
  is_active: true,
  created_at: new Date().toISOString(),
};

beforeEach(() => {
  vi.clearAllMocks();
  getWsMock.mockResolvedValue({
    id: 'ws-1',
    name: 'Demo',
    description: null,
    created_at: new Date().toISOString(),
    created_by: 'admin-id',
  });
  useAuthStore.setState({
    accessToken: 'token',
    refreshToken: null,
    user: {
      id: 'admin-id',
      email: 'admin@example.com',
      display_name: 'Admin',
      is_active: true,
      is_superuser: false,
      created_at: new Date().toISOString(),
    },
  });
});

describe('<WorkspaceMembers>', () => {
  it("as an admin, shows the invite form + can change a member's role", async () => {
    listMock.mockResolvedValue([
      { ...BASE_MEMBER, user_id: 'admin-id', email: 'admin@example.com', role: 'admin' },
      { ...BASE_MEMBER, user_id: 'alice-id', email: 'alice@example.com', role: 'member' },
    ]);

    render(wrap());

    // Invite form is visible (admin only).
    expect(await screen.findByRole('heading', { name: /invite an existing user/i })).toBeInTheDocument();

    // Change Alice's role to admin via the inline select.
    const selects = screen.getAllByRole('combobox');
    // Find Alice's row select — the 2nd combobox (1st is invite-role).
    const memberSelects = selects.filter((el) =>
      (el as HTMLSelectElement).value === 'member' || (el as HTMLSelectElement).value === 'admin',
    );
    // The invite role defaults to 'member' and appears first; the members
    // table rows come after. Pick the admin's select (disabled, because
    // last-admin guard) and Alice's select (enabled).
    const enabledMemberSelect = memberSelects.find(
      (el) =>
        !(el as HTMLSelectElement).disabled && (el as HTMLSelectElement).value === 'member' && (el as HTMLSelectElement).id !== 'inv-role',
    );
    expect(enabledMemberSelect).toBeDefined();

    changeRoleMock.mockResolvedValue({
      ...BASE_MEMBER,
      user_id: 'alice-id',
      email: 'alice@example.com',
      role: 'admin',
    });
    await userEvent.selectOptions(enabledMemberSelect!, 'admin');
    await waitFor(() =>
      expect(changeRoleMock).toHaveBeenCalledWith('ws-1', 'alice-id', {
        role: 'admin',
      }),
    );
  });

  it('disables the role select + remove button on the last admin', async () => {
    listMock.mockResolvedValue([
      { ...BASE_MEMBER, user_id: 'admin-id', email: 'admin@example.com', role: 'admin' },
    ]);
    render(wrap());
    await waitFor(() =>
      expect(screen.getAllByText('admin@example.com').length).toBeGreaterThan(0),
    );

    // The sole admin's select is disabled. It's the second combobox (invite-
    // role is first + enabled).
    const selects = screen.getAllByRole('combobox') as HTMLSelectElement[];
    const adminSelect = selects.find(
      (s) => s.id !== 'inv-role' && s.value === 'admin',
    );
    expect(adminSelect).toBeDefined();
    expect(adminSelect!).toBeDisabled();

    // Remove button also disabled.
    const removeBtn = screen.getByRole('button', { name: /^remove$/i });
    expect(removeBtn).toBeDisabled();
  });

  it('as a non-admin member, hides the invite form + action column', async () => {
    // Sign in as Alice, a member.
    useAuthStore.setState({
      accessToken: 'token',
      refreshToken: null,
      user: {
        id: 'alice-id',
        email: 'alice@example.com',
        display_name: 'Alice',
        is_active: true,
        is_superuser: false,
        created_at: new Date().toISOString(),
      },
    });
    listMock.mockResolvedValue([
      { ...BASE_MEMBER, user_id: 'admin-id', email: 'admin@example.com', role: 'admin' },
      { ...BASE_MEMBER, user_id: 'alice-id', email: 'alice@example.com', role: 'member' },
    ]);

    render(wrap());
    await waitFor(() =>
      expect(screen.getAllByText('alice@example.com').length).toBeGreaterThan(0),
    );

    expect(
      screen.queryByRole('heading', { name: /invite an existing user/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole('button', { name: /^remove$/i }),
    ).not.toBeInTheDocument();
  });

  it('invite submit fires the API with the chosen role', async () => {
    listMock.mockResolvedValue([
      { ...BASE_MEMBER, user_id: 'admin-id', email: 'admin@example.com', role: 'admin' },
    ]);
    inviteMock.mockResolvedValue({
      ...BASE_MEMBER,
      user_id: 'new-id',
      email: 'bob@example.com',
      role: 'admin',
    });
    const user = userEvent.setup();
    render(wrap());
    await waitFor(() =>
      expect(screen.getByLabelText(/^email/i)).toBeInTheDocument(),
    );

    await user.type(screen.getByLabelText(/^email/i), 'bob@example.com');
    // Pick the invite-form role select (id='inv-role').
    await user.selectOptions(screen.getByLabelText(/^role/i), 'admin');
    await user.click(screen.getByRole('button', { name: /^invite$/i }));

    await waitFor(() =>
      expect(inviteMock).toHaveBeenCalledWith('ws-1', {
        email: 'bob@example.com',
        role: 'admin',
      }),
    );
  });
});
