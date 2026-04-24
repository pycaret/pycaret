import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApiKeysScreen } from './ApiKeysScreen';

const listMock = vi.fn();
const createMock = vi.fn();
const revokeMock = vi.fn();

vi.mock('@/api/endpoints', () => ({
  apiKeysApi: {
    list: () => listMock(),
    create: (body: unknown) => createMock(body),
    revoke: (id: string) => revokeMock(id),
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

describe('<ApiKeysScreen>', () => {
  it('renders empty state hint when user has no keys', async () => {
    listMock.mockResolvedValue([]);
    render(wrap(<ApiKeysScreen />));
    expect(await screen.findByText(/no api keys yet/i)).toBeInTheDocument();
  });

  it('create flow shows the one-time plaintext panel on success', async () => {
    listMock.mockResolvedValue([]);
    createMock.mockResolvedValue({
      id: 'k-1',
      name: 'ci-bot',
      prefix: 'pck_abcd1234',
      workspace_id: null,
      scopes: null,
      expires_at: null,
      last_used_at: null,
      revoked_at: null,
      created_at: new Date().toISOString(),
      token: 'pck_abcd1234_secret_plaintext_do_not_show_again',
    });
    const user = userEvent.setup();
    render(wrap(<ApiKeysScreen />));
    await user.type(screen.getByLabelText(/^name/i), 'ci-bot');
    await user.click(screen.getByRole('button', { name: /create key/i }));

    // One-time panel appears with plaintext + warning.
    expect(
      await screen.findByText(/api key created — copy it now/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/pck_abcd1234_secret_plaintext_do_not_show_again/),
    ).toBeInTheDocument();
    expect(screen.getByText(/never/i)).toBeInTheDocument();
    expect(createMock).toHaveBeenCalledWith({
      name: 'ci-bot',
      expires_in_days: null,
    });
  });

  it("renders active / revoked status from the key's revoked_at field", async () => {
    listMock.mockResolvedValue([
      {
        id: 'k1',
        name: 'my-laptop',
        prefix: 'pck_a',
        workspace_id: null,
        scopes: null,
        expires_at: null,
        last_used_at: null,
        revoked_at: null,
        created_at: new Date().toISOString(),
      },
      {
        id: 'k2',
        name: 'old-ci-token',
        prefix: 'pck_b',
        workspace_id: null,
        scopes: null,
        expires_at: null,
        last_used_at: null,
        revoked_at: new Date().toISOString(),
        created_at: new Date().toISOString(),
      },
    ]);
    render(wrap(<ApiKeysScreen />));
    // Match on exact cell text to avoid collisions with key names.
    await waitFor(() =>
      expect(screen.getByText('active', { exact: true })).toBeInTheDocument(),
    );
    expect(screen.getByText('revoked', { exact: true })).toBeInTheDocument();
    // And the name column renders distinct names.
    expect(screen.getByText('my-laptop')).toBeInTheDocument();
    expect(screen.getByText('old-ci-token')).toBeInTheDocument();
  });
});
