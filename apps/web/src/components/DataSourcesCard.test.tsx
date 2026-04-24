import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DataSourcesCard } from './DataSourcesCard';

const listMock = vi.fn();
const uploadMock = vi.fn();
const removeMock = vi.fn();

vi.mock('@/api/endpoints', () => ({
  dataSourcesApi: {
    list: (id: string) => listMock(id),
    uploadCsv: (...args: unknown[]) => uploadMock(...args),
    remove: (id: string) => removeMock(id),
  },
}));

function wrap(children: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe('<DataSourcesCard>', () => {
  it('renders empty-state hint when workspace has no CSV uploads', async () => {
    listMock.mockResolvedValue([]);
    render(wrap(<DataSourcesCard workspaceId="ws-1" />));
    await waitFor(() => expect(listMock).toHaveBeenCalledWith('ws-1'));
    expect(
      await screen.findByText(/no csv uploads yet/i),
    ).toBeInTheDocument();
  });

  it('lists each CSV upload with its row count and approximate size', async () => {
    listMock.mockResolvedValue([
      {
        id: 'd-1',
        workspace_id: 'ws-1',
        name: 'iris.csv',
        kind: 'csv_upload',
        description: null,
        config: { rows: 150, size_bytes: 4620, columns: ['a', 'b', 'c'] },
        created_at: new Date().toISOString(),
        created_by: 'u-1',
      },
      {
        // A non-csv_upload kind should be filtered out.
        id: 'd-2',
        workspace_id: 'ws-1',
        name: 's3-thing',
        kind: 's3',
        description: null,
        config: {},
        created_at: new Date().toISOString(),
        created_by: 'u-1',
      },
    ]);
    render(wrap(<DataSourcesCard workspaceId="ws-1" />));
    expect(await screen.findByText('iris.csv')).toBeInTheDocument();
    expect(screen.getByText(/150 rows · 4\.5 kB · 3 cols/)).toBeInTheDocument();
    expect(screen.queryByText('s3-thing')).not.toBeInTheDocument();
  });

  it('disables upload button until a file is chosen', async () => {
    listMock.mockResolvedValue([]);
    render(wrap(<DataSourcesCard workspaceId="ws-1" />));
    const btn = await screen.findByRole('button', { name: /^upload$/i });
    expect(btn).toBeDisabled();

    const file = new File(['a,b\n1,2\n3,4'], 'sample.csv', { type: 'text/csv' });
    const input = screen.getByLabelText(/upload csv/i);
    await userEvent.upload(input, file);
    expect(btn).toBeEnabled();
  });
});
