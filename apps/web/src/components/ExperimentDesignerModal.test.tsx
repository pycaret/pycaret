import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ExperimentDesignerModal } from './ExperimentDesignerModal';

const listMock = vi.fn();
const designMock = vi.fn();

vi.mock('@/api/endpoints', () => ({
  dataSourcesApi: {
    list: (id: string) => listMock(id),
  },
  llmApi: {
    designExperiment: (body: unknown) => designMock(body),
  },
}));

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>;
}

describe('<ExperimentDesignerModal>', () => {
  it('is inert when closed', () => {
    const { container } = render(
      wrap(
        <ExperimentDesignerModal
          workspaceId="ws-1"
          open={false}
          onClose={() => {}}
        />,
      ),
    );
    expect(container.textContent).toBe('');
    expect(listMock).not.toHaveBeenCalled();
  });

  it('loads CSV options and keeps submit disabled until dataset + goal are filled', async () => {
    listMock.mockResolvedValue([
      {
        id: 'd-1',
        workspace_id: 'ws-1',
        name: 'iris.csv',
        kind: 'csv_upload',
        description: null,
        config: {},
        created_at: new Date().toISOString(),
        created_by: 'u',
      },
      {
        id: 'd-2',
        workspace_id: 'ws-1',
        name: 's3-thing',
        kind: 's3',
        description: null,
        config: {},
        created_at: new Date().toISOString(),
        created_by: 'u',
      },
    ]);

    render(
      wrap(
        <ExperimentDesignerModal
          workspaceId="ws-1"
          open
          onClose={() => {}}
        />,
      ),
    );

    // Only the CSV source renders as an option (s3 filtered out).
    expect(await screen.findByRole('option', { name: /iris\.csv/i })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /s3-thing/i })).not.toBeInTheDocument();

    const submit = screen.getByRole('button', { name: /design experiment/i });
    expect(submit).toBeDisabled();
  });

  it('fires designExperiment with the entered goal + picked dataset', async () => {
    listMock.mockResolvedValue([
      {
        id: 'd-1',
        workspace_id: 'ws-1',
        name: 'iris.csv',
        kind: 'csv_upload',
        description: null,
        config: {},
        created_at: new Date().toISOString(),
        created_by: 'u',
      },
    ]);
    designMock.mockResolvedValue({
      id: 'c-1',
      workspace_id: 'ws-1',
      project_id: null,
      experiment_id: null,
      run_id: null,
      type: 'experiment_design',
      provider: 'anthropic',
      model_name: 'claude-sonnet-4-5',
      prompt: '{...}',
      response_json: {
        suggested_config_json: { task_type: 'classification', target: 'target' },
        suggested_action: 'Run a compare of lr + rf + xgb.',
        reasoning_summary: 'Classification target with 3 balanced classes.',
        risk_flags: [],
      },
      generated_config_json: null,
      latency_ms: 50,
      error: null,
      created_at: new Date().toISOString(),
      created_by: 'u',
    });

    const user = userEvent.setup();
    render(
      wrap(
        <ExperimentDesignerModal
          workspaceId="ws-1"
          open
          onClose={() => {}}
        />,
      ),
    );

    await waitFor(() =>
      expect(screen.getByRole('option', { name: /iris\.csv/i })).toBeInTheDocument(),
    );
    await user.selectOptions(screen.getByLabelText(/dataset/i), 'd-1');
    await user.type(screen.getByLabelText(/goal/i), 'predict iris species');
    await user.click(screen.getByRole('button', { name: /design experiment/i }));

    await waitFor(() =>
      expect(designMock).toHaveBeenCalledWith({
        workspace_id: 'ws-1',
        data_source_id: 'd-1',
        goal: 'predict iris species',
        business_context: undefined,
        include_business_context: false,
      }),
    );
    expect(await screen.findByText(/Run a compare of lr \+ rf \+ xgb/)).toBeInTheDocument();
  });

  it('passes business_context and include_business_context when provided', async () => {
    listMock.mockResolvedValue([
      {
        id: 'd-1',
        workspace_id: 'ws-1',
        name: 'iris.csv',
        kind: 'csv_upload',
        description: null,
        config: {},
        created_at: new Date().toISOString(),
        created_by: 'u',
      },
    ]);
    designMock.mockResolvedValue({
      id: 'c-1',
      workspace_id: 'ws-1',
      project_id: null,
      experiment_id: null,
      run_id: null,
      type: 'experiment_design',
      provider: 'anthropic',
      model_name: 'claude-sonnet-4-5',
      prompt: '{...}',
      response_json: {
        suggested_config_json: { task_type: 'classification', target: 'target' },
        suggested_action: 'Run a compare focusing on recall.',
        reasoning_summary: 'Cost-sensitive classification.',
        risk_flags: [],
      },
      generated_config_json: null,
      latency_ms: 50,
      error: null,
      created_at: new Date().toISOString(),
      created_by: 'u',
    });

    const user = userEvent.setup();
    render(
      wrap(
        <ExperimentDesignerModal
          workspaceId="ws-1"
          open
          onClose={() => {}}
        />,
      ),
    );

    await waitFor(() =>
      expect(screen.getByRole('option', { name: /iris\.csv/i })).toBeInTheDocument(),
    );
    await user.selectOptions(screen.getByLabelText(/dataset/i), 'd-1');
    await user.type(screen.getByLabelText(/goal/i), 'predict iris species');
    await user.type(screen.getByRole('textbox', { name: /^business context/i }), 'False negatives cost 50x');
    await user.click(screen.getByLabelText(/include business context/i));
    await user.click(screen.getByRole('button', { name: /design experiment/i }));

    await waitFor(() =>
      expect(designMock).toHaveBeenCalledWith({
        workspace_id: 'ws-1',
        data_source_id: 'd-1',
        goal: 'predict iris species',
        business_context: 'False negatives cost 50x',
        include_business_context: true,
      }),
    );
  });
});
