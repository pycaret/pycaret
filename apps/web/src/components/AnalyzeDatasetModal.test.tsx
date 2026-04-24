import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AnalyzeDatasetModal } from './AnalyzeDatasetModal';
import type { LLMConsultationRead } from '@/api/types';

const analyzeMock = vi.fn();

vi.mock('@/api/endpoints', () => ({
  llmApi: {
    analyzeDataset: (body: unknown) => analyzeMock(body),
  },
}));

function wrap(children: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

const FAKE: LLMConsultationRead = {
  id: 'c-1',
  workspace_id: 'ws-1',
  project_id: null,
  experiment_id: null,
  run_id: null,
  type: 'dataset_analysis',
  provider: 'anthropic',
  model_name: 'claude-sonnet-4-5',
  prompt: '{...}',
  response_json: {
    suggested_config_json: { task_type: 'classification', target: 'churn' },
    suggested_action: 'Run a classification compare.',
    reasoning_summary: 'Target has two classes; dataset is balanced.',
    risk_flags: ['small_sample'],
  },
  generated_config_json: null,
  latency_ms: 123.4,
  error: null,
  created_at: new Date().toISOString(),
  created_by: 'u-1',
};

describe('<AnalyzeDatasetModal>', () => {
  it('returns null when closed', () => {
    const { container } = render(
      wrap(
        <AnalyzeDatasetModal
          workspaceId="ws-1"
          dataSourceId="d-1"
          dataSourceName="iris.csv"
          open={false}
          onClose={() => {}}
        />,
      ),
    );
    expect(container.textContent).toBe('');
    expect(analyzeMock).not.toHaveBeenCalled();
  });

  it('fires analyzeDataset on open and renders the advice envelope', async () => {
    analyzeMock.mockResolvedValue(FAKE);
    render(
      wrap(
        <AnalyzeDatasetModal
          workspaceId="ws-1"
          dataSourceId="d-1"
          dataSourceName="iris.csv"
          open
          onClose={() => {}}
        />,
      ),
    );
    await waitFor(() =>
      expect(analyzeMock).toHaveBeenCalledWith({
        workspace_id: 'ws-1',
        data_source_id: 'd-1',
        task_type_hint: null,
      }),
    );
    expect(await screen.findByText(/Run a classification compare/)).toBeInTheDocument();
    expect(screen.getByText(/target has two classes/i)).toBeInTheDocument();
    expect(screen.getByText('small_sample')).toBeInTheDocument();
    // Footer: provider + model + latency.
    expect(screen.getByText(/anthropic · claude-sonnet-4-5 · 123ms/i)).toBeInTheDocument();
  });

  it('calls onClose when Close button is clicked', async () => {
    analyzeMock.mockResolvedValue(FAKE);
    const onClose = vi.fn();
    render(
      wrap(
        <AnalyzeDatasetModal
          workspaceId="ws-1"
          dataSourceId="d-1"
          dataSourceName="iris.csv"
          open
          onClose={onClose}
        />,
      ),
    );
    // Two close buttons exist (✕ aria-label + footer text "Close"). Pick the
    // visible text one via exact regex match, ignoring aria-label cases.
    await waitFor(() =>
      expect(screen.getAllByRole('button', { name: /close/i }).length).toBeGreaterThanOrEqual(2),
    );
    const footerClose = screen
      .getAllByRole('button', { name: /close/i })
      .find((el) => el.textContent === 'Close');
    expect(footerClose).toBeTruthy();
    await userEvent.click(footerClose!);
    expect(onClose).toHaveBeenCalled();
  });
});
