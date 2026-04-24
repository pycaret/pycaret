import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DriftReportsCard } from './DriftReportsCard';
import type { DriftReportRead } from '@/api/types';

const listMock = vi.fn();
const createMock = vi.fn();
const analyzeMock = vi.fn();

vi.mock('@/api/endpoints', () => ({
  driftApi: {
    list: (id: string) => listMock(id),
    create: (id: string, body: unknown) => createMock(id, body),
    get: vi.fn(),
  },
  // DriftAnalysisModal imports llmApi.analyzeDrift.
  llmApi: {
    analyzeDrift: (body: unknown) => analyzeMock(body),
  },
}));

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>;
}

const REPORT: DriftReportRead = {
  id: 'r-1',
  deployment_id: 'd-1',
  baseline_artifact_id: null,
  window_start: '2026-04-17T00:00:00Z',
  window_end: '2026-04-24T00:00:00Z',
  drift_score: 0.31,
  drift_status: 'moderate',
  feature_drift_json: { amount: { score: 0.42, kind: 'missing_rate' } },
  prediction_drift_json: { kind: 'js', score: 0.02 },
  sample_size: 400,
  created_at: '2026-04-24T00:00:00Z',
  created_by: 'u',
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe('<DriftReportsCard>', () => {
  it('empty state shows the hint + the Record button', async () => {
    listMock.mockResolvedValue([]);
    render(wrap(<DriftReportsCard deploymentId="d-1" />));
    expect(
      await screen.findByText(/no drift reports yet/i),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /record snapshot/i })).toBeInTheDocument();
  });

  it('renders the list with status + opens the analysis modal on ✨ Analyze', async () => {
    listMock.mockResolvedValue([REPORT]);
    analyzeMock.mockResolvedValue({
      id: 'c',
      workspace_id: 'ws',
      project_id: null,
      experiment_id: null,
      run_id: null,
      type: 'drift_analysis',
      provider: 'anthropic',
      model_name: 'claude-sonnet-4-5',
      prompt: '{}',
      response_json: {
        suggested_config_json: {},
        suggested_action: 'INVESTIGATE: missing_rate on amount',
        reasoning_summary: 'Upstream ETL is the most likely cause.',
        risk_flags: ['missing_rate_spike'],
      },
      generated_config_json: null,
      latency_ms: 123,
      error: null,
      created_at: new Date().toISOString(),
      created_by: 'u',
    });

    render(wrap(<DriftReportsCard deploymentId="d-1" />));
    await waitFor(() => expect(listMock).toHaveBeenCalledWith('d-1'));
    // Status cell.
    expect(await screen.findByText('moderate')).toBeInTheDocument();
    // Score rendered.
    expect(screen.getByText('0.310')).toBeInTheDocument();

    // Click Analyze → modal appears + auto-fires.
    await userEvent.click(screen.getByRole('button', { name: /analyze/i }));
    await waitFor(() =>
      expect(analyzeMock).toHaveBeenCalledWith({ drift_report_id: 'r-1' }),
    );
    expect(await screen.findByText(/INVESTIGATE/)).toBeInTheDocument();
  });

  it('submits the create form after parsing the pasted JSON', async () => {
    listMock.mockResolvedValue([]);
    createMock.mockResolvedValue(REPORT);
    const user = userEvent.setup();
    render(wrap(<DriftReportsCard deploymentId="d-1" />));
    await user.click(screen.getByRole('button', { name: /record snapshot/i }));
    // The form textareas are pre-populated with valid JSON — just submit.
    await user.click(screen.getByRole('button', { name: /^record$/i }));

    await waitFor(() => expect(createMock).toHaveBeenCalled());
    const [did, body] = createMock.mock.calls[0];
    expect(did).toBe('d-1');
    expect(body.drift_score).toBeCloseTo(0.2);
    expect(body.feature_drift_json).toHaveProperty('amount');
  });

  it('shows a form error when drift_score is out of range', async () => {
    listMock.mockResolvedValue([]);
    const user = userEvent.setup();
    render(wrap(<DriftReportsCard deploymentId="d-1" />));
    await user.click(screen.getByRole('button', { name: /record snapshot/i }));
    const scoreInput = screen.getByLabelText(/drift_score/i);
    await user.clear(scoreInput);
    await user.type(scoreInput, '5');
    await user.click(screen.getByRole('button', { name: /^record$/i }));
    expect(
      await screen.findByText(/drift_score must be a number in/i),
    ).toBeInTheDocument();
    expect(createMock).not.toHaveBeenCalled();
  });
});
