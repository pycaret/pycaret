import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DriftAnalysisModal } from './DriftAnalysisModal';
import type { DriftReportRead } from '@/api/types';

const analyzeMock = vi.fn();

vi.mock('@/api/endpoints', () => ({
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
  feature_drift_json: {
    amount: { score: 0.42, kind: 'missing_rate' },
    age: { score: 0.05, kind: 'psi' },
  },
  prediction_drift_json: { kind: 'js', score: 0.02 },
  sample_size: 400,
  created_at: '2026-04-24T00:00:00Z',
  created_by: 'u',
};

describe('<DriftAnalysisModal>', () => {
  it('is inert when closed', () => {
    const { container } = render(
      wrap(<DriftAnalysisModal report={REPORT} open={false} onClose={() => {}} />),
    );
    expect(container.textContent).toBe('');
    expect(analyzeMock).not.toHaveBeenCalled();
  });

  it('auto-fires on open and tone-codes the RETRAIN NOW verdict as danger', async () => {
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
        suggested_config_json: { retrain_window_days: 30 },
        suggested_action: 'RETRAIN NOW: amount feature missing-rate 0.42',
        reasoning_summary:
          'Drift is concentrated in one feature with a missing-rate kind.',
        risk_flags: ['concentrated_drift', 'missing_rate_spike'],
      },
      generated_config_json: null,
      latency_ms: 210,
      error: null,
      created_at: new Date().toISOString(),
      created_by: 'u',
    });

    render(wrap(<DriftAnalysisModal report={REPORT} open onClose={() => {}} />));

    await waitFor(() =>
      expect(analyzeMock).toHaveBeenCalledWith({ drift_report_id: 'r-1' }),
    );
    const verdict = await screen.findByText(/RETRAIN NOW/);
    expect(verdict).toBeInTheDocument();
    expect(verdict).toHaveClass('text-danger-500');
    expect(screen.getByText('concentrated_drift')).toBeInTheDocument();
    // Feature rows rendered, sorted by score desc — 'amount' (0.42) first.
    const rows = screen.getAllByRole('row');
    // First row is the header; second is 'amount'.
    expect(rows[1]).toHaveTextContent('amount');
    expect(rows[2]).toHaveTextContent('age');
  });

  it('tone-codes NO ACTION as success', async () => {
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
        suggested_action: 'NO ACTION: drift within tolerance',
        reasoning_summary: 'All features under PSI 0.1.',
        risk_flags: [],
      },
      generated_config_json: null,
      latency_ms: 80,
      error: null,
      created_at: new Date().toISOString(),
      created_by: 'u',
    });

    render(wrap(<DriftAnalysisModal report={REPORT} open onClose={() => {}} />));
    const verdict = await screen.findByText(/NO ACTION/);
    expect(verdict).toHaveClass('text-success-500');
  });
});
