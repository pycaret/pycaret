import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DeploymentReviewModal } from './DeploymentReviewModal';

const reviewMock = vi.fn();

vi.mock('@/api/endpoints', () => ({
  llmApi: {
    reviewDeployment: (body: unknown) => reviewMock(body),
  },
}));

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>;
}

describe('<DeploymentReviewModal>', () => {
  it('is inert when closed', () => {
    const { container } = render(
      wrap(
        <DeploymentReviewModal
          pipelineId="p-1"
          pipelineName="iris-v1"
          open={false}
          onClose={() => {}}
        />,
      ),
    );
    expect(container.textContent).toBe('');
    expect(reviewMock).not.toHaveBeenCalled();
  });

  it('auto-fires on open and tone-codes the verdict', async () => {
    reviewMock.mockResolvedValue({
      id: 'c',
      workspace_id: 'ws',
      project_id: null,
      experiment_id: null,
      run_id: 'r',
      type: 'deployment_risk_review',
      provider: 'anthropic',
      model_name: 'claude-sonnet-4-5',
      prompt: '{}',
      response_json: {
        suggested_config_json: {},
        suggested_action: 'APPROVE WITH CAVEATS: small training sample',
        reasoning_summary: 'Only 150 rows; leaderboard looks reasonable but validate on prod data.',
        risk_flags: ['small_training_sample'],
      },
      generated_config_json: null,
      latency_ms: 99,
      error: null,
      created_at: new Date().toISOString(),
      created_by: 'u',
    });

    render(
      wrap(
        <DeploymentReviewModal
          pipelineId="p-1"
          pipelineName="iris-v1"
          open
          onClose={() => {}}
        />,
      ),
    );

    await waitFor(() =>
      expect(reviewMock).toHaveBeenCalledWith({ pipeline_id: 'p-1' }),
    );
    const verdict = await screen.findByText(/APPROVE WITH CAVEATS/);
    expect(verdict).toBeInTheDocument();
    // Verdict text is tone-coded with the warn color.
    expect(verdict).toHaveClass('text-warn-500');
    expect(screen.getByText(/only 150 rows/i)).toBeInTheDocument();
    expect(screen.getByText('small_training_sample')).toBeInTheDocument();
  });
});
