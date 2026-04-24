import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { FailureDebuggerCard } from './FailureDebuggerCard';

const debugMock = vi.fn();

vi.mock('@/api/endpoints', () => ({
  llmApi: {
    debugRun: (body: unknown) => debugMock(body),
  },
}));

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>;
}

describe('<FailureDebuggerCard>', () => {
  it('button renders but mutation does NOT auto-fire', () => {
    render(wrap(<FailureDebuggerCard runId="r-1" />));
    expect(screen.getByRole('button', { name: /^diagnose$/i })).toBeInTheDocument();
    expect(debugMock).not.toHaveBeenCalled();
  });

  it('click fires debugRun and renders diagnosis + risk flags', async () => {
    debugMock.mockResolvedValue({
      id: 'c-1',
      workspace_id: 'ws',
      project_id: 'p',
      experiment_id: 'e',
      run_id: 'r-1',
      type: 'failure_debugging',
      provider: 'anthropic',
      model_name: 'claude-sonnet-4-5',
      prompt: '{}',
      response_json: {
        suggested_config_json: { next_action: 'rename_target' },
        suggested_action: 'Rename target column from y to target.',
        reasoning_summary: 'DATA: target not found in dataset.',
        risk_flags: ['needs_dataset_inspection'],
      },
      generated_config_json: null,
      latency_ms: 42,
      error: null,
      created_at: new Date().toISOString(),
      created_by: 'u',
    });
    const user = userEvent.setup();
    render(wrap(<FailureDebuggerCard runId="r-1" />));
    await user.click(screen.getByRole('button', { name: /^diagnose$/i }));

    expect(debugMock).toHaveBeenCalledWith({ run_id: 'r-1' });
    expect(await screen.findByText(/rename target column/i)).toBeInTheDocument();
    expect(screen.getByText(/DATA: target not found/i)).toBeInTheDocument();
    expect(screen.getByText('needs_dataset_inspection')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /re-diagnose/i })).toBeInTheDocument();
  });
});
