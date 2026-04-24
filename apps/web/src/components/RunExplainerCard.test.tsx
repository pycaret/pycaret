import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RunExplainerCard } from './RunExplainerCard';

const explainMock = vi.fn();

vi.mock('@/api/endpoints', () => ({
  llmApi: {
    explainRun: (body: unknown) => explainMock(body),
  },
}));

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>;
}

describe('<RunExplainerCard>', () => {
  it('renders the button but does NOT fire the mutation on mount', () => {
    render(wrap(<RunExplainerCard runId="r-1" />));
    expect(screen.getByRole('button', { name: /^explain$/i })).toBeInTheDocument();
    expect(explainMock).not.toHaveBeenCalled();
  });

  it('fires explainRun with the right runId when clicked, and renders the result', async () => {
    const user = userEvent.setup();
    explainMock.mockResolvedValue({
      id: 'c-1',
      workspace_id: 'ws',
      project_id: 'p',
      experiment_id: 'e',
      run_id: 'r-1',
      type: 'run_summary',
      provider: 'anthropic',
      model_name: 'claude-sonnet-4-5',
      prompt: '{...}',
      response_json: {
        suggested_config_json: { next_actions: ['tune_rf', 'add_stratified_cv'] },
        suggested_action: 'Tune the top random forest next.',
        reasoning_summary: 'Random forest won with 0.96 AUC; gap to #2 is tight.',
        risk_flags: ['tiny_margin'],
      },
      generated_config_json: null,
      latency_ms: 111.1,
      error: null,
      created_at: new Date().toISOString(),
      created_by: 'u',
    });

    render(wrap(<RunExplainerCard runId="r-1" />));
    await user.click(screen.getByRole('button', { name: /^explain$/i }));

    expect(explainMock).toHaveBeenCalledWith({ run_id: 'r-1' });
    expect(await screen.findByText(/tune the top random forest next/i)).toBeInTheDocument();
    expect(screen.getByText(/random forest won/i)).toBeInTheDocument();
    expect(screen.getByText('tune_rf')).toBeInTheDocument();
    expect(screen.getByText('add_stratified_cv')).toBeInTheDocument();
    expect(screen.getByText('tiny_margin')).toBeInTheDocument();
    // Button label flips to "Re-explain" after first success.
    expect(screen.getByRole('button', { name: /re-explain/i })).toBeInTheDocument();
  });
});
