import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PredictTester } from './PredictTester';

vi.mock('@/api/endpoints', () => ({
  deploymentsApi: {
    predict: vi.fn().mockResolvedValue({
      deployment_id: 'd-1',
      endpoint_slug: 'iris-v1',
      predictions: [
        { index: 0, prediction: 1 },
        { index: 1, prediction: 2 },
      ],
      latency_ms: 3.14,
      request_id: 'req-abcdefgh-9999',
    }),
  },
}));

function wrap(children: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe('<PredictTester>', () => {
  it('renders a json textarea pre-filled with an iris-shaped payload', () => {
    render(wrap(<PredictTester endpointSlug="iris-v1" />));
    const textarea = screen.getByLabelText(/rows/i) as HTMLTextAreaElement;
    expect(textarea).toBeInTheDocument();
    // Asymmetric matchers don't work with toHaveValue; check the DOM value directly.
    expect(textarea.value).toContain('sepal length');
  });

  it('surfaces an inline parse-error hint for invalid JSON as the user types', () => {
    render(wrap(<PredictTester endpointSlug="iris-v1" />));
    // userEvent.type treats `{` as a special char — fireEvent.change is simpler
    // for pasting raw invalid JSON.
    const textarea = screen.getByLabelText(/rows/i);
    fireEvent.change(textarea, { target: { value: 'not json' } });
    expect(screen.getByText(/^JSON:/)).toBeInTheDocument();
    const submit = screen.getByRole('button', { name: /send request/i });
    expect(submit).toBeDisabled();
  });

  it('renders predictions + latency after a successful response', async () => {
    const user = userEvent.setup();
    render(wrap(<PredictTester endpointSlug="iris-v1" />));
    await user.click(screen.getByRole('button', { name: /send request/i }));
    // Response table renders — latency + both prediction rows.
    expect(await screen.findByText(/3\.1ms/)).toBeInTheDocument();
    expect(screen.getByText('Response')).toBeInTheDocument();
    expect(
      screen.getAllByRole('row').filter((r) => r.querySelector('td')).length,
    ).toBe(2);
  });
});
