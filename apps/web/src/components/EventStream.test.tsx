import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import { EventStream } from './EventStream';
import { useAuthStore } from '@/state/auth';

/**
 * We replace the global WebSocket with a controllable fake so the test can
 * drive open / message / close events synchronously and assert the rendered
 * output. This is the only way to exercise the component deterministically
 * without a real backend + live run.
 */

// ───────────────────────────────────────────────────────── fake WebSocket

type Handler = (ev: { data?: string; code?: number }) => void;

class FakeWebSocket {
  static READYSTATE_OPEN = 1;
  static READYSTATE_CLOSED = 3;

  static instances: FakeWebSocket[] = [];

  url: string;
  onopen: Handler | null = null;
  onmessage: Handler | null = null;
  onclose: Handler | null = null;
  onerror: Handler | null = null;
  readyState = 0;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  // Called by the test to simulate server behaviour.
  _open() {
    this.readyState = FakeWebSocket.READYSTATE_OPEN;
    this.onopen?.({});
  }
  _message(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
  _close(code = 1000) {
    this.readyState = FakeWebSocket.READYSTATE_CLOSED;
    this.onclose?.({ code });
  }
  close() {
    this.readyState = FakeWebSocket.READYSTATE_CLOSED;
    this.onclose?.({ code: 1000 });
  }
}

// ───────────────────────────────────────────────────────── tests

describe('<EventStream>', () => {
  const realWS = globalThis.WebSocket;

  beforeEach(() => {
    FakeWebSocket.instances = [];
    // @ts-expect-error — replacing for the test scope
    globalThis.WebSocket = FakeWebSocket;
    // Pretend the user is signed in; the component gates on accessToken presence.
    useAuthStore.setState({
      accessToken: 'test-access-token',
      refreshToken: null,
      user: null,
    });
  });

  afterEach(() => {
    globalThis.WebSocket = realWS;
    useAuthStore.setState({ accessToken: null, refreshToken: null, user: null });
    vi.useRealTimers();
  });

  it('connects to /api/v1/runs/:id/events/ws with the bearer token as query param', () => {
    render(<EventStream runId="abc-123" />);
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(FakeWebSocket.instances[0].url).toContain('/api/v1/runs/abc-123/events/ws');
    expect(FakeWebSocket.instances[0].url).toContain('token=test-access-token');
  });

  it('renders events as they arrive and flips the indicator to `live` on open', () => {
    render(<EventStream runId="run-1" />);
    const ws = FakeWebSocket.instances[0];

    // Before open — connecting indicator.
    expect(screen.getByText(/connecting/i)).toBeInTheDocument();

    act(() => {
      ws._open();
    });
    expect(screen.getByText(/live/i)).toBeInTheDocument();

    act(() => {
      ws._message({
        kind: 'experiment.started',
        message: 'Starting classification experiment',
        payload: {},
        duration_ms: null,
        timestamp: 1700000000,
        experiment_id: null,
      });
    });
    expect(screen.getByText(/Starting classification experiment/)).toBeInTheDocument();
    expect(screen.getByText(/EXPERIMENT STARTED/)).toBeInTheDocument();

    act(() => {
      ws._message({
        kind: 'experiment.fitted',
        message: 'Experiment fitted and ready',
        payload: {},
        duration_ms: 3450,
        timestamp: 1700000003,
        experiment_id: null,
      });
    });
    // Counter updates to 2 events.
    expect(screen.getByText(/2 events/)).toBeInTheDocument();
    // Duration rendered in short form
    expect(screen.getByText('3.5s')).toBeInTheDocument();
  });

  it('shows `closed` status after the terminal run.closed sentinel', () => {
    render(<EventStream runId="run-2" />);
    const ws = FakeWebSocket.instances[0];
    act(() => {
      ws._open();
      ws._message({
        kind: 'run.closed',
        message: '',
        payload: {},
        duration_ms: null,
        timestamp: 1700000010,
        experiment_id: null,
      });
      ws._close(1000);
    });
    // Status indicator renders inside a pill — find by the exact text "closed"
    // on a span carrying a pill class so we don't collide with the
    // "RUN CLOSED" event-log entry.
    const pill = screen
      .getAllByText('closed')
      .find((el) => el.className.includes('pill'));
    expect(pill).toBeTruthy();
  });

  it('surfaces an auth-failure close code as an error', () => {
    render(<EventStream runId="run-3" />);
    act(() => {
      FakeWebSocket.instances[0]._close(4401);
    });
    expect(screen.getByText(/auth failure \(4401\)/)).toBeInTheDocument();
  });
});
