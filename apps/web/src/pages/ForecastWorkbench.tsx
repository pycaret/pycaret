/**
 * Forecast Workbench — `/runs/:runId/forecast`.
 *
 * Time-series-specific dashboard for a fitted forecasting run. Shows:
 *  - Forecast vs. actual with optional CI band.
 *  - Trend / seasonal / residual decomposition.
 *  - Residual diagnostics (4-panel).
 *  - ACF + PACF of the training series.
 *
 * All charts come from `/api/v1/runs/:runId/plots/:kind`.
 */

import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';

import { plotsApi, runsApi } from '../api/endpoints';
import { PlotlyFigure } from '../components/PlotlyFigure';
import type { PlotEnvelope } from '../api/types';

function usePlot(runId: string, kind: string) {
  return useQuery<PlotEnvelope, Error>({
    queryKey: ['runs', runId, 'plots', kind],
    queryFn: () => plotsApi.forRun(runId, kind),
    enabled: !!runId,
    staleTime: 60_000,
  });
}

export function ForecastWorkbench() {
  const { runId = '' } = useParams<{ runId: string }>();
  const run = useQuery({
    queryKey: ['runs', runId],
    queryFn: () => runsApi.get(runId),
    enabled: !!runId,
  });

  const forecast = usePlot(runId, 'forecast');
  const decomposition = usePlot(runId, 'decomposition');
  const residuals = usePlot(runId, 'residual_diagnostics');
  const acf = usePlot(runId, 'acf');
  const pacf = usePlot(runId, 'pacf');

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <nav style={{ fontSize: 12, color: '#94A3B8' }}>
          <Link to={`/runs/${runId}`} style={{ color: 'inherit' }}>
            Run {runId.slice(0, 8)}…
          </Link>{' '}
          / Forecast workbench
        </nav>
        <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0, color: '#0F172A' }}>
          Forecast workbench
        </h1>
        <p style={{ color: '#64748B', fontSize: 13, margin: 0 }}>
          {run.data?.status === 'succeeded'
            ? 'Diagnostics for the time-series model.'
            : 'Awaiting run completion…'}
        </p>
      </header>

      <PlotlyFigure
        figure={forecast.data?.figure}
        loading={forecast.isLoading}
        error={forecast.error ?? undefined}
        onRetry={() => forecast.refetch()}
        title="Forecast vs. actual"
        caption="Point predictions over the holdout horizon, with optional confidence band."
        height={420}
      />

      <PlotlyFigure
        figure={decomposition.data?.figure}
        loading={decomposition.isLoading}
        error={decomposition.error ?? undefined}
        onRetry={() => decomposition.refetch()}
        title="Trend / seasonal / residual decomposition"
        height={620}
      />

      <PlotlyFigure
        figure={residuals.data?.figure}
        loading={residuals.isLoading}
        error={residuals.error ?? undefined}
        onRetry={() => residuals.refetch()}
        title="Residual diagnostics"
        height={560}
      />

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
          gap: 16,
        }}
      >
        <PlotlyFigure
          figure={acf.data?.figure}
          loading={acf.isLoading}
          error={acf.error ?? undefined}
          onRetry={() => acf.refetch()}
          title="Autocorrelation"
        />
        <PlotlyFigure
          figure={pacf.data?.figure}
          loading={pacf.isLoading}
          error={pacf.error ?? undefined}
          onRetry={() => pacf.refetch()}
          title="Partial autocorrelation"
        />
      </div>
    </div>
  );
}
