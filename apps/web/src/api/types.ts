/**
 * Hand-written type mirrors of the pycaret-server Pydantic schemas.
 *
 * These are kept in sync manually while the UI surface is small; once it
 * grows past this file, swap to `npm run gen:api` for generated types.
 */

// ───────────────────────────────────────────────────────────── auth / setup

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: 'bearer';
  expires_in: number;
}

export interface User {
  id: string;
  email: string;
  display_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
}

export interface SetupStatus {
  is_bootstrapped: boolean;
  user_count: number;
  workspace_count: number;
}

export interface BootstrapRequest {
  email: string;
  password: string;
  display_name?: string | null;
  workspace_name: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

// ───────────────────────────────────────────────────────────── workspaces

export interface Workspace {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  created_by: string;
}

export interface Project {
  id: string;
  workspace_id: string;
  name: string;
  description: string | null;
  tags: string[];
  created_at: string;
  created_by: string;
}

// ───────────────────────────────────────────────────────────── experiments + runs

export type TaskType =
  | 'classification'
  | 'regression'
  | 'clustering'
  | 'anomaly'
  | 'time_series';

export interface Experiment {
  id: string;
  project_id: string;
  name: string;
  task: TaskType;
  target: string | null;
  setup_params: Record<string, unknown>;
  data_source_id: string | null;
  created_at: string;
  created_by: string;
}

export type RunStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface Run {
  id: string;
  experiment_id: string;
  status: RunStatus;
  started_at: string | null;
  finished_at: string | null;
  duration_ms: number | null;
  error: string | null;
  leaderboard: Record<string, unknown>[] | null;
  metrics_summary: Record<string, unknown> | null;
  snapshot: Record<string, unknown> | null;
  created_at: string;
  created_by: string;
}

export interface RunEvent {
  id: string;
  run_id: string;
  kind: string;
  message: string | null;
  payload: Record<string, unknown> | null;
  duration_ms: number | null;
  emitted_at: string;
}
