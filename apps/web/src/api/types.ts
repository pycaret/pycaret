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

// ───────────────────────────────────────────────────────── engine introspection

/** One parameter descriptor in the engine's setup-params schema. */
export type ParamKind = 'bool' | 'int' | 'float' | 'enum' | 'column' | 'string';

export interface SetupParam {
  name: string;
  kind: ParamKind;
  default: unknown;
  description: string;
  choices: string[] | null;
  minimum: number | null;
  maximum: number | null;
  required: boolean;
  group: string;
}

export interface SetupParamSchema {
  task: TaskType;
  parameters: SetupParam[];
  /** Group labels in display order. */
  groups: string[];
}

export interface ModelCard {
  id: string;
  name: string;
  task: TaskType;
  description: string;
  library: string;
  gpu_enabled: boolean;
  is_turbo: boolean;
  is_available: boolean;
  hyperparameters: unknown[];
  tags: string[];
}

export interface MetricCard {
  id: string;
  name: string;
  task: TaskType;
  greater_is_better: boolean;
  description: string;
  is_default: boolean;
  is_available: boolean;
}

// ───────────────────────────────────────────────────────── create-experiment form

/** Payload for POST /projects/:id/experiments. */
export interface ExperimentCreate {
  name: string;
  task: TaskType;
  target?: string | null;
  setup_params?: Record<string, unknown>;
  data_source_id?: string | null;
}
