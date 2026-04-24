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

// ───────────────────────────────────────────────────────────── data sources

export type DataSourceKind = 'csv_upload' | 's3' | 'postgres';

export interface DataSource {
  id: string;
  workspace_id: string;
  name: string;
  kind: DataSourceKind;
  description: string | null;
  config: Record<string, unknown>;
  created_at: string;
  created_by: string;
}

// ───────────────────────────────────────────────────────────────── runs form

export type RunPlan = 'setup' | 'create' | 'compare';

/** Payload for POST /experiments/:id/runs. */
export interface RunCreate {
  plan: RunPlan;
  model_id?: string | null;
  plan_params?: Record<string, unknown>;
  sklearn_dataset?: string | null;
  data_inline?: Record<string, unknown>[] | null;
  data_source_id?: string | null;
  target?: string | null;
}

// ─────────────────────────────────────────────────────────── pipelines + deployments

export interface Pipeline {
  id: string;
  workspace_id: string;
  name: string;
  description: string | null;
  tags: string[];
  model_id: string | null;
  origin_run_id: string | null;
  stored_path: string;
  sha256: string | null;
  params: Record<string, unknown>;
  created_at: string;
  created_by: string;
}

export interface Deployment {
  id: string;
  workspace_id: string;
  pipeline_id: string;
  endpoint_slug: string;
  status: 'active' | 'paused' | 'archived';
  auth_mode: 'workspace' | 'api-key' | 'public';
  inference_count: number;
  last_inference_at: string | null;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
  error_count: number;
  created_at: string;
  created_by: string;
}

// ──────────────────────────────────────────────────────────── WebSocket payload

/** Event as delivered over the WebSocket (matches engine Event.to_dict()). */
export interface WsEvent {
  kind: string;
  message: string;
  payload: Record<string, unknown>;
  duration_ms: number | null;
  timestamp: number;
  experiment_id: string | null;
}

// ──────────────────────────────────────────────────────────── LLM advisory

export type LLMProviderName =
  | 'anthropic'
  | 'openai'
  | 'google'
  | 'azure_openai'
  | 'ollama'
  | 'custom_openai_compatible';

/** Canonical output envelope from every consultation. */
export interface LLMAdvice {
  suggested_config_json: Record<string, unknown>;
  suggested_action: string;
  reasoning_summary: string;
  risk_flags: string[];
}

export interface LLMProviderSettingRead {
  id: string;
  workspace_id: string;
  provider: LLMProviderName;
  base_url: string | null;
  model_name: string;
  enabled: boolean;
  config: Record<string, unknown> | null;
  /** Set server-side; we never ship the plaintext key back to the browser. */
  has_api_key: boolean;
  created_at: string;
  created_by: string;
}

export interface LLMProviderSettingWrite {
  provider: LLMProviderName;
  api_key?: string | null;
  base_url?: string | null;
  model_name: string;
  enabled?: boolean;
  config?: Record<string, unknown> | null;
}

export interface LLMConsultationRead {
  id: string;
  workspace_id: string;
  project_id: string | null;
  experiment_id: string | null;
  run_id: string | null;
  type: string;
  provider: string;
  model_name: string;
  prompt: string;
  response_json: LLMAdvice;
  generated_config_json: Record<string, unknown> | null;
  latency_ms: number | null;
  error: string | null;
  created_at: string;
  created_by: string;
}

export interface TestConnectionResponse {
  ok: boolean;
  provider: string;
  model_name: string;
  error: string | null;
  latency_ms: number | null;
}

// ──────────────────────────────────────────────────────────── API keys

export interface ApiKeyRead {
  id: string;
  name: string;
  prefix: string;
  workspace_id: string | null;
  scopes: string[] | null;
  expires_at: string | null;
  last_used_at: string | null;
  revoked_at: string | null;
  created_at: string;
}

/** Only present on the POST /auth/api-keys response. Never stored. */
export interface ApiKeyCreateResponse extends ApiKeyRead {
  token: string;
}

export interface ApiKeyCreateRequest {
  name: string;
  workspace_id?: string | null;
  expires_in_days?: number | null;
  scopes?: string[] | null;
}
