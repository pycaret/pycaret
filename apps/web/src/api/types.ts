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
  business_context?: string | null;
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

export interface RunStats {
  total: number;
  succeeded: number;
  failed: number;
  cancelled: number;
  running: number;
  queued: number;
  last_status: RunStatus | null;
  last_finished_at: string | null;
}

export interface Experiment {
  id: string;
  project_id: string;
  name: string;
  task: TaskType;
  target: string | null;
  setup_params: Record<string, unknown>;
  data_source_id: string | null;
  business_context?: string | null;
  created_at: string;
  created_by: string;
  run_stats: RunStats;
}

export type RunStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

/** What kind of training event produced a Trial row.
 *  - ``compare`` — born from a compare_models / search / standalone create.
 *  - ``tuned`` / ``ensembled`` — follow-on action on one source trial.
 *  - ``blended`` / ``stacked`` — follow-on action on multiple source trials.
 *  - ``manual`` — reserved for future hand-uploaded pipelines.
 */
export type TrialKind =
  | 'compare'
  | 'tuned'
  | 'ensembled'
  | 'blended'
  | 'stacked'
  | 'manual';

/** Per-Trial lifecycle. Phase 0-v2: each Trial is its own scheduled
 *  unit of work. Workers flip a Trial through ``queued → running →
 *  succeeded | failed | cancelled`` independently of every other Trial
 *  in its parent Run. The Run's status is derived from the aggregate. */
export type TrialStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled';

/** Native Phase 0-v2 Trial. Run owns N Trials directly via
 *  ``Trial.run_id`` — each Trial carries its own metrics + artifact +
 *  status. Matches `pycaret_server.api.trials._serialise`. */
export interface Trial {
  id: string;
  run_id: string | null;
  experiment_id: string | null;
  workspace_id: string;
  name: string | null;
  model_id: string;
  kind: TrialKind;
  status: TrialStatus;
  rank: number | null;
  is_best: boolean;
  metrics: Record<string, number>;
  params: Record<string, unknown>;
  started_at: string | null;
  finished_at: string | null;
  duration_ms: number | null;
  error: string | null;
  stored_path: string | null;
  sha256: string | null;
  size_bytes: number | null;
  has_artifact: boolean;
  parent_trial_ids: string[];
  created_by_action_id: string | null;
  fitted_pipeline_id: string | null;
  notes: string | null;
  created_at: string | null;
  updated_at: string | null;
}

// ─────────────────────────────────────────── Phase 4 data catalog

export interface Secret {
  id: string;
  workspace_id: string;
  name: string;
  kind: string;
  last4: string | null;
  created_at: string | null;
  created_by: string;
}

export interface Connection {
  id: string;
  workspace_id: string;
  name: string;
  kind: string;
  config: Record<string, unknown>;
  secret_id: string | null;
  last_tested_at: string | null;
  last_test_status: string | null;
  last_test_error: string | null;
  created_at: string | null;
  created_by: string;
}

export interface Dataset {
  id: string;
  workspace_id: string;
  data_source_id: string;
  version: number;
  name: string | null;
  schema_json: {
    columns: Array<{ name: string; dtype: string; nullable: boolean; sample: unknown }>;
    sample_rows: Array<Record<string, unknown>>;
  } | null;
  row_count: number | null;
  byte_count: number | null;
  snapshot_uri: string | null;
  sample_uri: string | null;
  created_at: string | null;
}

export interface LineageEdge {
  id: string;
  source: { kind: string; id: string };
  target: { kind: string; id: string };
  relation: string;
  metadata: Record<string, unknown> | null;
  created_at: string | null;
}

// ─────────────────────────────────────────── Phase 5 Git

export type GitProvider = 'github' | 'gitlab' | 'gitea' | 'bitbucket';

export interface GitRepository {
  id: string;
  workspace_id: string;
  project_id: string | null;
  provider: GitProvider;
  url: string;
  default_branch: string;
  path_prefix: string | null;
  secret_id: string | null;
  enabled: boolean;
  auto_publish: boolean;
  last_push_at: string | null;
  last_push_status: string | null;
  last_push_sha: string | null;
  last_push_error: string | null;
  created_at: string | null;
  created_by: string;
}

export interface PublishResult {
  ok: boolean;
  sha?: string | null;
  error?: string | null;
}

// ─────────────────────────────────────────── Phase 7 model registry

export type RegisteredModelVersionStatus = 'staging' | 'production' | 'archived';

export interface RegisteredModel {
  id: string;
  workspace_id: string;
  project_id: string | null;
  name: string;
  description: string | null;
  current_version_id: string | null;
  tags: string[];
  owner_user_id: string | null;
  created_at: string | null;
  created_by: string;
}

export interface RegisteredModelVersion {
  id: string;
  registered_model_id: string;
  version: number;
  run_id: string | null;
  trial_id: string | null;
  stored_path: string;
  sha256: string | null;
  size_bytes: number | null;
  params: Record<string, unknown>;
  metrics: Record<string, number>;
  status: RegisteredModelVersionStatus;
  promoted_by: string | null;
  promoted_at: string | null;
  notes: string | null;
  created_at: string | null;
}

// ─────────────────────────────────────────── Phase 10 monitoring

export type AlertComparator = 'gt' | 'gte' | 'lt' | 'lte' | 'eq';
export type AlertDestination = 'slack' | 'email' | 'webhook';

export interface AlertRule {
  id: string;
  workspace_id: string;
  deployment_id: string | null;
  name: string;
  metric: string;
  comparator: AlertComparator;
  threshold: number;
  window_seconds: number;
  destination_kind: AlertDestination;
  destination_config: Record<string, unknown>;
  enabled: boolean;
  last_fired_at: string | null;
  last_status: string | null;
  last_error: string | null;
  created_at: string | null;
  created_by: string;
}

export interface MetricPoint {
  metric: string;
  ts: string;
  value: number;
  count: number;
  extra: Record<string, unknown> | null;
}

// ─────────────────────────────────────────── Phase 12 governance

export type ApprovalStatus =
  | 'pending'
  | 'approved'
  | 'rejected'
  | 'executed'
  | 'cancelled';

export interface ApprovalWorkflow {
  id: string;
  workspace_id: string;
  target_kind: string;
  target_id: string | null;
  action: string;
  status: ApprovalStatus;
  required_approvals: number;
  approvals: Array<{
    user_id: string;
    approved_at?: string;
    rejected_at?: string;
    comment?: string | null;
  }>;
  request_payload: Record<string, unknown> | null;
  requested_by: string;
  created_at: string | null;
  updated_at: string | null;
}

// ─────────────────────────────────────────── Phase 14 queue admin

export interface QueueRow {
  name: string;
  queued: number;
  running: number;
  succeeded: number;
  failed: number;
  cancelled: number;
  recent_throughput_1h: number;
}

export interface WorkerRow {
  worker_id: string;
  running_jobs: number;
  last_lock_at: string | null;
}

// ─────────────────────────────────────────── Phase 8 notebook runtime

export interface Notebook {
  id: string;
  workspace_id: string;
  project_id: string;
  name: string;
  path: string | null;
  kernel: string;
  object_uri: string | null;
  description: string | null;
  last_executed_at: string | null;
  last_modified_at: string | null;
  tags: string[];
  created_at: string | null;
  created_by: string;
}

export type NotebookSessionStatus =
  | 'starting'
  | 'running'
  | 'stopping'
  | 'stopped'
  | 'failed';

export interface NotebookSessionRow {
  id: string;
  workspace_id: string;
  notebook_id: string;
  user_id: string;
  status: NotebookSessionStatus;
  container_id: string | null;
  port: number | null;
  started_at: string | null;
  last_active_at: string | null;
  stopped_at: string | null;
  idle_timeout_seconds: number;
  cpu_limit: number | null;
  memory_mb_limit: number | null;
  error: string | null;
}

/** Returned by `POST /notebooks/{id}/sessions` — token + iframe URL
 *  only surface here so the frontend embeds the session safely. */
export interface NotebookSessionStart extends NotebookSessionRow {
  url: string;
  token: string;
}

// ─────────────────────────────────────────── Phase 11 statistical computing

export type AnalysisKind =
  | 'ttest'
  | 'welch_ttest'
  | 'paired_ttest'
  | 'mannwhitney'
  | 'anova_oneway'
  | 'kruskal'
  | 'chi2'
  | 'ols'
  | 'kaplan_meier'
  | 'logrank'
  | 'cox_ph'
  | 'arima'
  | 'prophet';

export interface Analysis {
  id: string;
  workspace_id: string;
  project_id: string;
  name: string;
  description: string | null;
  kind: AnalysisKind;
  params: Record<string, unknown>;
  data_source_id: string | null;
  created_at: string | null;
  created_by: string;
}

/** Uniform result envelope every analysis procedure returns. */
export interface AnalysisResult {
  test_statistic: number | null;
  p_value: number | null;
  effect_size: number | null;
  effect_size_name: string | null;
  ci_low: number | null;
  ci_high: number | null;
  table: Array<Record<string, unknown>>;
  interpretation: string;
  figure: PlotlyFigure | null;
  extra: Record<string, unknown>;
}

export interface AnalysisRunResponse {
  analysis_id: string;
  run_id: string;
  duration_ms: number;
  result: AnalysisResult;
}

export interface AnalysisRunRecord {
  run_id: string;
  status: string;
  started_at: string | null;
  duration_ms: number | null;
  metrics: AnalysisResult;
  params: Record<string, unknown>;
}

export interface Run {
  id: string;
  experiment_id: string;
  project_id: string | null;
  workspace_id: string | null;
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

export type RunPlan = 'setup' | 'create' | 'compare' | 'search';

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
  family_id: string | null;
  version: number;
  created_at: string;
  created_by: string;
  // Session-56 UI merge: every new promote creates BOTH a Pipeline and a
  // RegisteredModelVersion. The BE serializer back-fills these so the FE
  // can deep-link straight to /workspaces/:wsId/models/:registered_model_id
  // without an extra round-trip. Pre-session-56 Pipelines return null.
  registered_model_id: string | null;
  registered_model_version_id: string | null;
}

export interface Deployment {
  id: string;
  workspace_id: string;
  // pipeline_id is the legacy artifact pointer (still used by the predict
  // path); nullable for registry-only deploys created via the version
  // endpoint. Most flows now also populate the registry IDs below.
  pipeline_id: string | null;
  registered_model_id: string | null;
  registered_model_version_id: string | null;
  trial_id: string | null;
  run_id: string | null;
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

// ──────────────────────────────────────────────────────────── workspace members

export type WorkspaceRole =
  | 'owner'
  | 'admin'
  | 'project_admin'
  | 'ml_engineer'
  | 'data_scientist'
  | 'viewer'
  | 'service_account'
  | 'member';

/** Admin-class roles — match the backend's `ADMIN_ROLES` set. */
export const ADMIN_WORKSPACE_ROLES: ReadonlySet<WorkspaceRole> = new Set([
  'owner',
  'admin',
  'project_admin',
]);

export interface MemberRead {
  user_id: string;
  email: string;
  display_name: string | null;
  role: WorkspaceRole;
  is_active: boolean;
  created_at: string;
}

export interface InviteRequest {
  email: string;
  role?: WorkspaceRole;
}

export interface PatchRoleRequest {
  role: WorkspaceRole;
}

// ──────────────────────────────────────────────────────────── drift reports

export type DriftStatus = 'none' | 'mild' | 'moderate' | 'severe';

export type DriftKind = 'psi' | 'ks' | 'chi2' | 'missing_rate';

export interface FeatureDriftEntry {
  score: number;
  kind: DriftKind;
}

export interface PredictionDrift {
  kind: 'js' | 'ks';
  score: number;
  baseline_mean?: number;
  current_mean?: number;
}

export interface DriftReportRead {
  id: string;
  deployment_id: string;
  baseline_artifact_id: string | null;
  window_start: string;
  window_end: string;
  drift_score: number;
  drift_status: DriftStatus;
  feature_drift_json: Record<string, FeatureDriftEntry>;
  prediction_drift_json: PredictionDrift | null;
  sample_size: number | null;
  created_at: string;
  created_by: string;
}

export interface DriftReportCreate {
  window_start: string;
  window_end: string;
  drift_score: number;
  feature_drift_json: Record<string, FeatureDriftEntry>;
  prediction_drift_json?: PredictionDrift | null;
  sample_size?: number | null;
  baseline_artifact_id?: string | null;
}

// ──────────────────────────────────────────────────────────── audit logs

export interface AuditLogRead {
  id: string;
  workspace_id: string | null;
  user_id: string | null;
  action: string;
  method: string;
  path: string;
  target_type: string | null;
  target_id: string | null;
  status_code: number | null;
  payload: Record<string, unknown> | null;
  ip_address: string | null;
  user_agent: string | null;
  created_at: string;
}

export interface AuditLogFilters {
  action?: string;
  user_id?: string;
  workspace_id?: string;
  target_type?: string;
  target_id?: string;
  since?: string;
  until?: string;
  limit?: number;
  offset?: number;
}

// ──────────────────────────────────────────────────────────── plots

/** A Plotly figure as JSON — pass directly to `<Plot data={fig.data} layout={fig.layout} />`. */
export interface PlotlyFigure {
  data: Array<Record<string, unknown>>;
  layout: Record<string, unknown>;
  frames?: Array<Record<string, unknown>>;
  config?: Record<string, unknown>;
}

export interface PlotEnvelope {
  kind: string;
  task: string;
  figure: PlotlyFigure;
  generated_at: string;
}

export interface PlotKindDetail {
  requires: string[];
  binary_only?: boolean;
}

export interface PlotRegistry {
  tasks: Record<string, string[]>;
  details: Record<string, Record<string, PlotKindDetail>>;
}

/**
 * Recursive serialization of a fitted sklearn Pipeline. The shape is
 * driven by what the trial-detail endpoint returns; the React diagram
 * mirrors the discriminated union directly.
 */
export type PipelineNodeType =
  | 'pipeline'
  | 'column_transformer'
  | 'feature_union'
  | 'leaf'
  | 'passthrough';

export interface PipelineNodeBase {
  name: string;
  class: string;
  module: string;
  params: Record<string, unknown>;
  is_estimator?: boolean;
  is_root?: boolean;
}

export interface PipelineLeafNode extends PipelineNodeBase {
  type: 'leaf' | 'passthrough';
}

export interface PipelineCompositeNode extends PipelineNodeBase {
  type: 'pipeline';
  children: PipelineNode[];
}

export interface PipelineColumnTransformerBranch {
  name: string;
  columns: (string | number | boolean)[];
  transformer: PipelineNode;
}

export interface PipelineColumnTransformerNode extends PipelineNodeBase {
  type: 'column_transformer';
  branches: PipelineColumnTransformerBranch[];
}

export interface PipelineFeatureUnionBranch {
  name: string;
  transformer: PipelineNode;
}

export interface PipelineFeatureUnionNode extends PipelineNodeBase {
  type: 'feature_union';
  branches: PipelineFeatureUnionBranch[];
}

export type PipelineNode =
  | PipelineLeafNode
  | PipelineCompositeNode
  | PipelineColumnTransformerNode
  | PipelineFeatureUnionNode;
