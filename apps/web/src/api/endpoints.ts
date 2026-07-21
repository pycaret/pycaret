/**
 * Typed API methods. One function per route the UI uses.
 *
 * Keep call sites small — TanStack Query wraps these.
 */

import { api } from './client';
import type {
  ApiKeyCreateRequest,
  ApiKeyCreateResponse,
  ApiKeyRead,
  AuditLogFilters,
  AuditLogRead,
  BootstrapRequest,
  DataSource,
  Deployment,
  DriftReportCreate,
  DriftReportRead,
  Experiment,
  ExperimentCreate,
  InviteRequest,
  LLMConsultationRead,
  LLMProviderSettingRead,
  LLMProviderSettingWrite,
  LoginRequest,
  MemberRead,
  MetricCard,
  ModelCard,
  PatchRoleRequest,
  Pipeline,
  PipelineNode,
  PlotEnvelope,
  Project,
  Run,
  RunCreate,
  RunEvent,
  SetupParamSchema,
  SetupStatus,
  TaskType,
  TestConnectionResponse,
  TokenPair,
  AlertRule,
  Analysis,
  AnalysisKind,
  AnalysisResult,
  AnalysisRunRecord,
  AnalysisRunResponse,
  ApprovalWorkflow,
  Connection,
  Dataset,
  GitRepository,
  LineageEdge,
  MetricPoint,
  Notebook,
  NotebookSessionRow,
  NotebookSessionStart,
  PublishResult,
  QueueRow,
  RegisteredModel,
  RegisteredModelVersion,
  Secret,
  Trial,
  TrialKind,
  User,
  WorkerRow,
  Workspace,
} from './types';

// ───────────────────────────── setup

export const setupApi = {
  status: () => api.get<SetupStatus>('/setup/status').then((r) => r.data),
  bootstrap: (body: BootstrapRequest) =>
    api.post<TokenPair>('/setup/bootstrap', body).then((r) => r.data),
};

// ───────────────────────────── auth

export const authApi = {
  login: (body: LoginRequest) => api.post<TokenPair>('/auth/login', body).then((r) => r.data),
  refresh: (refresh_token: string) =>
    api.post<TokenPair>('/auth/refresh', { refresh_token }).then((r) => r.data),
  logout: () => api.post<void>('/auth/logout').then((r) => r.data),
  me: () => api.get<User>('/auth/me').then((r) => r.data),
};

// ───────────────────────────── workspaces

export const workspacesApi = {
  list: () => api.get<Workspace[]>('/workspaces').then((r) => r.data),
  get: (id: string) => api.get<Workspace>(`/workspaces/${id}`).then((r) => r.data),
  create: (body: { name: string; description?: string }) =>
    api.post<Workspace>('/workspaces', body).then((r) => r.data),
  remove: (id: string) => api.delete<void>(`/workspaces/${id}`).then((r) => r.data),
};

// ───────────────────────────── projects

export const projectsApi = {
  list: (workspace_id: string) =>
    api.get<Project[]>(`/workspaces/${workspace_id}/projects`).then((r) => r.data),
  get: (workspace_id: string, project_id: string) =>
    api
      .get<Project>(`/workspaces/${workspace_id}/projects/${project_id}`)
      .then((r) => r.data),
  create: (
    workspace_id: string,
    body: { name: string; description?: string; tags?: string[] },
  ) =>
    api
      .post<Project>(`/workspaces/${workspace_id}/projects`, body)
      .then((r) => r.data),
  remove: (workspace_id: string, project_id: string) =>
    api
      .delete<void>(`/workspaces/${workspace_id}/projects/${project_id}`)
      .then((r) => r.data),
};

// ───────────────────────────── experiments

export const experimentsApi = {
  list: (project_id: string) =>
    api.get<Experiment[]>(`/projects/${project_id}/experiments`).then((r) => r.data),
  get: (project_id: string, experiment_id: string) =>
    api
      .get<Experiment>(`/projects/${project_id}/experiments/${experiment_id}`)
      .then((r) => r.data),
  create: (project_id: string, body: ExperimentCreate) =>
    api
      .post<Experiment>(`/projects/${project_id}/experiments`, body)
      .then((r) => r.data),
  remove: (project_id: string, experiment_id: string) =>
    api
      .delete<void>(`/projects/${project_id}/experiments/${experiment_id}`)
      .then((r) => r.data),
};

// ───────────────────────────── engine introspection (drives dynamic form + dropdowns)

export const describeApi = {
  setupParams: (task: TaskType) =>
    api
      .get<SetupParamSchema>('/describe/setup-params', { params: { task } })
      .then((r) => r.data),
  models: (task: TaskType) =>
    api.get<ModelCard[]>('/describe/models', { params: { task } }).then((r) => r.data),
  metrics: (task: TaskType) =>
    api.get<MetricCard[]>('/describe/metrics', { params: { task } }).then((r) => r.data),
};

// ───────────────────────────── runs

export const runsApi = {
  listForExperiment: (experiment_id: string) =>
    api.get<Run[]>(`/experiments/${experiment_id}/runs`).then((r) => r.data),
  submit: (experiment_id: string, body: RunCreate) =>
    api.post<Run>(`/experiments/${experiment_id}/runs`, body).then((r) => r.data),
  get: (run_id: string) => api.get<Run>(`/runs/${run_id}`).then((r) => r.data),
  events: (
    run_id: string,
    opts?: { after_id?: string; limit?: number; tail?: boolean },
  ) =>
    api
      .get<RunEvent[]>(`/runs/${run_id}/events`, { params: opts })
      .then((r) => r.data),
  cancel: (run_id: string) => api.post<Run>(`/runs/${run_id}/cancel`).then((r) => r.data),
  wait: (run_id: string, timeout_s = 30) =>
    api
      .post<Run>(`/runs/${run_id}/wait`, null, { params: { timeout_s } })
      .then((r) => r.data),
  promote: (run_id: string, body: { name: string; description?: string; tags?: string[] }) =>
    api.post<Pipeline>(`/runs/${run_id}/promote`, body).then((r) => r.data),
  trials: (run_id: string) =>
    api
      .get<{
        run_id: string;
        items: Array<{
          id: string;
          run_id: string | null;
          model_id: string;
          name: string | null;
          rank: number | null;
          metrics: Record<string, number>;
          is_best: boolean;
          status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
          started_at: string | null;
          finished_at: string | null;
          duration_ms: number | null;
          error: string | null;
          fitted_pipeline_id: string | null;
          has_artifact: boolean;
          size_bytes: number | null;
          kind: TrialKind;
          parent_trial_ids: string[];
          created_at: string | null;
        }>;
      }>(`/runs/${run_id}/trials`)
      .then((r) => r.data),
  trial: (run_id: string, trial_id: string) =>
    api
      .get<{
        id: string;
        run_id: string;
        workspace_id: string;
        model_id: string;
        rank: number;
        metrics: Record<string, number>;
        is_best: boolean;
        fitted_pipeline_id: string | null;
        params: Record<string, unknown>;
        pipeline_steps: Array<{
          index: number;
          name: string;
          class: string;
          module: string;
          params: Record<string, unknown>;
          is_estimator: boolean;
        }>;
        pipeline_tree: PipelineNode | null;
        available_plots: string[];
        task: string | null;
        run_snapshot: Record<string, unknown>;
        notes: string | null;
        input_schema: {
          columns: Array<{ name: string; dtype: string }>;
          sample_row: Record<string, unknown>;
          target: string | null;
        } | null;
        stored_path: string | null;
        sha256: string | null;
        size_bytes: number | null;
        has_artifact: boolean;
        kind: TrialKind;
        parent_trial_ids: string[];
        created_at: string | null;
      }>(`/runs/${run_id}/trials/${trial_id}`)
      .then((r) => r.data),
  trialDownload: async (
    run_id: string,
    trial_id: string,
    filename: string,
  ): Promise<void> => {
    // Plain `<a href download>` skips the bearer token; we have to fetch
    // through axios so the auth interceptor attaches the header, then
    // hand the blob to a click-once anchor.
    const r = await api.get<Blob>(
      `/runs/${run_id}/trials/${trial_id}/download`,
      { responseType: 'blob' },
    );
    const blob = r.data instanceof Blob ? r.data : new Blob([r.data]);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  },
  trialPlot: (run_id: string, trial_id: string, kind: string) =>
    api
      .get<PlotEnvelope>(
        `/runs/${encodeURIComponent(run_id)}/trials/${encodeURIComponent(trial_id)}/plots/${encodeURIComponent(kind)}`,
      )
      .then((r) => r.data),
  trialPromote: (
    run_id: string,
    trial_id: string,
    body: { name: string; description?: string; tags?: string[] },
  ) =>
    api
      .post<Pipeline>(`/runs/${run_id}/trials/${trial_id}/promote`, body)
      .then((r) => r.data),
  trialUnpromote: (run_id: string, trial_id: string) =>
    api
      .delete<void>(`/runs/${run_id}/trials/${trial_id}/promote`)
      .then((r) => r.data),
  trialTune: (
    run_id: string,
    trial_id: string,
    body: {
      n_iter?: number;
      optimize?: string;
      fold?: number;
    },
  ) =>
    api
      .post<{
        action: string;
        run_id: string;
        source_trial_id: string;
        status: string;
      }>(`/runs/${run_id}/trials/${trial_id}/tune`, body)
      .then((r) => r.data),
  trialEnsemble: (
    run_id: string,
    trial_id: string,
    body: {
      method: 'Bagging' | 'Boosting';
      n_estimators?: number;
      fold?: number;
    },
  ) =>
    api
      .post<{
        action: string;
        run_id: string;
        source_trial_id: string;
        status: string;
      }>(`/runs/${run_id}/trials/${trial_id}/ensemble`, body)
      .then((r) => r.data),
  runBlend: (
    run_id: string,
    body: {
      source_trial_ids: string[];
      method?: 'auto' | 'hard' | 'soft';
      weights?: number[];
      fold?: number;
    },
  ) =>
    api
      .post<{
        action: string;
        run_id: string;
        source_trial_ids: string[];
        status: string;
      }>(`/runs/${run_id}/blend`, body)
      .then((r) => r.data),
  runStack: (
    run_id: string,
    body: {
      source_trial_ids: string[];
      meta_model?: string;
      fold?: number;
    },
  ) =>
    api
      .post<{
        action: string;
        run_id: string;
        source_trial_ids: string[];
        status: string;
      }>(`/runs/${run_id}/stack`, body)
      .then((r) => r.data),
  trialPatch: (
    run_id: string,
    trial_id: string,
    body: { notes?: string | null },
  ) =>
    api
      .patch<{ id: string; notes: string | null }>(
        `/runs/${run_id}/trials/${trial_id}`,
        body,
      )
      .then((r) => r.data),
  trialPredict: (
    run_id: string,
    trial_id: string,
    body: { rows: Record<string, unknown>[] },
  ) =>
    api
      .post<{
        predictions: (string | number | boolean | null)[];
        probabilities?: number[][];
        classes?: (string | number | boolean | null)[];
        scores?: number[];
      }>(`/runs/${run_id}/trials/${trial_id}/predict`, body)
      .then((r) => r.data),
  trialCv: (run_id: string, trial_id: string, folds = 5) =>
    api
      .get<{
        task: string;
        folds: number;
        scoring: string[];
        rows: Array<{ fold: number } & Record<string, number>>;
        summary: Record<
          string,
          { mean: number; std: number; min: number; max: number }
        >;
      }>(`/runs/${run_id}/trials/${trial_id}/cv`, { params: { folds } })
      .then((r) => r.data),
  trialCohorts: (run_id: string, trial_id: string, column: string) =>
    api
      .get<{
        task: string;
        column: string | null;
        rows: Array<{
          value: string;
          n: number;
          metrics: Record<string, number>;
        }>;
        available_columns: string[];
      }>(`/runs/${run_id}/trials/${trial_id}/cohorts`, {
        params: column ? { column } : undefined,
      })
      .then((r) => r.data),
};

// ─────────────────────── Phase 0-v2 trial-centric read API
//
// Trials are owned by Runs directly (``Trial.run_id``). The
// ``runsApi.trials(...)`` route is the primary list surface. These
// endpoints exist for cross-experiment views + direct deep-linking.

export const trialsApi = {
  /** Every Trial in an Experiment, optionally narrowed by kind or run. */
  forExperiment: (
    experiment_id: string,
    opts?: { kind?: TrialKind; run_id?: string; limit?: number },
  ) =>
    api
      .get<{ experiment_id: string; items: Trial[] }>(
        `/experiments/${experiment_id}/trials`,
        { params: opts },
      )
      .then((r) => r.data),

  /** Fetch one Trial by id — no Run umbrella required. */
  get: (trial_id: string) =>
    api.get<Trial>(`/trials/${trial_id}`).then((r) => r.data),

  /** Update user-editable fields (``name``, ``notes``). */
  patch: (trial_id: string, body: { name?: string | null; notes?: string | null }) =>
    api.patch<Trial>(`/trials/${trial_id}`, body).then((r) => r.data),

  /** Delete the Trial. 409 if a Pipeline still references it via
   *  ``fitted_pipeline_id``. */
  delete: (trial_id: string) =>
    api.delete<void>(`/trials/${trial_id}`).then(() => undefined),
};

// ───────────────────────────── Phase 4: data catalog (secrets / connections / datasets / lineage)

export const secretsApi = {
  list: (workspace_id: string) =>
    api.get<Secret[]>(`/workspaces/${workspace_id}/secrets`).then((r) => r.data),
  create: (workspace_id: string, body: { name: string; value: string; kind?: string }) =>
    api.post<Secret>(`/workspaces/${workspace_id}/secrets`, body).then((r) => r.data),
  delete: (workspace_id: string, secret_id: string) =>
    api
      .delete<void>(`/workspaces/${workspace_id}/secrets/${secret_id}`)
      .then(() => undefined),
};

export const connectionsApi = {
  list: (workspace_id: string) =>
    api
      .get<Connection[]>(`/workspaces/${workspace_id}/connections`)
      .then((r) => r.data),
  create: (
    workspace_id: string,
    body: {
      name: string;
      kind: string;
      config?: Record<string, unknown>;
      secret_id?: string | null;
    },
  ) =>
    api
      .post<Connection>(`/workspaces/${workspace_id}/connections`, body)
      .then((r) => r.data),
  test: (workspace_id: string, connection_id: string) =>
    api
      .post<{ ok: boolean; error: string | null }>(
        `/workspaces/${workspace_id}/connections/${connection_id}/test`,
      )
      .then((r) => r.data),
  delete: (workspace_id: string, connection_id: string) =>
    api
      .delete<void>(`/workspaces/${workspace_id}/connections/${connection_id}`)
      .then(() => undefined),
  kinds: () =>
    api.get<{ kinds: string[] }>('/datasource-kinds').then((r) => r.data),
  /** List the tables / objects exposed by a Connection's backend. The
   *  "Register as DataSource" flow drives off this list. */
  tables: (workspace_id: string, connection_id: string) =>
    api
      .get<{ connection_id: string; kind: string; tables: string[] }>(
        `/workspaces/${workspace_id}/connections/${connection_id}/tables`,
      )
      .then((r) => r.data),
  /** Register a Connection table as a DataSource. */
  registerAsDataSource: (
    workspace_id: string,
    body: {
      connection_id: string;
      table: string;
      name?: string;
      description?: string;
    },
  ) =>
    api
      .post(`/workspaces/${workspace_id}/data-sources/from-connection`, body)
      .then((r) => r.data),
};

export const datasetsApi = {
  list: (data_source_id: string) =>
    api
      .get<Dataset[]>(`/data-sources/${data_source_id}/datasets`)
      .then((r) => r.data),
  refresh: (data_source_id: string) =>
    api
      .post<Dataset>(`/data-sources/${data_source_id}/refresh`)
      .then((r) => r.data),
};

export const lineageApi = {
  forWorkspace: (
    workspace_id: string,
    opts?: { node_kind?: string; node_id?: string; depth?: number },
  ) =>
    api
      .get<{ edges: LineageEdge[]; nodes?: { kind: string; id: string }[] }>(
        `/workspaces/${workspace_id}/lineage`,
        { params: opts },
      )
      .then((r) => r.data),
};

// ───────────────────────────── Phase 5: Git integration

export const gitApi = {
  list: (workspace_id: string) =>
    api
      .get<GitRepository[]>(`/workspaces/${workspace_id}/git-repositories`)
      .then((r) => r.data),
  create: (
    workspace_id: string,
    body: {
      provider: GitRepository['provider'];
      url: string;
      project_id?: string;
      default_branch?: string;
      path_prefix?: string;
      secret_id?: string;
      auto_publish?: boolean;
    },
  ) =>
    api
      .post<GitRepository>(`/workspaces/${workspace_id}/git-repositories`, body)
      .then((r) => r.data),
  patch: (
    workspace_id: string,
    repo_id: string,
    body: Partial<{
      enabled: boolean;
      auto_publish: boolean;
      default_branch: string;
      path_prefix: string | null;
      project_id: string | null;
      secret_id: string | null;
    }>,
  ) =>
    api
      .patch<GitRepository>(
        `/workspaces/${workspace_id}/git-repositories/${repo_id}`,
        body,
      )
      .then((r) => r.data),
  delete: (workspace_id: string, repo_id: string) =>
    api
      .delete<void>(`/workspaces/${workspace_id}/git-repositories/${repo_id}`)
      .then(() => undefined),
  publish: (
    repo_id: string,
    body?: { commit_message?: string; project_id?: string },
  ) =>
    api.post<PublishResult>(`/git-repositories/${repo_id}/publish`, body).then((r) => r.data),
};

// ───────────────────────────── Phase 7: model registry

export const registryApi = {
  list: (workspace_id: string) =>
    api
      .get<RegisteredModel[]>(`/workspaces/${workspace_id}/registered-models`)
      .then((r) => r.data),
  create: (
    workspace_id: string,
    body: {
      name: string;
      description?: string;
      project_id?: string;
      tags?: string[];
    },
  ) =>
    api
      .post<RegisteredModel>(`/workspaces/${workspace_id}/registered-models`, body)
      .then((r) => r.data),
  get: (model_id: string) =>
    api.get<RegisteredModel>(`/registered-models/${model_id}`).then((r) => r.data),
  versions: (model_id: string) =>
    api
      .get<RegisteredModelVersion[]>(`/registered-models/${model_id}/versions`)
      .then((r) => r.data),
  // Removed in session 56: the standalone "create a new version" call.
  // Versions are now created automatically by runsApi.trialPromote — see
  // services/api/pycaret_server/api/runs.py::promote_trial. The old form
  // was always-broken (read attributes off Run that live on Trial) and had
  // zero test coverage, so its removal is forward-only.
  setStatus: (
    model_id: string,
    version_id: string,
    body: { status: RegisteredModelVersion['status']; set_current?: boolean },
  ) =>
    api
      .post<RegisteredModelVersion>(
        `/registered-models/${model_id}/versions/${version_id}/promote`,
        body,
      )
      .then((r) => r.data),
  rollback: (model_id: string, version_id: string) =>
    api
      .post<RegisteredModelVersion>(
        `/registered-models/${model_id}/versions/${version_id}/rollback`,
      )
      .then((r) => r.data),
  /** Phase 12: open an ApprovalWorkflow to promote-to-production.
   *  Requester is auto-signed; with ``required_approvals=1`` this flips
   *  the workflow to ``approved`` immediately so the next step is just
   *  hitting Execute. */
  requestPromotion: (
    model_id: string,
    version_id: string,
    body?: { required_approvals?: number; comment?: string },
  ) =>
    api
      .post<{
        id: string;
        status: 'pending' | 'approved' | 'rejected' | 'executed' | 'cancelled';
        required_approvals: number;
        approvals: Array<{ user_id: string; approved_at?: string; comment?: string | null }>;
      }>(
        `/registered-models/${model_id}/versions/${version_id}/request-promotion`,
        body || {},
      )
      .then((r) => r.data),
  delete: (model_id: string) =>
    api
      .delete<void>(`/registered-models/${model_id}`)
      .then(() => undefined),
  /** Create a Deployment directly off a RegisteredModelVersion (Phase 7
   *  flow — no legacy pipeline detour). */
  deploy: (
    model_id: string,
    version_id: string,
    body: { endpoint_slug: string; auth_mode?: 'workspace' | 'api-key' | 'public' },
  ) =>
    api
      .post(
        `/registered-models/${model_id}/versions/${version_id}/deployments`,
        body,
      )
      .then((r) => r.data),
};

// ───────────────────────────── Phase 10: monitoring

export const monitoringApi = {
  listRules: (workspace_id: string) =>
    api
      .get<AlertRule[]>(`/workspaces/${workspace_id}/alert-rules`)
      .then((r) => r.data),
  createRule: (
    workspace_id: string,
    body: Omit<AlertRule, 'id' | 'workspace_id' | 'last_fired_at' | 'last_status' | 'last_error' | 'created_at' | 'created_by' | 'enabled'> & { enabled?: boolean },
  ) =>
    api
      .post<AlertRule>(`/workspaces/${workspace_id}/alert-rules`, body)
      .then((r) => r.data),
  patchRule: (
    rule_id: string,
    body: Partial<Pick<AlertRule, 'enabled' | 'threshold' | 'comparator' | 'window_seconds' | 'destination_config'>>,
  ) => api.patch<AlertRule>(`/alert-rules/${rule_id}`, body).then((r) => r.data),
  deleteRule: (rule_id: string) =>
    api.delete<void>(`/alert-rules/${rule_id}`).then(() => undefined),
  metrics: (
    deployment_id: string,
    opts?: { metric?: string; since_seconds?: number; limit?: number },
  ) =>
    api
      .get<{
        deployment_id: string;
        since: string;
        points: MetricPoint[];
      }>(`/deployments/${deployment_id}/metrics`, { params: opts })
      .then((r) => r.data),
  ingest: (
    deployment_id: string,
    body: { metric: string; value: number; ts?: string; count?: number; extra?: Record<string, unknown> },
  ) =>
    api
      .post<{ id: string; ts: string }>(
        `/deployments/${deployment_id}/metrics`,
        body,
      )
      .then((r) => r.data),
};

// ───────────────────────────── Phase 12: governance

export const governanceApi = {
  list: (workspace_id: string, status?: string) =>
    api
      .get<ApprovalWorkflow[]>(`/workspaces/${workspace_id}/approvals`, {
        params: status ? { status_filter: status } : undefined,
      })
      .then((r) => r.data),
  open: (
    workspace_id: string,
    body: {
      target_kind: string;
      target_id?: string;
      action: string;
      required_approvals?: number;
      request_payload?: Record<string, unknown>;
    },
  ) =>
    api
      .post<ApprovalWorkflow>(`/workspaces/${workspace_id}/approvals`, body)
      .then((r) => r.data),
  approve: (approval_id: string, body?: { comment?: string }) =>
    api
      .post<ApprovalWorkflow>(`/approvals/${approval_id}/approve`, body || {})
      .then((r) => r.data),
  reject: (approval_id: string, body?: { comment?: string }) =>
    api
      .post<ApprovalWorkflow>(`/approvals/${approval_id}/reject`, body || {})
      .then((r) => r.data),
  execute: (approval_id: string) =>
    api
      .post<ApprovalWorkflow>(`/approvals/${approval_id}/execute`)
      .then((r) => r.data),
};

// ───────────────────────────── Phase 14: queue admin

export interface SystemInfo {
  runs_backend: string;
  redis: { url: string | null; healthy: boolean; error: string | null };
  gpu: {
    available: boolean;
    count: number;
    devices: string[];
    source: string;
    error: string | null;
  };
  worker_queues: string[];
}

export const queueAdminApi = {
  queues: () =>
    api.get<{ queues: QueueRow[]; as_of: string }>('/admin/queues').then((r) => r.data),
  workers: () =>
    api.get<{ workers: WorkerRow[] }>('/admin/workers').then((r) => r.data),
  system: () =>
    api.get<SystemInfo>('/admin/system').then((r) => r.data),
};

// ───────────────────────────── Phase 8: notebooks

export const notebooksApi = {
  forProject: (project_id: string) =>
    api
      .get<Notebook[]>(`/projects/${project_id}/notebooks`)
      .then((r) => r.data),
  create: (
    project_id: string,
    body: {
      name: string;
      kernel?: string;
      path?: string;
      description?: string;
      tags?: string[];
    },
  ) =>
    api
      .post<Notebook>(`/projects/${project_id}/notebooks`, body)
      .then((r) => r.data),
  get: (notebook_id: string) =>
    api.get<Notebook>(`/notebooks/${notebook_id}`).then((r) => r.data),
  patch: (
    notebook_id: string,
    body: Partial<Pick<Notebook, 'name' | 'description' | 'kernel' | 'path' | 'tags'>>,
  ) => api.patch<Notebook>(`/notebooks/${notebook_id}`, body).then((r) => r.data),
  delete: (notebook_id: string) =>
    api.delete<void>(`/notebooks/${notebook_id}`).then(() => undefined),

  startSession: (
    notebook_id: string,
    body?: {
      cpu_limit?: number;
      memory_mb_limit?: number;
      idle_timeout_seconds?: number;
      env?: Record<string, string>;
    },
  ) =>
    api
      .post<NotebookSessionStart>(
        `/notebooks/${notebook_id}/sessions`,
        body || {},
      )
      .then((r) => r.data),
  listSessions: (notebook_id: string) =>
    api
      .get<NotebookSessionRow[]>(`/notebooks/${notebook_id}/sessions`)
      .then((r) => r.data),
  heartbeat: (session_id: string) =>
    api
      .post<{ ok: boolean; last_active_at: string }>(
        `/sessions/${session_id}/heartbeat`,
      )
      .then((r) => r.data),
  stopSession: (session_id: string) =>
    api.delete<void>(`/sessions/${session_id}`).then(() => undefined),
};

// ───────────────────────────── Phase 11: analyses (statistical computing)

export const analysesApi = {
  kinds: () =>
    api.get<{ kinds: AnalysisKind[] }>('/analysis-kinds').then((r) => r.data),
  forProject: (project_id: string) =>
    api.get<Analysis[]>(`/projects/${project_id}/analyses`).then((r) => r.data),
  create: (
    project_id: string,
    body: {
      name: string;
      kind: AnalysisKind;
      params: Record<string, unknown>;
      description?: string;
      data_source_id?: string;
    },
  ) =>
    api
      .post<Analysis>(`/projects/${project_id}/analyses`, body)
      .then((r) => r.data),
  get: (analysis_id: string) =>
    api.get<Analysis>(`/analyses/${analysis_id}`).then((r) => r.data),
  patch: (
    analysis_id: string,
    body: Partial<Pick<Analysis, 'name' | 'description' | 'params'>>,
  ) => api.patch<Analysis>(`/analyses/${analysis_id}`, body).then((r) => r.data),
  delete: (analysis_id: string) =>
    api.delete<void>(`/analyses/${analysis_id}`).then(() => undefined),

  run: (
    analysis_id: string,
    body?: {
      params_override?: Record<string, unknown>;
      inline_csv?: string;
      data_source_id?: string;
    },
  ) =>
    api
      .post<AnalysisRunResponse>(`/analyses/${analysis_id}/run`, body || {})
      .then((r) => r.data),

  runOnce: (body: {
    kind: AnalysisKind;
    params: Record<string, unknown>;
    inline_csv?: string;
    data_source_id?: string;
  }) =>
    api
      .post<{ kind: AnalysisKind; result: AnalysisResult }>(
        '/analyses/run-once',
        body,
      )
      .then((r) => r.data),

  results: (analysis_id: string, limit = 50) =>
    api
      .get<AnalysisRunRecord[]>(`/analyses/${analysis_id}/results`, {
        params: { limit },
      })
      .then((r) => r.data),
};

// ───────────────────────────── pipelines (workspace-scoped fitted-model registry)

export const pipelinesApi = {
  list: (workspace_id: string) =>
    api
      .get<Pipeline[]>(`/workspaces/${workspace_id}/pipelines`)
      .then((r) => r.data),
  get: (pipeline_id: string) =>
    api.get<Pipeline>(`/pipelines/${pipeline_id}`).then((r) => r.data),
  versions: (pipeline_id: string) =>
    api
      .get<{ family_id: string | null; items: Pipeline[] }>(
        `/pipelines/${pipeline_id}/versions`,
      )
      .then((r) => r.data),
  inputSchema: (pipeline_id: string) =>
    api
      .get<{
        columns: Array<{ name: string; dtype: string }>;
        sample_row: Record<string, unknown>;
        target: string | null;
      }>(`/pipelines/${pipeline_id}/input-schema`)
      .then((r) => r.data),
  remove: (pipeline_id: string) =>
    api.delete<void>(`/pipelines/${pipeline_id}`).then((r) => r.data),
};

// ───────────────────────────── deployments (in-house serving)

export interface PredictRequest {
  rows: Record<string, unknown>[];
}
export interface PredictResponse {
  deployment_id: string;
  endpoint_slug: string;
  predictions: Array<{
    index: number;
    prediction: unknown;
    /** Per-class likelihood map. Present iff the underlying classifier
     *  supports predict_proba. Keys are class labels (string-coerced). */
    probabilities?: Record<string, number>;
  }>;
  latency_ms: number;
  request_id: string;
}

export const deploymentsApi = {
  list: (workspace_id: string) =>
    api
      .get<Deployment[]>(`/workspaces/${workspace_id}/deployments`)
      .then((r) => r.data),
  get: (deployment_id: string) =>
    api.get<Deployment>(`/deployments/${deployment_id}`).then((r) => r.data),
  create: (
    pipeline_id: string,
    body: {
      endpoint_slug: string;
      auth_mode?: 'workspace' | 'api-key' | 'public';
    },
  ) =>
    api
      .post<Deployment>(`/pipelines/${pipeline_id}/deployments`, body)
      .then((r) => r.data),
  remove: (deployment_id: string) =>
    api.delete<void>(`/deployments/${deployment_id}`).then((r) => r.data),
  predict: (endpoint_slug: string, body: PredictRequest) =>
    api
      .post<PredictResponse>(`/deployments/${endpoint_slug}/predict`, body)
      .then((r) => r.data),
  predictionLogs: (
    deployment_id: string,
    opts?: { limit?: number; offset?: number; status_filter?: 'ok' | 'error' },
  ) =>
    api
      .get<{
        deployment_id: string;
        limit: number;
        offset: number;
        items: Array<{
          id: string;
          request_id: string;
          created_at: string;
          n_rows: number;
          latency_ms: number | null;
          status: 'ok' | 'error';
          error: string | null;
          request_sample: Array<Record<string, unknown>> | null;
          response_sample: Array<{ index: number; prediction: unknown }> | null;
          user_id: string | null;
        }>;
      }>(`/deployments/${deployment_id}/prediction-logs`, { params: opts })
      .then((r) => r.data),
  rollback: (deployment_id: string, body: { pipeline_id: string }) =>
    api
      .post<Deployment>(`/deployments/${deployment_id}/rollback`, body)
      .then((r) => r.data),
};

// ───────────────────────────── schedules

export type ScheduleKind =
  | 'drift_monitor'
  | 'retrain'
  | 'drift_check'
  | 'batch_predict'
  | 'dataset_refresh';

export interface Schedule {
  id: string;
  workspace_id: string;
  kind: ScheduleKind;
  target_id: string | null;
  schedule: { interval_seconds?: number; cron?: string };
  spec: Record<string, unknown> | null;
  enabled: boolean;
  last_run_at: string | null;
  last_status: string | null;
  last_error: string | null;
  last_run_run_id: string | null;
}

export const schedulesApi = {
  list: (workspace_id: string) =>
    api
      .get<{ items: Schedule[] }>(`/workspaces/${workspace_id}/schedules`)
      .then((r) => r.data),
  create: (
    workspace_id: string,
    body: {
      kind: ScheduleKind;
      target_id: string;
      schedule: { interval_seconds?: number; cron?: string };
      spec?: Record<string, unknown> | null;
      enabled?: boolean;
    },
  ) =>
    api
      .post<Schedule>(`/workspaces/${workspace_id}/schedules`, body)
      .then((r) => r.data),
  patch: (
    job_id: string,
    body: {
      schedule?: { interval_seconds?: number; cron?: string };
      spec?: Record<string, unknown> | null;
      enabled?: boolean;
    },
  ) => api.patch<Schedule>(`/schedules/${job_id}`, body).then((r) => r.data),
  remove: (job_id: string) =>
    api.delete<void>(`/schedules/${job_id}`).then((r) => r.data),
  runNow: (job_id: string) =>
    api.post<Schedule>(`/schedules/${job_id}/run-now`).then((r) => r.data),
};

// ───────────────────────────── experiment templates

export interface ExperimentTemplate {
  id: string;
  workspace_id: string;
  name: string;
  description: string | null;
  task: string;
  setup_params: Record<string, unknown>;
  plan_params: Record<string, unknown> | null;
  created_at: string | null;
  updated_at: string | null;
}

export const templatesApi = {
  list: (workspace_id: string, task?: string) =>
    api
      .get<{ items: ExperimentTemplate[] }>(
        `/workspaces/${workspace_id}/experiment-templates`,
        { params: task ? { task } : undefined },
      )
      .then((r) => r.data),
  create: (
    workspace_id: string,
    body: {
      name: string;
      task: string;
      setup_params: Record<string, unknown>;
      description?: string;
      plan_params?: Record<string, unknown> | null;
    },
  ) =>
    api
      .post<ExperimentTemplate>(
        `/workspaces/${workspace_id}/experiment-templates`,
        body,
      )
      .then((r) => r.data),
  patch: (
    template_id: string,
    body: Partial<{
      name: string;
      description: string | null;
      setup_params: Record<string, unknown>;
      plan_params: Record<string, unknown> | null;
    }>,
  ) =>
    api
      .patch<ExperimentTemplate>(
        `/experiment-templates/${template_id}`,
        body,
      )
      .then((r) => r.data),
  remove: (template_id: string) =>
    api
      .delete<void>(`/experiment-templates/${template_id}`)
      .then((r) => r.data),
};

// ───────────────────────────── webhooks

export interface Webhook {
  id: string;
  workspace_id: string;
  url: string;
  event_types: string[];
  has_secret: boolean;
  filters: Record<string, unknown> | null;
  enabled: boolean;
  last_fired_at: string | null;
  last_status_code: number | null;
  last_error: string | null;
}

export const webhooksApi = {
  list: (workspace_id: string) =>
    api
      .get<{ items: Webhook[] }>(`/workspaces/${workspace_id}/webhooks`)
      .then((r) => r.data),
  create: (
    workspace_id: string,
    body: {
      url: string;
      event_types: string[];
      secret?: string;
      filters?: Record<string, unknown>;
      enabled?: boolean;
    },
  ) =>
    api
      .post<Webhook>(`/workspaces/${workspace_id}/webhooks`, body)
      .then((r) => r.data),
  patch: (
    webhook_id: string,
    body: Partial<{
      url: string;
      event_types: string[];
      secret: string | null;
      filters: Record<string, unknown> | null;
      enabled: boolean;
    }>,
  ) => api.patch<Webhook>(`/webhooks/${webhook_id}`, body).then((r) => r.data),
  remove: (webhook_id: string) =>
    api.delete<void>(`/webhooks/${webhook_id}`).then((r) => r.data),
  test: (webhook_id: string) =>
    api.post<Webhook>(`/webhooks/${webhook_id}/test`).then((r) => r.data),
};

// ───────────────────────────── model library (workspace-scoped, editable)

export interface ModelLibraryRow {
  id: string;
  workspace_id: string;
  task_type: string;
  model_id: string;
  name: string;
  enabled: boolean;
  custom_params: Record<string, unknown> | null;
  created_at: string | null;
  updated_at: string | null;
}

export const modelLibraryApi = {
  list: (workspace_id: string, task?: string) =>
    api
      .get<{ workspace_id: string; items: ModelLibraryRow[] }>(
        `/workspaces/${workspace_id}/model-library`,
        { params: task ? { task } : undefined },
      )
      .then((r) => r.data),
  patch: (
    workspace_id: string,
    row_id: string,
    body: { enabled?: boolean; custom_params?: Record<string, unknown> | null },
  ) =>
    api
      .patch<ModelLibraryRow>(
        `/workspaces/${workspace_id}/model-library/${row_id}`,
        body,
      )
      .then((r) => r.data),
  sync: (workspace_id: string, task?: string) =>
    api
      .post<{
        workspace_id: string;
        synced_tasks: string[];
        had_existing_rows: boolean;
      }>(
        `/workspaces/${workspace_id}/model-library/sync`,
        null,
        { params: task ? { task } : undefined },
      )
      .then((r) => r.data),
};

// ───────────────────────────── platform admin (superuser-only)

export interface UserAdminRead {
  id: string;
  email: string;
  display_name: string | null;
  is_superuser: boolean;
  is_active: boolean;
  workspace_count: number;
  created_at: string | null;
}

export const adminApi = {
  listUsers: (opts?: { limit?: number; offset?: number }) =>
    api
      .get<{ items: UserAdminRead[]; limit: number; offset: number }>(
        '/admin/users',
        { params: opts },
      )
      .then((r) => r.data),
  patchUser: (
    user_id: string,
    body: { is_superuser?: boolean; is_active?: boolean },
  ) =>
    api
      .patch<UserAdminRead>(`/admin/users/${user_id}`, body)
      .then((r) => r.data),
};

// ───────────────────────────── data sources (listing + register; upload is separate)

// ───────────────────────────── LLM advisory surface

export const llmApi = {
  getSettings: (workspace_id: string) =>
    api
      .get<LLMProviderSettingRead | null>(
        `/workspaces/${workspace_id}/llm/settings`,
      )
      .then((r) => r.data),
  upsertSettings: (workspace_id: string, body: LLMProviderSettingWrite) =>
    api
      .put<LLMProviderSettingRead>(
        `/workspaces/${workspace_id}/llm/settings`,
        body,
      )
      .then((r) => r.data),
  deleteSettings: (workspace_id: string) =>
    api
      .delete<void>(`/workspaces/${workspace_id}/llm/settings`)
      .then((r) => r.data),
  testConnection: (workspace_id: string) =>
    api
      .post<TestConnectionResponse>(
        `/workspaces/${workspace_id}/llm/test-connection`,
      )
      .then((r) => r.data),
  analyzeDataset: (body: {
    workspace_id: string;
    data_source_id: string;
    task_type_hint?: string | null;
  }) =>
    api
      .post<LLMConsultationRead>('/llm/analyze-dataset', body)
      .then((r) => r.data),
  designExperiment: (body: {
    workspace_id: string;
    data_source_id: string;
    goal: string;
    business_context?: string | null;
    include_business_context?: boolean;
  }) =>
    api
      .post<LLMConsultationRead>('/llm/design-experiment', body)
      .then((r) => r.data),
  explainRun: (body: { run_id: string }) =>
    api.post<LLMConsultationRead>('/llm/explain-run', body).then((r) => r.data),
  debugRun: (body: { run_id: string }) =>
    api.post<LLMConsultationRead>('/llm/debug-run', body).then((r) => r.data),
  reviewDeployment: (body: { pipeline_id: string }) =>
    api
      .post<LLMConsultationRead>('/llm/review-deployment', body)
      .then((r) => r.data),
  analyzeDrift: (body: { drift_report_id: string }) =>
    api.post<LLMConsultationRead>('/llm/analyze-drift', body).then((r) => r.data),
  listConsultations: (workspace_id: string, limit = 50) =>
    api
      .get<LLMConsultationRead[]>(
        `/workspaces/${workspace_id}/llm/consultations`,
        { params: { limit } },
      )
      .then((r) => r.data),
  getConsultation: (id: string) =>
    api.get<LLMConsultationRead>(`/llm/consultations/${id}`).then((r) => r.data),
};

// ───────────────────────────── workspace members

export const membersApi = {
  list: (workspace_id: string) =>
    api
      .get<MemberRead[]>(`/workspaces/${workspace_id}/members`)
      .then((r) => r.data),
  invite: (workspace_id: string, body: InviteRequest) =>
    api
      .post<MemberRead>(`/workspaces/${workspace_id}/members`, body)
      .then((r) => r.data),
  changeRole: (workspace_id: string, user_id: string, body: PatchRoleRequest) =>
    api
      .patch<MemberRead>(
        `/workspaces/${workspace_id}/members/${user_id}`,
        body,
      )
      .then((r) => r.data),
  remove: (workspace_id: string, user_id: string) =>
    api
      .delete<void>(`/workspaces/${workspace_id}/members/${user_id}`)
      .then((r) => r.data),
};

// ───────────────────────────── API keys (personal programmatic tokens)

export const apiKeysApi = {
  list: () => api.get<ApiKeyRead[]>('/auth/api-keys').then((r) => r.data),
  create: (body: ApiKeyCreateRequest) =>
    api.post<ApiKeyCreateResponse>('/auth/api-keys', body).then((r) => r.data),
  revoke: (id: string) =>
    api.delete<void>(`/auth/api-keys/${id}`).then((r) => r.data),
};

// ───────────────────────────── drift reports + audit logs (session 21)

export const driftApi = {
  list: (deployment_id: string, limit = 50) =>
    api
      .get<DriftReportRead[]>(
        `/deployments/${deployment_id}/drift-reports`,
        { params: { limit } },
      )
      .then((r) => r.data),
  create: (deployment_id: string, body: DriftReportCreate) =>
    api
      .post<DriftReportRead>(
        `/deployments/${deployment_id}/drift-reports`,
        body,
      )
      .then((r) => r.data),
  get: (report_id: string) =>
    api.get<DriftReportRead>(`/drift-reports/${report_id}`).then((r) => r.data),
};

export const auditApi = {
  listAdmin: (filters?: AuditLogFilters) =>
    api
      .get<AuditLogRead[]>('/admin/audit-logs', { params: filters })
      .then((r) => r.data),
  listForWorkspace: (workspace_id: string, filters?: AuditLogFilters) =>
    api
      .get<AuditLogRead[]>(`/workspaces/${workspace_id}/audit-logs`, {
        params: filters,
      })
      .then((r) => r.data),
};

export const dataSourcesApi = {
  list: (workspace_id: string) =>
    api
      .get<DataSource[]>(`/workspaces/${workspace_id}/data-sources`)
      .then((r) => r.data),
  get: (id: string) => api.get<DataSource>(`/data-sources/${id}`).then((r) => r.data),
  remove: (id: string) => api.delete<void>(`/data-sources/${id}`).then((r) => r.data),
  /**
   * CSV upload — multipart/form-data. Returns the new DataSource row.
   * Axios sets the Content-Type + boundary automatically when we pass FormData.
   */
  uploadCsv: (workspace_id: string, file: File, name: string, description?: string) => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('name', name);
    if (description) fd.append('description', description);
    return api
      .post<DataSource>(`/workspaces/${workspace_id}/data-sources/upload`, fd)
      .then((r) => r.data);
  },
  profile: (id: string, sampleRows = 10) =>
    api
      .get<DatasetProfile>(`/data-sources/${id}/profile`, {
        params: { sample_rows: sampleRows },
      })
      .then((r) => r.data),
};

// ──────────────────────────────────────────── bundled sample datasets

export interface SampleDataset {
  name: string;
  data_types: string;
  default_task: string;
  target_1: string | null;
  target_2: string | null;
  instances: number;
  attributes: number;
  has_missing: boolean;
  size_bytes: number | null;
  available: boolean;
}

export const sampleDatasetsApi = {
  list: () =>
    api.get<SampleDataset[]>('/sample-datasets').then((r) => r.data),
  register: (workspace_id: string, sample_name: string, name?: string) =>
    api
      .post<DataSource>(
        `/workspaces/${workspace_id}/data-sources/from-sample`,
        { sample_name, name: name ?? null },
      )
      .then((r) => r.data),
};

// Schema for the rich EDA profile endpoint. See
// services/api/pycaret_server/api/data_sources.py:_compute_dataset_profile.
export type DatasetColumnKind =
  | 'numeric'
  | 'categorical'
  | 'datetime'
  | 'boolean'
  | 'text';

export interface DatasetColumn {
  name: string;
  dtype: string;
  kind: DatasetColumnKind;
  missing: number;
  missing_pct: number;
  unique: number;
  cardinality_pct: number;
  is_constant: boolean;
  is_id_like: boolean;
  stats: Record<string, number | string | null> | null;
  histogram: Array<{ bin_start: number; bin_end: number; count: number }>;
  top_values: Array<{ value: string; count: number; pct: number }>;
  sample_values: Array<string | number | boolean | null>;
}

export interface DatasetProfile {
  name: string | null;
  shape: { rows: number; cols: number };
  memory_bytes: number;
  duplicates: number;
  missing_total: number;
  missing_pct: number;
  type_counts: Record<string, number>;
  columns: DatasetColumn[];
  correlations: { columns: string[]; matrix: number[][] } | null;
  warnings: Array<{
    kind: string;
    severity: 'info' | 'warn' | 'error';
    column: string | null;
    message: string;
  }>;
  sample: Array<Record<string, unknown>>;
}

// ──────────────────────────────────────────────────────────── plots

import type { PlotRegistry } from './types';

/**
 * Note: the api client's baseURL already includes `/api/v1`. Other
 * endpoints in this file use bare paths like `/workspaces/...`.
 */
export const plotsApi = {
  registry: () => api.get<PlotRegistry>('/plots/registry').then((r) => r.data),
  forRun: (runId: string, kind: string) =>
    api
      .get<PlotEnvelope>(
        `/runs/${encodeURIComponent(runId)}/plots/${encodeURIComponent(kind)}`,
      )
      .then((r) => r.data),
  forDataset: (
    dataSourceId: string,
    kind: string,
    params?: { column?: string; feature?: string; target?: string },
  ) => {
    const q = new URLSearchParams();
    if (params?.column) q.set('column', params.column);
    if (params?.feature) q.set('feature', params.feature);
    if (params?.target) q.set('target', params.target);
    const suffix = q.toString() ? `?${q.toString()}` : '';
    return api
      .get<PlotEnvelope>(
        `/datasets/${encodeURIComponent(dataSourceId)}/plots/eda/${encodeURIComponent(kind)}${suffix}`,
      )
      .then((r) => r.data);
  },
};
