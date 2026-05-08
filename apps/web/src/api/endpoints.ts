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
  Project,
  Run,
  RunCreate,
  RunEvent,
  SetupParamSchema,
  SetupStatus,
  TaskType,
  TestConnectionResponse,
  TokenPair,
  User,
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
  events: (run_id: string, opts?: { after_id?: string; limit?: number }) =>
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
          model_id: string;
          rank: number;
          metrics: Record<string, number>;
          is_best: boolean;
          fitted_pipeline_id: string | null;
          created_at: string | null;
        }>;
      }>(`/runs/${run_id}/trials`)
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
  predictions: Array<{ index: number; prediction: unknown }>;
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

export interface Schedule {
  id: string;
  workspace_id: string;
  kind: 'drift_monitor' | 'retrain';
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
      kind: 'drift_monitor' | 'retrain';
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
};

// ──────────────────────────────────────────────────────────── plots

import type { PlotEnvelope, PlotRegistry } from './types';

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
