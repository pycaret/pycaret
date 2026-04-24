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
};

// ───────────────────────────── pipelines (workspace-scoped fitted-model registry)

export const pipelinesApi = {
  list: (workspace_id: string) =>
    api
      .get<Pipeline[]>(`/workspaces/${workspace_id}/pipelines`)
      .then((r) => r.data),
  get: (pipeline_id: string) =>
    api.get<Pipeline>(`/pipelines/${pipeline_id}`).then((r) => r.data),
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
