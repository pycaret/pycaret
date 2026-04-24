/**
 * Typed API methods. One function per route the UI uses.
 *
 * Keep call sites small — TanStack Query wraps these.
 */

import { api } from './client';
import type {
  BootstrapRequest,
  LoginRequest,
  Project,
  SetupStatus,
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
