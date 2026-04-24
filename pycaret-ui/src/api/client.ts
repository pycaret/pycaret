/**
 * Thin axios-based API client for pycaret-server.
 *
 * Why not Orval/OpenAPI codegen on day one: we want a tight hand-written
 * client for the 4 screens session 12 ships, with room to swap in a
 * generated client (`npm run gen:api`) as the surface grows.
 *
 * The `request` function is the seam everything else goes through. It:
 *  - attaches the bearer token from the auth store
 *  - auto-refreshes once on 401 (single-flight, no thundering-herd)
 *  - parses JSON and returns typed data
 */

import axios from 'axios';
import type { AxiosError, AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import { useAuthStore } from '@/state/auth';

const baseURL = '/api/v1';

function createClient(): AxiosInstance {
  const c = axios.create({ baseURL, timeout: 30_000 });

  // Attach bearer token on every request.
  c.interceptors.request.use((config: InternalAxiosRequestConfig) => {
    const token = useAuthStore.getState().accessToken;
    if (token) {
      config.headers.set('Authorization', `Bearer ${token}`);
    }
    return config;
  });

  // Single-flight refresh. `refreshing` holds the in-flight promise so
  // multiple concurrent 401s share one refresh call.
  let refreshing: Promise<boolean> | null = null;

  c.interceptors.response.use(
    (r) => r,
    async (error: AxiosError) => {
      const original = error.config as (InternalAxiosRequestConfig & { _retried?: boolean }) | undefined;
      if (!original || original._retried) throw error;
      if (error.response?.status !== 401) throw error;

      // Don't try to refresh the refresh endpoint itself.
      if (original.url?.startsWith('/auth/refresh')) throw error;

      if (!refreshing) {
        refreshing = useAuthStore
          .getState()
          .refresh()
          .finally(() => {
            refreshing = null;
          });
      }
      const ok = await refreshing;
      if (!ok) throw error;

      original._retried = true;
      const token = useAuthStore.getState().accessToken;
      if (token) original.headers.set('Authorization', `Bearer ${token}`);
      return c.request(original);
    },
  );
  return c;
}

export const api = createClient();

/** Pretty-print an axios error for toast/form display. */
export function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data as { detail?: string } | undefined;
    return detail?.detail ?? err.message;
  }
  return err instanceof Error ? err.message : String(err);
}
