/**
 * Authentication state. Single source of truth for tokens + the current user.
 *
 * Persistence: we keep the refresh token in localStorage so a page reload
 * doesn't kick the user back to /login. Access token lives in memory only;
 * a refresh restores it from the refresh token on every reload.
 *
 * This file is imported by the axios interceptor in src/api/client.ts, so it
 * must not import from there (would circular).
 */

import axios from 'axios';
import { create } from 'zustand';
import type { TokenPair, User } from '@/api/types';

const REFRESH_KEY = 'pycaret.refresh_token';

type AuthState = {
  accessToken: string | null;
  refreshToken: string | null;
  user: User | null;
  // setTokens is the only legit way to mutate token state.
  setTokens: (pair: TokenPair) => void;
  setUser: (u: User | null) => void;
  clear: () => void;
  /** Returns true if refresh succeeded. */
  refresh: () => Promise<boolean>;
};

export const useAuthStore = create<AuthState>((set, get) => ({
  accessToken: null,
  refreshToken: localStorage.getItem(REFRESH_KEY),
  user: null,

  setTokens: (pair) => {
    localStorage.setItem(REFRESH_KEY, pair.refresh_token);
    set({ accessToken: pair.access_token, refreshToken: pair.refresh_token });
  },

  setUser: (u) => set({ user: u }),

  clear: () => {
    localStorage.removeItem(REFRESH_KEY);
    set({ accessToken: null, refreshToken: null, user: null });
  },

  refresh: async () => {
    const token = get().refreshToken;
    if (!token) return false;
    try {
      // Use a bare axios call to avoid the interceptor recursion on /auth/refresh.
      const { data } = await axios.post<TokenPair>(
        '/api/v1/auth/refresh',
        { refresh_token: token },
        { timeout: 10_000 },
      );
      get().setTokens(data);
      return true;
    } catch {
      get().clear();
      return false;
    }
  },
}));

/** True once the bootstrap + `me` resolution has completed (used by router). */
export function isAuthenticated(): boolean {
  return useAuthStore.getState().accessToken !== null;
}
