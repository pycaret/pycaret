import { beforeEach, describe, expect, it } from 'vitest';
import { useAuthStore } from './auth';

function reset() {
  localStorage.clear();
  useAuthStore.setState({ accessToken: null, refreshToken: null, user: null });
}

describe('auth store', () => {
  beforeEach(reset);

  it('persists refresh token to localStorage and clears on sign-out', () => {
    useAuthStore.getState().setTokens({
      access_token: 'A',
      refresh_token: 'R',
      token_type: 'bearer',
      expires_in: 3600,
    });
    expect(useAuthStore.getState().accessToken).toBe('A');
    expect(useAuthStore.getState().refreshToken).toBe('R');
    expect(localStorage.getItem('pycaret.refresh_token')).toBe('R');

    useAuthStore.getState().clear();
    expect(useAuthStore.getState().accessToken).toBeNull();
    expect(localStorage.getItem('pycaret.refresh_token')).toBeNull();
  });

  it('returns false from refresh() when no refresh token is present', async () => {
    const ok = await useAuthStore.getState().refresh();
    expect(ok).toBe(false);
  });
});
