import { describe, expect, it, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { AuthGate } from './AuthGate';
import { useAuthStore } from '@/state/auth';

function reset() {
  localStorage.clear();
  useAuthStore.setState({ accessToken: null, refreshToken: null, user: null });
}

function setup(initialPath = '/') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route
          path="/"
          element={
            <AuthGate>
              <div>HOME</div>
            </AuthGate>
          }
        />
        <Route path="/login" element={<div>LOGIN</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('<AuthGate>', () => {
  beforeEach(reset);

  it('redirects to /login when no tokens are present', () => {
    setup('/');
    expect(screen.getByText('LOGIN')).toBeInTheDocument();
    expect(screen.queryByText('HOME')).not.toBeInTheDocument();
  });

  it('renders children when an access token is present', () => {
    useAuthStore.setState({ accessToken: 'valid', refreshToken: null });
    setup('/');
    expect(screen.getByText('HOME')).toBeInTheDocument();
  });
});
