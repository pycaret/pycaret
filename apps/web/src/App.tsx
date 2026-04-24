import { Route, Routes } from 'react-router-dom';
import { AuthGate } from '@/components/AuthGate';
import { Layout } from '@/components/Layout';
import { Setup } from '@/pages/Setup';
import { Login } from '@/pages/Login';
import { Workspaces } from '@/pages/Workspaces';
import { WorkspaceDetail } from '@/pages/WorkspaceDetail';

export default function App() {
  return (
    <Routes>
      <Route path="/setup" element={<Setup />} />
      <Route path="/login" element={<Login />} />
      <Route
        element={
          <AuthGate>
            <Layout />
          </AuthGate>
        }
      >
        <Route index element={<Workspaces />} />
        <Route path="/workspaces/:id" element={<WorkspaceDetail />} />
      </Route>
      <Route
        path="*"
        element={
          <div className="min-h-screen flex items-center justify-center">
            <div className="text-center">
              <h1 className="text-2xl font-semibold">Not found</h1>
              <p className="mt-2 text-sm text-ink-200/70">That page doesn't exist.</p>
            </div>
          </div>
        }
      />
    </Routes>
  );
}
