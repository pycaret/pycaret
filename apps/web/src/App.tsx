import { Route, Routes } from 'react-router-dom';
import { AuthGate } from '@/components/AuthGate';
import { Layout } from '@/components/Layout';
import { Setup } from '@/pages/Setup';
import { Login } from '@/pages/Login';
import { Workspaces } from '@/pages/Workspaces';
import { WorkspaceDetail } from '@/pages/WorkspaceDetail';
import { ProjectDetail } from '@/pages/ProjectDetail';
import { NewExperiment } from '@/pages/NewExperiment';
import { ExperimentDetail } from '@/pages/ExperimentDetail';
import { RunDetail } from '@/pages/RunDetail';
import { Pipelines } from '@/pages/Pipelines';
import { PipelineDetail } from '@/pages/PipelineDetail';
import { Deployments } from '@/pages/Deployments';
import { DeploymentDetail } from '@/pages/DeploymentDetail';
import { LLMSettings } from '@/pages/LLMSettings';
import { ApiKeysScreen } from '@/pages/ApiKeysScreen';
import { WorkspaceMembers } from '@/pages/WorkspaceMembers';
import { AuditLogViewer } from '@/pages/AuditLogViewer';

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
        <Route
          path="/workspaces/:wsId/projects/:projectId"
          element={<ProjectDetail />}
        />
        <Route
          path="/workspaces/:wsId/projects/:projectId/experiments/new"
          element={<NewExperiment />}
        />
        <Route
          path="/workspaces/:wsId/projects/:projectId/experiments/:experimentId"
          element={<ExperimentDetail />}
        />
        <Route path="/runs/:runId" element={<RunDetail />} />
        <Route path="/workspaces/:wsId/pipelines" element={<Pipelines />} />
        <Route
          path="/workspaces/:wsId/pipelines/:pipelineId"
          element={<PipelineDetail />}
        />
        <Route path="/workspaces/:wsId/deployments" element={<Deployments />} />
        <Route path="/deployments/:deploymentId" element={<DeploymentDetail />} />
        <Route path="/workspaces/:wsId/llm" element={<LLMSettings />} />
        <Route
          path="/workspaces/:wsId/members"
          element={<WorkspaceMembers />}
        />
        <Route path="/account/api-keys" element={<ApiKeysScreen />} />
        <Route path="/admin/audit" element={<AuditLogViewer />} />
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
