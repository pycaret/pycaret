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
import { TrialDetail } from '@/pages/TrialDetail';
import { TrialCompare } from '@/pages/TrialCompare';
import { ModelCard } from '@/pages/ModelCard';
import { DataProfile } from '@/pages/DataProfile';
import { WorkspaceHome } from '@/pages/WorkspaceHome';
import { ForecastWorkbench } from '@/pages/ForecastWorkbench';
import { DriftDashboard } from '@/pages/DriftDashboard';
import { Deployments } from '@/pages/Deployments';
import { DeploymentDetail } from '@/pages/DeploymentDetail';
import { LLMSettings } from '@/pages/LLMSettings';
import { ApiKeysScreen } from '@/pages/ApiKeysScreen';
import { WorkspaceMembers } from '@/pages/WorkspaceMembers';
import { AuditLogViewer } from '@/pages/AuditLogViewer';
import { AdminUsers } from '@/pages/AdminUsers';
import { AdminWorkspace } from '@/pages/AdminWorkspace';
import { AdminIntegrations } from '@/pages/AdminIntegrations';
import { Schedules } from '@/pages/Schedules';
import { ExperimentTemplates } from '@/pages/ExperimentTemplates';
import { Webhooks } from '@/pages/Webhooks';
// Phase 0/14 admin/observability surfaces shipped post-MVP.
import { QueueAdmin } from '@/pages/QueueAdmin';
import { Approvals } from '@/pages/Approvals';
import { Notebooks } from '@/pages/Notebooks';
import { Analyses } from '@/pages/Analyses';
import { RegisteredModels } from '@/pages/RegisteredModels';
import { RegisteredModelDetail } from '@/pages/RegisteredModelDetail';
import { Monitoring } from '@/pages/Monitoring';
import { Lineage } from '@/pages/Lineage';
import { Secrets } from '@/pages/Secrets';
import { Connections } from '@/pages/Connections';
import { GitRepositories } from '@/pages/GitRepositories';
import { Datasets } from '@/pages/Datasets';
import { AllDatasets } from '@/pages/AllDatasets';

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
        <Route
          path="/runs/:runId/trials/:trialId"
          element={<TrialDetail />}
        />
        <Route path="/runs/:runId/compare" element={<TrialCompare />} />
        <Route path="/runs/:runId/model-card" element={<ModelCard />} />
        <Route path="/workspaces/:wsId/home" element={<WorkspaceHome />} />
        <Route
          path="/workspaces/:wsId/datasets/:dataSourceId/profile"
          element={<DataProfile />}
        />
        <Route path="/runs/:runId/forecast" element={<ForecastWorkbench />} />
        <Route path="/workspaces/:wsId/drift" element={<DriftDashboard />} />
        <Route path="/workspaces/:wsId/deployments" element={<Deployments />} />
        <Route path="/deployments/:deploymentId" element={<DeploymentDetail />} />
        <Route path="/workspaces/:wsId/llm" element={<LLMSettings />} />
        <Route
          path="/workspaces/:wsId/members"
          element={<WorkspaceMembers />}
        />
        <Route path="/account/api-keys" element={<ApiKeysScreen />} />
        <Route path="/admin/audit" element={<AuditLogViewer />} />
        <Route path="/admin/users" element={<AdminUsers />} />
        <Route path="/workspaces/:wsId/admin" element={<AdminWorkspace />} />
        <Route
          path="/workspaces/:wsId/admin/integrations"
          element={<AdminIntegrations />}
        />
        <Route path="/workspaces/:wsId/schedules" element={<Schedules />} />
        <Route
          path="/workspaces/:wsId/templates"
          element={<ExperimentTemplates />}
        />
        <Route path="/workspaces/:wsId/webhooks" element={<Webhooks />} />
        {/* Phase 7 — model registry. */}
        <Route
          path="/workspaces/:wsId/models"
          element={<RegisteredModels />}
        />
        <Route
          path="/workspaces/:wsId/models/:modelId"
          element={<RegisteredModelDetail />}
        />
        {/* Phase 8 — notebooks (project-scoped). */}
        <Route
          path="/workspaces/:wsId/projects/:projectId/notebooks"
          element={<Notebooks />}
        />
        {/* Phase 10 — monitoring dashboard. */}
        <Route
          path="/workspaces/:wsId/monitoring"
          element={<Monitoring />}
        />
        {/* Phase 11 — statistical analyses. */}
        <Route
          path="/workspaces/:wsId/projects/:projectId/analyses"
          element={<Analyses />}
        />
        {/* Phase 12 — governance / approvals. */}
        <Route
          path="/workspaces/:wsId/approvals"
          element={<Approvals />}
        />
        {/* Phase 4 / 12 — lineage graph. */}
        <Route path="/workspaces/:wsId/lineage" element={<Lineage />} />
        {/* Phase 14 — queue + worker admin. */}
        <Route path="/admin/queues" element={<QueueAdmin />} />
        {/* Phase 4 — secrets / connections / git / datasets. */}
        <Route path="/workspaces/:wsId/secrets" element={<Secrets />} />
        <Route
          path="/workspaces/:wsId/connections"
          element={<Connections />}
        />
        <Route path="/workspaces/:wsId/git" element={<GitRepositories />} />
        <Route path="/workspaces/:wsId/datasets" element={<AllDatasets />} />
        <Route
          path="/workspaces/:wsId/datasets/:dataSourceId"
          element={<Datasets />}
        />
      </Route>
      <Route
        path="*"
        element={
          <div className="min-h-screen flex items-center justify-center">
            <div className="text-center">
              <h1 className="text-2xl font-semibold">Not found</h1>
              <p className="mt-2 text-sm text-ink-500">That page doesn't exist.</p>
            </div>
          </div>
        }
      />
    </Routes>
  );
}
