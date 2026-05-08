"""``ControlPlane`` — typed-ish HTTP client for the PyCaret backend.

Hand-written for clarity. A future revision may regenerate from the
running OpenAPI spec; the namespacing here matches the eventual generator
output (``cp.workspaces.list()``, ``cp.runs.submit(...)``, etc.) so call
sites won't need to migrate.

Auth modes:
  * ``ControlPlane.connect(base_url, email=, password=)`` — interactive
    bootstrap, exchanges credentials for a Bearer token.
  * ``ControlPlane(base_url, token="...")`` — Bearer token directly.
  * ``ControlPlane(base_url, api_key="...")`` — programmatic key (sent as
    ``X-PyCaret-Key``).

Errors surface as ``ControlPlaneError`` carrying the upstream JSON detail.
"""

from __future__ import annotations

import time
from typing import Any

import requests


class ControlPlaneError(RuntimeError):
    """Wraps any non-2xx response from the API."""

    def __init__(self, status_code: int, detail: Any, *, request_id: str | None = None) -> None:
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id
        super().__init__(f"HTTP {status_code}: {detail}")


class _Endpoint:
    """Helper base — owns a back-reference to the ControlPlane parent."""

    def __init__(self, cp: "ControlPlane") -> None:
        self._cp = cp


class _Auth(_Endpoint):
    def me(self) -> dict:
        return self._cp._get("/auth/me")

    def login(self, email: str, password: str) -> dict:
        body = {"email": email, "password": password}
        return self._cp._post("/auth/login", json=body, authed=False)

    def logout(self) -> None:
        self._cp._post("/auth/logout", authed=True)


class _Workspaces(_Endpoint):
    def list(self) -> list[dict]:
        return self._cp._get("/workspaces")

    def create(self, *, name: str, description: str | None = None) -> dict:
        return self._cp._post("/workspaces", json={"name": name, "description": description})


class _Projects(_Endpoint):
    def list(self, workspace_id: str) -> list[dict]:
        return self._cp._get(f"/workspaces/{workspace_id}/projects")

    def create(self, *, workspace_id: str, name: str, description: str | None = None) -> dict:
        return self._cp._post(
            f"/workspaces/{workspace_id}/projects",
            json={"name": name, "description": description},
        )


class _Experiments(_Endpoint):
    def list(self, project_id: str) -> list[dict]:
        return self._cp._get(f"/projects/{project_id}/experiments")

    def create(
        self,
        *,
        project_id: str,
        name: str,
        task: str,
        target: str | None = None,
        setup_params: dict | None = None,
    ) -> dict:
        body = {
            "name": name,
            "task": task,
            "target": target,
            "setup_params": setup_params or {},
        }
        return self._cp._post(f"/projects/{project_id}/experiments", json=body)


class _Runs(_Endpoint):
    def list(self, experiment_id: str) -> list[dict]:
        return self._cp._get(f"/experiments/{experiment_id}/runs")

    def submit(
        self,
        *,
        experiment_id: str,
        plan: str,
        model_id: str | None = None,
        plan_params: dict | None = None,
        sklearn_dataset: str | None = None,
        data_source_id: str | None = None,
        data_inline: list[dict] | None = None,
    ) -> dict:
        body = {
            "plan": plan,
            "model_id": model_id,
            "plan_params": plan_params,
            "sklearn_dataset": sklearn_dataset,
            "data_source_id": data_source_id,
            "data_inline": data_inline,
        }
        body = {k: v for k, v in body.items() if v is not None}
        return self._cp._post(f"/experiments/{experiment_id}/runs", json=body)

    def get(self, run_id: str) -> dict:
        return self._cp._get(f"/runs/{run_id}")

    def wait(self, run_id: str, *, timeout: int = 600, poll: float = 2.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            run = self.get(run_id)
            if run["status"] in ("succeeded", "failed", "cancelled"):
                return run
            time.sleep(poll)
        raise TimeoutError(f"run {run_id} did not finish within {timeout}s")

    def cancel(self, run_id: str) -> dict:
        return self._cp._post(f"/runs/{run_id}/cancel")

    def trials(self, run_id: str) -> dict:
        return self._cp._get(f"/runs/{run_id}/trials")

    def promote(
        self,
        run_id: str,
        *,
        name: str,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        return self._cp._post(
            f"/runs/{run_id}/promote",
            json={"name": name, "description": description, "tags": tags or []},
        )


class _Pipelines(_Endpoint):
    def list(self, workspace_id: str) -> list[dict]:
        return self._cp._get(f"/workspaces/{workspace_id}/pipelines")

    def get(self, pipeline_id: str) -> dict:
        return self._cp._get(f"/pipelines/{pipeline_id}")

    def versions(self, pipeline_id: str) -> dict:
        return self._cp._get(f"/pipelines/{pipeline_id}/versions")


class _Deployments(_Endpoint):
    def list(self, workspace_id: str) -> list[dict]:
        return self._cp._get(f"/workspaces/{workspace_id}/deployments")

    def create(
        self,
        *,
        pipeline_id: str,
        endpoint_slug: str,
        auth_mode: str = "workspace",
    ) -> dict:
        return self._cp._post(
            f"/pipelines/{pipeline_id}/deployments",
            json={"endpoint_slug": endpoint_slug, "auth_mode": auth_mode},
        )

    def predict(
        self, *, endpoint_slug: str, rows: list[dict]
    ) -> dict:
        return self._cp._post(
            f"/deployments/{endpoint_slug}/predict",
            json={"rows": rows},
        )

    def rollback(self, deployment_id: str, *, pipeline_id: str) -> dict:
        return self._cp._post(
            f"/deployments/{deployment_id}/rollback",
            json={"pipeline_id": pipeline_id},
        )

    def prediction_logs(
        self,
        deployment_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        status_filter: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status_filter:
            params["status_filter"] = status_filter
        return self._cp._get(
            f"/deployments/{deployment_id}/prediction-logs", params=params
        )


class _Schedules(_Endpoint):
    def list(self, workspace_id: str) -> dict:
        return self._cp._get(f"/workspaces/{workspace_id}/schedules")

    def create(
        self,
        *,
        workspace_id: str,
        kind: str,
        target_id: str,
        schedule: dict,
        spec: dict | None = None,
        enabled: bool = True,
    ) -> dict:
        body = {
            "kind": kind,
            "target_id": target_id,
            "schedule": schedule,
            "spec": spec,
            "enabled": enabled,
        }
        return self._cp._post(f"/workspaces/{workspace_id}/schedules", json=body)

    def patch(self, job_id: str, **patch: Any) -> dict:
        return self._cp._patch(f"/schedules/{job_id}", json=patch)

    def delete(self, job_id: str) -> None:
        self._cp._delete(f"/schedules/{job_id}")

    def run_now(self, job_id: str) -> dict:
        return self._cp._post(f"/schedules/{job_id}/run-now")


class _Webhooks(_Endpoint):
    def list(self, workspace_id: str) -> dict:
        return self._cp._get(f"/workspaces/{workspace_id}/webhooks")

    def create(
        self,
        *,
        workspace_id: str,
        url: str,
        event_types: list[str],
        secret: str | None = None,
        filters: dict | None = None,
        enabled: bool = True,
    ) -> dict:
        body = {
            "url": url,
            "event_types": event_types,
            "secret": secret,
            "filters": filters,
            "enabled": enabled,
        }
        return self._cp._post(f"/workspaces/{workspace_id}/webhooks", json=body)

    def delete(self, webhook_id: str) -> None:
        self._cp._delete(f"/webhooks/{webhook_id}")


class _Templates(_Endpoint):
    def list(self, workspace_id: str, *, task: str | None = None) -> dict:
        params = {"task": task} if task else None
        return self._cp._get(
            f"/workspaces/{workspace_id}/experiment-templates", params=params
        )

    def create(
        self,
        *,
        workspace_id: str,
        name: str,
        task: str,
        setup_params: dict,
        description: str | None = None,
        plan_params: dict | None = None,
    ) -> dict:
        body = {
            "name": name,
            "task": task,
            "setup_params": setup_params,
            "description": description,
            "plan_params": plan_params,
        }
        return self._cp._post(
            f"/workspaces/{workspace_id}/experiment-templates", json=body
        )


class _ModelLibrary(_Endpoint):
    def list(self, workspace_id: str, *, task: str | None = None) -> dict:
        params = {"task": task} if task else None
        return self._cp._get(
            f"/workspaces/{workspace_id}/model-library", params=params
        )

    def patch(
        self,
        *,
        workspace_id: str,
        row_id: str,
        enabled: bool | None = None,
        custom_params: dict | None = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if enabled is not None:
            body["enabled"] = enabled
        if custom_params is not None:
            body["custom_params"] = custom_params
        return self._cp._patch(
            f"/workspaces/{workspace_id}/model-library/{row_id}", json=body
        )


class ControlPlane:
    """Top-level SDK handle. Construct directly or via ``connect()``."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/api/v1"
        self.timeout = timeout
        self._session = requests.Session()
        self._token = token
        self._api_key = api_key

        # Namespaced endpoints.
        self.auth = _Auth(self)
        self.workspaces = _Workspaces(self)
        self.projects = _Projects(self)
        self.experiments = _Experiments(self)
        self.runs = _Runs(self)
        self.pipelines = _Pipelines(self)
        self.deployments = _Deployments(self)
        self.schedules = _Schedules(self)
        self.webhooks = _Webhooks(self)
        self.templates = _Templates(self)
        self.model_library = _ModelLibrary(self)

    # ------------------------------------------------------ class methods

    @classmethod
    def connect(
        cls,
        base_url: str,
        *,
        email: str,
        password: str,
        timeout: float = 30.0,
    ) -> "ControlPlane":
        """Construct + log in via /auth/login. Returns an authed client."""
        cp = cls(base_url, timeout=timeout)
        tokens = cp.auth.login(email=email, password=password)
        cp._token = tokens["access_token"]
        return cp

    # -------------------------------------------------------- thin verbs

    def _headers(self, *, authed: bool = True) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if authed:
            if self._token:
                h["Authorization"] = f"Bearer {self._token}"
            elif self._api_key:
                h["X-PyCaret-Key"] = self._api_key
        return h

    def _check(self, resp: requests.Response) -> Any:
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:  # noqa: BLE001
                detail = resp.text
            raise ControlPlaneError(
                resp.status_code,
                detail,
                request_id=resp.headers.get("X-Request-ID"),
            )
        if not resp.content:
            return None
        try:
            return resp.json()
        except Exception:  # noqa: BLE001
            return resp.text

    def _get(self, path: str, *, params: dict | None = None) -> Any:
        return self._check(
            self._session.get(
                self.base_url + path,
                headers=self._headers(),
                params=params,
                timeout=self.timeout,
            )
        )

    def _post(
        self, path: str, *, json: Any = None, authed: bool = True
    ) -> Any:
        return self._check(
            self._session.post(
                self.base_url + path,
                headers=self._headers(authed=authed),
                json=json,
                timeout=self.timeout,
            )
        )

    def _patch(self, path: str, *, json: Any = None) -> Any:
        return self._check(
            self._session.patch(
                self.base_url + path,
                headers=self._headers(),
                json=json,
                timeout=self.timeout,
            )
        )

    def _delete(self, path: str) -> Any:
        return self._check(
            self._session.delete(
                self.base_url + path,
                headers=self._headers(),
                timeout=self.timeout,
            )
        )
