# PyCaret 4.0 — Vision

## What we're building

**PyCaret is an open-source, self-hosted ML platform for tabular data — a credible alternative to DataRobot and H2O.ai that teams actually own.**

Two layers, one brand:

- **PyCaret Engine** — `pip install pycaret`. Config-driven, stateless, notebook-friendly. Built on scikit-learn 1.7+. Runs AutoML, preprocessing pipelines, model training, evaluation. Produces immutable pipeline artifacts.

- **PyCaret Control Plane** — the full web application that wraps the engine: auth, workspaces, projects, datasets, experiments, runs, artifact registry, deployment & serving, monitoring, drift detection, LLM-assisted experiment design. Self-hosted. Runs on a laptop, on one Docker host, or on Kubernetes + Terraform in the cloud.

## Who it's for

- **Data scientists** who want AutoML without lock-in.
- **ML engineers** who want a platform they can deploy inside their company and own the artifacts, logs, and pipelines.
- **Small teams** (≤20 people) who need the whole loop — train → deploy → monitor → improve — without Databricks licenses.
- **Enterprises** who need SSO, audit logs, and multi-cloud deployment in the same code they started prototyping with.

## The product loop we're optimising for

```
Create project
  Upload dataset
    Create experiment
      Run AutoML
        Compare pipelines
          Save artifact
            Deploy endpoint
              Monitor predictions
                Ask AI what to improve next
```

Every step beautiful, keyboard-first, dark-mode. No marketing chrome. No mystery meat. Single-column forms.

## What we are deliberately not

- **Not a hosted SaaS.** You run it. We might run one later (dual BSL-1.1 license in the platform packages preserves that option).
- **Not Databricks.** We don't manage clusters or notebooks or data warehouses. We own the path from "I have a CSV" to "I have a live prediction endpoint."
- **Not tied to any one LLM vendor.** Claude and OpenAI are both first-class via a provider router; bring your own key.
- **Not wedded to MLflow / Comet / Weights & Biases.** Their value is fine; their opinionated grip on the pipeline is not. Everything we need we own.
- **Not a feature store / data platform.** Integrations with Snowflake / Postgres / S3 / BigQuery, not a reimplementation.

## Deployment modes we ship

| Mode | For | Ships as |
|---|---|---|
| Local developer | A notebook user who wants the UI too | `uv run pycaret-server serve` + `npm run dev` |
| Local desktop *(V2)* | Analyst who wants a native app | Electron installer (mac / win / linux) |
| Single-server self-hosted | Small team | `docker compose up` |
| Kubernetes enterprise *(V2)* | Ops team | `helm install pycaret` + Terraform (AWS / GCP / Azure) |

## Three engineering principles

1. **Engine is stateless.** One call: `result = engine.run(config)`. No hidden globals, no `setup()`-then-`compare_models()` chain that breaks when you reorder cells.
2. **Config is the contract.** A single `RunConfig` JSON drives notebook runs, API runs, UI-wizard runs, LLM-generated runs. Same schema, four interfaces.
3. **Artifacts are immutable, deployments are versioned.** You never mutate a trained pipeline. You never deploy a moving target. Every production prediction is traceable to a specific `pipeline_artifact` row.

## What success looks like

- A data scientist can `pip install pycaret`, open a notebook, and train a model in 5 lines — the way 3.x worked, but faster and on modern sklearn.
- A team can `git clone`, `docker compose up`, and have the full Control Plane on `localhost:3000` in under 5 minutes.
- An enterprise can deploy to their AWS account with `terraform apply`, get SSO + audit logs, and satisfy compliance review.
- An analyst can upload a CSV, click "AutoML", get a leaderboard, click "Deploy", and ship a prediction endpoint — without writing Python.
- An LLM agent can read the dataset profile, propose a RunConfig, the user approves, and the deterministic engine runs it.

That's the whole product.

---

For the full specification see [`CONTROL_PLANE_SPEC.md`](CONTROL_PLANE_SPEC.md). For where we are and what's next see [`ROADMAP.md`](ROADMAP.md) and [`STATUS.md`](STATUS.md).
