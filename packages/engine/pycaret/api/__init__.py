"""PyCaret 4.0 introspection API — designed for the React UI and LLM agents.

This package is the *typed, side-effect-free* view of the engine. It lets an
external process (a UI server, an LLM agent, a docs generator) answer
questions like:

- What models does PyCaret support for classification?
- What are the hyperparameters of XGBoost in this version, with ranges and
  defaults?
- What parameters does `setup(...)` accept, in what shape would a React form
  render them?
- What metrics are available for regression, and which is the default?

Nothing in here trains a model. Every function returns a dataclass (or list of
dataclasses) that is JSON-serializable and independently useful.

Public surface:

    # Work without an Experiment (static — for form rendering and docs)
    list_models(TaskType.CLASSIFICATION) -> list[ModelCard]
    describe_model(TaskType.CLASSIFICATION, "lr") -> ModelCard
    list_metrics(TaskType.CLASSIFICATION) -> list[MetricCard]
    describe_setup_params(TaskType.CLASSIFICATION) -> SetupParamSchema

    # Work with a live Experiment (includes runtime state — e.g. which
    # optional packages are installed, which engines are active)
    list_available_models(experiment) -> list[ModelCard]
"""

from pycaret.api.cards import MetricCard, ModelCard, ParameterCard, ParameterKind
from pycaret.api.describe import (
    describe_model,
    describe_setup_params,
    list_available_models,
    list_metrics,
    list_models,
)
from pycaret.api.schemas import SetupParamSchema

__all__ = [
    "MetricCard",
    "ModelCard",
    "ParameterCard",
    "ParameterKind",
    "SetupParamSchema",
    "describe_model",
    "describe_setup_params",
    "list_available_models",
    "list_metrics",
    "list_models",
]
