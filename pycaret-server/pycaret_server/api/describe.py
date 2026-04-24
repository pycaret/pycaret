"""Engine-introspection proxy routes.

Thin wrappers over ``pycaret.api`` — the React UI / LLM agents hit these to
render forms and dropdowns without shipping a Python runtime to the browser.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pycaret.api import (
    describe_model,
    describe_setup_params,
    list_metrics,
    list_models,
)
from pycaret.core.errors import ConfigurationError, UnknownModelError

router = APIRouter(prefix="/describe", tags=["describe"])


@router.get("/models")
def list_models_route(task: str) -> list[dict]:
    """List all registered models for a task — returns ModelCard.to_dict()."""
    try:
        return [m.to_dict() for m in list_models(task)]
    except ConfigurationError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e


@router.get("/models/{model_id}")
def describe_model_route(task: str, model_id: str) -> dict:
    try:
        return describe_model(task, model_id).to_dict()
    except UnknownModelError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    except ConfigurationError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e


@router.get("/metrics")
def list_metrics_route(task: str) -> list[dict]:
    try:
        return [m.to_dict() for m in list_metrics(task)]
    except ConfigurationError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e


@router.get("/setup-params")
def describe_setup_params_route(task: str) -> dict:
    """The schema a React form renders from."""
    try:
        return describe_setup_params(task).to_dict()
    except ConfigurationError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
