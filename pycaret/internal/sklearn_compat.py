import inspect
from typing import Any


def _supports_kw(callable_obj: Any, kw: str) -> bool:
    try:
        return kw in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        # Some callables (e.g. C-extensions) may not have introspectable
        # signatures. In that case, assume the kw is not supported.
        return False


def _finite_check_param_name() -> str:
    """Return the kwarg name used by sklearn's check_array for finite checks."""

    from sklearn.utils.validation import check_array

    # sklearn<1.6 uses `force_all_finite`, sklearn>=1.6 uses `ensure_all_finite`.
    return (
        "ensure_all_finite"
        if _supports_kw(check_array, "ensure_all_finite")
        else "force_all_finite"
    )


def validate_data(
    estimator: Any,
    X: Any,
    *,
    allow_nan: bool,
    **kwargs: Any,
):
    """Validate X across sklearn versions.

    scikit-learn renamed the finite-check kwarg from `force_all_finite` to
    `ensure_all_finite` (and introduced the standalone `validate_data` helper).
    This wrapper keeps PyCaret compatible with both APIs.
    """

    finite_value = "allow-nan" if allow_nan else True
    kwargs[_finite_check_param_name()] = finite_value

    try:
        from sklearn.utils.validation import validate_data as sklearn_validate_data
    except ImportError:
        return estimator._validate_data(X, **kwargs)

    return sklearn_validate_data(estimator, X, **kwargs)
