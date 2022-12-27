from pycaret.utils._dependencies import _check_soft_dependencies

if _check_soft_dependencies("fugue", extra="parallel", severity="error"):
    from .fugue_backend import FugueBackend
