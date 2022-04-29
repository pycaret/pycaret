from pycaret.utils._dependencies import _check_soft_dependencies

if _check_soft_dependencies("fugue", extra="others", severity="warning"):
    from .fugue_backend import FugueBackend
