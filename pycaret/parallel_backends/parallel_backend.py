from typing import Any, Callable, Dict, List, Optional, Union

from pycaret.internal.Display import Display
from pycaret.internal.tabular import _get_setup_signature


class NoDisplay(Display):
    def can_display(self, override):
        return False


class ParallelBackend:
    def __init__(self) -> None:
        self._signature: Optional[str] = None
        self._setup_func: Optional[Callable] = None
        self._setup_params: Optional[Dict[str, Any]] = None

    def attach(self, setup_func: Callable, setup_params: Dict[str, Any]):
        self._signature: Optional[str] = _get_setup_signature()
        self._setup_func = setup_func
        self._setup_params = {k: v for k, v in setup_params.items() if v is not None}

    def remote_setup(self):
        if self._signature != _get_setup_signature():
            params = dict(self._setup_params)
            params["silent"] = True
            params["verbose"] = False
            params["html"] = False
            self._setup_func(**params)

    def compare_models(
        self, func: Callable, params: Dict[str, Any]
    ) -> Union[Any, List[Any]]:
        raise NotImplementedError
