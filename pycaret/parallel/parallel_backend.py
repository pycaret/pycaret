from typing import Any, Callable, Dict, List, Optional, Union

from pycaret.internal.Display import Display
from pycaret.internal.tabular import _get_setup_signature


class NoDisplay(Display):
    """The Display class to completely turn off all displays"""

    def can_display(self, override):
        return False


class ParallelBackend:
    """The parallel backend base class for different parallel implementations.
    None of the methods of this class should be called by users.
    """

    def __init__(self) -> None:
        self._signature: Optional[str] = None
        self._setup_func: Optional[Callable] = None
        self._setup_params: Optional[Dict[str, Any]] = None

    def attach(self, setup_func: Callable, setup_params: Dict[str, Any]):
        """Attach the current setup function to this backend.

        setup_func: Callable
            The ``setup`` function used

        setup_params: Dict[str, Any]
            The parameters used to call the ``setup`` function
        """
        self._signature: Optional[str] = _get_setup_signature()
        self._setup_func = setup_func
        self._setup_params = {k: v for k, v in setup_params.items() if v is not None}

    def remote_setup(self):
        """Call setup on a worker."""
        if self._signature != _get_setup_signature():
            params = dict(self._setup_params)
            params["silent"] = True
            params["verbose"] = False
            params["html"] = False
            self._setup_func(**params)

    def compare_models(
        self, func: Callable, params: Dict[str, Any]
    ) -> Union[Any, List[Any]]:
        """Distributed ``compare_models`` wrapper.

        func: Callable
            The ``compare_models`` function used

        params: Dict[str, Any]
            The parameters used to call the ``compare_models`` function
        """
        raise NotImplementedError
