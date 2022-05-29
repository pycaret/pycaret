from typing import Any, Dict, List, Optional, Type, Union

from pycaret.internal.Display import CommonDisplay


class NoDisplay(CommonDisplay):
    """The Display class to completely turn off all displays"""

    def can_display(self, override):
        return False


class ParallelBackend:
    """The parallel backend base class for different parallel implementations.
    None of the methods of this class should be called by users.
    """

    def __init__(self) -> None:
        self._exp_class: Optional[Type] = None
        self._instance_pack: Any = None

    def attach(self, instance: Any):
        """Attach the current setup function to this backend.

        instance: Any
            The ``_PyCaretExperiment`` instance
        """
        self._instance_pack = instance._pack_for_remote()
        self._exp_class = type(instance)

    def remote_setup(self) -> Any:
        """Call setup on a worker."""
        instance = self._exp_class()
        instance._unpack_at_remote(self._instance_pack)
        params = dict(instance._setup_params)
        params["silent"] = True
        params["verbose"] = False
        params["html"] = False
        instance.setup(**params)
        return instance

    def compare_models(
        self, instance: Any, params: Dict[str, Any]
    ) -> Union[Any, List[Any]]:
        """Distributed ``compare_models`` wrapper.

        instance: Any
            The ``_PyCaretExperiment`` instance

        params: Dict[str, Any]
            The parameters used to call the ``compare_models`` function
        """
        raise NotImplementedError
