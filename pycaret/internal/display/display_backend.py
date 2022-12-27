from abc import ABC, abstractmethod
from pprint import pprint
from typing import Any, Optional, Union

import pandas as pd

if "get_ipython" not in globals():
    from IPython import get_ipython

from IPython.display import HTML, DisplayHandle, clear_output
from IPython.display import display as ipython_display
from pandas.io.formats.style import Styler

try:
    import dbruntime.display

    IN_DATABRICKS = True
except ImportError:
    IN_DATABRICKS = False

try:
    import google.colab

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

COLAB_ENABLED = False
EMPTY_HTML = HTML("")
EMPTY = ""


def _enable_matplotlib_inline():
    get_ipython().run_line_magic("matplotlib", "inline")


def _enable_colab():
    global COLAB_ENABLED
    if not COLAB_ENABLED:
        from google.colab import output

        output.enable_custom_widget_manager()
        COLAB_ENABLED = True


class DisplayBackend(ABC):
    id: str
    can_update_text: bool
    can_update_rich: bool

    @abstractmethod
    def display(self, obj: Any, *, final_display: bool = True) -> None:
        """Display obj.

        Args:
            final_display: If True, this is considered the final
            display of the caller. Set to False if display
            will be updated. Allows for special Databricks
            logic."""
        pass

    @abstractmethod
    def clear_display(self) -> None:
        """Clear current display (not entire cell)."""
        pass

    @abstractmethod
    def clear_output(self) -> None:
        """Clear entire cell."""
        pass


class SilentBackend(DisplayBackend):
    id: str = "silent"
    can_update_text: bool = False
    can_update_rich: bool = False

    def display(self, obj: Any, *, final_display: bool = True) -> None:
        pass

    def clear_display(self) -> None:
        pass

    def clear_output(self) -> None:
        pass


class CLIBackend(DisplayBackend):
    id: str = "cli"
    can_update_text: bool = False
    can_update_rich: bool = False

    def display(self, obj: Any, *, final_display: bool = True) -> None:
        obj = self._handle_input(obj)
        if obj is not None:
            if hasattr(obj, "show"):
                obj.show()
                return
            pprint(obj)

    def clear_display(self) -> None:
        pass

    def clear_output(self) -> None:
        pass

    def _handle_input(self, obj: Any) -> Any:
        if isinstance(obj, Styler):
            obj = obj.data
        if isinstance(obj, (pd.Series, pd.DataFrame)) and obj.empty:
            return None
        return obj


class JupyterBackend(DisplayBackend):
    id: str = "jupyter"
    can_update_text: bool = True
    can_update_rich: bool = True
    _empty_content = EMPTY_HTML

    def __init__(self) -> None:
        _enable_matplotlib_inline()
        self._display_ref: Optional[DisplayHandle] = None

    def display(self, obj: Any, *, final_display: bool = True) -> None:
        self._display(obj)

    def _display(self, obj: Any, **display_kwargs):
        obj = self._handle_input(obj)
        if not self._display_ref:
            if obj is not None:
                self._display_ref = ipython_display(
                    obj, display_id=True, **display_kwargs
                )
            else:
                self._display_ref = ipython_display(
                    self._empty_content, display_id=True, **display_kwargs
                )
        elif obj is not None:
            self._display_ref.update(obj, **display_kwargs)

    def clear_display(self) -> None:
        if self._display_ref:
            self._display_ref.update(self._empty_content)

    def clear_output(self) -> None:
        clear_output(wait=True)

    def _handle_input(self, obj: Any) -> Any:
        return obj


class ColabBackend(JupyterBackend):
    id: str = "colab"

    def __init__(self) -> None:
        _enable_colab()
        super().__init__()


class DatabricksBackend(JupyterBackend):
    id: str = "databricks"
    can_update_rich: bool = False
    _empty_content = EMPTY

    def display(self, obj: Any, *, final_display: bool = True) -> None:
        if not final_display:
            display_kwargs = dict(include=["text/plain"])
            self._display(obj, **display_kwargs)
        else:
            self._display(obj)

    def _handle_input(self, obj: Any) -> Any:
        if isinstance(obj, Styler):
            return obj.data

        return obj


backends = [CLIBackend, JupyterBackend, ColabBackend, DatabricksBackend, SilentBackend]
backends = {b.id: b for b in backends}


def detect_backend(
    backend: Optional[Union[str, DisplayBackend]] = None
) -> DisplayBackend:
    if backend is None:
        class_name = ""

        if IN_DATABRICKS:
            return DatabricksBackend()

        if IN_COLAB:
            return ColabBackend()

        try:
            ipython = get_ipython()
            assert ipython
            class_name = str(ipython.__class__)
            is_notebook = True if "Terminal" not in class_name else False
        except Exception:
            is_notebook = False

        if not is_notebook:
            return CLIBackend()
        return JupyterBackend()

    if isinstance(backend, str):
        backend_id = backend.lower()
        backend = backends.get(backend_id, None)
        if not backend:
            raise ValueError(
                f"Wrong backend id. Got {backend_id}, expected one of {list(backends.keys())}."
            )
        return backend()

    if isinstance(backend, DisplayBackend):
        return backend

    raise TypeError(
        f"Wrong backend type. Expected None, str or DisplayBackend, got {type(backend)}."
    )


detect_backend()
