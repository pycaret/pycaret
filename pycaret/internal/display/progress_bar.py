from abc import ABC, abstractmethod
from typing import Optional, Union

import tqdm.notebook
import tqdm.std

from pycaret.internal.display.display_backend import (
    DatabricksBackend,
    DisplayBackend,
    JupyterBackend,
)
from pycaret.internal.display.display_component import DisplayComponent


class ProgressBarBackend(ABC):
    def __init__(
        self,
        value: int = 0,
        min: int = 0,
        max: int = 10,
        description: str = "Processing: ",
        backend: Optional[Union[str, DisplayBackend]] = None,
    ) -> None:
        self.value = value
        self.min = min
        self.max = max
        self.description = description
        self.backend = backend
        self._closed = False

    @abstractmethod
    def open(self):
        self._pbar = tqdm.std.tqdm()

    def step(self, value: int):
        if not self._closed:
            self._pbar.update(value)
            if self._pbar.total and self._pbar.n >= self._pbar.total:
                self.close()

    def close(self):
        self._pbar.close()
        self._closed = True

    def __del__(self):
        self.close()


class CLIProgressBarBackend(ProgressBarBackend):
    def open(self):
        self._pbar = tqdm.std.tqdm(
            desc=self.description,
            total=self.max,
            initial=self.min,
            leave=False,
        )


class CustomDisplayNotebookTqdm(tqdm.notebook.tqdm):
    def __init__(self, *args, **kwargs):
        self.display_backend: DisplayBackend = kwargs.pop("display_backend")
        super().__init__(*args, **kwargs)

    def display(
        self, msg=None, pos=None, close=False, bar_style=None, check_delay=True
    ):
        # trick tqdm into doing all the updating without displaying in its own display
        original_displayed = self.displayed
        self.displayed = True
        _, pbar, _ = self.container.children
        super().display(msg, pos, close, bar_style, check_delay)
        self.displayed = original_displayed
        if check_delay and self.delay > 0 and not self.displayed:
            self.display_backend.display(self.container)
            self.displayed = True

        if close:
            try:
                self.container.close()
            except AttributeError:
                self.container.visible = False
            self.display_backend.clear_display()

    def close(self):
        if self.disable:
            return
        super().close()
        # Try to detect if there was an error or KeyboardInterrupt
        # in manual mode: if n < total, things probably got wrong
        if self.leave:
            self.disp(bar_style="success", check_delay=False)
        else:
            self.disp(close=True, check_delay=False)


class JupyterProgressBarBackend(ProgressBarBackend):
    def open(self):
        self._pbar = CustomDisplayNotebookTqdm(
            desc=self.description,
            total=self.max,
            initial=self.min,
            display_backend=self.backend,
            display=False,
            leave=False,
        )
        self._pbar.delay = 1
        self._pbar.display()


class ProgressBarDisplay(DisplayComponent):
    def __init__(
        self,
        value: int = 0,
        min: int = 0,
        max: int = 10,
        description: str = "Processing: ",
        *,
        verbose: bool = True,
        backend: Optional[Union[str, DisplayBackend]] = None
    ) -> None:
        super().__init__(verbose=verbose, backend=backend)
        if isinstance(self.backend, JupyterBackend) and not isinstance(
            self.backend, DatabricksBackend
        ):
            self.pbar_backend_cls = JupyterProgressBarBackend
        else:
            self.pbar_backend_cls = CLIProgressBarBackend
        self.pbar_backend = None
        self.value = value
        self.min = min
        self.max = max
        self.description = description

    def display(self):
        if not self.verbose:
            return
        if self.pbar_backend:
            self.pbar_backend.close()
        self.pbar_backend = self.pbar_backend_cls(
            value=self.value,
            min=self.min,
            max=self.max,
            description=self.description,
            backend=self.backend,
        )
        self.pbar_backend.open()

    def step(self, value: int = 1):
        if not self.verbose:
            return
        self.pbar_backend.step(value)

    def close(self):
        if not self.verbose:
            return
        self.pbar_backend.close()
        return super().close()
