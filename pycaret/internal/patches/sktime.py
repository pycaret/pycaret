# pycaret/internal/patches/sktime.py
"""
Monkey temporary patches to fix sktime issues until PR #7903 is merged.
- Fix LossySetitemError in SummaryTransformer.
- Fix infinite recursion in _custom_showwarning.
Todo: Remove this file when sktime is updated to a version that includes SKTIME PR #7903.
Author: [CelestinoXP]
"""

import warnings
import pandas as pd
from sktime.transformations.series.summarize import SummaryTransformer
from sktime.utils.warnings import _SuppressWarningPattern

# 1.Patch for LossySetitemError in SummaryTransformer
original_summary_fit = SummaryTransformer.fit

def patched_summary_fit(self, X, y=None):
    """Patched fit method para evitar LossySetitemError."""
    self._setup(X)
    func_dict = self.func_dict.copy()
    func_dict["window"] = func_dict["window"].astype("object", copy=False)
    self.func_dict_ = func_dict
    return self

SummaryTransformer.fit = patched_summary_fit

#2. Patch for infinite recursion in _custom_showwarning
original_custom_showwarning = _SuppressWarningPattern._custom_showwarning

def patched_custom_showwarning(self, message, category, filename, lineno, file=None, line=None):
    """Patched _custom_showwarning para evitar recursão infinita."""
    if not hasattr(self, "_in_warning"):
        self._in_warning = True
        try:
            right_type = issubclass(category, self.warning_type)
            fits_pattern = self.message_pattern.search(str(message))
            if not (right_type and fits_pattern):
                self.original_showwarning(message, category, filename, lineno, file, line)
        finally:
            del self._in_warning

_SuppressWarningPattern._custom_showwarning = patched_custom_showwarning