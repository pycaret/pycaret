# pycaret/internal/patches/sktime.py
"""
Monkey temporary patches to fix sktime issues until PR #7903 is merged.
- Fix LossySetitemError and FutureWarning in SummaryTransformer.
- Fix infinite recursion in _custom_showwarning.
Todo: Remove this file when sktime is updated to a version that includes SKTIME PR #7903.
Author: [CelestinoXP]
"""

import warnings

import sktime.utils.warnings
from sktime.transformations.series.summarize import SummaryTransformer

# 1. Patch for LossySetitemError and FutureWarning in SummaryTransformer
original_summary_fit = SummaryTransformer.fit


def patched_summary_fit(self, X, y=None):
    """Patched fit method para evitar LossySetitemError e FutureWarning."""
    # Recall the original setting with warning suppression
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="Setting an item of incompatible dtype is deprecated",
        )
        original_summary_fit(self, X, y)
    # Fix func_dict_ directly
    if hasattr(self, "func_dict_"):
        func_dict = self.func_dict_.copy()
        func_dict["window"] = func_dict["window"].astype("object", copy=False)
        self.func_dict_ = func_dict
    return self


SummaryTransformer.fit = patched_summary_fit


# 2. Patch for infinite recursion in _custom_showwarning
def patched_custom_showwarning(
    self, message, category, filename, lineno, file=None, line=None
):
    """Patched _custom_showwarning para evitar recursão infinita."""
    if not hasattr(self, "_in_warning"):
        self._in_warning = True
        try:
            # Use Python's default showwarning as a safe fallback
            warnings._showwarnmsg(
                warnings.WarningMessage(message, category, filename, lineno, file, line)
            )
        finally:
            del self._in_warning


# Directly replaces the sktime.utils.warnings module
sktime.utils.warnings._SuppressWarningPattern._custom_showwarning = (
    patched_custom_showwarning
)
