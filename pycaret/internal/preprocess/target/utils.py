from pycaret.utils.generic import get_all_object_vars_and_properties, is_fit_var


class TargetTransformerMixin:
    @property
    def estimator(self):
        if hasattr(self, "regressor_"):
            return self.regressor_
        if hasattr(self, "regressor"):
            return self.regressor
        if hasattr(self, "classifier_"):
            return self.classifier_
        if hasattr(self, "classifier"):
            return self.classifier
        return self

    def _carry_over_estimator_fit_vars(self, fitted_estimator, ignore: list = None):
        if not ignore:
            ignore = set()
        else:
            ignore = set(ignore)
        if not hasattr(self, "_fit_vars"):
            self._fit_vars = set()
        self._clear_estimator_fit_vars(fitted_estimator)
        for k, v in get_all_object_vars_and_properties(fitted_estimator).items():
            if is_fit_var(k) and k not in ignore:
                try:
                    setattr(self, k, v)
                    self._fit_vars.add(k)
                except Exception:
                    pass

    def _clear_estimator_fit_vars(self, fitted_estimator, all: bool = False):
        if not hasattr(self, "_fit_vars"):
            self._fit_vars = set()
        vars_to_remove = []
        try:
            for var in self._fit_vars:
                if all or var not in get_all_object_vars_and_properties(
                    fitted_estimator
                ):
                    vars_to_remove.append(var)
            for var in vars_to_remove:
                try:
                    delattr(self, var)
                    self._fit_vars.remove(var)
                except Exception:
                    pass
        except Exception:
            pass


def get_estimator_from_meta_estimator(estimator):
    """
    If ``estimator`` is a meta estimator, get estimator inside.
    Otherwise return ``estimator``. Will try to return the fitted
    estimator first.
    """
    try:
        return estimator.estimator
    except Exception:
        return estimator
