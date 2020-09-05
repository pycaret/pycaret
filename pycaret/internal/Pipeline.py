import imblearn.pipeline


class Pipeline(imblearn.pipeline.Pipeline):
    def fit(self, X, y=None, **fit_params):
        result = super().fit(X, y=y, **fit_params)

        try:
            self.coef_ = self.steps[-1][-1].coef_
        except:
            pass
        try:
            self.feature_importances_ = self.steps[-1][-1].feature_importances_
        except:
            pass
        return result
