import pandas as pd

from pycaret.internal.pycaret_experiment.supervised_experiment import (
    _SupervisedExperiment,
)
from pycaret.utils.time_series.forecasting.pipeline import _pipeline_transform


class _TSSupervisedExperiment(_SupervisedExperiment):
    @property
    def X(self):
        X = self.dataset.drop(self.target_param, axis=1)
        if X.empty and self.fe_exogenous is None:
            return None
        else:
            # If X is not empty or empty but self.fe_exogenous is provided
            # Return X instead of None, since the index can be used to
            # generate features using self.fe_exogenous
            return X

    @property
    def dataset_transformed(self):
        # Use fully trained pipeline to get the requested data
        return pd.concat(
            [
                *_pipeline_transform(
                    pipeline=self.pipeline_fully_trained, y=self.y, X=self.X
                )
            ],
            axis=1,
        )

    @property
    def X_train_transformed(self):
        # Use pipeline trained on training data only to get the requested data
        # In time series, the order of arguments and returns may be reversed.
        return _pipeline_transform(
            pipeline=self.pipeline, y=self.y_train, X=self.X_train
        )[1]

    @property
    def train_transformed(self):
        # Use pipeline trained on training data only to get the requested data
        # In time series, the order of arguments and returns may be reversed.
        return pd.concat(
            [
                *_pipeline_transform(
                    pipeline=self.pipeline, y=self.y_train, X=self.X_train
                )
            ],
            axis=1,
        )

    @property
    def X_transformed(self):
        # Use fully trained pipeline to get the requested data
        # In time series, the order of arguments and returns may be reversed.
        return _pipeline_transform(
            pipeline=self.pipeline_fully_trained, y=self.y, X=self.X
        )[1]

    @property
    def X_train(self):
        X_train = self.train.drop(self.target_param, axis=1)

        if X_train.empty and self.fe_exogenous is None:
            return None
        else:
            # If X_train is not empty or empty but self.fe_exogenous is provided
            # Return X_train instead of None, since the index can be used to
            # generate features using self.fe_exogenous
            return X_train

    @property
    def X_test(self):
        # Use index for y_test (idx 2) to get the data
        test = self.dataset.loc[self.idx[2], :]
        X_test = test.drop(self.target_param, axis=1)

        if X_test.empty and self.fe_exogenous is None:
            return None
        else:
            # If X_test is not empty or empty but self.fe_exogenous is provided
            # Return X_test instead of None, since the index can be used to
            # generate features using self.fe_exogenous
            return X_test

    @property
    def test(self):
        # Return the y_test indices not X_test indices.
        # X_test indices are expanded indices for handling FH with gaps.
        # But if we return X_test indices, then we will get expanded test
        # indices even for univariate time series without exogenous variables
        # which would be confusing. Hence, we return y_test indices here and if
        # we want to get X_test indices, then we use self.X_test directly.
        # Refer:
        # https://github.com/sktime/sktime/issues/2598#issuecomment-1203308542
        # https://github.com/sktime/sktime/blob/4164639e1c521b112711c045d0f7e63013c1e4eb/sktime/forecasting/model_evaluation/_functions.py#L196
        return self.dataset.loc[self.idx[1], :]

    @property
    def test_transformed(self):
        # When transforming the test set, we can and should use all data before that
        # In time series, the order of arguments and returns may be reversed.
        all_data = pd.concat(
            [
                *_pipeline_transform(
                    pipeline=self.pipeline_fully_trained,
                    y=self.y,
                    X=self.X,
                )
            ],
            axis=1,
        )
        # Return the y_test indices not X_test indices.
        # X_test indices are expanded indices for handling FH with gaps.
        # But if we return X_test indices, then we will get expanded test
        # indices even for univariate time series without exogenous variables
        # which would be confusing. Hence, we return y_test indices here and if
        # we want to get X_test indices, then we use self.X_test directly.
        # Refer:
        # https://github.com/sktime/sktime/issues/2598#issuecomment-1203308542
        # https://github.com/sktime/sktime/blob/4164639e1c521b112711c045d0f7e63013c1e4eb/sktime/forecasting/model_evaluation/_functions.py#L196
        return all_data.loc[self.idx[1]]

    @property
    def y_transformed(self):
        # Use fully trained pipeline to get the requested data
        # In time series, the order of arguments and returns may be reversed.
        return _pipeline_transform(
            pipeline=self.pipeline_fully_trained, y=self.y, X=self.X
        )[0]

    @property
    def X_test_transformed(self):
        # In time series, the order of arguments and returns may be reversed.
        # When transforming the test set, we can and should use all data before that
        _, X = _pipeline_transform(
            pipeline=self.pipeline_fully_trained, y=self.y, X=self.X
        )

        if X is None:
            return None
        else:
            return X.loc[self.idx[2]]

    @property
    def y_train_transformed(self):
        # Use pipeline trained on training data only to get the requested data
        # In time series, the order of arguments and returns may be reversed.
        return _pipeline_transform(
            pipeline=self.pipeline, y=self.y_train, X=self.X_train
        )[0]

    @property
    def y_test_transformed(self):
        # In time series, the order of arguments and returns may be reversed.
        # When transforming the test set, we can and should use all data before that
        y, _ = _pipeline_transform(
            pipeline=self.pipeline_fully_trained, y=self.y, X=self.X
        )
        return y.loc[self.idx[1]]

    def _create_model_get_train_X_y(self, X_train, y_train):
        """Return appropriate training X and y values depending on whether
        X_train and y_train are passed or not. If X_train and y_train are not
        passes, internal self.X_train and self.y_train are returned. If they are
        passed, then a copy of them is returned."""
        data_X = self.X_train if X_train is None else X_train.copy()
        data_y = self.y_train if y_train is None else y_train.copy()
        return data_X, data_y
