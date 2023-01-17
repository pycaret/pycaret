import pandas as pd

from pycaret.internal.pycaret_experiment.supervised_experiment import (
    _SupervisedExperiment,
)


class _NonTSSupervisedExperiment(_SupervisedExperiment):
    def __init__(self) -> None:
        super().__init__()

    @property
    def X(self):
        """Feature set."""
        return self.dataset.drop(self.target_param, axis=1)

    @property
    def dataset_transformed(self):
        """Transformed dataset."""
        return pd.concat([self.train_transformed, self.test_transformed])

    @property
    def X_train_transformed(self):
        """Transformed feature set of the training set."""
        return self.pipeline.transform(
            X=self.X_train,
            y=self.y_train,
            filter_train_only=False,
        )[0]

    @property
    def train_transformed(self):
        """Transformed training set."""
        return pd.concat(
            [self.X_train_transformed, self.y_train_transformed],
            axis=1,
        )

    @property
    def X_transformed(self):
        """Transformed feature set."""
        return pd.concat([self.X_train_transformed, self.X_test_transformed])

    @property
    def X_train(self):
        """Feature set of the training set."""
        return self.train.drop(self.target_param, axis=1)

    @property
    def X_test(self):
        """Feature set of the test set."""
        return self.test.drop(self.target_param, axis=1)

    @property
    def test(self):
        """Test set."""
        return self.dataset.loc[self.idx[1], :]

    @property
    def test_transformed(self):
        """Transformed test set."""
        return pd.concat(
            [self.X_test_transformed, self.y_test_transformed],
            axis=1,
        )

    @property
    def y_transformed(self):
        """Transformed target column."""
        return pd.concat([self.y_train_transformed, self.y_test_transformed])

    @property
    def X_test_transformed(self):
        """Transformed feature set of the test set."""
        return self.pipeline.transform(self.X_test)

    @property
    def y_train_transformed(self):
        """Transformed target column of the training set."""
        return self.pipeline.transform(
            X=self.X_train,
            y=self.y_train,
            filter_train_only=False,
        )[1]

    @property
    def y_test_transformed(self):
        """Transformed target column of the test set."""
        return self.pipeline.transform(y=self.y_test)

    def _create_model_get_train_X_y(self, X_train, y_train):
        """Return appropriate training X and y values depending on whether
        X_train and y_train are passed or not. If X_train and y_train are not
        passes, internal self.X_train and self.y_train are returned. If they are
        passed, then a copy of them is returned.
        """
        if X_train is not None:
            data_X = X_train.copy()
        else:
            if self.X_train is None:
                data_X = None
            else:
                data_X = self.X_train
        data_y = self.y_train if y_train is None else y_train.copy()
        return data_X, data_y
