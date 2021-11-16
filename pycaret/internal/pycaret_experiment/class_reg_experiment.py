import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

from pycaret.internal.logging import get_logger
from pycaret.internal.pycaret_experiment.supervised_experiment import (
    _SupervisedExperiment,
)
from pycaret.internal.utils import (
    get_columns_to_stratify_by,
)

warnings.filterwarnings("ignore")
LOGGER = get_logger()


class ClassRegExperiment(_SupervisedExperiment):

    def _get_data_and_idx(self, train_size, test_data, shuffle, stratify):
        if test_data is None:
            train, test = train_test_split(
                self.data,
                test_size=1 - train_size,
                stratify=get_columns_to_stratify_by(self.X, self.y, stratify),
                random_state=self.seed,
                shuffle=shuffle,
            )
            data = pd.concat([train, test]).reset_index(drop=True)
            idx = [train.index, test.index]

        else:  # test_data is provided
            data = pd.concat([data, test_data]).reset_index(drop=True)
            idx = [data.index, test_data.index]

        return data, idx
