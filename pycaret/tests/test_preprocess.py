import os
import sys
import pandas as pd
import numpy as np
import pytest
import pycaret.datasets
import pycaret.internal.preprocess
import pycaret.classification

sys.path.insert(0, os.path.abspath(".."))


def test_auto_infer_label():
    # loading dataset
    data = pycaret.datasets.get_data("juice")
    data.loc[:, 'test_target'] = np.random.randint(5, 8, data.shape[0])
    data.loc[:, 'test_target'] = data.loc[:, 'test_target'].astype(np.int64)  # should not encode
    target = 'test_target'

    # init setup
    _ = pycaret.classification.setup(
        data,
        target=target,
        log_experiment=True,
        silent=True,
        html=False,
        session_id=123,
        n_jobs=1
    )

    with pytest.raises(AttributeError):
        _ = pycaret.classification.get_config('prep_pipe').named_steps["dtypes"].replacement


def test():
    # loading dataset
    data = pycaret.datasets.get_data("juice")
    target = "Purchase"

    # preprocess all in one
    pipe = pycaret.internal.preprocess.Preprocess_Path_One(
        train_data=data, target_variable=target, display_types=False
    )
    X = pipe.fit_transform(data)
    assert isinstance(X, pd.core.frame.DataFrame)

    assert 1 == 1


if __name__ == "__main__":
    test()
