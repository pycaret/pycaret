import uuid

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

import pycaret.datasets
import pycaret.nlp


@pytest.fixture(scope="module")
def kiva_dataframe():
    # loading dataset
    data = pycaret.datasets.get_data("kiva")
    data = data.head(1000)
    assert isinstance(data, pd.DataFrame)
    return pycaret.datasets.get_data("kiva")


def test_nlp(kiva_dataframe):
    data = kiva_dataframe.head(1000)
    assert isinstance(data, pd.DataFrame)

    # init setup
    nlp1 = pycaret.nlp.setup(
        data=data,
        target="en",
        log_experiment=True,
        html=False,
        session_id=123,
    )
    assert isinstance(nlp1, tuple)
    assert isinstance(nlp1[0], list)
    assert isinstance(nlp1[1], pd.DataFrame)
    assert isinstance(nlp1[2], list)
    assert isinstance(nlp1[4], int)
    assert isinstance(nlp1[5], str)
    assert isinstance(nlp1[6], list)
    assert isinstance(nlp1[7], str)
    assert isinstance(nlp1[8], bool)
    assert isinstance(nlp1[9], bool)

    # create model
    lda = pycaret.nlp.create_model("lda")

    # assign model
    lda_results = pycaret.nlp.assign_model(lda)
    assert isinstance(lda_results, pd.DataFrame)

    # evaluate model
    pycaret.nlp.evaluate_model(lda)

    # save model
    pycaret.nlp.save_model(lda, "lda_model_23122019")

    # load model
    pycaret.nlp.load_model("lda_model_23122019")

    # returns table of models
    all_models = pycaret.nlp.models()
    assert isinstance(all_models, pd.DataFrame)

    # get config
    text = pycaret.nlp.get_config("text")
    assert isinstance(text, list)

    # set config
    pycaret.nlp.set_config("seed", 124)
    seed = pycaret.nlp.get_config("seed")
    assert seed == 124

    assert 1 == 1


class TestNLPExperimentCustomTags:
    def test_nlp_setup_fails_with_experiment_custom_tags(self, kiva_dataframe):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.nlp.setup(
                data=kiva_dataframe,
                target="en",
                log_experiment=True,
                html=False,
                session_id=123,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags="custom_tag",
            )

    def test_nlp_create_model_fails_with_experiment_custom_tags(self, kiva_dataframe):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.nlp.setup(
                data=kiva_dataframe,
                target="en",
                log_experiment=True,
                html=False,
                session_id=123,
                experiment_name=uuid.uuid4().hex,
            )
            _ = pycaret.nlp.create_model(
                "lda", experiment_custom_tags=("pytest", "testing")
            )

    @pytest.mark.parametrize("custom_tag", [1, ("pytest", "True"), True, 1000.0])
    def test_nlp_setup_fails_with_experiment_custom_multiples_inputs(self, custom_tag):
        with pytest.raises(TypeError):
            # init setup
            _ = pycaret.nlp.setup(
                data=kiva_dataframe,
                target="en",
                log_experiment=True,
                html=False,
                session_id=123,
                experiment_name=uuid.uuid4().hex,
                experiment_custom_tags=custom_tag,
            )

    def test_nlp_setup_with_experiment_custom_tags(self, kiva_dataframe):
        experiment_name = uuid.uuid4().hex
        # init setup
        _ = pycaret.nlp.setup(
            data=kiva_dataframe,
            target="en",
            log_experiment=True,
            html=False,
            session_id=123,
            experiment_name=experiment_name,
            experiment_custom_tags={"pytest": "testing"},
        )

        # get experiment data
        tracking_api = MlflowClient()
        experiment = [
            e for e in tracking_api.list_experiments() if e.name == experiment_name
        ][0]
        experiment_id = experiment.experiment_id
        # get run's info
        experiment_run = tracking_api.list_run_infos(experiment_id)[0]
        # get run id
        run_id = experiment_run.run_id
        # get run data
        run_data = tracking_api.get_run(run_id)
        # assert that custom tag was inserted
        assert "testing" == run_data.to_dictionary().get("data").get("tags").get(
            "pytest"
        )

    def test_nlp_create_models_with_experiment_custom_tags(self, kiva_dataframe):
        experiment_name = uuid.uuid4().hex
        # init setup
        _ = pycaret.nlp.setup(
            data=kiva_dataframe,
            target="en",
            log_experiment=True,
            html=False,
            session_id=123,
            experiment_name=experiment_name,
        )
        _ = pycaret.nlp.create_model(
            "lda", experiment_custom_tags={"pytest": "testing"}
        )

        # get experiment data
        tracking_api = MlflowClient()
        experiment = [
            e for e in tracking_api.list_experiments() if e.name == experiment_name
        ][0]
        experiment_id = experiment.experiment_id
        # get run's info
        experiment_run = tracking_api.list_run_infos(experiment_id)[0]
        # get run id
        run_id = experiment_run.run_id
        # get run data
        run_data = tracking_api.get_run(run_id)
        # assert that custom tag was inserted
        assert "testing" == run_data.to_dictionary().get("data").get("tags").get(
            "pytest"
        )


if __name__ == "__main__":
    test_nlp()
    TestNLPExperimentCustomTags()
