import os
from pathlib import Path
from pycaret.loggers.mlflow_logger import MlflowLogger

try:
    import dagshub
    import mlflow
except ImportError:
    dagshub = None


class DagshubLogger(MlflowLogger):
    def __init__(self, remote=None) -> None:
        if dagshub is None:
            raise ImportError(
                "DagshubLogger requires dagshub. Install using `pip install dagshub`"
            )
        super().__init__()

        # check token exist or not:
        from dagshub.auth.tokens import _get_token_storage

        is_token_set = (
            "https://dagshub.com" in _get_token_storage()._load_cache_file().keys()
        )  # Check OAuth
        if not is_token_set:
            dagshub.login()

        # Check mlflow environment variable is set:
        is_mlflow_set = (
            os.getenv("MLFLOW_TRACKING_URI") is not None
            and os.getenv("MLFLOW_TRACKING_USERNAME") is not None
            and os.getenv("MLFLOW_TRACKING_PASSWORD") is not None
        )
        if not is_mlflow_set:
            os.environ["REPO_OWNER"] = input(
                "Please insert your repository owner name:"
            )
            os.environ["REPO_NAME"] = input(
                "Please insert your repository project name:"
            )
            dagshub.init(
                repo_name=os.getenv("REPO_NAME"), repo_owner=os.getenv("REPO_OWNER")
            )

        from dagshub.upload import Repo

        self.run = None

        self.remote = remote
        self.__dvc_fld_path = Path("artifacts")
        self.__remote_model_root = Path("models")
        self.__remote_rawdata_root = Path("data/raw")
        self.__remote_procdata_root = Path("data/process")
        owner_name = os.getenv("MLFLOW_TRACKING_URI").split(os.sep)[-2]
        repo_name = (
            os.getenv("MLFLOW_TRACKING_URI").split(os.sep)[-1].replace(".mlflow", "")
        )
        branch = "main" if os.getenv("BRANCH") is None else os.getenv("BRANCH")
        self.repo = Repo(
            owner=owner_name,
            name=repo_name,
            branch=branch,
        )
        self.dvc_fld = self.repo.directory(str(self.__dvc_fld_path))

    def init_experiment(self, exp_name_log, full_name=None):
        mlflow.set_tracking_uri(self.remote)
        super().init_experiment(exp_name_log, full_name)

    def log_artifact(self, file, type="artifact"):
        def dvc_upload(local_path="", remote_path="", commit=""):
            assert os.path.isfile(local_path), FileExistsError(
                f"Invalid file path: {local_path}"
            )
            self.dvc_fld.add(file=local_path, path=remote_path)
            self.dvc_fld.commit(commit, versioning="dvc", force=True)

        if type == "model":
            if not file.endswith("Transformation Pipeline.pkl"):
                remote_filename = os.path.join(self.__remote_model_root, file)
                dvc_upload(
                    local_path=file,
                    remote_path=remote_filename,
                    commit="update new trained model",
                )

        elif type in [
            "train_data_remote",
            "train_transform_data_remote",
            "test_data_remote",
            "test_transform_data_remote",
        ]:
            data_type = type.split("_")[0].lower()
            is_transformed = "transform" in type
            transformed = "transformed " if is_transformed else ""
            remote_dir = (
                self.__remote_procdata_root
                if is_transformed
                else self.__remote_rawdata_root
            )
            remote_filename = os.path.join(remote_dir, file.split(os.sep)[-1])
            dvc_upload(
                local_path=file,
                remote_path=remote_filename,
                commit=f"update {transformed}{data_type} data",
            )
        else:
            mlflow.log_artifact(file)
