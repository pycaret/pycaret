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
        from dagshub.auth.tokens import TokenStorage, _get_token_storage

        token_dict = _get_token_storage()._load_cache_file()
        is_token_set = not TokenStorage._is_expired(token_dict)  # Check OAuth
        if not is_token_set:
            dagshub.auth.get_token()

        # Check mlflow environment variable is set:
        is_mlflow_set = (
            os.getenv("MLFLOW_TRACKING_URI") is not None
            and os.getenv("MLFLOW_TRACKING_USERNAME") is not None
            and os.getenv("MLFLOW_TRACKING_PASSWORD") is not None
        )

        if not is_mlflow_set or remote is None:
            prompt_in = input(
                "Please insert your repository owner_name/repo_name:"
            ).split("/")
            assert (
                len(prompt_in) == 2
            ), f"Invalid input, should be owner_name/repo_name, but get {prompt_in} instead"

            dagshub.init(repo_name=prompt_in[0], repo_owner=prompt_in[1])
            remote = os.getenv("MLFLOW_TRACKING_URI")

        from dagshub.upload import Repo

        self.run = None

        self.remote = remote
        self.__dvc_folder_path = Path("artifacts")
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
        self.dvc_folder = self.repo.directory(str(self.__dvc_folder_path))
        self.__commit_data_type = []

    def init_experiment(self, exp_name_log, full_name=None):
        mlflow.set_tracking_uri(self.remote)
        super().init_experiment(exp_name_log, full_name)

    def _dvc_add(self, local_path="", remote_path=""):
        assert os.path.isfile(local_path), FileExistsError(
            f"Invalid file path: {local_path}"
        )
        self.dvc_folder.add(file=local_path, path=remote_path)

    def _dvc_commit(self, commit=""):
        self.dvc_folder.commit(commit, versioning="dvc", force=True)

    def log_artifact(self, file, type="artifact"):
        if type == "model":
            if not file.endswith("Transformation Pipeline.pkl"):
                remote_filename = os.path.join(self.__remote_model_root, file)
                self._dvc_add(
                    local_path=file,
                    remote_path=remote_filename,
                )
                self._dvc_commit(commit="update new trained model")
        elif type == "data":
            self.__commit_data_type.append(file.split(os.sep)[-1].lower())
            is_transformed = "transform" in self.__commit_data_type[-1]
            remote_dir = (
                self.__remote_procdata_root
                if is_transformed
                else self.__remote_rawdata_root
            )

            remote_filename = os.path.join(remote_dir, self.__commit_data_type[-1])
            self._dvc_add(local_path=file, remote_path=remote_filename)
        elif type == "data_commit":
            commit_msg = "update data: " + ", ".join(self.__commit_data_type)
            self._dvc_commit(commit=commit_msg)
            self.__commit_data_type = []
        else:
            mlflow.log_artifact(file)
