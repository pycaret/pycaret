import os
from pathlib import Path

from pycaret.loggers.mlflow_logger import MlflowLogger

try:
    import dagshub
    import mlflow
    from dagshub.upload import Repo
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
        dagshub.auth.get_token()

        # Check mlflow environment variable is set:
        if (
            len(
                {
                    "MLFLOW_TRACKING_URI",
                    "MLFLOW_TRACKING_USERNAME",
                    "MLFLOW_TRACKING_PASSWORD",
                }.difference(os.environ)
            )
            > 0
            or not remote
        ):
            prompt_in = input(
                "Please insert your repository owner_name/repo_name:"
            ).split("/")
            assert (
                len(prompt_in) == 2
            ), f"Invalid input, should be owner_name/repo_name, but get {prompt_in} instead"

            dagshub.init(repo_name=prompt_in[1], repo_owner=prompt_in[0])
            remote = os.getenv("MLFLOW_TRACKING_URI")

        self.run = None
        self.remote = remote
        self.paths = {
            "dvc_directory": Path("artifacts"),
            "models": self.paths["dvc_directory"] / "models",
            "data": self.paths["dvc_directory"] / "data",
            "raw_data": self.paths["data"] / "raw",
            "processed_data": self.paths["data"] / "processed",
        }

        self.repo = Repo(
            owner=self.remote.split(os.sep)[-2],
            name=self.remote.split(os.sep)[-1].replace(".mlflow"),
            branch=os.getenv("BRANCH", "main"),
        )
        self.dvc_folder = self.repo.directory(self.paths["dvc_directory"])
        self.__commit_data_type = []

    def init_experiment(self, **args):
        mlflow.set_tracking_uri(self.remote)
        super().init_experiment(**args)

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
                self._dvc_add(
                    local_path=file,
                    remote_path=os.path.join(self.__remote_model_root, file),
                )
                self._dvc_commit(commit="added new trained model")
        elif type == "data":
            self.__commit_data_type.append(file.split(os.sep)[-1].lower())
            remote_dir = (
                self.__remote_procdata_root
                if "transform" in self.__commit_data_type[-1]
                else self.__remote_rawdata_root
            )

            self._dvc_add(
                local_path=file,
                remote_path=os.path.join(remote_dir, self.__commit_data_type[-1]),
            )
        elif type == "data_commit":
            self._dvc_commit(
                commit="update data: " + ", ".join(self.__commit_data_type)
            )
            self.__commit_data_type = []
        else:
            mlflow.log_artifact(file)
