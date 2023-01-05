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
    def __init__(self, remote=None, repo=None) -> None:
        super().__init__()
        if dagshub is None:
            raise ImportError(
                "DagshubLogger requires dagshub. Install using `pip install dagshub`"
            )

        self.run = None
        self.remote = remote
        self.paths = {
            "dvc_directory": Path("artifacts"),
            "models": Path("models"),
            "raw_data": Path("data") / "raw",
            "processed_data": Path("data") / "processed",
        }

        if repo:
            self.repo_name, self.repo_owner = self.splitter(repo)
        else:
            self.repo_name, self.repo_owner = None, None

        self.__commit_data_type = []

    @staticmethod
    def splitter(repo):
        splitted = repo.split("/")
        if len(splitted) != 2:
            raise ValueError(
                f"Invalid input, should be owner_name/repo_name, but got {repo} instead"
            )
        return splitted[1], splitted[0]

    def init_experiment(self, *args, **kargs):
        # check token exist or not:
        token = dagshub.auth.get_token()
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        # Check mlflow environment variable is set:
        if not self.repo_name or not self.repo_owner:
            self.repo_name, self.repo_owner = self.splitter(
                input("Please insert your repository owner_name/repo_name:")
            )

        if not self.remote or "dagshub" not in os.getenv("MLFLOW_TRACKING_URI"):
            dagshub.init(repo_name=self.repo_name, repo_owner=self.repo_owner)
            self.remote = os.getenv("MLFLOW_TRACKING_URI")

        self.repo = Repo(
            owner=self.remote.split(os.sep)[-2],
            name=self.remote.split(os.sep)[-1].replace(".mlflow", ""),
            branch=os.getenv("BRANCH", "main"),
        )
        self.dvc_folder = self.repo.directory(str(self.paths["dvc_directory"]))

        mlflow.set_tracking_uri(self.remote)
        super().init_experiment(*args, **kargs)

    def _dvc_add(self, local_path="", remote_path=""):
        if not os.path.isfile(local_path):
            FileExistsError(f"Invalid file path: {local_path}")
        self.dvc_folder.add(file=local_path, path=remote_path)

    def _dvc_commit(self, commit=""):
        self.dvc_folder.commit(commit, versioning="dvc", force=True)

    def log_artifact(self, file, type="artifact"):
        if type == "model":
            if not file.endswith("Transformation Pipeline.pkl"):
                self._dvc_add(
                    local_path=file,
                    remote_path=os.path.join(self.paths["models"], file),
                )
                self._dvc_commit(commit="added new trained model")
        elif type == "data":
            self.__commit_data_type.append(file.split(os.sep)[-1].lower())
            remote_dir = (
                self.paths["processed_data"]
                if "transform" in self.__commit_data_type[-1]
                else self.paths["raw_data"]
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
