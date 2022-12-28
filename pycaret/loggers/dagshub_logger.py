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
        from dagshub.upload import Repo

        self.run = None

        self.remote = remote
        self.__dvc_fld_path = Path("artifacts")
        self.__remote_model_root = Path("models")
        self.__remote_rawdata_root = Path("data/raw")
        self.__remote_procdata_root = Path("data/process")
        owner_name = os.getenv("REPO_OWNER")
        repo_name = os.getenv("REPO_NAME")
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
