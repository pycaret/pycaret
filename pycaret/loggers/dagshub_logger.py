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
        self.remote_model_root = Path("artifacts/models")
        self.remote_rawdata_root = Path("artifacts/data/raw")
        self.remote_procdata_root = Path("artifacts/data/process")
        owner_name = os.getenv("REPO_OWNER")
        repo_name = os.getenv("REPO_NAME")
        branch = "main" if os.getenv("BRANCH") is None else os.getenv("BRANCH")
        self.repo = Repo(
            owner=owner_name,
            name=repo_name,
            branch=branch,
        )

    def init_experiment(self, exp_name_log, full_name=None):
        mlflow.set_tracking_uri(self.remote)
        super().init_experiment(exp_name_log, full_name)

    def log_artifact(self, file, type="artifact"):
        if type == "model":
            if not file.endswith("Transformation Pipeline.pkl"):
                remote_filename = os.path.join(self.remote_model_root, file)
                self.repo.upload(
                    file=file,
                    path=remote_filename,
                    versioning="dvc",
                    commit_message="update new trained model",
                    force=True,
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
                self.remote_procdata_root
                if is_transformed
                else self.remote_rawdata_root
            )
            remote_filename = os.path.join(remote_dir, file.split(os.sep)[-1])
            self.repo.upload(
                file=file,
                path=remote_filename,
                versioning="dvc",
                commit_message=f"update {transformed}{data_type} data",
                force=True,
            )
        else:
            mlflow.log_artifact(file)
