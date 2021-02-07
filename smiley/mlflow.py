import os

import mlflow
from mlflow.tracking import MlflowClient

from .utils import Paths


class MlFlowTracker(MlflowClient):
    def __init__(self, root_dir: str = None):
        if root_dir is None:
            self.root_dir = Paths().root_dir
        else:
            self.root_dir = root_dir
        os.chdir(self.root_dir)
        super().__init__()

    @staticmethod
    def get_experiment_id(experiment_name):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        return experiment_id

    def get_best_model(self, experiment_name, metric="val_accuracy"):
        experiment_id = self.get_experiment_id(experiment_name)
        runs = self.search_runs(experiment_id, order_by=[f"metrics.{metric} DESC"])

        return runs[0]

    def get_best_model_metric(self, experiment_name, metric="val_accuracy"):
        best_model = self.get_best_model(experiment_name, metric=metric)
        best_model_dict = best_model.to_dictionary()
        return best_model_dict["data"]["metrics"].get(metric)

    def get_new_version(self, experiment_name):
        n_version = self.get_last_version(experiment_name)
        return n_version + 1

    def get_last_version(self, experiment_name):
        experiment_id = self.get_experiment_id(experiment_name)
        runs = self.search_runs(experiment_id)
        n_version = len(runs)
        return n_version