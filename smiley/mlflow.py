"""This module contains one only class to deal with MlFlow environment and tracking server: `MlFlowTracker`. It inherits from base `mlflow.tracking.MlflowClient` but is augmented with few in-class methods:

 - `get_experiment_id`: to get the `experiment_id`based on the `experiment_name` provided
 - `get_best_model`: to find the best model depending on the provided `metrics` for the provided `experiment_name`
 - `gest_best_model_metrics`: to find the raw metrics for the best model found with previous arguments
 - `get_last_version`: to find the number of versions already trained for the provided `experiment_name`
 - `get_new_version`: to find the new number of version for the in-progress training.
"""

# Authors: Léo Grosjean <leo.grosjean@live.fr>
# License: GPL3

import os

import mlflow
from mlflow.tracking import MlflowClient

from .utils import Paths


class MlFlowTracker(MlflowClient):
    """
    Base class to deal with MlFlow environment.
    """

    def __init__(self, root_dir: str = None):
        """Initialize class based on `root_dir` input. This argument is essential to tell the class where to find the `mlruns` default folder to look for previous experiments.

        Args:
            root_dir (str, optional): The folder containing the `mlruns` folder. Defaults to None. If None, set to default root folder.
        """
        if root_dir is None:
            self.root_dir = Paths().root_dir
        else:
            self.root_dir = root_dir
        os.chdir(self.root_dir)
        super().__init__()

    @staticmethod
    def get_experiment_id(experiment_name: str) -> str:
        """Get the experiment_id of an experimentation based on its `experiment_name`

        Args:
            experiment_name ([str]): The experiment_name to look for its id

        Returns:
            [str]: The experiment_id
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        return experiment_id

    def get_best_model(self, experiment_name: str, metric: str = "val_accuracy"):
        experiment_id = self.get_experiment_id(experiment_name)
        runs = self.search_runs(experiment_id, order_by=[f"metrics.{metric} DESC"])

        return runs[0]

    def get_best_model_metric(
        self, experiment_name: str, metric: str = "val_accuracy"
    ) -> float:
        best_model = self.get_best_model(experiment_name, metric=metric)
        best_model_dict = best_model.to_dictionary()
        return best_model_dict["data"]["metrics"].get(metric)

    def get_new_version(self, experiment_name: str) -> int:
        n_version = self.get_last_version(experiment_name)
        return n_version + 1

    def get_last_version(self, experiment_name: str) -> int:
        experiment_id = self.get_experiment_id(experiment_name)
        runs = self.search_runs(experiment_id)
        n_version = len(runs)
        return n_version