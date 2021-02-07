import os
import logging
import time

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.keras
import tensorflow as tf

from .utils import InputError, Paths
from .learner import Model1Learner, Model2Learner
from .mlflow import MlFlowTracker

MODEL_DICT = {"model1": Model1Learner, "model2": Model2Learner}
"""
Root variable to map input model name to the corresponding Learner: 

 - `model1`: `Model1Learner`
 - `model2`: `Model2Learner`
"""


def _check_input(model: str):
    """Check if inputs are part of MODEL_DICT keys.

    Args:
        model (str): name of the model to train

    Raises:
        InputError: Raise Error if not present in keys.
    """
    if not model in MODEL_DICT:
        raise InputError(
            f"The input model {model} is not existing (possible: {list(MODEL_DICT.keys())}."
        )


def train(model, experiment_name, best_metric="val_accuracy", **kwargs):
    _check_input(model)

    owd = os.getcwd()
    root_dir = Paths().root_dir
    os.chdir(root_dir)

    mlflow.set_experiment(experiment_name)
    tracker = MlFlowTracker()

    timestamp = time.strftime("%Y%m%d%H%M")
    run_name = f"{experiment_name}_{timestamp}"

    learner = MODEL_DICT.get(model)()
    print(learner.name)

    version = tracker.get_new_version(experiment_name)
    logging.info(version)

    with mlflow.start_run(run_name=run_name):
        run_uuid = mlflow.active_run().info.run_uuid
        logging.info(f"MLflow Run ID: {run_uuid}")

        learner.train(**kwargs)

        # Get training params
        params = learner.get_params()

        # Log parameters
        mlflow.log_params(params)

        # calculate metrics
        metrics = {}
        for metric in learner.metrics:
            metrics[metric] = learner.history[metric][-1]
            metrics[f"val_{metric}"] = learner.history[f"val_{metric}"][-1]
        metrics["loss"] = learner.history["loss"][-1]
        metrics["val_loss"] = learner.history["val_loss"][-1]

        final_metric = metrics.get(best_metric)

        # log metrics
        mlflow.log_metrics(metrics)

        # log model
        model_name = learner.model.name
        X_train = learner.X_train
        y_pred = learner.predict(X_train)
        signature = infer_signature(X_train, y_pred)
        mlflow.keras.log_model(
            learner.model.model, model_name, signature=signature, save_format="tf"
        )

    models_path = Paths().model / "models"
    if not models_path.exists():
        models_path.mkdir()

    final_metric_best = tracker.get_best_model_metric(
        experiment_name, metric=best_metric
    )

    if final_metric >= final_metric_best:
        logging.info(
            "Best model found. Saving to model dir to use with Tensorflow Serving"
        )
        model_path = os.path.join(str(models_path), model)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            logging.info(f"Folder ")
        if model == "model2":
            tfmodel = TFModel(learner.model.model)
            tf.saved_model.save(
                tfmodel.model,
                os.path.join(model_path, "0"),
                signatures={"serving_default": tfmodel.prediction},
            )
            print(tfmodel)
        else:
            learner.model.model.save(os.path.join(model_path, "0"))
        logging.info(f"Model exported at {model_path}.")
    else:
        logging.info(
            f"Model logged but best performance not improved for experiment {experiment_name} (current version: {version})."
        )

    os.chdir(owd)


class TFModel(tf.Module):
    def __init__(self, model: tf.keras.Model) -> None:
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(1,), dtype=tf.string)])
    def prediction(self, review: str):
        return {
            "prediction": self.model(review),
        }
