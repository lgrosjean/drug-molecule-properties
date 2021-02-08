"""
Core model to train our models. Two models can be trained :

 - `model1`: will train the `Model1Learner`
 - `model2`: will train the `Model2Learner`

It also creates Mlflow tracking experience to quicker check if performance has improved or not. All the tracking experiments can be found through: 

```sh
$ mlflow ui --port 54180
```

It will also convert Keras model into TfSavedModel to be integrated in Tensorflow Serving services, to request predictions. By defaults, each model will be saved in the corresponding folder in the `model/models` folder.

In particular, the model2 model has to be transform in a specific `TFModel` in order to understand raw string with shape=1 as inputs during serving.
"""
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


def train(
    model: str, experiment_name: str = None, best_metric="val_accuracy", **kwargs
):
    """Base method to train a model. Will train the model input based on `MODEL_DICT` correspondance, and define the `experiment_name` in MlFlow tracking.

    Args:
        model (str): the model to train. Only two choices: `model1` or `model2`.
        experiment_name (str, optional): The experiment name to define in MlFlow tracking server. Defaults to None. If None, will be define with `model` value.
        best_metric (str, optional): The metrics on which performing evaluation of the model, and to check if performance has improved since best last model. Defaults to "val_accuracy".
    """
    _check_input(model)

    if experiment_name is None:
        experiment_name = model

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
    """Specific class to transform `tf.keras.Model` into `tf.Module` to understand raw string input during prediction. Its two attributes will be integrated in the saving, especially the `prediction` which will be integrated in the signatures of the SavedModel."""

    def __init__(self, model: tf.keras.Model) -> None:
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(1,), dtype=tf.string)])
    def prediction(self, smile: str):
        """The base prediction model to define input_signature. Will return a dictionnary as prediction in the format: `{"prediction": prediction}`

        Args:
            smile (str): The input raw smile as string.

        Returns:
            dict: A dict for the prediction containing only the key "prediction"
        """
        return {
            "prediction": self.model(smile),
        }
