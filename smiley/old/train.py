import os
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight as _class_weight

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tqdm.keras import TqdmCallback

import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

from .dataset import Dataset
from .utils import Paths


def scheduler(epoch, lr, coef=0.1):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-coef)


class SingleModel:
    def __init__(
        self,
        input_shape: int,
        name: str = "single_model",
        optimizer: str = "adam",
        loss="binary_crossentropy",
        metrics: list = ["accuracy"],
        weighted_metrics: list = None,
    ):
        self.input_shape = input_shape
        self.name = name
        self.optimizer = optimizer
        self.metrics = metrics
        self.weighted_metrics = weighted_metrics
        self.model = None
        self.loss = loss
        self.create()
        self.compile()

    def set_params(self, params: dict = None):
        if params is not None:
            for param in params:
                setattr(self, param, params.get(param))

    def create(self, params=None):
        self.set_params(params)
        inputs = Input(shape=(self.input_shape,))
        norm = Normalization()(inputs)
        dense_1 = Dense(32, activation="relu")(norm)
        relu_1 = ReLU()(dense_1)
        dense_2 = Dense(32, activation="relu")(relu_1)
        outputs = Dense(1, activation="sigmoid")(dense_2)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)

    def set_model(self, model, compile=True, compile_params: dict = None):
        self.model = model
        self.compile(compile_params)

    def compile(self, params=None):
        self.set_params(params)
        if self.model is None:
            self.create(params)
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics,
            weighted_metrics=self.weighted_metrics,
        )


class SingleTrainer:
    def __init__(
        self, dataset=None, data_dir=None, params=None, load=True, transform=True
    ):
        self.X = None
        self.y = None
        self.model = None
        self.epochs = 10
        self.batch_size = 64
        self.verbose = 0
        self.shuffle = True
        self.test_size = 0.2
        self.val_size = 0.2
        self.class_weight_dict = None
        self.weighted = True
        self.random_state = 54
        self.metrics = []
        if dataset:
            self.dataset = dataset
        else:
            logging.info("Setting default Dataset to Trainer")
            self.dataset = Dataset(data_dir=data_dir, mode="single", target_col="P1")
        self.data_dir = self.dataset.data_dir
        self.scheduler = scheduler
        if params:
            self.set_params(params)
        if load:
            self.dataset.load()
        if not self.dataset.is_transformed() and transform:
            self.X, self.y = self.dataset.transform()
        elif self.dataset.is_transformed():
            logging.info("Dataset already transformed.")
            self.X, self.y = self.dataset.X_, self.dataset.y_
        else:
            raise AttributeError("Please transform your Dataset b efore")

    def get_params(self):
        params = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "weighted": self.weighted,
            "optimizer": self.model.optimizer,
            "loss": self.model.loss,
            "random_state": self.random_state,
        }
        return params

    def set_params(self, params: dict):
        for param in params:
            setattr(self, param, params.get(param))
            logging.debug(f"Parameters {param} set to {params.get(param)}.")

    def set_model(self, *args, **kwargs):
        single_model = SingleModel(*args, **kwargs)
        self.model = single_model
        self.metrics = self.model.metrics

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def prepare(self, class_weight=True, random_state=54):
        self.random_state = random_state
        if self.X is None and self.y is None:
            if not self.dataset.is_loaded():
                self.dataset.load()
            self.dataset.transform()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            stratify=self.y,
            random_state=self.random_state,
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=self.val_size,
            stratify=self.y_train,
            random_state=self.random_state,
        )
        self.validation_data = (self.X_val, self.y_val)
        self.input_shape = self.dataset.input_shape
        if class_weight:
            self.weighted = True
            self.class_weights = _class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(np.array(self.y_train)),
                y=np.array(self.y_train),
            )
            self.class_weight_dict = dict(enumerate(self.class_weights))

    def reset_model(self):
        self.model = None

    def reset_data(self):
        self.prepare()

    def fit(
        self,
        X_train=None,
        y_train=None,
        validation_data=None,
        early_stopping: int = None,
        tqdm=True,
        scheduler=True,
        plot=True,
        **params,
    ):

        if X_train is None and y_train is None:
            self.prepare()

        self.callbacks_ = []
        if early_stopping:
            self.early_stopping = early_stopping
            es = early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.early_stopping
            )
            self.callbacks_.append(es)
        if tqdm:
            tqdm_nb = TqdmCallback()
            self.callbacks_.append(tqdm_nb)
        if scheduler:
            lr_scheduler = keras.callbacks.LearningRateScheduler(self.scheduler)
            self.callbacks_.append(lr_scheduler)

        if X_train is not None:
            self.X_train = X_train
        if y_train is not None:
            self.y_train = y_train
        if validation_data is not None:
            self.validation_data = validation_data

        if self.model is None:
            self.input_shape = self.X_train.shape[1]
            self.set_model(input_shape=self.input_shape)

        self.set_params(params)
        print(self.get_params())

        logging.info("Start training")

        history = self.model.model.fit(
            x=self.X_train,
            y=self.y_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            epochs=self.epochs,
            verbose=self.verbose,
            validation_data=self.validation_data,
            class_weight=self.class_weight_dict,
            callbacks=self.callbacks_,
        )

        self.history = history.history

        if plot:
            self.plot_history()

    def plot_history(self):
        for metric in self.metrics:
            plt.plot(self.history[metric])
            plt.plot(self.history[f"val_{metric}"])
            plt.title(f"model {metric}")
            plt.ylabel(f"{metric}")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            plt.show()
        # summarize history for loss
        plt.plot(self.history["loss"])
        plt.plot(self.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

    def evaluate(self, function, X_test=None, y_test=None, *args, **kwargs):
        if self.model is None:
            raise AttributeError("Model is not fitted.")

        if X_test is not None:
            self.X_test = X_test
        if y_test is not None:
            self.y_test = y_test

        y_pred = self.model.model.predict(self.X_test)
        y_pred = y_pred > 0.5

        return function(self.y_test, y_pred, *args, **kwargs)


def train(trainer, experiment_name, version="1", *args, **kwargs):

    owd = os.getcwd()
    os.chdir(Paths().root_dir)

    mlflow.set_experiment(experiment_name)
    timestamp = time.strftime("%Y%m%d%H%M")
    run_name = f"{experiment_name}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        run_uuid = mlflow.active_run().info.run_uuid
        logging.info(f"MLflow Run ID: {run_uuid}")

        trainer.fit(*args, **kwargs)

        # Get training params
        params = trainer.get_params()

        # Log parameters
        mlflow.log_params(params)

        # calculate metrics
        metrics = {}
        for metric in trainer.metrics:
            metrics[metric] = trainer.history[metric][-1]
            metrics[f"val_{metric}"] = trainer.history[f"val_{metric}"][-1]
        metrics["loss"] = trainer.history["loss"][-1]
        metrics["val_loss"] = trainer.history["val_loss"][-1]

        # log metrics
        mlflow.log_metrics(metrics)

        # log model
        model = trainer.model.model
        model_name = trainer.model.name
        X_train = trainer.X_train
        y_pred = trainer.model.model.predict(X_train)
        signature = infer_signature(X_train, y_pred)
        mlflow.keras.log_model(model, model_name, signature=signature)
        models_path = Paths().model / "models"
        if not models_path.exists():
            models_path.mkdir()
        model_path = models_path / model_name / version
        model.save(model_path)
        logging.info(f"Model exported at {model_path}.")

    os.chdir(owd)
