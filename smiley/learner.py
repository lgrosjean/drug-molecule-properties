import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight as _class_weight
from sklearn.metrics import f1_score
from tensorflow import keras as keras
from tqdm.keras import TqdmCallback

from .dataset import Dataset
from .model import Model1, Model2, Model3


class BaseTrainer:
    def __init__(
        self,
        dataset: Dataset = None,
        smile_col="smiles",
        data_dir=None,
        params=None,
        load=True,
        **kwargs,
    ):
        self.X = None
        self.y = None
        self.smile_col = smile_col
        self.model = None
        self.epochs = 10
        self.batch_size = 64
        self.verbose = 0
        self.shuffle = True
        self.test_size = 0.2
        self.val_size = 0.2
        self.class_weight_dict = None
        self.weighted = True
        self.metrics = []
        self.data_dir = data_dir
        self.random_state = 54
        self.X = None
        self.y = None
        self.callbacks = []
        self.set_dataset(dataset)
        self.set_params(params)
        self.set_params(kwargs)

    def is_fitted(self):
        return (not self.X is None) and (not self.y is None)

    def set_random_state(self, random_state: int = None):
        if not random_state is None:
            self.random_state = random_state

    def set_dataset(self, dataset: Dataset):
        if dataset:
            self.dataset = dataset
        else:
            logging.info("Importing dataset.")
            self.dataset = Dataset(self.data_dir)

    def set_params(self, params: dict):
        if not params is None:
            for param in params:
                value = params.get(param)
                setattr(self, param, value)
                logging.debug(f"Parameter {param} set to {value}.")

    def get_params(self):
        params = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "weighted": self.weighted,
        }
        return params

    def set_model(self, model, **kwargs):
        self.set_params(kwargs)
        self.model = model(input_shape=self.input_shape, **kwargs)
        self.metrics = self.model.metrics

    def reset_model(self):
        self.model = None

    def set_X_y(self, X_col, y_col):
        self.X = self.dataset.data.get(X_col)
        self.y = self.dataset.data.get(y_col)

    def set_input_shape(self, input_shape: int):
        self.input_shape = input_shape


class OneColLearner(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_col = "P1"
        self.input_col = None

    def prepare(self, weighted: bool = None, random_state=None):
        self.set_random_state(random_state=random_state)

        self.set_X_y(self.input_col, self.output_col)

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
        self.set_class_weight(weighted=weighted)

    def set_class_weight(self, weighted: bool = None):
        if weighted is not None:
            self.weighted = weighted
        if self.weighted:
            self.class_weights = _class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(np.array(self.y_train)),
                y=np.array(self.y_train),
            )
            self.class_weight_dict = dict(enumerate(self.class_weights))
        else:
            self.class_weights = None
            self.class_weight_dict = None

    def train(
        self,
        X_train=None,
        y_train=None,
        validation_data=None,
        early_stopping: int = None,
        tqdm=True,
        **params,
    ):
        if X_train is not None:
            self.X_train = X_train
        if y_train is not None:
            self.y_train = y_train
        if validation_data is not None:
            self.validation_data = validation_data

        if early_stopping:
            self.early_stopping_ = early_stopping
            es = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.early_stopping
            )
            self.callbacks.append(es)
        if tqdm:
            tqdm_nb = TqdmCallback()
            self.callbacks.append(tqdm_nb)

        self.set_params(params)
        logging.info("Start training")
        history = self.model.model.fit(
            x=self.X_train.values,
            y=self.y_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            epochs=self.epochs,
            verbose=self.verbose,
            validation_data=self.validation_data,
            class_weight=self.class_weight_dict,
            callbacks=self.callbacks,
        )
        self.history = history.history

    def evaluate(self, function=f1_score, X_test=None, y_test=None, *args, **kwargs):
        if X_test is not None:
            self.X_test = X_test
        if y_test is not None:
            self.y_test = y_test

        y_pred = self.model.model.predict(self.X_test)
        y_pred = y_pred > 0.5

        return function(self.y_test, y_pred, *args, **kwargs)

    def predict_proba(self, X_inputs):
        return self.model.model.predict(X_inputs)

    def predict(self, X_inputs, thresh: int = 0.5):
        pred = self.predict_proba(X_inputs)
        pred = pred > thresh
        return pred


class Model1Learner(OneColLearner):
    def __init__(self, fit=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Model1_Learner"
        self.input_col = [i for i in range(0, 2048)]
        self.set_input_shape(len(self.input_col))
        self.set_model(Model1)
        self.fit()
        self.prepare()

    def fit(self):
        self.dataset.transform_smile()


class Model2Learner(OneColLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Model2_Learner"
        self.input_col = self.smile_col
        self.set_input_shape(len(self.input_col))
        self.set_model(Model2)
        self.prepare()