"""
This module contains all the Learner to prepare data, prepare model et start training. There is a root learner: `BaseTrainer` and two childrens:

- `Model1Learner` whose aim is to deal with the model 1 problem
- `Model2Learner` whose aim is to deal with the model 2 problem
"""
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
    """
    Base trainer to define inputs, hyperparameters and internal methods
    """

    def __init__(
        self,
        dataset: Dataset = None,
        smile_col="smiles",
        data_dir: str = None,
        params: dict = None,
        **kwargs,
    ):
        """Initialization of Base Trainer

        Args:
            dataset (Dataset, optional): Input dataset for trainer. Defaults to None. If None, Dataset will be generated with default arguments.
            smile_col (str, optional): The column containing the smiles attributes. Defaults to "smiles".
            data_dir (str, optional): Location of the data for training. Defaults to None.
            params (dict, optional): List of hyperparameters to fit to dataset or model. Defaults to None.
        """
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

    def is_fitted(self) -> bool:
        """Simple method to check if the attributes X & y exists, and so the dataset has been fitted

        Returns:
            bool: The learner is fitted
        """
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

    def get_params(self) -> dict:
        """Get the hyperparameters of the training

        Returns:
            dict: the hyperparamters dictionnary.
        """
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
    """Parent class to define more precisely a Learner for a one-column target. Inherits from Base Learner."""

    def __init__(self, output_col: str = "P1", input_col: str = None, *args, **kwargs):
        """Initialize One-column learner with inherited attributes.

        Args:
            output_col (str, optional): The output column for training. Defaults to "P1".
            input_col (str, optional): The input column for training. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.output_col = output_col
        self.input_col = input_col

    def prepare(self, weighted: bool = None, random_state: int = None):
        """Prepare the Learner and the inputs for learning. Transform the X,y attributes to X_train, y_train, X_test, y_test, X_val, y_val. Also create weighted dictionnary if requested to weigth training if the dataset is imbalanced.

        Args:
            weighted (bool, optional): [description]. Defaults to None.
            random_state (int, optional): Random state for split and shuffling. Defaults to None.
        """
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
        """Base method to train input model. Based on Keras hyperparameters

        Args:
            X_train ([pd.DataFrame, np.array], optional): the Keras `x` input. Defaults to None. If None, the learner will take the attribute `X_train` fitted in the class.
            y_train ([pd.DataFrame, np.array], optional): the Keras `y` input. Defaults to None. If None, the learner will take the attribute `y_train` fitted in the class.
            validation_data ([pd.DataFrame, np.array], optional): the Keras `validation_data` input. Defaults to None. If None, the learner will take the attribute `validation_data` fitted in the class.
            early_stopping (int, optional): Number of epochs before Early Stopping. Defaults to None. If None, the training will last `epochs`.
            tqdm (bool, optional): To create `tqdm` progress bar during training or not. Defaults to True.
        """
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
        """Evaluate learner base on sklearen function, on `X_test` and `y_test`

        Args:
            function (functuin, optional): the metric function. From `sklearn.metrics`. Defaults to f1_score.
            X_test ([pd.DataFrame, np.array], optional): The data on which evaluate learner. Defaults to None. If `None`, will take the default `X_test`.
            y_test ([pd.DataFrame, np.array], optional): The target on which evaluate the model. Defaults to None. If `None`, will take the default `y_test`.

        Returns:
            [float, array of float]: the metrics inputs.
        """
        if X_test is not None:
            self.X_test = X_test
        if y_test is not None:
            self.y_test = y_test

        y_pred = self.model.model.predict(self.X_test)
        y_pred = y_pred > 0.5

        return function(self.y_test, y_pred, *args, **kwargs)

    def predict_proba(self, X_inputs):
        """Prediction on `X_inputs` with raw outputs (between 0 & 1)

        Args:
            X_inputs ([pd.DataFrame, np.array, list]): the inputs on which make the prediction

        Returns:
            [np.array, list]: List of predictions
        """
        return self.model.model.predict(X_inputs)

    def predict(self, X_inputs, thresh: int = 0.5):
        """Prediction on `X_inputs` with predicted class, based on input `thresh`.

        Args:
            X_inputs ([pd.DataFrame, np.array, list]): the inputs on which make the prediction
            thresh (int, optional): The threshold between class 0 & 1. Defaults to 0.5.

        Returns:
            [type]: [description]
        """
        pred = self.predict_proba(X_inputs)
        pred = pred > thresh
        return pred


class Model1Learner(OneColLearner):
    """Model1Learner to use to solve the model 1 problem with converted smiles as input."""

    def __init__(self, *args, **kwargs):
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
    """Model2Learner to use to solve the model 2 problem with the raw smile string as input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Model2_Learner"
        self.input_col = self.smile_col
        self.set_input_shape(len(self.input_col))
        self.set_model(Model2)
        self.prepare()