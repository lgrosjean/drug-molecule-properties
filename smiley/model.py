"""This module contains all the classes defining the Keras model to solve the two problems: Model 1 and Model 2.

It defines a base class: `BaseModel` to instanciate default arguments and methods.

The most important parts are defined in the two mains classes: 

 - `Model1`
 - `Model2`
"""

# Authors: Léo Grosjean <leo.grosjean@live.fr>
# License: GPL3

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Dense,
    ReLU,
    Embedding,
    Dropout,
    Conv1D,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.layers.experimental.preprocessing import (
    Normalization,
    TextVectorization,
)


class ModelCreationError(Exception):
    """An exception raised when a model is not created."""


class BaseModel:
    def __init__(
        self,
        input_shape: int = None,
        name: str = "model",
        optimizer: str = "adam",
        loss: str = "binary_crossentropy",
        metrics: list = ["accuracy"],
        weighted_metrics: list = None,
        model: Model = None,
    ):
        """Generate base model for learning.

        Args:
            input_shape (int): input shape for Keras model.
            name (str, optional): Name for Keras model. Defaults to "model".
            optimizer (str, optional): Optimizer of Keras model. Defaults to "adam".
            loss (str, optional): Loss for Keras model. Defaults to 'binary_crossentropy'.
            metrics (list, optional): List of metrics for Keras model training. Defaults to ["accuracy"].
            weighted_metrics (list, optional): List of weighted metrics to define for Keras model compilation. Defaults to None.
            model (Model, optional): The Keras model. Defaults to None.
        """
        self.input_shape = input_shape
        self.name = name
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.weighted_metrics = weighted_metrics
        self.model = model
        self.create()
        self.compile()

    def set_params(self, params: dict = None):
        if params is not None:
            for param in params:
                setattr(self, param, params.get(param))

    def set_model(self, model, compile=True, compile_params: dict = None):
        self.model = model
        self.compile(compile_params)

    def compile(self, params=None):
        self.set_params()
        if self.model is None:
            raise ModelCreationError("Model is not fitted yet")
        else:
            self.model.compile(
                loss=self.loss,
                optimizer=self.optimizer,
                metrics=self.metrics,
                weighted_metrics=self.weighted_metrics,
            )

    def create(self):
        pass


class Model1(BaseModel):
    """Model class which defines the core Keras model for the problem 1. The model is the following:

    ```python
    inputs = Input(shape=(self.input_shape,))
    norm = Normalization()(inputs)
    dense_1 = Dense(32, activation="relu")(norm)
    relu_1 = ReLU()(dense_1)
    dense_2 = Dense(32, activation="relu")(relu_1)
    outputs = Dense(1, activation="sigmoid")(dense_2)

    self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = "model_1"

    def create(self, params=None, **kwargs):
        self.set_params(params)
        self.set_params(kwargs)
        inputs = Input(shape=(self.input_shape,))
        norm = Normalization()(inputs)
        dense_1 = Dense(32, activation="relu")(norm)
        relu_1 = ReLU()(dense_1)
        dense_2 = Dense(32, activation="relu")(relu_1)
        outputs = Dense(1, activation="sigmoid")(dense_2)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)


class Model2(BaseModel):
    """Model class which defines the core Keras model for the problem 2. The model is the following:

    ```python
    inputs = Input(shape=(1,), dtype=tf.string, name="inputs")
    x = TextVectorization(
        max_tokens=5000,
        ngrams=self.ngrams,
        output_sequence_length=self.vocab_size,
        name="text_vectorization",
    )(inputs)
    x = Embedding(
        input_dim=self.vocab_size,
        output_dim=self.emb_output_dim,
        input_length=self.emb_input_length,
        name="embedding",
    )(x)
    x = Dropout(0.3, name="dropout_1")(x)
    x = Conv1D(filters=32, kernel_size=5, activation="relu", name="conv1d")(x)
    x = MaxPooling1D(pool_size=2, name="max_pooling")(x)
    x = LSTM(self.lstm_cell, name="lstm")(x)

    output = Dense(1, activation="sigmoid", name="output_P1")(x)
    self.model = Model(inputs=inputs, outputs=output)
    ```
    """

    def __init__(
        self,
        embedding_vector_length: int = 32,
        emb_output_dim: int = 64,
        lstm_cell: int = 32,
        emb_input_length=72,
        ngrams=3,
        *args,
        **kwargs
    ):
        self.model_type = "model_2"
        self.embedding_vecor_length = embedding_vector_length
        self.vocab_size = 128
        self.ngrams = ngrams
        self.emb_output_dim = emb_output_dim
        self.lstm_cell = lstm_cell
        self.emb_input_length = emb_input_length
        super().__init__(*args, **kwargs)

    def create(self, params=None, **kwargs):
        self.set_params(params)
        self.set_params(kwargs)
        inputs = Input(shape=(1,), dtype=tf.string, name="inputs")
        x = TextVectorization(
            max_tokens=5000,
            ngrams=self.ngrams,
            output_sequence_length=self.vocab_size,
            name="text_vectorization",
        )(inputs)
        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.emb_output_dim,
            input_length=self.emb_input_length,
            name="embedding",
        )(x)
        x = Dropout(0.3, name="dropout_1")(x)
        x = Conv1D(filters=32, kernel_size=5, activation="relu", name="conv1d")(x)
        x = MaxPooling1D(pool_size=2, name="max_pooling")(x)
        x = LSTM(self.lstm_cell, name="lstm")(x)

        output = Dense(1, activation="sigmoid", name="output_P1")(x)
        self.model = Model(inputs=inputs, outputs=output)


class Model3(BaseModel):
    def __init__(self, *args, **kwargs):
        super(*args, **kwargs).__init__()
        self.model_type = "model_3"


# embedding_vecor_length = 32
# vocab_size = 128 # number of ASCII chars
# output_dim = 64

# inputs = Input(shape=(maxlen,), dtype='int32', name='inputs')
# x = Embedding(
#     input_dim=vocab_size,
#     output_dim=output_dim,
#     input_length=maxlen,
#     name='embedding'
# )(inputs)
# x = Dropout(0.3, name='dropout_1')(x)
# x = Conv1D(filters=32, kernel_size=5, activation='relu', name='conv1d')(x)
# x = MaxPooling1D(pool_size=2, name="max_pooling")(x)
# x = LSTM(32, name='lstm')(x)

# output_array = []
# metrics_array = {}
# loss_array = {}

# for target_col in target_cols:
#     name = f'binary_output_{target_col}'
#     # A Dense Layer is created for each output
#     binary_output = Dense(1, activation='sigmoid', name=name)(x)
#     output_array.append(binary_output)
#     metrics_array[name] = 'binary_accuracy'
#     loss_array[name] = 'binary_crossentropy'

#     model = Model(inputs=inputs, outputs=output_array)

#     model.compile(optimizer='adam',
#               loss=loss_array,
#               metrics=metrics_array)

#               history = model.fit(
#     X_train,
#     y_train_output,
#     epochs=20,
#     validation_split=0.1,
#     batch_size=32
#     ,verbose=0)

#     y_train_output = []
# y_test_output = []
# for col in target_cols:
#     y_train_output.append(y_train[col])
#     y_test_output.append(y_test[col])