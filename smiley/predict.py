"""
This modules contains one only function `predict` to help predicting with a trained model. It takes as input :

  - `model`: the name of the model to evalute (within `model1` or `model2` choices)
  - `X_inputs`: the inputs to fill in the model to predict on. Depends on the format. For model1: array of shape (None, 2048), for model2: raw string.
  - `data_dir`: the location of the input data. By default, will look in the `$ROOT_dir/data` folder
  - `root_dir`: the location of the directory containing models, data and mlflow directories.

Example:

  - model1
    ```python
    >>> from smiley.predict import predict
    >>> model = 'model1`# To evalute model1
    >>> data_dir = './data'
    >>> root_dir = '.'
    >>> X_inputs = np.array([np.random.randint(0,2,2048)])
    >>> predict(model=model, data_dir=data_dir, root_dir=root_dir)
    O.876
    ```
  - model2
    ```python
    >>> from smiley.predict import predict
    >>> model = 'model2`# To evalute model1
    >>> data_dir = './data'
    >>> root_dir = '.'
    >>> X_inputs = ['Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C']
    >>> predict(model=model, data_dir=data_dir, root_dir=root_dir)
    O.976
    ```
"""

# Authors: Léo Grosjean <leo.grosjean@live.fr>
# License: GPL3

import tensorflow as tf

from .train import MODEL_DICT, _check_input
from .utils import Paths


def predict(model: str, X_inputs, data_dir=None, root_dir=None):
    _check_input(model)
    models_path = (Paths(root_dir=root_dir)["model"] / "models").absolute()
    learner = MODEL_DICT.get(model)(data_dir=data_dir)
    model_path = models_path / model / "0"
    new_model = tf.keras.models.load_model(model_path)
    learner.model.set_model(new_model, compile=False)
    pred = learner.predict_proba(X_inputs=X_inputs)
    return pred
