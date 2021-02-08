"""
This modules contains one only function `evaluate` to help evaluating trained model. It takes as input :

  - `model`: the name of the model to evalute (within `model1` or `model2` choices)
  - `function`: the metric function to use during evaluation. The default metric is `sklearn.metrics.f1_score`.
  - `data_dir`: the location of the input data. By default, will look in the `$ROOT_dir/data` folder
  - `root_dir`: the location of the directory containing models, data and mlflow directories.

Example:
```python
>>> from smiley.evaluate import evaluate
>>> model = 'model1`# To evalute model1
>>> data_dir = './data'
>>> root_dir = '.'
>>> evalute(model=model, data_dir=data_dir, root_dir=root_dir)
O.875
```
"""

# Authors: Léo Grosjean <leo.grosjean@live.fr>
# License: GPL3

import tensorflow as tf

from .learner import f1_score
from .train import MODEL_DICT, _check_input
from .utils import Paths


def evaluate(
    model: str,
    function=f1_score,
    X_test=None,
    y_test=None,
    data_dir=None,
    root_dir=None,
):
    _check_input(model)
    models_path = Paths(root_dir=root_dir)["model"] / "models"
    learner = MODEL_DICT.get(model)(data_dir=data_dir, fit=False)
    model_path = (models_path / model / "0").absolute()
    new_model = tf.keras.models.load_model(model_path)
    learner.model.set_model(new_model, compile=False)
    if X_test is not None and y_test is not None:
        evaluation = learner.evaluate(X_test=X_test, y_test=y_test, function=function)
    else:
        if model == "model1":
            learner.fit()
        evaluation = learner.evaluate(function=function)
    return evaluation
