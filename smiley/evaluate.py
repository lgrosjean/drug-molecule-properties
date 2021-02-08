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
