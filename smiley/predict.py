import tensorflow as tf

from .train import MODEL_DICT, _check_input
from .utils import Paths


def predict(model: str, X_inputs, data_dir=None):
    _check_input(model)
    models_path = Paths()["model"] / "models"
    learner = MODEL_DICT.get(model)(data_dir=data_dir)
    model_path = models_path / model / "0"
    new_model = tf.keras.models.load_model(model_path)
    learner.model.set_model(new_model, compile=False)
    pred = learner.predict_proba(X_inputs=X_inputs)
    return pred
