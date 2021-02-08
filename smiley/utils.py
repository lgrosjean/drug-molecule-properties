from os.path import abspath
from pathlib import Path


from .feature_extractor import fingerprint_features


class LoadError(Exception):
    """Raised when the data is not loaded"""

    pass


class InputError(Exception):
    """Raised when the input model required is not existing"""


class Paths:
    """
    Base object to centralize all paths and directory for projets
    """

    def __init__(
        self,
        root_dir=None,
        src_dirname="smiley",
        data_dirname="data",
        app_dirname="app",
        model_dirname="model",
        mlruns_dirname="mlruns",
        force=False,
    ):
        """Initialize Paths object for project

        Args:
            src_dirname (str, optional): Name of `src`folder containing scripts. Defaults to "smiley".
            data_dirname (str, optional): Name of data folder, containing all data for training. Defaults to "data".
            app_dirname (str, optional): Name of the `app` folder containing all the requirements for Flask application. Defaults to "app".
            model_dirname (str, optional): Name of the `model` folder containing all resources to deploy Tensorflow Serving models. Defaults to "model".
            mlruns_dirname (str, optional): Name of the MlFlow runs directory to save trainings. Defaults to "mlruns".
            force (bool, optional): [description]. Defaults to False.
        """
        if root_dir is None:
            self.root_dir = Path(abspath(__file__)).parents[1]
        else:
            self.root_dir = Path(root_dir).absolute()
        self.src_dirname = src_dirname
        self.data_dirname = data_dirname
        self.app_dirname = app_dirname
        self.model_dirname = model_dirname
        self.mlruns_dirname = mlruns_dirname
        self.set_list_dir()
        if not self.data_dirname in self.list_dir:
            self.root_dir = self.root_dir.parents[0]
            self.set_list_dir()
        self.d = {
            "src": self.root_dir / self.src_dirname,
            "data": self.root_dir / self.data_dirname,
            "model": self.root_dir / self.model_dirname,
            # 'mlruns': self.root_dir / self.mlruns_dirname
        }
        self.dirs = list(self.d.keys())

    def __getattr__(self, name):
        if name in self.d:
            return self.d.get(name)
        else:
            raise AttributeError(
                f"Attribute {name} does not exists (existing: {self.dirs})."
            )

    def __getitem__(self, name):
        return getattr(self, name)

    def set_list_dir(self):
        """Set the list of directories in the Project folder."""
        self.list_dir = [p.name for p in self.root_dir.iterdir()]

    def get_list_dir(self) -> list:
        """Return list of directories in the project folder.

        Returns:
            list: list of subdirectories.
        """
        return self.list_dir


def smile2bytes(smile: str) -> list:
    """Convert a smile string into a bytes-vector list.

    Args:
        smile (str): The string representation of a smile

    Returns:
        list: List of bytes representing the smile
    """
    bit_vect = fingerprint_features(smile)
    return list(bit_vect)
