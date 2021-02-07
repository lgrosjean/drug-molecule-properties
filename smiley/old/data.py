import logging

import pandas as pd
from tqdm.auto import tqdm

from ..utils import LoadError, Paths
from ..feature_extractor import fingerprint_features


def smile2bytes(smile: str) -> list:
    """Convert a smile string into a bytes-vector list.

    Args:
        smile (str): The string representation of a smile

    Returns:
        list: List of bytes representing the smile
    """
    bit_vect = fingerprint_features(smile)
    return list(bit_vect)


class Dataset:

    valid_mode = ["multi", "single"]

    def __init__(
        self,
        data_dir=None,
        mode="single",
        target_col="P1",
        smile_col="smiles",
        load=True,
        transform=True,
    ):
        if not mode in self.valid_mode:
            raise LoadError(
                f"Please provide a mode from {self.valid_mode} (provided: {mode})"
            )
        self.mode = mode
        self.target_col = target_col
        self.smile_col = smile_col
        self.X_ = None
        self.y_ = None
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = Paths()["data"]
        self.path = self.data_dir / f"dataset_{self.mode}.csv"
        self.data = None
        if load:
            self.load()
        if transform:
            self.X_, self.y_ = self.transform()

    def is_loaded(self):
        return not (self.data is None)

    def is_transformed(self):
        return self.y_ is not None and self.X_ is not None

    def _check_load(self):
        if not self.is_loaded():
            raise LoadError("Please load data.")

    def load(self):
        data = pd.read_csv(self.path)
        self.data = data.copy()
        self.X = self.data.drop(self.target_col, axis=1)
        self.y = self.data[self.target_col]
        self.smiles = self.data[self.smile_col]
        self.input_shape = self.X.shape[1]

    def get(self):
        self._check_load()
        return self.data

    def transform(self):
        self._check_load()
        bytes_list = []
        smiles_list = list(self.data["smiles"])
        logging.info("Converting smiles to bytes...")
        for smile in tqdm(smiles_list):
            byte = smile2bytes(smile)
            bytes_list.append(byte)
        df_bytes = pd.DataFrame.from_records(bytes_list)
        self.X_ = df_bytes
        self.y_ = self.y
        return self.X_, self.y_