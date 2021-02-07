"""
Module to define Dataset class, and simplify data loading into memory. A unique class is defined here: `Dataset`.
"""
from pathlib import Path
import logging

import pandas as pd
from tqdm.auto import tqdm

from .utils import Paths, smile2bytes


class LoadError(Exception):
    """An exception raised when the data is not loaded."""


class Dataset:
    """Class Dataset to load data into memory."""

    def __init__(self, data_dir: Path = None, load=True):
        """Initialize Dataset class.

        Args:
            data_dir (Path, optional): Location for data directory containing raw csv file. Defaults to None. If None, the default argument will be set as default data_dir for project.
            load (bool, optional): Load dataframe into memoery. Defaults to True.
        """
        self.set_data_dir(data_dir)
        self.path = self.data_dir / f"dataset_multi.csv"
        self.data = None
        if load:
            self.load()

    def _is_loaded(self) -> bool:
        """Intern function to check if data has been loaded into memory

        Returns:
            bool: the dataset is loaded
        """
        return not (self.data is None)

    def _check_loaded(self):
        """Check if the input data is loaded or not.

        Raises:
            LoadError: Data has to be loaded before action.
        """
        if not self._is_loaded():
            raise LoadError("Your input data is not loaded.")

    def set_data_dir(self, data_dir):
        """Set data_dir attributes.

        Args:
            data_dir (str, pathlib.Path): The location of the data_dir if not by default.
        """
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = Paths()["data"]

    def load(self):
        """Load csv located at `self.path` into class memory."""
        logging.info(f"Loading dataset from {self.path}")
        data = pd.read_csv(self.path)
        self.data = data.copy()
        self.input_shape = self.data.shape[1]

    def transform_smile(self, smile_col: str = "smiles"):
        self._check_loaded()
        if not "smiles" in self.data.columns:
            raise AttributeError(
                f"The provided smile_col {smile_col} does not exist in the dataset."
            )
        bytes_list = []
        smiles_list = list(self.data.get(smile_col))
        logging.info(f"Start conversion of smiles into bytes...")
        for smile in tqdm(smiles_list):
            byte = smile2bytes(smile)
            bytes_list.append(byte)
        df_bytes = pd.DataFrame.from_records(bytes_list)
        self.data = pd.concat([self.data, df_bytes], axis=1)