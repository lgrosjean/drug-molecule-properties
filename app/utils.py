import requests

from rdkit.Chem import (  # pylint: disable=no-name-in-module
    rdMolDescriptors,
    MolFromSmiles,
    rdmolfiles,
    rdmolops,
)


def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=size, useChirality=True, useBondTypes=True, useFeatures=False
    )


def smile2bytes(smile: str) -> list:
    """Convert a smile string into a bytes-vector list.

    Args:
        smile (str): The string representation of a smile

    Returns:
        list: List of bytes representing the smile
    """
    bit_vect = fingerprint_features(smile)
    return list(bit_vect)


class Prediction:
    def __init__(self, model: str, inputs: str = None):
        self.model = model
        self.inputs = inputs
        self.set_url(model=model)

    def set_url(self, model: str = None):
        if not model is None:
            self.model = model
        self.url = f"http://localhost:8501/v1/models/{self.model}:predict"

    def predict(self, inputs: str = None) -> dict:
        if not inputs is None:
            self.inputs = inputs
        print(self.inputs)
        if self.model == "model1":
            inputs_ = list(smile2bytes(self.inputs))
        else:
            inputs_ = [self.inputs]
        payload = {"instances": inputs_}
        res = requests.post(self.url, json=payload)
        return res.json()
