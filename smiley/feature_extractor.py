# Authors: Léo Grosjean <leo.grosjean@live.fr>
# License: GPL3

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
