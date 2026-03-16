# features.py
# Computes molecular descriptors AND Morgan fingerprints from SMILES strings.
#
# Why Morgan fingerprints?
#   Our 8 basic descriptors (LogP, MolWeight etc.) describe general properties.
#   Morgan fingerprints encode EXACT structural patterns — which atoms are
#   connected to which, what rings are present, what functional groups exist.
#   This is what professional cheminformatics tools use for toxicity prediction.
#
# Features generated:
#   - 8 molecular descriptors (same as before)
#   - 128 Morgan fingerprint bits (new — encodes structural patterns)
#   Total: 136 features per molecule
#
# Run after load_data.py:
#   python src/features.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem


def get_features(smiles):
    """
    Takes a SMILES string and returns 136 molecular features.
    - 8 standard descriptors (drug-likeness properties)
    - 128 Morgan fingerprint bits (structural patterns)
    Returns None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ── 8 standard molecular descriptors ─────────────────────────────────────
    features = {
        "MolWeight"       : round(Descriptors.MolWt(mol), 3),
        "LogP"            : round(Descriptors.MolLogP(mol), 3),
        "TPSA"            : round(Descriptors.TPSA(mol), 3),
        "NumHDonors"      : Descriptors.NumHDonors(mol),
        "NumHAcceptors"   : Descriptors.NumHAcceptors(mol),
        "NumRotBonds"     : Descriptors.NumRotatableBonds(mol),
        "NumRings"        : Descriptors.RingCount(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
    }

    # ── 128 Morgan fingerprint bits ───────────────────────────────────────────
    # radius=2 means it looks 2 bonds away from each atom
    # nBits=128 gives 128 binary features (0 or 1)
    # each bit represents whether a specific structural pattern is present
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
    for i, bit in enumerate(fp):
        features[f"fp_{i}"] = int(bit)

    return features


def get_features_list(smiles):
    """returns features as a plain list — needed by sklearn/xgboost models"""
    f = get_features(smiles)
    if f is None:
        return None
    return list(f.values())


# ── run directly to generate features for the full tox21 dataset ─────────────
if __name__ == "__main__":

    print("loading tox21.csv...")
    df = pd.read_csv("data/tox21.csv")
    print(f"  loaded {len(df)} molecules")

    print("computing features (8 descriptors + 128 fingerprint bits)...")
    rows    = []
    skipped = 0

    for i, smiles in enumerate(df["smiles"]):
        feat = get_features(smiles)
        if feat:
            rows.append(feat)
        else:
            rows.append({})
            skipped += 1

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1} / {len(df)} done...")

    features_df = pd.DataFrame(rows)
    features_df.fillna(features_df.mean(), inplace=True)

    output = pd.concat([df.reset_index(drop=True),
                        features_df.reset_index(drop=True)], axis=1)

    output.to_csv("data/features_tox21.csv", index=False)

    print(f"\ndone!")
    print(f"  features per molecule : {len(features_df.columns)}")
    print(f"    → 8 descriptors + 128 fingerprint bits = 136 total")
    print(f"  skipped {skipped} invalid SMILES out of {len(df)}")
    print(f"  saved → data/features_tox21.csv")
    print(f"\nnext step: python src/train.py")