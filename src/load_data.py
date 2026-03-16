# load_data.py
# This script downloads and saves all datasets we need for ToxiScan.
# Run this once before running anything else.
# Datasets used:
#   - Tox21     : main dataset for training (12 toxicity labels)
#   - ZINC250k  : used for molecular property comparison
#   - ChEMBL    : queried live via API when needed (see chembl_lookup.py)

import os
import glob
import pandas as pd

os.environ["KAGGLE_KEY"] = os.environ.get("KAGGLE_API_TOKEN")
os.environ["KAGGLE_USERNAME"] = os.environ.get("KAGGLE_USERNAME")
import kagglehub

# create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)


def download_kaggle(dataset_id, filename):
    """
    Downloads a kaggle dataset and saves it as a CSV in the data folder.
    Skips download if the file already exists (saves time on reruns).
    """
    save_path = f"data/{filename}"

    # if already downloaded before, just load it
    if os.path.exists(save_path):
        print(f"  already downloaded, loading from {save_path}")
        return pd.read_csv(save_path)

    # download from kaggle
    folder = kagglehub.dataset_download(dataset_id)

    # find the csv inside the downloaded folder
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"no csv found in {folder}, files: {os.listdir(folder)}")

    df = pd.read_csv(csv_files[0])
    df.to_csv(save_path, index=False)
    return df


# ── Tox21 ──────────────────────────────────────────────────────────────────
print("loading tox21...")
tox21 = download_kaggle("epicskills/tox21-dataset", "tox21.csv")
print(f"  tox21 ready — {tox21.shape[0]} rows, {tox21.shape[1]} columns")

# ── ZINC250k ───────────────────────────────────────────────────────────────
print("loading zinc250k...")
zinc = download_kaggle("basu369victor/zinc250k", "zinc250k.csv")
print(f"  zinc250k ready — {zinc.shape[0]} rows, {zinc.shape[1]} columns")

# ── ChEMBL ─────────────────────────────────────────────────────────────────
# we don't bulk download ChEMBL — it's queried live per compound
# check src/chembl_lookup.py for how we use it
print("chembl — using live API (no download needed)")

# ── done ───────────────────────────────────────────────────────────────────
print("\nall datasets ready!")
print(f"  data/tox21.csv    — {tox21.shape}")
print(f"  data/zinc250k.csv — {zinc.shape}")
print(f"  chembl            — live API")