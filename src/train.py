# train.py
# Patient-safety focused model training.
#
# Key principle:
#   In drug safety, missing a toxic compound (false negative) is far more
#   dangerous than flagging a safe compound as toxic (false positive).
#   This script is tuned to MAXIMIZE RECALL (sensitivity) for toxic compounds.
#
# Improvements over basic version:
#   1. Morgan fingerprints — encode exact structural patterns (128 bits)
#   2. XGBoost — better than Random Forest on imbalanced data
#   3. SMOTE — creates synthetic toxic examples to fix class imbalance
#   4. Low decision threshold (0.3) — flags more compounds as potentially toxic
#   5. Reports both AUC-ROC and Recall so we can see patient safety coverage
#
# Run after features.py:
#   python src/train.py

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, recall_score,
                             precision_score, f1_score)
from xgboost import XGBClassifier

os.makedirs("models", exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────

# decision threshold — lower = catches more toxic compounds
# 0.3 means: "if model is 30% sure it's toxic, flag it"
# standard is 0.5 — we use 0.3 for patient safety
SAFETY_THRESHOLD = 0.3

TARGET_COLS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

# ── load data ─────────────────────────────────────────────────────────────────
print("loading features_tox21.csv...")
df = pd.read_csv("data/features_tox21.csv")

# get all feature columns automatically
# includes both the 8 descriptors and Morgan fingerprint bits
FEATURE_COLS = [c for c in df.columns
                if c not in TARGET_COLS
                and c not in ["smiles", "mol_id"]
                and not c.startswith("Unnamed")]

print(f"  features available: {len(FEATURE_COLS)}")
print(f"  molecules loaded  : {len(df)}\n")

# ── train one model per assay ──────────────────────────────────────────────────
results = {}

for target in TARGET_COLS:
    if target not in df.columns:
        print(f"  {target:20s} — skipped (column not found)")
        continue

    # keep only rows where this target has a label
    mask = df[target].notna()
    X    = df.loc[mask, FEATURE_COLS].values
    y    = df.loc[mask, target].values.astype(int)

    if len(np.unique(y)) < 2:
        print(f"  {target:20s} — skipped (only one class)")
        continue

    n_toxic    = sum(y == 1)
    n_nontoxic = sum(y == 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── SMOTE — fix class imbalance ───────────────────────────────────────────
    # creates synthetic toxic examples so model trains on balanced data
    # only apply if we have enough toxic samples
    try:
        from imblearn.over_sampling import SMOTE
        k = min(3, sum(y_train == 1) - 1)
        if k >= 1:
            smote    = SMOTE(random_state=42, k_neighbors=k)
            X_train, y_train = smote.fit_resample(X_train, y_train)
    except Exception:
        pass  # if SMOTE fails, continue without it

    # ── XGBoost model ────────────────────────────────────────────────────────
    # scale_pos_weight handles remaining imbalance after SMOTE
    pos_weight = n_nontoxic / max(n_toxic, 1)

    model = XGBClassifier(
        n_estimators     = 200,
        max_depth        = 6,
        learning_rate    = 0.05,
        scale_pos_weight = pos_weight,
        random_state     = 42,
        eval_metric      = "auc",
        verbosity        = 0,
        n_jobs           = -1
    )
    model.fit(X_train, y_train)

    # ── evaluate ──────────────────────────────────────────────────────────────
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    # use low threshold for patient safety — catch more toxic compounds
    y_pred_safety = (y_proba >= SAFETY_THRESHOLD).astype(int)
    recall    = recall_score(y_test,    y_pred_safety, zero_division=0)
    precision = precision_score(y_test, y_pred_safety, zero_division=0)
    f1        = f1_score(y_test,        y_pred_safety, zero_division=0)

    results[target] = {
        "AUC"      : round(auc,       4),
        "Recall"   : round(recall,    4),
        "Precision": round(precision, 4),
        "F1"       : round(f1,        4),
    }

    # save model and threshold together
    model_data = {
        "model"    : model,
        "threshold": SAFETY_THRESHOLD,
        "features" : FEATURE_COLS
    }
    with open(f"models/{target}.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print(f"  {target:20s} — AUC={auc:.3f}  "
          f"Recall={recall:.3f}  "
          f"Precision={precision:.3f}  ✅")

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\ntraining complete!")
print(f"  models trained     : {len(results)}")
print(f"  decision threshold : {SAFETY_THRESHOLD} "
      f"(lower = safer for patients)")
print(f"\n  average AUC       : "
      f"{np.mean([v['AUC']       for v in results.values()]):.4f}")
print(f"  average Recall    : "
      f"{np.mean([v['Recall']    for v in results.values()]):.4f}  "
      f"← most important for patient safety")
print(f"  average Precision : "
      f"{np.mean([v['Precision'] for v in results.values()]):.4f}")
print(f"\n  what Recall means:")
print(f"  0.80 recall = catches 80% of truly toxic compounds")
print(f"  0.20 missed = those 20% could harm patients if approved")
print(f"\nnext step: python src/features.py then python src/train.py")
print(f"           then python -m streamlit run app.py")