# charts.py
# This script generates all visualizations for ToxiScan.
# Charts are saved as PNG files in the charts/ folder.
# These same charts are displayed inside the Streamlit app.
#
# Charts generated:
#   1. AUC scores per toxicity target
#   2. Feature importance (which molecular property matters most)
#   3. LogP distribution — toxic vs non-toxic
#   4. Confusion matrix for best model (SR-MMP)
#
# Run this after train.py:
#   python src/charts.py

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

os.makedirs("charts", exist_ok=True)

FEATURE_COLS = [
    "MolWeight", "LogP", "TPSA", "NumHDonors",
    "NumHAcceptors", "NumRotBonds", "NumRings", "NumAromaticRings"
]

TARGET_COLS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

# ── load data ─────────────────────────────────────────────────────────────────
print("loading data...")
df = pd.read_csv("data/features_tox21.csv")


# ── chart 1 — AUC scores per target ──────────────────────────────────────────
print("generating chart 1 — AUC scores...")

auc_scores = {}
for target in TARGET_COLS:
    try:
        with open(f"models/{target}.pkl", "rb") as f:
            model = pickle.load(f)
        mask = df[target].notna()
        X    = df.loc[mask, FEATURE_COLS].values
        y    = df.loc[mask, target].values.astype(int)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        auc_scores[target] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except Exception as e:
        print(f"  skipping {target}: {e}")

# color bars by score — green = good, orange = ok, red = weak
colors = [
    "#2ecc71" if v >= 0.80 else
    "#e67e22" if v >= 0.70 else
    "#e74c3c"
    for v in auc_scores.values()
]

plt.figure(figsize=(12, 5))
plt.bar(auc_scores.keys(), auc_scores.values(), color=colors, edgecolor="white", width=0.6)
plt.axhline(0.80, color="#2ecc71", linestyle="--", linewidth=1, alpha=0.7, label="Good (0.80)")
plt.axhline(0.70, color="#e67e22", linestyle="--", linewidth=1, alpha=0.7, label="Acceptable (0.70)")
plt.title("Model performance across 12 toxicity targets", fontsize=14, pad=12)
plt.ylabel("AUC-ROC Score")
plt.ylim(0.5, 1.0)
plt.xticks(rotation=40, ha="right", fontsize=9)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("charts/auc_scores.png", dpi=150)
plt.close()
print("  saved → charts/auc_scores.png ✅")


# ── chart 2 — feature importance ─────────────────────────────────────────────
print("generating chart 2 — feature importance...")

# use SR-MMP since it had the best AUC (0.9057)
with open("models/SR-MMP.pkl", "rb") as f:
    best_model = pickle.load(f)

importance  = best_model.feature_importances_
sorted_idx  = np.argsort(importance)
sorted_feats = [FEATURE_COLS[i] for i in sorted_idx]
sorted_vals  = [importance[i]   for i in sorted_idx]

plt.figure(figsize=(8, 5))
plt.barh(sorted_feats, sorted_vals, color="#3498db", edgecolor="white")
plt.title("Which molecular property predicts SR-MMP toxicity most?", fontsize=13, pad=10)
plt.xlabel("Feature importance score")
plt.tight_layout()
plt.savefig("charts/feature_importance.png", dpi=150)
plt.close()
print("  saved → charts/feature_importance.png ✅")


# ── chart 3 — LogP distribution toxic vs non-toxic ───────────────────────────
print("generating chart 3 — LogP distribution...")

# use NR-AhR as example (good AUC, decent sample size)
mask     = df["NR-AhR"].notna()
toxic    = df.loc[mask & (df["NR-AhR"] == 1), "LogP"].dropna()
nontoxic = df.loc[mask & (df["NR-AhR"] == 0), "LogP"].dropna()

plt.figure(figsize=(9, 4))
plt.hist(nontoxic, bins=50, alpha=0.6, color="#2ecc71", label=f"non-toxic  (n={len(nontoxic)})")
plt.hist(toxic,    bins=50, alpha=0.6, color="#e74c3c", label=f"toxic      (n={len(toxic)})")
plt.axvline(nontoxic.mean(), color="#27ae60", linestyle="--", linewidth=1.5,
            label=f"non-toxic mean = {nontoxic.mean():.2f}")
plt.axvline(toxic.mean(),    color="#c0392b", linestyle="--", linewidth=1.5,
            label=f"toxic mean = {toxic.mean():.2f}")
plt.title("LogP distribution — toxic vs non-toxic (NR-AhR assay)", fontsize=13, pad=10)
plt.xlabel("LogP  (lipophilicity — higher = more fat-soluble)")
plt.ylabel("number of compounds")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("charts/logp_distribution.png", dpi=150)
plt.close()
print("  saved → charts/logp_distribution.png ✅")


# ── chart 4 — confusion matrix ────────────────────────────────────────────────
print("generating chart 4 — confusion matrix...")

mask   = df["SR-MMP"].notna()
X_all  = df.loc[mask, FEATURE_COLS].values
y_all  = df.loc[mask, "SR-MMP"].values.astype(int)
_, X_test, _, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)
y_pred = best_model.predict(X_test)
cm     = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["non-toxic", "toxic"]).plot(
    ax=ax, cmap="Blues", colorbar=False
)
ax.set_title("Confusion matrix — SR-MMP (best model)", fontsize=12, pad=10)
plt.tight_layout()
plt.savefig("charts/confusion_matrix.png", dpi=150)
plt.close()
print("  saved → charts/confusion_matrix.png ✅")


# ── done ──────────────────────────────────────────────────────────────────────
print("\nall charts saved in charts/ folder!")
print("  charts/auc_scores.png")
print("  charts/feature_importance.png")
print("  charts/logp_distribution.png")
print("  charts/confusion_matrix.png")
print("\nnext step: python app.py")