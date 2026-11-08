"""
IJIES Response follow-up, Task 3(b): compare plausible alternative
preprocessing choices against the manuscript's own log1p+StandardScaler
pipeline, all under the identical fixed K=3, K-Means++ (n_init=10,
random_state=42) configuration, on the same cleaned UCI RFM data.

Tests two alternatives (both cheap to implement cleanly):
  (i)  No log1p transform -- StandardScaler directly on raw RFM.
  (ii) log1p + RobustScaler instead of StandardScaler.

Isolated script: reads only outputs/results/pca_official/rfm_raw.csv.
Writes only outputs/results/alternative_preprocessing_uci.csv. Does not
modify any existing script or result.

Run with:
    python rfm_alternative_preprocessing_uci.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
RESULTS_PATH = config.RESULTS_PATH
RAW_PATH = os.path.join(RESULTS_PATH, "pca_official", "rfm_raw.csv")
COLS = ["Recency", "Frequency", "Monetary"]


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def fit_and_score(X, label):
    km = KMeans(n_clusters=3, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    print(f"{label:<40} | Silhouette={sil:.4f} | DBI={dbi:.4f}")
    return sil, dbi


_section("ALTERNATIVE PREPROCESSING COMPARISON -- UCI, fixed K=3, K-Means++")

rfm = pd.read_csv(RAW_PATH)
print(f"Loaded {RAW_PATH}: {len(rfm)} customers.\n")

rows = []

# Reference: manuscript's own pipeline
X_ref = StandardScaler().fit_transform(rfm[COLS].apply(np.log1p))
sil, dbi = fit_and_score(X_ref, "REFERENCE: log1p + StandardScaler (manuscript)")
rows.append({"Configuration": "log1p + StandardScaler (manuscript's own pipeline)", "Silhouette": sil, "DBI": dbi})

# Alternative (i): no log1p, StandardScaler on raw RFM
X_i = StandardScaler().fit_transform(rfm[COLS])
sil, dbi = fit_and_score(X_i, "(i) No log1p -- StandardScaler on raw RFM")
rows.append({"Configuration": "No log1p transform, StandardScaler on raw RFM", "Silhouette": sil, "DBI": dbi})

# Alternative (iii): log1p + RobustScaler
X_iii = RobustScaler().fit_transform(rfm[COLS].apply(np.log1p))
sil, dbi = fit_and_score(X_iii, "(iii) log1p + RobustScaler")
rows.append({"Configuration": "log1p + RobustScaler", "Silhouette": sil, "DBI": dbi})

table = pd.DataFrame(rows)
os.makedirs(RESULTS_PATH, exist_ok=True)
out_path = os.path.join(RESULTS_PATH, "alternative_preprocessing_uci.csv")
table.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")

_section("SUMMARY")
print(table.to_string(index=False))
ref_sil = rows[0]["Silhouette"]
for r in rows[1:]:
    diff = r["Silhouette"] - ref_sil
    print(f"{r['Configuration']}: {diff:+.4f} vs. the manuscript's own log1p+StandardScaler baseline")
print(f"\nFor context, the PCA-enhanced result (2 components, on top of log1p+StandardScaler) is 0.3754.")
