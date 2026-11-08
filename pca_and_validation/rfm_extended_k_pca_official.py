"""
Verification script: the criteria table (literature_comparison_criteria.csv)
claims "29 K values tested (K=2 to K=30)" for the official model. That claim
was accurate for the ORIGINAL 3D K-Means++ model (see
outputs/results/k_extended_2_to_30.csv, K=3 Silhouette=0.3365, matching the
old official result) -- but was never re-verified after PCA+K-Means++
became the official model (rfm_analysis_pca_official.py only tests K=2-10
via the standard find_optimal_k(), config.MAX_K=10).

This script closes that gap honestly: it reproduces the exact same PCA
projection as rfm_analysis_pca_official.py, then tests K=2 through K=30 in
that actual official clustering space, so the "29 values tested" claim can
either be reaffirmed (if it still holds) or corrected (if it doesn't) --
decided by the result, not assumed.

Does not modify rfm_analysis.py, rfm_analysis_pca_official.py, or any
existing output. Writes only to a new file,
outputs/results/pca_official/k_extended_2_to_30.csv.

Run with:
    python rfm_extended_k_pca_official.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
OUT_PATH = os.path.join(config.RESULTS_PATH, "pca_official", "k_extended_2_to_30.csv")


def _section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


_section("VERIFYING 'K=2 TO K=30 TESTED' CLAIM FOR THE OFFICIAL PCA MODEL")

rfm = pd.read_csv(os.path.join(config.RESULTS_PATH, "pca_official", "rfm_raw.csv"))
X_scaled = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p))
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

k_values = list(range(2, 31))
rows = []
for k in k_values:
    km = KMeans(n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_pca)
    rows.append({
        "K": k, "WCSS": km.inertia_,
        "Silhouette": silhouette_score(X_pca, labels),
        "DBI": davies_bouldin_score(X_pca, labels),
    })

table = pd.DataFrame(rows)
print(table.to_string(index=False))

k3_row = table[table["K"] == 3].iloc[0]
print(f"\nK=3 Silhouette in this extended run: {k3_row['Silhouette']:.4f}")
print("Previously recorded official PCA result (K=3): 0.3754")
print(f"Match: {abs(k3_row['Silhouette'] - 0.3754) < 0.0005}")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
table.to_csv(OUT_PATH, index=False)
print(f"\nSaved {OUT_PATH}")

_section("VERDICT")
print(f"29 K values (K=2 to K=30) tested in the ACTUAL current official clustering "
      f"space (PCA 2D): {len(k_values) == 29}")
