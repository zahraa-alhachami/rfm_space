"""
IJIES Response, Task A.2/B.2: a contemporary same-data, same-methodology
comparator. Substitutes scikit-learn's BisectingKMeans for K-Means++ on the
IDENTICAL cleaned UCI Online Retail RFM data (log1p + StandardScaler,
K=2-10 scan, random_state=42), everything else held constant.

Isolated script: reads only outputs/results/rfm_raw.csv (already-verified
4,338-customer cleaned RFM table, identical to the manuscript's Section 3.1
pipeline output). Writes only outputs/results/bisecting_kmeans_uci_kscan.csv.
Does not modify rfm_analysis.py or any existing result.

Run with:
    python rfm_bisecting_kmeans_uci.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
RAW_PATH = os.path.join(config.RESULTS_PATH, "rfm_raw.csv")


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


_section("TASK 2: BISECTING K-MEANS COMPARATOR ON UCI (same-data, same-method)")

rfm = pd.read_csv(RAW_PATH)
n = len(rfm)
print(f"Loaded {RAW_PATH}: {n} customers.")
if n == 4338:
    print("MATCHES the manuscript's reported cleaned customer count (4,338) exactly.")
else:
    print(f"DOES NOT MATCH the manuscript's reported count (4,338). Difference: {n - 4338:+d}. "
          "Reporting honestly rather than adjusting.")

X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p))

rows = []
for k in range(2, 11):
    bkm = BisectingKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = bkm.fit_predict(X)
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    rows.append({"K": k, "Silhouette": sil, "DBI": dbi})
    print(f"K={k:2d} | Silhouette={sil:.4f} | DBI={dbi:.4f}")

table = pd.DataFrame(rows)
os.makedirs(config.RESULTS_PATH, exist_ok=True)
out_path = os.path.join(config.RESULTS_PATH, "bisecting_kmeans_uci_kscan.csv")
table.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")

best_row = table.loc[table["Silhouette"].idxmax()]
best_k = int(best_row["K"])
best_sil = best_row["Silhouette"]
best_dbi = best_row["DBI"]

_section("COMPARISON: BISECTING K-MEANS vs. K-MEANS++ (this paper's own result)")
kmeans_pp_k3_sil, kmeans_pp_k3_dbi = 0.3365, 1.0483
bkm_at_k3 = table[table["K"] == 3].iloc[0]

print(f"K-Means++ (this paper, K=3):        Silhouette={kmeans_pp_k3_sil:.4f}  DBI={kmeans_pp_k3_dbi:.4f}")
print(f"Bisecting K-Means (Silhouette-best, K={best_k}): Silhouette={best_sil:.4f}  DBI={best_dbi:.4f}")
print(f"Bisecting K-Means AT K=3 (matched):  Silhouette={bkm_at_k3['Silhouette']:.4f}  DBI={bkm_at_k3['DBI']:.4f}")

diff_matched = bkm_at_k3["Silhouette"] - kmeans_pp_k3_sil
diff_best = best_sil - kmeans_pp_k3_sil
verdict_matched = "outperforms" if diff_matched > 0 else ("underperforms" if diff_matched < 0 else "ties")
verdict_best = "outperforms" if diff_best > 0 else ("underperforms" if diff_best < 0 else "ties")

print(f"\nAt matched K=3: Bisecting K-Means {verdict_matched} K-Means++ by {diff_matched:+.4f} "
      f"({diff_matched/kmeans_pp_k3_sil*100:+.2f}%).")
print(f"At each method's own best K: Bisecting K-Means (K={best_k}) {verdict_best} K-Means++ (K=3) "
      f"by {diff_best:+.4f} ({diff_best/kmeans_pp_k3_sil*100:+.2f}%).")
