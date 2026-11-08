"""
IJIES Response follow-up, Reviewer 1 Point 7 residual gap: the manuscript's
per-dataset protocol table (Table 3b) needs a genuine 3D-baseline K=2-10
candidate scan for the Bakery dataset, matching Table 1's methodology on
UCI. The only K-scan previously saved for Bakery
(outputs/results/bakery/k_selection_table.csv) was produced by
rfm_bakery_pipeline_official.py, which runs the scan IN PCA(2D) SPACE
(K=3 Silhouette=0.3922 there matches the known PCA-official result) --
not the 3D baseline space, so it is not the right comparison point here.

This script reproduces the exact same cleaning as rfm_bakery.py's Stage A
(CustomerID != 0, Quantity>0, ItemValue>0, full-row dedup,
TotalPrice=Quantity*ItemValue), computes RFM, applies log1p+StandardScaler
(no PCA), and scans K=2-10 -- the direct 3D-baseline analogue of Table 1.

Isolated script: reads only data/bakery/bakery.csv (read-only). Writes
only outputs/results/bakery/k_selection_table_3d_baseline.csv. Does not
modify rfm_bakery.py, rfm_bakery_pipeline_official.py, or any existing
result.

Run with:
    python rfm_bakery_3d_kscan.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "bakery", "bakery.csv")
RESULTS_PATH = os.path.join(config.RESULTS_PATH, "bakery")


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


_section("BAKERY 3D-BASELINE K-SCAN (K=2-10, no PCA) -- Table 1 analogue")

df = pd.read_csv(DATA_PATH)
df["SaleDate"] = pd.to_datetime(df["SaleDate"])
d = df.dropna(subset=["CustomerID", "SaleDate", "ItemValue"])
d = d[d["CustomerID"] != 0]
d = d[d["Quantity"] > 0]
d = d[d["ItemValue"] > 0]
d = d.drop_duplicates()
d["TotalPrice"] = d["Quantity"] * d["ItemValue"]

snapshot_date = d["SaleDate"].max() + pd.Timedelta(days=1)
rfm = d.groupby("CustomerID").agg(
    Recency=("SaleDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("SaleID", "nunique"),
    Monetary=("TotalPrice", "sum"),
).reset_index()
print(f"Customers: {len(rfm)} (reference: 9,573)")

X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p))

rows = []
print(f"\n{'K':>3} | {'WCSS':>14} | {'Silhouette':>11} | {'DBI':>8}")
for k in range(2, 11):
    km = KMeans(n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    wcss = km.inertia_
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    print(f"{k:>3} | {wcss:>14.2f} | {sil:>11.4f} | {dbi:>8.4f}")
    rows.append({"K": k, "WCSS": wcss, "Silhouette_Score": sil, "Davies_Bouldin_Index": dbi})

table = pd.DataFrame(rows)
os.makedirs(RESULTS_PATH, exist_ok=True)
out_path = os.path.join(RESULTS_PATH, "k_selection_table_3d_baseline.csv")
table.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")

best_sil_k = int(table.loc[table["Silhouette_Score"].idxmax(), "K"])
best_dbi_k = int(table.loc[table["Davies_Bouldin_Index"].idxmin(), "K"])
sil_at_3 = table.loc[table["K"] == 3, "Silhouette_Score"].iloc[0]
sil_at_best = table["Silhouette_Score"].max()
print(f"\nBest K by Silhouette: {best_sil_k} (Silhouette={sil_at_best:.4f})")
print(f"Best K by DBI: {best_dbi_k}")
print(f"K=3 Silhouette: {sil_at_3:.4f} (diff from best: {sil_at_3 - sil_at_best:+.4f})")
