"""
Produces the standard dashboard-facing artifacts for the bakery dataset
(4th dataset in the Flask dashboard, alongside UCI/Retail II/Olist):
cluster_profiles.csv, rfm_clustered.csv, k_selection_table.csv, and the
standard figure set (optimal_k, 3d_clusters, cluster_sizes, cluster_heatmap,
boxplots) -- using the OFFICIAL adopted model (PCA(2D) + K-Means++, K=3).

Does NOT modify rfm_bakery.py (which remains the authoritative source for
the Stage A/B literature-comparison analysis in
outputs/results/bakery/bakery_comparison_summary.csv). This script
reproduces the same cleaning rules as rfm_bakery.py's Stage A
(deterministic, verified identical) purely so the dashboard has the
standard per-segment profile/figure artifacts that the other three
datasets already have.

Reuses rfm_analysis.py's profile_clusters/visualize_clusters/find_optimal_k
/run_kmeans functions unmodified (imported, not duplicated) -- these are
fully generic over any Recency/Frequency/Monetary/Cluster/Segment table,
regardless of clustering space or source dataset.

Run with:
    python rfm_bakery_pipeline_official.py
"""

import os
import sys

import numpy as np
import pandas as pd

import config

RESULTS_PATH = os.path.join(config.RESULTS_PATH, "bakery") + os.sep
FIGURES_PATH = os.path.join(config.FIGURES_PATH, "bakery") + os.sep
config.RESULTS_PATH = RESULTS_PATH
config.FIGURES_PATH = FIGURES_PATH

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Same reason as in rfm_analysis_pca_official.py: this script imports
# rfm_analysis.py as a library. Here they are in the same folder, and running
# from the package root does not put that folder on sys.path, so it is added.
# No computation changes.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rfm_analysis import (
    _section,
    find_optimal_k,
    run_kmeans,
    profile_clusters,
    visualize_clusters,
    SELECTED_K,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "bakery", "bakery.csv")


def load_and_clean():
    """Identical cleaning rules to rfm_bakery.py's Stage A (our official
    framework's own conventions), reproduced here read-only for determinism."""
    df = pd.read_csv(DATA_PATH)
    df["SaleDate"] = pd.to_datetime(df["SaleDate"])
    d = df.dropna(subset=["CustomerID", "SaleDate", "ItemValue"])
    d = d[d["CustomerID"] != 0]
    d = d[d["Quantity"] > 0]
    d = d[d["ItemValue"] > 0]
    d = d.drop_duplicates()
    d["TotalPrice"] = d["Quantity"] * d["ItemValue"]
    return d


def compute_rfm(d):
    snapshot_date = d["SaleDate"].max() + pd.Timedelta(days=1)
    rfm = d.groupby("CustomerID").agg(
        Recency=("SaleDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("SaleID", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()
    return rfm


def main():
    for path in (RESULTS_PATH, FIGURES_PATH):
        os.makedirs(path, exist_ok=True)

    _section("BAKERY DATASET — OFFICIAL MODEL DASHBOARD ARTIFACTS (PCA + K-Means++)")

    d = load_and_clean()
    rfm = compute_rfm(d)
    print(f"Unique customers: {len(rfm)}")

    out_raw = os.path.join(RESULTS_PATH, "rfm_raw_bakery.csv")
    rfm.to_csv(out_raw, index=False)
    print(f"Saved {out_raw}")

    X_scaled = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p))
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    rfm_pca = pca.fit_transform(X_scaled)
    print(f"PCA variance retained (PC1+PC2): {pca.explained_variance_ratio_.sum() * 100:.2f}%")

    k_results = find_optimal_k(rfm_pca)
    optimal_k = k_results["selected_k"]
    assert optimal_k == SELECTED_K == 3

    rfm, kmeans = run_kmeans(rfm_pca, rfm, optimal_k)
    rfm, profile = profile_clusters(rfm)
    visualize_clusters(rfm)

    _section("DONE")
    print(f"All bakery dashboard artifacts saved under {RESULTS_PATH} and {FIGURES_PATH}")


if __name__ == "__main__":
    main()
