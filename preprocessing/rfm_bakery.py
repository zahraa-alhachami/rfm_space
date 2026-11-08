"""
Fifth generalization dataset: a public Brazilian bakery transactional dataset
(189,107 line-item records, Jan-Apr 2025), published as open-access
supplementary material for:

    Carvalho, H., & Maciel, F. (2026). RFM-Based Customer Segmentation: A
    Pedagogical Case Study for Marketing Analytics Education. Journal for
    Advancement of Marketing Education (JAME), 34(1), 14-26.
    https://doi.org/10.63963/001c.157562

Citation verified via DOI resolution and direct fetch of the article page
before this script was written; both matched the numbers supplied by the
researcher exactly.

This script runs TWO clearly separated analyses, isolated from every other
script in this project (reads config.py only; writes only new "bakery"-
prefixed paths; modifies nothing else):

  STAGE A - "Framework generalization": the OFFICIAL adopted thesis pipeline
  (log1p -> StandardScaler -> PCA(2D) -> K-Means++(K=3)) applied to this
  dataset using THIS project's own RFM/cleaning conventions, exactly as was
  done for Online Retail II / Olist / Superstore.

  STAGE B - "Exact methodological replication" of Carvalho & Maciel's own
  published code (the Python_Code_Guide.docx supplied alongside bakery.csv):
  np.log (not log1p), StandardScaler, K-Means fit directly on the 3D scaled
  RFM data (PCA used ONLY for visualization in their code, never for
  clustering), K=3 and K=4, same cleaning rules they used (drop
  CustomerID==0, dedupe on SaleID+CustomerID+ProductID, Frequency =
  nunique(SaleDate)). This stage first reproduces their reported Silhouette
  scores as a sanity check before anything is reported as a comparison.

Because Stage A and Stage B differ in what space K-Means actually clusters
in (2D PCA-projected vs. full 3D standardized RFM), they are NOT directly
comparable to each other, and this is reported explicitly rather than
glossed over -- the same methodological-fairness discipline applied to
every other external comparison in this project.

Decision criteria fixed BEFORE running:
  - Stage B counts as a successful reproduction if its Silhouette values are
    within +/-0.01 of the paper's reported 0.3383 (K=3) and 0.3872 (K=4).
  - No result is reframed after being seen; every outcome (favorable or not)
    is reported as-is.

Run with:
    python rfm_bakery.py
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
DATA_PATH = os.path.join("data", "bakery", "bakery.csv")
RESULTS_PATH = os.path.join(config.RESULTS_PATH, "bakery")

# Reported in Carvalho & Maciel (2026), verified via DOI + article-page fetch.
PAPER_SILHOUETTE = {3: 0.3383, 4: 0.3872, 5: 0.3393, 6: 0.3309, 7: 0.3381}
REPRO_TOLERANCE = 0.01

# Official thesis reference (UCI, PCA+K-Means++, adopted as of this session).
OFFICIAL_UCI_PCA_SILHOUETTE = 0.3754


def _section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_raw():
    df = pd.read_csv(DATA_PATH)
    df["SaleDate"] = pd.to_datetime(df["SaleDate"])
    return df


# ---------------------------------------------------------------------------
# STAGE A: our official framework's own cleaning/RFM conventions
# ---------------------------------------------------------------------------
def stage_a_framework_generalization(df):
    _section("STAGE A: FRAMEWORK GENERALIZATION (our official PCA + K-Means++ pipeline)")

    d = df.copy()
    before = len(d)
    d = d.dropna(subset=["CustomerID", "SaleDate", "ItemValue"])
    d = d[d["CustomerID"] != 0]  # not a real, trackable customer (walk-in code)
    d = d[d["Quantity"] > 0]
    d = d[d["ItemValue"] > 0]
    d = d.drop_duplicates()
    d["TotalPrice"] = d["Quantity"] * d["ItemValue"]
    print(f"Records: {before} -> {len(d)} after cleaning (our standard rules)")

    snapshot_date = d["SaleDate"].max() + pd.Timedelta(days=1)
    rfm = d.groupby("CustomerID").agg(
        Recency=("SaleDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("SaleID", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()
    print(f"Unique customers: {len(rfm)}")

    os.makedirs(RESULTS_PATH, exist_ok=True)
    rfm.to_csv(os.path.join(RESULTS_PATH, "rfm_raw_bakery.csv"), index=False)

    X_scaled = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p))
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var_total = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA variance retained (PC1+PC2): {var_total:.2f}%")

    kmeans = KMeans(n_clusters=3, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    dbi = davies_bouldin_score(X_pca, labels)

    sizes = pd.Series(labels).value_counts().sort_index()
    pct = (sizes / len(labels) * 100).round(2)
    print(f"\nSilhouette (PCA 2D, K=3): {sil:.4f}")
    print(f"Davies-Bouldin: {dbi:.4f}")
    print("Cluster sizes:", dict(sizes), " pct:", dict(pct))

    balanced = pct.min() >= 5.0  # same informal validity bar used in prior sessions
    print(f"Cluster-balance validity check (min segment >= 5%): {'PASS' if balanced else 'FAIL - degenerate split'}")

    return {"n_customers": len(rfm), "silhouette": sil, "dbi": dbi, "sizes": dict(sizes),
            "pct": dict(pct), "balanced": balanced, "pca_variance_pct": var_total}


# ---------------------------------------------------------------------------
# STAGE B: exact replication of Carvalho & Maciel's own code
# ---------------------------------------------------------------------------
def stage_b_exact_replication(df):
    _section("STAGE B: EXACT REPLICATION OF Carvalho & Maciel (2026) CODE")

    d = df.copy()
    before = len(d)
    d = d.dropna(subset=["CustomerID", "SaleDate", "ItemValue", "ItemDescription"])
    d = d[d["CustomerID"] != 0]
    d = d.drop_duplicates(subset=["SaleID", "CustomerID", "ProductID"])
    print(f"Records: {before} -> {len(d)} after their exact cleaning rules")

    snapshot_date = d["SaleDate"].max() + pd.Timedelta(days=1)
    rfm = d.groupby("CustomerID").agg(
        Recency=("SaleDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("SaleDate", "nunique"),
        Monetary=("ItemValue", "sum"),
    ).reset_index()
    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    print(f"Unique customers: {len(rfm)}")

    # Their code uses np.log (not log1p); only valid if no zeros are present.
    zero_check = (rfm[["Recency", "Frequency", "Monetary"]] <= 0).sum()
    print("Zero/negative values per column (np.log requires none):", dict(zero_check))

    X_transformed = rfm[["Recency", "Frequency", "Monetary"]].apply(np.log)
    X_scaled = StandardScaler().fit_transform(X_transformed)

    repro_results = {}
    print(f"\n{'K':>3} | {'Our Silhouette':>15} | {'Paper Reported':>15} | {'Diff':>8} | Reproduced?")
    for k in (3, 4):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        reported = PAPER_SILHOUETTE[k]
        diff = abs(sil - reported)
        reproduced = diff <= REPRO_TOLERANCE
        repro_results[k] = {"our_sil": sil, "paper_sil": reported, "diff": diff, "reproduced": reproduced}
        print(f"{k:>3} | {sil:>15.4f} | {reported:>15.4f} | {diff:>8.4f} | {'YES' if reproduced else 'NO'}")

    return repro_results


def main():
    _section("EXPLORATORY: 5TH GENERALIZATION DATASET (BRAZILIAN BAKERY)")
    print(
        "Citation verified via DOI resolution + article-page fetch before any code was written. "
        "Purely exploratory / generalization-testing. Does not modify any prior script or saved "
        "result. Every outcome reported honestly, favorable or not."
    )

    df = load_raw()
    print(f"\nLoaded {len(df)} raw transaction records ({df['SaleDate'].min()} to {df['SaleDate'].max()})")

    stage_a = stage_a_framework_generalization(df)
    stage_b = stage_b_exact_replication(df)

    _section("FINAL SUMMARY")
    print("Stage B reproduction of the paper's own numbers (sanity check on our understanding of their code):")
    for k, r in stage_b.items():
        print(f"  K={k}: reproduced={r['reproduced']} (ours={r['our_sil']:.4f}, theirs={r['paper_sil']:.4f})")

    print(f"\nStage A (our official PCA+K-Means++ framework, K=3): Silhouette = {stage_a['silhouette']:.4f}")
    print(f"Paper's own K=3 result (their 3D space, no PCA in clustering): {PAPER_SILHOUETTE[3]:.4f}")
    print(f"Paper's own adopted/best result (K=4, their 3D space): {PAPER_SILHOUETTE[4]:.4f}")
    print(
        "\nIMPORTANT CAVEAT: Stage A clusters in 2D PCA-projected space; the paper's numbers "
        "(Stage B target) come from clustering in full 3D standardized RFM space with PCA used "
        "only for visualization. These are NOT a like-for-like methodological comparison -- "
        "reported side by side for transparency, not as a claim of direct superiority."
    )

    summary_rows = [
        {"Comparison": "Our Stage A (PCA 2D + K-Means++, K=3) vs paper K=3 (3D, their space)",
         "Ours": round(stage_a["silhouette"], 4), "Paper": PAPER_SILHOUETTE[3],
         "Directly comparable?": "No - different clustering space (2D PCA vs 3D)"},
        {"Comparison": "Our Stage B replication (3D, their space) vs paper K=3",
         "Ours": round(stage_b[3]["our_sil"], 4), "Paper": PAPER_SILHOUETTE[3],
         "Directly comparable?": "Yes - exact same method"},
        {"Comparison": "Our Stage B replication (3D, their space) vs paper K=4 (their adopted best)",
         "Ours": round(stage_b[4]["our_sil"], 4), "Paper": PAPER_SILHOUETTE[4],
         "Directly comparable?": "Yes - exact same method"},
    ]
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + summary_df.to_string(index=False))

    os.makedirs(RESULTS_PATH, exist_ok=True)
    out_path = os.path.join(RESULTS_PATH, "bakery_comparison_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
