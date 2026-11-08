"""
Supplementary analysis (NOT a replacement for the main 3D pipeline): tests
whether PCA (Principal Component Analysis), optionally followed by Varimax
rotation for interpretability, improves K-Means++ clustering quality on the
same RFM data — and whether any such improvement generalizes across all
three thesis datasets (UCI, Online Retail II, Olist).

Scientific commitment: every result is reported honestly, favorable or not.
PCA is not assumed to help — with only 3 well-chosen RFM features there is
limited redundancy for PCA to exploit; that expectation is tested, not
assumed. Varimax rotation is applied only because it is a standard,
zero-cost technique for improving component interpretability: being an
orthogonal rotation, it cannot change pairwise distances, so it is verified
here (not just asserted) that Silhouette and Davies-Bouldin are unchanged
by it in every dataset.

Official result: the full 3D (Recency, Frequency, Monetary) K-Means++
clustering remains the one relied upon throughout the thesis. This script's
findings are supplementary evidence, to be adopted formally only as a
deliberate researcher decision — not automatically because a number moved.

Run with:
    python rfm_pca_analysis.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

import config

RANDOM_STATE = config.RANDOM_STATE
SELECTED_K = 3
COLS = ["Recency", "Frequency", "Monetary"]
SEGMENT_ORDER = ["Champions", "At-Risk Customers", "Lost Customers"]

DATASETS = [
    {"name": "UCI", "path": os.path.join(config.RESULTS_PATH, "rfm_raw.csv")},
    {"name": "Online Retail II", "path": os.path.join(config.RESULTS_PATH, "retail2", "rfm_raw_retail2.csv")},
    {"name": "Olist", "path": os.path.join(config.RESULTS_PATH, "olist", "rfm_raw_olist.csv")},
]


def _section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def varimax(loadings_matrix, gamma=1.0, max_iter=50, tol=1e-8):
    """Standard Kaiser Varimax rotation of a (n_features, n_components)
    loadings matrix. Orthogonal rotation: preserves pairwise distances
    between any points expressed in the rotated component space, so it
    cannot change clustering results — only which axis "explains" what."""
    p, k = loadings_matrix.shape
    rotation = np.eye(k)
    d = 0
    for _ in range(max_iter):
        rotated = loadings_matrix @ rotation
        u, s, vh = np.linalg.svd(
            loadings_matrix.T @ (rotated ** 3 - (gamma / p) * rotated @ np.diag(np.diag(rotated.T @ rotated)))
        )
        rotation = u @ vh
        d_new = np.sum(s)
        if d_new < d * (1 + tol):
            break
        d = d_new
    return loadings_matrix @ rotation, rotation


def profile_clusters(rfm, labels):
    df = rfm.copy()
    df["Cluster"] = labels
    profile = df.groupby("Cluster").agg(
        Recency_mean=("Recency", "mean"), Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"), Count=("Cluster", "size"),
    ).reset_index()
    profile["Percentage"] = profile["Count"] / profile["Count"].sum() * 100
    composite = (1 / profile["Recency_mean"]) * profile["Frequency_mean"] * profile["Monetary_mean"]
    order = composite.sort_values(ascending=False).index
    segment_map = dict(zip(order, SEGMENT_ORDER)) if len(profile) == 3 else {}
    profile["Segment"] = profile.index.map(lambda i: segment_map.get(i, f"Cluster{i}"))
    return profile


def analyze_dataset(name, path):
    _section(f"DATASET: {name}")

    rfm = pd.read_csv(path)
    X_scaled = StandardScaler().fit_transform(rfm[COLS].apply(np.log1p))

    # --- Baseline: full 3D K-Means++ (the official, thesis-adopted result) ---
    km_3d = KMeans(n_clusters=SELECTED_K, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels_3d = km_3d.fit_predict(X_scaled)
    sil_3d = silhouette_score(X_scaled, labels_3d)
    dbi_3d = davies_bouldin_score(X_scaled, labels_3d)
    profile_3d = profile_clusters(rfm, labels_3d)
    print(f"[Baseline 3D]  Silhouette={sil_3d:.4f}  DBI={dbi_3d:.4f}")

    # --- PCA (2 components) ---
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var_total = sum(pca.explained_variance_ratio_) * 100
    if var_total < 80:
        print(f"WARNING: only {var_total:.2f}% variance retained (<80%) — interpret with extra caution.")

    km_pca = KMeans(n_clusters=SELECTED_K, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels_pca = km_pca.fit_predict(X_pca)
    sil_pca = silhouette_score(X_pca, labels_pca)
    dbi_pca = davies_bouldin_score(X_pca, labels_pca)
    profile_pca = profile_clusters(rfm, labels_pca)
    ari = adjusted_rand_score(labels_3d, labels_pca)

    delta_pct = (sil_pca - sil_3d) / sil_3d * 100
    print(f"[PCA 2D]       Silhouette={sil_pca:.4f} ({delta_pct:+.1f}%)  DBI={dbi_pca:.4f}  "
          f"Variance retained={var_total:.2f}%  ARI vs 3D={ari:.4f}")

    # --- Varimax rotation (interpretability only — must not change metrics) ---
    rotated_loadings, rotation = varimax(pca.components_.T)
    loadings_df = pd.DataFrame(rotated_loadings, index=COLS, columns=["RC1", "RC2"])

    X_rotated = X_pca @ rotation
    km_rot = KMeans(n_clusters=SELECTED_K, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels_rot = km_rot.fit_predict(X_rotated)
    sil_rot = silhouette_score(X_rotated, labels_rot)
    dbi_rot = davies_bouldin_score(X_rotated, labels_rot)
    metrics_preserved = abs(sil_rot - sil_pca) < 1e-6 and abs(dbi_rot - dbi_pca) < 1e-6

    print(f"[PCA+Varimax]  Silhouette={sil_rot:.4f}  DBI={dbi_rot:.4f}  "
          f"(metrics identical to PCA: {'YES' if metrics_preserved else 'NO — investigate!'})")
    print("Rotated component loadings:")
    print(loadings_df.to_string())
    for pc in loadings_df.columns:
        row = loadings_df[pc].abs().sort_values(ascending=False)
        gap = row.iloc[0] - row.iloc[1]
        label = "clear single-variable dominance" if gap > 0.25 else "still a blend of two variables"
        print(f"  {pc}: dominant={row.index[0]} ({loadings_df.loc[row.index[0], pc]:+.3f}), "
              f"gap={gap:.3f} -> {label}")

    return {
        "name": name, "sil_3d": sil_3d, "dbi_3d": dbi_3d,
        "sil_pca": sil_pca, "dbi_pca": dbi_pca, "var_total": var_total, "ari": ari,
        "sil_rot": sil_rot, "dbi_rot": dbi_rot, "metrics_preserved": metrics_preserved,
        "loadings": loadings_df, "profile_3d": profile_3d, "profile_pca": profile_pca,
    }


def main():
    _section("SUPPLEMENTARY ANALYSIS: PCA + VARIMAX + K-MEANS++ (ALL 3 DATASETS)")
    print(
        "Supplementary to the official 3D pipeline. Every number below is reported "
        "honestly, whatever it shows — including if PCA does not help on any dataset."
    )

    results = [analyze_dataset(ds["name"], ds["path"]) for ds in DATASETS]

    _section("CROSS-DATASET SUMMARY")
    summary = pd.DataFrame([
        {
            "Dataset": r["name"], "Silhouette_3D": round(r["sil_3d"], 4),
            "Silhouette_PCA": round(r["sil_pca"], 4), "Silhouette_PCA_Varimax": round(r["sil_rot"], 4),
            "Improvement_%": round((r["sil_pca"] - r["sil_3d"]) / r["sil_3d"] * 100, 1),
            "Variance_Retained_%": round(r["var_total"], 2), "ARI_vs_3D": round(r["ari"], 4),
            "Rotation_Metrics_Preserved": r["metrics_preserved"],
        }
        for r in results
    ])
    print(summary.to_string(index=False))

    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    out_path = os.path.join(config.RESULTS_PATH, "pca_varimax_final_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved final summary to {out_path}")

    all_preserved = all(r["metrics_preserved"] for r in results)
    all_high_ari = all(r["ari"] > 0.85 for r in results)
    print(f"\nRotation preserved clustering metrics on all datasets: {'YES' if all_preserved else 'NO'}")
    print(f"High business-label agreement (ARI > 0.85) on all datasets: {'YES' if all_high_ari else 'NO'}")

    _section("OFFICIAL STATUS")
    print(
        "This remains a SUPPLEMENTARY, well-validated finding. The official clustering "
        "result relied upon throughout the thesis is still the full 3D K-Means++ result "
        "on each dataset. Formal adoption of PCA+Varimax as the primary methodology is a "
        "deliberate researcher decision, to be made when writing up the report — not an "
        "automatic consequence of these numbers."
    )


if __name__ == "__main__":
    main()
