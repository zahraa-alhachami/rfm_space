"""
OFFICIAL thesis pipeline, PCA + K-Means++ variant.

Per the supervisor's direction, PCA + K-Means++ (2D) is now the adopted
official model for the thesis, replacing the original full-dimensional
(3D) K-Means++ result as the primary reported result. This script produces
the complete official output set (segment profiles, revenue/temporal/
geographic/product analyses, marketing recommendations, figures, K-Means
vs K-Means++ stability comparison) under this newly adopted methodology.

Design choice: this script does NOT modify rfm_analysis.py. It imports its
already-tested, unchanged functions (load/preprocess/RFM/transform/profile/
visualize/marketing/revenue/temporal/geographic/product/algorithm-
comparison) and only replaces the clustering INPUT (2D PCA-projected data
instead of the raw 3D scaled data). This keeps the original 3D result
fully intact and independently reproducible at outputs/results/ /
outputs/figures/ (the historical baseline, still auditable), while this
script's outputs land in a separate outputs/results/pca_official/ and
outputs/figures/pca_official/ tree -- nothing here overwrites prior work.

Run with:
    python rfm_analysis_pca_official.py
"""

import os
import sys

import config

# Redirect all downstream output paths BEFORE calling any imported function
# (every function in rfm_analysis.py reads config.RESULTS_PATH /
# config.FIGURES_PATH at call time, so reassigning here is sufficient and
# does not require touching rfm_analysis.py itself).
config.RESULTS_PATH = os.path.join("outputs", "results", "pca_official") + os.sep
config.FIGURES_PATH = os.path.join("outputs", "figures", "pca_official") + os.sep

from sklearn.decomposition import PCA

# rfm_analysis.py is this script's library, and it lives in preprocessing/ while
# this script lives in pca_and_validation/. Running from the package root puts
# the root on sys.path but not its subfolders, so the sibling folder is added
# here. This changes no computation: the imported functions are unmodified.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "preprocessing"))

from rfm_analysis import (
    _section,
    load_data,
    preprocess_data,
    compute_rfm,
    transform_rfm,
    find_optimal_k,
    run_kmeans,
    profile_clusters,
    visualize_clusters,
    marketing_recommendations,
    revenue_analysis,
    temporal_analysis,
    geographic_analysis,
    product_analysis,
    algorithm_comparison,
    SELECTED_K,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42


def main():
    for path in (config.RESULTS_PATH, config.FIGURES_PATH):
        os.makedirs(path, exist_ok=True)

    _section("OFFICIAL MODEL: PCA + K-MEANS++ (adopted per supervisor direction)")
    print(
        "This run replaces the original 3D K-Means++ result as the thesis's primary "
        "reported result. The original 3D result remains untouched and available at "
        "outputs/results/ / outputs/figures/ for audit and comparison."
    )

    df = load_data()
    df = preprocess_data(df)
    rfm = compute_rfm(df)
    rfm_scaled, scaler = transform_rfm(rfm)

    _section("STEP D2: PCA DIMENSIONALITY REDUCTION (2 COMPONENTS)")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    rfm_pca = pca.fit_transform(rfm_scaled)
    var_ratio = pca.explained_variance_ratio_
    print(f"PC1 explained variance: {var_ratio[0] * 100:.2f}%")
    print(f"PC2 explained variance: {var_ratio[1] * 100:.2f}%")
    print(f"Total variance retained: {var_ratio.sum() * 100:.2f}%")

    loadings_path = os.path.join(config.RESULTS_PATH, "pca_loadings.csv")
    import pandas as pd
    loadings = pd.DataFrame(
        pca.components_, columns=["Recency", "Frequency", "Monetary"], index=["PC1", "PC2"]
    )
    loadings.to_csv(loadings_path)
    print(f"Saved PCA loadings to {loadings_path}")

    # K-selection is now conducted IN THE ACTUAL CLUSTERING SPACE (2D PCA),
    # not the original 3D space, since that is what K-Means++ is fit on.
    k_results = find_optimal_k(rfm_pca)
    optimal_k = k_results["selected_k"]
    assert optimal_k == SELECTED_K == 3

    rfm, kmeans = run_kmeans(rfm_pca, rfm, optimal_k)
    rfm, profile = profile_clusters(rfm)
    visualize_clusters(rfm)
    marketing_recommendations(rfm, df)
    revenue_analysis(rfm, df)
    temporal_analysis(rfm, df)
    geographic_analysis(rfm, df)
    product_analysis(rfm, df)
    algorithm_comparison(rfm_pca, rfm)

    _section("OFFICIAL PCA + K-MEANS++ PIPELINE COMPLETE")
    print(f"All outputs saved under {config.FIGURES_PATH} and {config.RESULTS_PATH}")


if __name__ == "__main__":
    main()
