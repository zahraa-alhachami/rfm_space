"""
Exploratory analysis (fully independent of rfm_analysis.py and every other
prior script — reads their saved CSVs but modifies nothing): tests the
STABILITY of the official K-Means++ (K=3) clustering solution under
resampling, using bootstrap resampling + Adjusted Rand Index (ARI).

This addresses a limitation explicitly named in the thesis reports
("stability under resampling not yet tested" — see Limitation 3 of the
practical-work report). It is not an attempt to change or improve the
Silhouette Score; it answers a different question entirely: if a slightly
different sample of customers had been drawn, would the SAME three
segments (Champions / At-Risk / Lost) still emerge, with the SAME
customers assigned to them?

Method
------
1. Reproduce the official K-Means++ (K=3) fit on the full dataset exactly
   (sanity check against the recorded baseline Silhouette Score).
2. For b = 1..B: draw a bootstrap resample (same size n, sampled WITH
   replacement) from the scaled RFM data, fit a fresh K-Means++ (K=3) model
   on that resample only.
3. Use the resulting bootstrap centroids to assign a cluster label to
   EVERY customer in the ORIGINAL (full, non-resampled) dataset — i.e. each
   customer is assigned to its nearest bootstrap centroid.
4. Compute the Adjusted Rand Index (ARI) between this bootstrap-derived
   labeling of the full customer base and the reference (full-data) K-Means++
   labeling. ARI is invariant to arbitrary cluster-label permutation, so no
   manual label alignment is needed.
5. Repeat for B bootstrap iterations and report the distribution of ARI
   (mean, std, min, max, median).

This is run independently on all three primary datasets (UCI, Online Retail
II, Olist) for full generalization, consistent with every other exploratory
path in this thesis.

Decision criteria (fixed BEFORE running, not adjusted after seeing results):
    Mean ARI >= 0.75  -> "Strong stability"   (segments are robust to resampling)
    Mean ARI 0.50-0.75 -> "Moderate stability" (3-segment structure holds;
                                                 some individual assignments
                                                 are resampling-sensitive)
    Mean ARI <  0.50  -> "Weak/unstable"
These bands are informal reference points commonly used when interpreting
ARI in clustering-stability contexts; they are not a claim of formal
statistical significance and are stated here before any result is computed.

Isolation guarantee: only reads from outputs/results/*.csv, never writes
into files with existing names, never imports or modifies any prior script.
All new outputs use a "bootstrap_" prefix.

Run with:
    python rfm_bootstrap_stability_analysis.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
N_BOOTSTRAP = 100
COLS = ["Recency", "Frequency", "Monetary"]

# Established baseline Silhouette (K=3, 3D, full data) from prior sessions,
# reproduced here READ-ONLY as a sanity check before proceeding.
BASELINE_SIL_3D = {"UCI": 0.3365, "Online Retail II": 0.3477, "Olist": 0.4203}

DATASETS = [
    {"name": "UCI", "path": os.path.join(config.RESULTS_PATH, "rfm_raw.csv")},
    {"name": "Online Retail II", "path": os.path.join(config.RESULTS_PATH, "retail2", "rfm_raw_retail2.csv")},
    {"name": "Olist", "path": os.path.join(config.RESULTS_PATH, "olist", "rfm_raw_olist.csv")},
]


def _section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_scaled(path):
    rfm = pd.read_csv(path)
    X_scaled = StandardScaler().fit_transform(rfm[COLS].apply(np.log1p))
    return rfm, X_scaled


def stage1_reproduce_baseline(dataset_name, X_scaled):
    reference = KMeans(n_clusters=3, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    reference_labels = reference.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, reference_labels)
    recorded = BASELINE_SIL_3D[dataset_name]
    diff = abs(sil - recorded)

    print(f"Reproduced Silhouette: {sil:.4f}")
    print(f"Previously recorded:   {recorded:.4f}")
    print(f"Absolute difference:   {diff:.4f}")

    if diff > 0.01:
        print("\nSTOP CONDITION TRIGGERED: reproduction differs by more than 0.01 from the "
              "recorded value. Halting for this dataset — would need investigation before "
              "trusting the bootstrap result.")
        return None, None
    print("Reproduction confirmed within tolerance. Proceeding to bootstrap stage.")
    return reference, reference_labels


def bootstrap_stability(X_scaled, reference_labels, dataset_name, n_bootstrap=N_BOOTSTRAP):
    _section(f"BOOTSTRAP STABILITY — {dataset_name} (B={n_bootstrap})")

    n = X_scaled.shape[0]
    rng = np.random.default_rng(RANDOM_STATE)
    ari_scores = []

    for b in range(n_bootstrap):
        boot_idx = rng.integers(0, n, size=n)
        X_boot = X_scaled[boot_idx]

        boot_model = KMeans(n_clusters=3, init="k-means++", random_state=RANDOM_STATE + b, n_init=10)
        boot_model.fit(X_boot)

        # Assign EVERY original customer to its nearest bootstrap centroid.
        labels_full_via_boot = boot_model.predict(X_scaled)
        ari = adjusted_rand_score(reference_labels, labels_full_via_boot)
        ari_scores.append(ari)

        if (b + 1) % 20 == 0:
            print(f"  ...{b + 1}/{n_bootstrap} bootstrap iterations complete")

    ari_scores = np.array(ari_scores)
    mean_ari = ari_scores.mean()
    std_ari = ari_scores.std()
    median_ari = np.median(ari_scores)
    min_ari = ari_scores.min()
    max_ari = ari_scores.max()
    pct_above_075 = (ari_scores >= 0.75).mean() * 100

    if mean_ari >= 0.75:
        verdict = "Strong stability"
    elif mean_ari >= 0.50:
        verdict = "Moderate stability"
    else:
        verdict = "Weak/unstable"

    print(f"\nMean ARI:    {mean_ari:.4f}")
    print(f"Std ARI:     {std_ari:.4f}")
    print(f"Median ARI:  {median_ari:.4f}")
    print(f"Min / Max:   {min_ari:.4f} / {max_ari:.4f}")
    print(f"% runs with ARI >= 0.75: {pct_above_075:.1f}%")
    print(f"Verdict (fixed thresholds, decided before running): {verdict}")

    return {
        "Dataset": dataset_name, "B": n_bootstrap,
        "Mean_ARI": round(mean_ari, 4), "Std_ARI": round(std_ari, 4),
        "Median_ARI": round(median_ari, 4), "Min_ARI": round(min_ari, 4),
        "Max_ARI": round(max_ari, 4), "Pct_ARI_ge_0.75": round(pct_above_075, 1),
        "Verdict": verdict,
    }, ari_scores


def main():
    _section("EXPLORATORY: BOOTSTRAP RESAMPLING STABILITY (ARI)")
    print(
        "Tests whether the official K-Means++ (K=3) segment assignments are stable "
        "under resampling. Addresses the previously documented 'stability under "
        "resampling not yet tested' limitation. Does not modify any prior script or "
        "saved result. Every outcome reported honestly."
    )
    print(f"\nDecision criteria (fixed before running):")
    print("  Mean ARI >= 0.75  -> Strong stability")
    print("  Mean ARI 0.50-0.75 -> Moderate stability")
    print("  Mean ARI <  0.50  -> Weak/unstable")

    summary_rows = []
    raw_rows = []

    for ds in DATASETS:
        _section(f"DATASET: {ds['name']}")
        rfm, X_scaled = load_scaled(ds["path"])
        print(f"Customers: {len(rfm)}")

        reference, reference_labels = stage1_reproduce_baseline(ds["name"], X_scaled)
        if reference is None:
            continue

        summary, ari_scores = bootstrap_stability(X_scaled, reference_labels, ds["name"])
        summary_rows.append(summary)
        for i, a in enumerate(ari_scores):
            raw_rows.append({"Dataset": ds["name"], "Iteration": i + 1, "ARI": round(float(a), 4)})

    _section("FINAL SUMMARY — BOOTSTRAP STABILITY ACROSS ALL DATASETS")
    summary_table = pd.DataFrame(summary_rows)
    print(summary_table.to_string(index=False))

    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    out_summary = os.path.join(config.RESULTS_PATH, "bootstrap_ari_summary.csv")
    summary_table.to_csv(out_summary, index=False)
    print(f"\nSaved {out_summary}")

    raw_table = pd.DataFrame(raw_rows)
    out_raw = os.path.join(config.RESULTS_PATH, "bootstrap_ari_raw.csv")
    raw_table.to_csv(out_raw, index=False)
    print(f"Saved {out_raw}")

    _section("INTERPRETATION")
    for row in summary_rows:
        print(f"{row['Dataset']}: mean ARI={row['Mean_ARI']:.4f} -> {row['Verdict']}")


if __name__ == "__main__":
    main()
