"""
Exploratory analysis (fully independent of rfm_analysis.py and
rfm_pca_analysis.py — reads their saved CSVs but modifies nothing): tests
Fuzzy C-Means (FCM) clustering, first alone, then combined with PCA if
Stage 1 is promising enough to warrant it.

This is pure exploration for the researcher's own decision-making. Nothing
here is an official/adopted result. Every result is reported honestly,
including if FCM performs worse than the existing K-Means++ baseline.

Isolation guarantee: this script only reads from outputs/results/*.csv
(never writes into files with existing names) and never imports or
modifies rfm_analysis.py / rfm_pca_analysis.py. All new outputs are saved
with an "fcm_" prefix.

Run with:
    python rfm_fcm_analysis.py
"""

import os

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

import config

RANDOM_STATE = 42
C = 3  # number of fuzzy clusters, matches K=3 used throughout
FCM_M = 2.0  # standard fuzziness parameter
FCM_ERROR = 0.005
FCM_MAXITER = 1000

COLS = ["Recency", "Frequency", "Monetary"]
SEGMENT_ORDER = ["Champions", "At-Risk Customers", "Lost Customers"]

# Established baselines from prior sessions, reproduced here READ-ONLY for
# honest side-by-side comparison — not recomputed differently, not altered.
BASELINES = {
    "UCI": {"kmeans_3d": 0.3365, "pca_2d": 0.3754},
    "Online Retail II": {"kmeans_3d": 0.3477, "pca_2d": 0.3817},
    "Olist": {"kmeans_3d": 0.4203, "pca_2d": 0.4815},
}

DATASETS = [
    {"name": "UCI", "path": os.path.join(config.RESULTS_PATH, "rfm_raw.csv")},
    {"name": "Online Retail II", "path": os.path.join(config.RESULTS_PATH, "retail2", "rfm_raw_retail2.csv")},
    {"name": "Olist", "path": os.path.join(config.RESULTS_PATH, "olist", "rfm_raw_olist.csv")},
]

FCM_RESULTS_DIR = config.RESULTS_PATH  # new files use "fcm_" prefix, no overwrite risk


def _section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_fcm(X):
    """X: (n_samples, n_features). skfuzzy expects (n_features, n_samples)."""
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, c=C, m=FCM_M, error=FCM_ERROR, maxiter=FCM_MAXITER, seed=RANDOM_STATE
    )
    return cntr, u, fpc  # u shape: (c, n_samples)


def defuzzify(u):
    return np.argmax(u, axis=0)  # hard label per sample


def fuzzy_gap_classification(u):
    """For each customer, gap between top-1 and top-2 membership scores."""
    sorted_u = np.sort(u, axis=0)[::-1, :]  # descending per column
    top1, top2 = sorted_u[0, :], sorted_u[1, :]
    gap = top1 - top2

    category = np.where(gap > 0.5, "Clear", np.where(gap >= 0.2, "Moderate", "Fuzzy"))
    return gap, category


def most_fuzzy_pair(u, category, labels):
    """Among 'Fuzzy' customers, which pair of segments (by hard label vs
    runner-up label) do they split between most often?"""
    fuzzy_idx = np.where(category == "Fuzzy")[0]
    if len(fuzzy_idx) == 0:
        return "N/A (no fuzzy customers)"
    sorted_idx = np.argsort(-u, axis=0)  # descending, per column: cluster index order
    top1_cluster = sorted_idx[0, fuzzy_idx]
    top2_cluster = sorted_idx[1, fuzzy_idx]
    pairs = [tuple(sorted((a, b))) for a, b in zip(top1_cluster, top2_cluster)]
    pair_counts = pd.Series(pairs).value_counts()
    return pair_counts.index[0], pair_counts.iloc[0], len(fuzzy_idx)


def profile_and_name_segments(rfm, labels):
    df = rfm.copy()
    df["Cluster"] = labels
    profile = df.groupby("Cluster").agg(
        R=("Recency", "mean"), F=("Frequency", "mean"), M=("Monetary", "mean"), Count=("Cluster", "size"),
    ).reset_index()
    composite = (1 / profile["R"]) * profile["F"] * profile["M"]
    order = composite.sort_values(ascending=False).index
    segment_map = dict(zip(order, SEGMENT_ORDER)) if len(profile) == 3 else {}
    # segment_map keys are positional indices into `profile`, aligned with
    # profile.index (both derived from the same groupby result).
    profile["Segment"] = [segment_map.get(i, f"C{i}") for i in profile.index]
    # Re-key by actual Cluster label (what labels_hard contains) for lookup elsewhere.
    cluster_to_segment = dict(zip(profile["Cluster"], profile["Segment"]))
    return profile, cluster_to_segment


def stage1_and_2(name, path):
    _section(f"STAGE 1+2: {name} — FCM ON ORIGINAL 3D DATA")

    rfm = pd.read_csv(path)
    X_scaled = StandardScaler().fit_transform(rfm[COLS].apply(np.log1p))

    cntr, u, fpc = run_fcm(X_scaled)
    labels_hard = defuzzify(u)

    sil = silhouette_score(X_scaled, labels_hard)
    dbi = davies_bouldin_score(X_scaled, labels_hard)
    baseline_sil = BASELINES[name]["kmeans_3d"]

    print(f"Fuzzy Partition Coefficient (FPC, higher=crisper): {fpc:.4f}")
    print(f"\n{'Method':<28} | {'Silhouette':>10} | {'DBI':>8}")
    print(f"{'K-Means++ (original, 3D)':<28} | {baseline_sil:>10.4f} | {'—':>8}")
    print(f"{'FCM (after defuzzify)':<28} | {sil:>10.4f} | {dbi:>8.4f}")

    delta_pct = (sil - baseline_sil) / baseline_sil * 100
    if sil > baseline_sil:
        print(f"\nRESULT: FCM is BETTER by {delta_pct:+.1f}%.")
    elif sil > baseline_sil * 0.95:
        print(f"\nRESULT: FCM is close to baseline ({delta_pct:+.1f}%) — within 5% margin.")
    else:
        print(f"\nRESULT: FCM is WORSE by {delta_pct:+.1f}%. Reported honestly, no adjustment.")

    # Stage 2: fuzzy-customer analysis
    gap, category = fuzzy_gap_classification(u)
    cat_counts = pd.Series(category).value_counts()
    cat_pct = (cat_counts / len(category) * 100).round(2)

    print("\n--- STAGE 2: Membership fuzziness distribution ---")
    for cat in ["Clear", "Moderate", "Fuzzy"]:
        c = int(cat_counts.get(cat, 0))
        p = cat_pct.get(cat, 0.0)
        print(f"  {cat:<10}: {c:>6} customers ({p:.2f}%)")

    pair_info = most_fuzzy_pair(u, category, labels_hard)
    if pair_info != "N/A (no fuzzy customers)":
        pair, count, total_fuzzy = pair_info
        # Map cluster indices to segment names for readability
        _, segment_map = profile_and_name_segments(rfm, labels_hard)
        seg_a = segment_map.get(pair[0], f"Cluster{pair[0]}")
        seg_b = segment_map.get(pair[1], f"Cluster{pair[1]}")
        print(f"  Most common fuzzy pair: {seg_a} <-> {seg_b} ({count}/{total_fuzzy} of fuzzy customers)")
        pair_label = f"{seg_a} <-> {seg_b}"
    else:
        print(f"  {pair_info}")
        pair_label = pair_info

    return {
        "name": name, "sil": sil, "dbi": dbi, "fpc": fpc, "baseline_sil": baseline_sil,
        "clear_pct": cat_pct.get("Clear", 0.0), "moderate_pct": cat_pct.get("Moderate", 0.0),
        "fuzzy_pct": cat_pct.get("Fuzzy", 0.0), "most_fuzzy_pair": pair_label,
        "rfm": rfm, "X_scaled": X_scaled,
    }


def stage4_fcm_pca(name, rfm, X_scaled, fcm_3d_sil):
    _section(f"STAGE 4: {name} — FCM ON PCA-PROJECTED (2D) DATA")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var_total = sum(pca.explained_variance_ratio_) * 100

    cntr, u, fpc = run_fcm(X_pca)
    labels_hard = defuzzify(u)
    sil = silhouette_score(X_pca, labels_hard)
    dbi = davies_bouldin_score(X_pca, labels_hard)

    print(f"Variance retained (PC1+PC2): {var_total:.2f}%")
    print(f"\n{'Method':<28} | {'Silhouette':>10}")
    print(f"{'K-Means++ (3D, original)':<28} | {BASELINES[name]['kmeans_3d']:>10.4f}")
    print(f"{'PCA+K-Means++ (2D)':<28} | {BASELINES[name]['pca_2d']:>10.4f}")
    print(f"{'FCM (3D, Stage 1)':<28} | {fcm_3d_sil:>10.4f}")
    print(f"{'FCM+PCA (2D, this stage)':<28} | {sil:>10.4f}")

    gap, category = fuzzy_gap_classification(u)
    cat_pct = (pd.Series(category).value_counts() / len(category) * 100).round(2)
    print("\nFuzziness distribution after PCA:")
    for cat in ["Clear", "Moderate", "Fuzzy"]:
        print(f"  {cat:<10}: {cat_pct.get(cat, 0.0):.2f}%")

    return {"name": name, "sil": sil, "dbi": dbi, "var_total": var_total,
            "fuzzy_pct": cat_pct.get("Fuzzy", 0.0)}


def main():
    _section("EXPLORATORY: FUZZY C-MEANS (FCM) ANALYSIS")
    print(
        "Purely exploratory. Does not modify rfm_analysis.py, rfm_pca_analysis.py, "
        "or any existing saved result. Every outcome reported honestly."
    )

    stage1_2_results = []
    for ds in DATASETS:
        result = stage1_and_2(ds["name"], ds["path"])
        stage1_2_results.append(result)

    _section("STAGE 3: CROSS-DATASET SUMMARY (STAGES 1+2)")
    summary_1_2 = pd.DataFrame([
        {
            "Dataset": r["name"], "Silhouette_FCM": round(r["sil"], 4),
            "Baseline_KMeans_3D": r["baseline_sil"], "Clear_%": r["clear_pct"],
            "Moderate_%": r["moderate_pct"], "Fuzzy_%": r["fuzzy_pct"],
            "Most_Fuzzy_Pair": r["most_fuzzy_pair"],
        }
        for r in stage1_2_results
    ])
    print(summary_1_2.to_string(index=False))
    out1 = os.path.join(FCM_RESULTS_DIR, "fcm_stage1_2_summary.csv")
    summary_1_2.to_csv(out1, index=False)
    print(f"\nSaved {out1}")

    # Decision gate for Stage 4, based on UCI (primary dataset)
    uci_result = stage1_2_results[0]
    proceed_to_stage4 = uci_result["sil"] >= uci_result["baseline_sil"] * 0.95

    _section("DECISION GATE FOR STAGE 4")
    print(f"UCI FCM Silhouette (3D): {uci_result['sil']:.4f}")
    print(f"UCI K-Means++ baseline: {uci_result['baseline_sil']:.4f}")
    print(f"Threshold to proceed (>=95% of baseline): {uci_result['baseline_sil'] * 0.95:.4f}")

    if not proceed_to_stage4:
        print(
            "\nDECISION: NOT proceeding to Stage 4. FCM on 3D data underperforms the "
            "K-Means++ baseline by more than 5%, so combining it with PCA is unlikely "
            "to be a fruitful direction. Reported honestly — stopping here as instructed."
        )
        stage4_results = []
    else:
        print("\nDECISION: Proceeding to Stage 4 (FCM performance is close to or better than baseline).")
        stage4_results = [
            stage4_fcm_pca(r["name"], r["rfm"], r["X_scaled"], r["sil"])
            for r in stage1_2_results
        ]
        if stage4_results:
            summary_4 = pd.DataFrame([
                {"Dataset": r["name"], "Silhouette_FCM_PCA": round(r["sil"], 4),
                 "Variance_Retained_%": round(r["var_total"], 2), "Fuzzy_%_after_PCA": r["fuzzy_pct"]}
                for r in stage4_results
            ])
            out4 = os.path.join(FCM_RESULTS_DIR, "fcm_stage4_pca_summary.csv")
            summary_4.to_csv(out4, index=False)
            print(f"\nSaved {out4}")

    # ------------------------------------------------------------------
    # FINAL COMPREHENSIVE TABLE (all methods tested so far, UCI reference)
    # ------------------------------------------------------------------
    _section("FINAL COMPREHENSIVE COMPARISON TABLE (KNOWLEDGE ONLY — NOT AN OFFICIAL UPDATE)")
    uci_fcm_pca_sil = next((r["sil"] for r in stage4_results if r["name"] == "UCI"), None)

    rows = [
        ["K-Means++ (3D, official thesis result)", 0.3365, "Adopted in thesis — UNCHANGED"],
        ["PCA+K-Means++ (2D)", 0.3754, "Supplementary, tested previously — UNCHANGED"],
        ["PCA+Varimax", 0.3754, "Same numbers, better interpretability — UNCHANGED"],
        ["FCM (3D, after defuzzify)", round(uci_result["sil"], 4), "NEW this session"],
    ]
    if uci_fcm_pca_sil is not None:
        rows.append(["FCM+PCA (2D, after defuzzify)", round(uci_fcm_pca_sil, 4), "NEW this session"])
    else:
        rows.append(["FCM+PCA (2D)", "NOT TESTED", "Stage 1 result did not meet the threshold to proceed"])

    final_table = pd.DataFrame(rows, columns=["Method", "Silhouette (UCI)", "Status"])
    print(final_table.to_string(index=False))
    print(
        "\nNOTE: this table is for researcher knowledge and comparison only. It does "
        "NOT change the official result adopted in the thesis (K-Means++, 3D, 0.3365)."
    )

    _section("CENTRAL QUESTION: DOES ANY OFFICIAL-GRADE METHOD BEAT BOTH PAPERS AT ONCE?")
    anitha, shirole = 0.362, 0.442
    print(f"Anitha & Patil (2022): {anitha}    Shirole et al. (2021): {shirole}\n")
    for _, row in final_table.iterrows():
        val = row["Silhouette (UCI)"]
        if isinstance(val, (int, float)):
            beats_both = val > anitha and val > shirole
            print(f"  {row['Method']:<42}: {val:.4f}  "
                  f"{'BEATS BOTH' if beats_both else ('beats Anitha only' if val > anitha else 'beats neither')}")
        else:
            print(f"  {row['Method']:<42}: {val}")


if __name__ == "__main__":
    main()
