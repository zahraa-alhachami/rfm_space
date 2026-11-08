# -*- coding: utf-8 -*-
"""IJIES second review, item 3: every bootstrap in the project, documented apart.

One row per bootstrap analysis, answering the five questions asked, with the
script that implements it so each answer can be checked:

  estimand            what quantity the resampling is an interval for
  refit policy        whether log1p, the scaler and PCA are recomputed inside
                      each resample, or fixed once on the full data
  evaluated rows      which observations the statistic is computed on - the
                      resample itself (duplicates included) or the original
                      customers
  interval            how the interval is formed
  Olist               whether Olist went through this same procedure

B and the dataset list in each row are read from the output file rather than
from the script's constant, so a stale constant cannot make this table wrong.

One finding is uniform and worth stating once: in EVERY analysis, log1p and
StandardScaler are fitted once on the full dataset and the resample draws rows
of the already-transformed matrix. So no interval here carries the uncertainty
of the standardisation itself. PCA is different - two analyses refit it inside
each resample and one holds the projection fixed - and that difference is
recorded per row.

Writes only outputs/results/rev2_* files. Changes no existing result.

Usage:  python rev2_bootstrap_provenance.py
"""
import io
import os
import sys

import pandas as pd

import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
R = config.RESULTS_PATH

FIXED_SCALING = ("log1p and StandardScaler fitted ONCE on the full dataset; the "
                 "resample draws rows of that matrix")

ANALYSES = [
    dict(
        Analysis="Stability ARI per dataset",
        Script="rfm_bootstrap_stability_analysis.py",
        Files=["bootstrap_ari_raw.csv", "bootstrap_ari_summary.csv"],
        Estimand="the expected ARI between the reference partition and a partition "
                 "refitted on a resample of the same size",
        Scaling=FIXED_SCALING,
        PCA="not applicable - the 3D space only",
        Evaluated_rows="the ORIGINAL customers: the resample's model predicts every "
                       "customer of the full dataset, and the ARI is computed there, "
                       "so no customer is counted twice",
        Interval="no interval is published; mean, SD, median, min, max and the share "
                 "of resamples at or above 0.75",
        Seeds="numpy default_rng(42) for the indices; KMeans random_state=42+b",
        Olist="YES - same procedure, B=100"),
    dict(
        Analysis="K=3 vs K=4 paired difference (UCI)",
        Script="rfm_bootstrap_silhouette_uncertainty.py",
        Files=["bootstrap_silhouette_k3_vs_k4.csv"],
        Estimand="Silhouette(K=4) - Silhouette(K=3), both fitted on the SAME resample",
        Scaling=FIXED_SCALING,
        PCA="not used in this arm",
        Evaluated_rows="the RESAMPLE itself, duplicates included. Duplicated customers "
                       "sit at distance zero from each other, which raises cohesion, so "
                       "this level is optimistic; the paired difference cancels most of "
                       "that but not all of it. The out-of-bag check in "
                       "rev2_paired_bootstrap.py exists for this reason",
        Interval="percentile: the 2.5th and 97.5th of the 100 differences",
        Seeds="numpy default_rng(42) for the indices; KMeans random_state=42+b",
        Olist="NO - UCI only. rev2_paired_bootstrap.py extends it to Olist and to "
              "three other datasets"),
    dict(
        Analysis="PCA vs 3D paired difference (UCI)",
        Script="rfm_bootstrap_silhouette_uncertainty.py",
        Files=["bootstrap_silhouette_pca_vs_3d.csv"],
        Estimand="Silhouette(PCA partition, scored in PCA) - Silhouette(3D partition, "
                 "scored in 3D) on the same resample - i.e. the gain a paper would "
                 "report, NOT the partition effect",
        Scaling=FIXED_SCALING,
        PCA="REFIT inside every resample (PCA(2).fit_transform of the resample)",
        Evaluated_rows="the resample itself, duplicates included - and the two terms "
                       "are scored in DIFFERENT spaces, which is the conflation the "
                       "common-space decomposition exists to separate",
        Interval="percentile: the 2.5th and 97.5th of the 100 differences",
        Seeds="numpy default_rng(42) for the indices; KMeans random_state=42+b; "
              "PCA random_state=42",
        Olist="NO - UCI only"),
    dict(
        Analysis="Partition effect in a common space (UCI)",
        Script="rfm_commonspace_pca_test.py",
        Files=["commonspace_bootstrap.csv"],
        Estimand="Silhouette(PCA partition scored in 3D) - Silhouette(3D partition "
                 "scored in 3D): both terms in ONE space, so this is the partition "
                 "effect and not the reported gain",
        Scaling=FIXED_SCALING,
        PCA="REFIT inside every resample, then its labels are scored in the 3D space",
        Evaluated_rows="the resample itself, duplicates included; both terms on the "
                       "same rows in the same space",
        Interval="percentile: the 2.5th and 97.5th of the 100 differences",
        Seeds="numpy default_rng(42) for the indices; KMeans random_state=42+b",
        Olist="NO - UCI only"),
    dict(
        Analysis="K-Means++ vs Bisecting K-Means (UCI)",
        Script="rfm_bootstrap_kmeanspp_vs_bisecting.py",
        Files=["bootstrap_kmeanspp_vs_bisecting.csv"],
        Estimand="Silhouette(K-Means++) - Silhouette(BisectingKMeans) at K=3 on the "
                 "same resample",
        Scaling=FIXED_SCALING,
        PCA="not used - the 3D space, deliberately",
        Evaluated_rows="the resample itself, duplicates included",
        Interval="percentile - but the difference is BIMODAL (the comparator collapses "
                 "to a different split in 28 of 100 resamples), so the interval and the "
                 "mean both fall between two modes. See rev2_bisecting_bootstrap_detail.csv",
        Seeds="numpy default_rng(42), the same indices as the K=3/K=4 study; "
              "random_state=42+b for both models",
        Olist="NO - UCI only"),
    dict(
        Analysis="Stability inside the K-selection criteria",
        Script="rfm_k_selection_null_corrected.py (function stability)",
        Files=["ksel_criteria_by_k.csv"],
        Estimand="raw stability = mean ARI(refit on resample, predicted on all "
                 "customers; reference labels). Corrected = (raw - null)/(1 - null) "
                 "where the null is a random labelling with the reference's exact "
                 "cluster sizes",
        Scaling=FIXED_SCALING,
        PCA="both spaces are analysed, but each is built once on the full data and "
            "held fixed across resamples",
        Evaluated_rows="the ORIGINAL customers, as in the stability analysis above",
        Interval="no interval; the mean over 100 resamples is the statistic",
        Seeds="numpy default_rng(42); KMeans random_state=42+b",
        Olist="NO - excluded from the whole stage, because the Dunn index and the gap "
              "statistic need the full pairwise distance matrix at n=93,358"),
    dict(
        Analysis="K=2 sensitivity (UCI)",
        Script="rfm_k2_sensitivity_uci.py",
        Files=["k2_sensitivity_bootstrap_ari.csv"],
        Estimand="the expected ARI between the K=2 reference partition and a K=2 refit "
                 "on a resample",
        Scaling=FIXED_SCALING,
        PCA="not used",
        Evaluated_rows="the ORIGINAL customers",
        Interval="no interval; mean and SD",
        Seeds="numpy default_rng(42); KMeans random_state=42+b",
        Olist="NO - UCI only"),
    dict(
        Analysis="LRFM replication stability (Olist and UCI)",
        Script="rfm_lrfm_olist_paper2.py (function bootstrap_ari)",
        Files=["lrfm_paper2_bootstrap_ari.csv"],
        Estimand="the expected ARI between the reference partition and a refit on a "
                 "resample, for the LRFM feature set",
        Scaling=FIXED_SCALING,
        PCA="not used",
        Evaluated_rows="the ORIGINAL customers",
        Interval="no interval; mean and SD",
        Seeds="numpy default_rng(42); KMeans random_state=42+b",
        Olist="YES - this analysis is primarily about Olist"),
    dict(
        Analysis="Projection stability (UCI)",
        Script="rfm_projection2d_analysis.py",
        Files=["projection2d_stability.csv"],
        Estimand="the expected ARI between a projection's reference partition and a "
                 "refit on a resample, inside that same projection",
        Scaling=FIXED_SCALING + " - and the scaler is fitted on the TWO columns the "
                                "projection uses, not on all three",
        PCA="the projection is fixed, built once on the full data; only KMeans is refit",
        Evaluated_rows="the ORIGINAL customers",
        Interval="no interval; mean, SD and min",
        Seeds="numpy default_rng(42); KMeans random_state=42+b",
        Olist="NO - UCI only"),
    dict(
        Analysis="K=3 vs K=4 across five datasets, with out-of-bag (NEW, this review)",
        Script="rev2_paired_bootstrap.py",
        Files=["rev2_paired_bootstrap_per_iteration.csv",
               "rev2_paired_bootstrap_summary.csv"],
        Estimand="Silhouette(K=4) - Silhouette(K=3) and DBI(K=4) - DBI(K=3) on the "
                 "same resample, per dataset; and for UCI the same differences "
                 "evaluated only on the customers the resample did NOT draw",
        Scaling=FIXED_SCALING + " (kept identical to the published protocol so the "
                                "numbers are comparable)",
        PCA="not used - the 3D space, as in the published K=3/K=4 study",
        Evaluated_rows="both: the resample itself, AND the out-of-bag customers for "
                       "UCI, which share no row with the fitting set",
        Interval="percentile: the 2.5th and 97.5th of the 100 differences, plus an "
                 "equivalence verdict against a +/- 0.01 Silhouette margin declared "
                 "in advance",
        Seeds="numpy default_rng(42) per dataset; KMeans random_state=42+b; the "
              "Silhouette subsample seed also moves with b",
        Olist="YES - and its Silhouette is an ESTIMATE on a 10,000-point subsample "
              "(n=93,358 > the 20,000 exact threshold in rfm_scoring.py). Within one "
              "resample the same subsample serves both K values, so the pairing is "
              "exact even though the level is an estimate"),
]

rows = []
print("=" * 104)
print("EVERY BOOTSTRAP IN THE PROJECT, ONE BLOCK EACH")
print("=" * 104)
for a in ANALYSES:
    b_obs, datasets = [], []
    for f in a["Files"]:
        p = os.path.join(R, f)
        if not os.path.exists(p):
            b_obs.append("%s: MISSING" % f)
            continue
        d = pd.read_csv(p)
        if "B" in d.columns:
            b_obs.append("%s: B=%s" % (f, ", ".join(str(int(v))
                                                    for v in sorted(d.B.unique()))))
        elif "Iteration" in d.columns:
            if "Dataset" in d.columns:
                per = d.groupby("Dataset").Iteration.max()
                b_obs.append("%s: %s" % (f, ", ".join("%s B=%d" % (k, int(v))
                                                      for k, v in per.items())))
            else:
                b_obs.append("%s: B=%d" % (f, int(d.Iteration.max())))
        else:
            b_obs.append("%s: %d rows" % (f, len(d)))
        if "Dataset" in d.columns:
            datasets += sorted(map(str, d.Dataset.unique()))
    a = dict(a)
    a["B_observed_in_file"] = "; ".join(b_obs)
    a["Datasets_in_file"] = ", ".join(sorted(set(datasets))) if datasets else \
        "not recorded in the file"
    a["Files"] = ", ".join(a["Files"])
    rows.append(a)

    print("\n%s" % a["Analysis"])
    print("   script          : %s" % a["Script"])
    print("   files           : %s" % a["Files"])
    print("   B (from file)   : %s" % a["B_observed_in_file"])
    print("   datasets        : %s" % a["Datasets_in_file"])
    print("   estimand        : %s" % a["Estimand"])
    print("   scaling refit   : %s" % a["Scaling"])
    print("   PCA refit       : %s" % a["PCA"])
    print("   evaluated rows  : %s" % a["Evaluated_rows"])
    print("   interval        : %s" % a["Interval"])
    print("   seeds           : %s" % a["Seeds"])
    print("   Olist           : %s" % a["Olist"])

out = os.path.join(R, "rev2_bootstrap_provenance.csv")
pd.DataFrame(rows)[["Analysis", "Script", "Files", "B_observed_in_file",
                    "Datasets_in_file", "Estimand", "Scaling", "PCA",
                    "Evaluated_rows", "Interval", "Seeds", "Olist"]].to_csv(
    out, index=False)

print()
print("=" * 104)
print("WHAT IS UNIFORM, AND WHAT IS NOT")
print("=" * 104)
print("""  uniform : B = 100 everywhere; numpy default_rng(42) for the indices;
            KMeans random_state = 42 + b; log1p and the scaler fitted once on
            the full data and never inside a resample.
  NOT uniform:
    - the evaluation set. The ARI-stability analyses score the ORIGINAL
      customers; the Silhouette-difference analyses score the RESAMPLE, so
      duplicated customers contribute to cohesion. Only the new out-of-bag
      arm scores customers the resample never drew.
    - PCA. Refit inside each resample in the two UCI PCA analyses; held fixed
      in the projection analysis; absent elsewhere.
    - Olist. Present in the stability ARI (B=100) and in the LRFM replication;
      absent from every UCI Silhouette-difference bootstrap until this review;
      excluded from the K-selection stage for a stated computational reason.
      Wherever Olist's Silhouette is computed, it is a 10,000-point subsample
      estimate, not an exact value.""")
print()
print("written: %s" % out)
