# -*- coding: utf-8 -*-
"""IJIES second review, items 2, 7 and the out-of-bag part of item 3.

What this answers
-----------------
Item 2: the K=3 versus K=4 difference in BOTH Silhouette and Davies-Bouldin,
        with a 95% paired-bootstrap interval, on each of the four additional
        datasets. UCI is run too, under the identical protocol, so the five
        appear on one footing and so this script's UCI column can be checked
        against the published bootstrap_silhouette_k3_vs_k4.csv.
Item 7: an equivalence margin of +/- 0.01 Silhouette, declared in advance, and
        each dataset's paired interval judged against it.
Item 3 (last part): an out-of-bag comparison for UCI - the paired difference
        re-evaluated on the customers a resample did NOT draw.

Protocol, identical to rfm_bootstrap_silhouette_uncertainty.py
--------------------------------------------------------------
  * estimand: the difference between two configurations fitted on the SAME
    bootstrap resample of customers, so the pairing is within-resample.
  * log1p and StandardScaler are fitted ONCE on the full dataset; the resample
    draws rows of that already-scaled matrix. They are NOT refitted per
    resample. This is the existing protocol and is kept so the numbers are
    comparable; its consequence is documented rather than changed.
  * resample indices: numpy default_rng(42), n draws with replacement from n.
  * KMeans(init="k-means++", n_init=10, random_state=42+b) for resample b.
  * interval: the 2.5th and 97.5th percentiles of the paired difference.
  * Silhouette via rfm_scoring.silhouette, the project's single estimator:
    exact up to 20,000 customers, and on Olist (93,358) an estimate on a
    10,000-point subsample. Within one resample the SAME subsample serves both
    K values, so the pairing is exact even where the level is an estimate; the
    subsample seed moves with the iteration, so the interval carries that
    sampling noise rather than hiding it.
  * Davies-Bouldin is exact everywhere - it needs no pairwise distance matrix.

Writes only outputs/results/rev2_* files. Changes no existing result.

Usage:  python rev2_paired_bootstrap.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler

import config
from rfm_scoring import silhouette

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
N_BOOTSTRAP = 100
EQUIV_MARGIN = 0.01          # item 7: declared before the intervals are read
COLS = ["Recency", "Frequency", "Monetary"]
R = config.RESULTS_PATH

# The five datasets the manuscript's tables cover. Dunnhumby is deliberately
# absent: it is not in the manuscript, and adding it here would answer a
# question no reviewer asked.
DATASETS = [
    ("UCI", os.path.join(R, "pca_official", "rfm_raw.csv")),
    ("Online Retail II", os.path.join(R, "retail2", "rfm_raw_retail2.csv")),
    ("Olist", os.path.join(R, "olist", "rfm_raw_olist.csv")),
    ("Superstore", os.path.join(R, "superstore", "rfm_raw_superstore.csv")),
    ("Bakery", os.path.join(R, "bakery", "rfm_raw_bakery.csv")),
]


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78, flush=True)


def ci(series):
    return float(series.quantile(0.025)), float(series.quantile(0.975))


def verdict(lo, hi, margin=EQUIV_MARGIN):
    """Three outcomes, not two: a wide interval is not evidence of equivalence."""
    if lo > 0 or hi < 0:
        inside = (abs(lo) < margin and abs(hi) < margin)
        return ("different but within the margin" if inside
                else "different and outside the margin")
    if abs(lo) < margin and abs(hi) < margin:
        return "equivalent within the margin"
    return "inconclusive: interval wider than the margin"


per_iter, summary = [], []

for name, path in DATASETS:
    _section("%s" % name)
    if not os.path.exists(path):
        print("  MISSING %s - skipped" % path)
        continue
    rfm = pd.read_csv(path)
    n = len(rfm)
    X = StandardScaler().fit_transform(rfm[COLS].apply(np.log1p))
    print("  %s: %d customers" % (path, n), flush=True)

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        seed = RANDOM_STATE + b

        km3 = KMeans(n_clusters=3, init="k-means++", random_state=seed, n_init=10)
        lab3 = km3.fit_predict(Xb)
        km4 = KMeans(n_clusters=4, init="k-means++", random_state=seed, n_init=10)
        lab4 = km4.fit_predict(Xb)

        # one subsample seed per iteration, shared by both K values
        s3 = silhouette(Xb, lab3, seed=seed)
        s4 = silhouette(Xb, lab4, seed=seed)
        d3 = davies_bouldin_score(Xb, lab3)
        d4 = davies_bouldin_score(Xb, lab4)

        row = dict(Dataset=name, Iteration=b + 1, N=n,
                   Silhouette_K3=s3, Silhouette_K4=s4,
                   Diff_Sil_K4_minus_K3=s4 - s3,
                   DBI_K3=d3, DBI_K4=d4, Diff_DBI_K4_minus_K3=d4 - d3)

        # Out-of-bag, UCI only (item 3): score the customers this resample did
        # not draw, using the labels the resample's model assigns them. The
        # evaluation set then shares no row with the fitting set, so duplicate
        # rows cannot inflate cohesion.
        if name == "UCI":
            oob = np.setdiff1d(np.arange(n), np.unique(idx))
            Xo = X[oob]
            o3, o4 = km3.predict(Xo), km4.predict(Xo)
            row.update(OOB_n=len(oob),
                       OOB_Silhouette_K3=silhouette(Xo, o3, seed=seed),
                       OOB_Silhouette_K4=silhouette(Xo, o4, seed=seed),
                       OOB_DBI_K3=davies_bouldin_score(Xo, o3),
                       OOB_DBI_K4=davies_bouldin_score(Xo, o4))
            row["OOB_Diff_Sil_K4_minus_K3"] = (row["OOB_Silhouette_K4"]
                                               - row["OOB_Silhouette_K3"])
            row["OOB_Diff_DBI_K4_minus_K3"] = row["OOB_DBI_K4"] - row["OOB_DBI_K3"]
        rows.append(row)
        if (b + 1) % 20 == 0:
            print("    ...%d/%d" % (b + 1, N_BOOTSTRAP), flush=True)

    df = pd.DataFrame(rows)
    per_iter.append(df)

    for metric, col, better in [("Silhouette", "Diff_Sil_K4_minus_K3", "higher"),
                                ("Davies-Bouldin", "Diff_DBI_K4_minus_K3", "lower")]:
        d = df[col]
        lo, hi = ci(d)
        summary.append(dict(
            Dataset=name, N=n, Metric=metric, Better=better,
            Mean_K3=float(df["Silhouette_K3" if metric == "Silhouette" else "DBI_K3"].mean()),
            Mean_K4=float(df["Silhouette_K4" if metric == "Silhouette" else "DBI_K4"].mean()),
            Mean_diff=float(d.mean()), SD_diff=float(d.std(ddof=1)),
            CI95_low=lo, CI95_high=hi,
            Pct_K4_better=float((d > 0).mean() * 100) if better == "higher"
            else float((d < 0).mean() * 100),
            CI_excludes_zero=bool(lo > 0 or hi < 0),
            Equivalence_margin=EQUIV_MARGIN if metric == "Silhouette" else np.nan,
            Equivalence_verdict=(verdict(lo, hi) if metric == "Silhouette"
                                 else "not applicable: the margin is in Silhouette units"),
            Silhouette_estimator=("estimate (10,000-point subsample)"
                                  if metric == "Silhouette" and n > 20000 else
                                  ("exact" if metric == "Silhouette" else "exact")),
            Scope="in-bag (full resample)"))
        print("  %-15s K3 %.4f  K4 %.4f  diff %+.4f  95%% CI [%+.4f, %+.4f]  %s"
              % (metric, summary[-1]["Mean_K3"], summary[-1]["Mean_K4"],
                 summary[-1]["Mean_diff"], lo, hi,
                 summary[-1]["Equivalence_verdict"]), flush=True)

    if name == "UCI":
        for metric, col in [("Silhouette", "OOB_Diff_Sil_K4_minus_K3"),
                            ("Davies-Bouldin", "OOB_Diff_DBI_K4_minus_K3")]:
            d = df[col]
            lo, hi = ci(d)
            better = "higher" if metric == "Silhouette" else "lower"
            summary.append(dict(
                Dataset=name, N=n, Metric=metric, Better=better,
                Mean_K3=float(df["OOB_Silhouette_K3" if metric == "Silhouette"
                                 else "OOB_DBI_K3"].mean()),
                Mean_K4=float(df["OOB_Silhouette_K4" if metric == "Silhouette"
                                 else "OOB_DBI_K4"].mean()),
                Mean_diff=float(d.mean()), SD_diff=float(d.std(ddof=1)),
                CI95_low=lo, CI95_high=hi,
                Pct_K4_better=float((d > 0).mean() * 100) if better == "higher"
                else float((d < 0).mean() * 100),
                CI_excludes_zero=bool(lo > 0 or hi < 0),
                Equivalence_margin=EQUIV_MARGIN if metric == "Silhouette" else np.nan,
                Equivalence_verdict=(verdict(lo, hi) if metric == "Silhouette"
                                     else "not applicable: the margin is in Silhouette units"),
                Silhouette_estimator="exact" if metric == "Silhouette" else "exact",
                Scope="out-of-bag (%.0f customers on average)" % df.OOB_n.mean()))
            print("  OOB %-11s K3 %.4f  K4 %.4f  diff %+.4f  95%% CI [%+.4f, %+.4f]"
                  % (metric, summary[-1]["Mean_K3"], summary[-1]["Mean_K4"],
                     summary[-1]["Mean_diff"], lo, hi), flush=True)

out_iter = os.path.join(R, "rev2_paired_bootstrap_per_iteration.csv")
out_sum = os.path.join(R, "rev2_paired_bootstrap_summary.csv")
pd.concat(per_iter, ignore_index=True).to_csv(out_iter, index=False)
pd.DataFrame(summary).to_csv(out_sum, index=False)

# ------------------------------------------------- check against the published file
_section("CHECK: does the UCI Silhouette column reproduce the published run?")
pub = os.path.join(R, "bootstrap_silhouette_k3_vs_k4.csv")
if os.path.exists(pub):
    old = pd.read_csv(pub)
    new = pd.concat(per_iter, ignore_index=True)
    new = new[new.Dataset == "UCI"]
    m = old.merge(new, on="Iteration", suffixes=("_pub", "_new"))
    for col_pub, col_new in [("Silhouette_K3", "Silhouette_K3"),
                             ("Silhouette_K4", "Silhouette_K4"),
                             ("Diff_K4_minus_K3", "Diff_Sil_K4_minus_K3")]:
        diff = (m[col_pub + "_pub" if col_pub == col_new else col_pub]
                - m[col_new + "_new" if col_pub == col_new else col_new]).abs().max()
        print("  max |published - recomputed| on %-22s %.3e" % (col_pub, diff))
else:
    print("  %s not found" % pub)

print()
print("written: %s" % out_iter)
print("written: %s" % out_sum)
