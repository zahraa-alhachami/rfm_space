# -*- coding: utf-8 -*-
"""IJIES second review, item 13: the Bisecting K-Means configuration, and 0.3567 vs 0.3561.

The manuscript reports 0.3567. Running scikit-learn's BisectingKMeans with its
defaults at random_state=42 gives 0.3561. The gap is one parameter: the project
passes n_init=10 and the library's default is n_init=1. This script proves that
by sweeping n_init with everything else held, and quantifies exactly how many
customers change cluster between the two answers.

It also prints the complete configuration - every parameter of the estimator,
marked as set by the project or left at the library default - and the bootstrap
procedure used for the comparison, so none of it has to be inferred from the
call site.

Writes only outputs/results/rev2_* files. Changes no existing result.

Usage:  python rev2_bisecting_config.py
"""
import inspect
import io
import os
import sys

import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RANDOM_STATE = 42
K = 3
COLS = ["Recency", "Frequency", "Monetary"]
R = config.RESULTS_PATH
RAW = os.path.join(R, "pca_official", "rfm_raw.csv")
PROJECT_CALL = dict(n_clusters=K, n_init=10, random_state=RANDOM_STATE)

rfm = pd.read_csv(RAW)
n = len(rfm)
X = StandardScaler().fit_transform(np.log1p(rfm[COLS].to_numpy(float)))

# ------------------------------------------------------- the full configuration
print("=" * 104)
print("1. THE COMPLETE CONFIGURATION  (scikit-learn %s, UCI n=%d, log1p + StandardScaler)"
      % (sklearn.__version__, n))
print("=" * 104)
print("  the project's call: BisectingKMeans(%s)"
      % ", ".join("%s=%r" % kv for kv in PROJECT_CALL.items()))
print()
print("  %-22s %-24s %s" % ("parameter", "effective value", "where it comes from"))
print("  " + "-" * 94)
rows = []
for name, p in inspect.signature(BisectingKMeans.__init__).parameters.items():
    if name == "self":
        continue
    if name in PROJECT_CALL:
        value, source = PROJECT_CALL[name], "SET BY THE PROJECT"
    else:
        value, source = p.default, "library default, not set by the project"
    print("  %-22s %-24r %s" % (name, value, source))
    rows.append(dict(Parameter=name, Effective_value=repr(value), Source=source,
                     Sklearn_version=sklearn.__version__))
pd.DataFrame(rows).to_csv(os.path.join(R, "rev2_bisecting_full_config.csv"),
                          index=False)
print("""
  The four the review asks about, explicitly:
    bisecting_strategy  'biggest_inertia'  - LIBRARY DEFAULT, never set by the
                        project. It splits the cluster with the largest inertia.
                        The alternative, 'largest_cluster', splits by size and
                        gives 0.2824 on this data - worse than K-Means++ - so
                        this unstated default is load-bearing.
    n_init              10                 - SET by the project; the library
                        default is 1. This is the whole of the 0.3567/0.3561 gap.
    max_iter            300                - library default, for the inner
                        2-means on each split, not the number of splits.
    random_state        42                 - set by the project, as everywhere.
    init                'random'           - library default. K-Means++ in the
                        same comparison was given init='k-means++'; measured
                        separately, this difference is worth 0.0002 (see
                        rev2_bisecting_runs.csv), so it is not the explanation
                        for anything.""")

# ------------------------------------------------------------- the n_init sweep
print()
print("=" * 104)
print("2. WHY 0.3567 AND NOT 0.3561: THE n_init SWEEP, EVERYTHING ELSE HELD")
print("=" * 104)
print("  %-9s %12s %12s %-20s %10s %12s"
      % ("n_init", "Silhouette", "DBI", "cluster sizes", "ARI vs 1", "relabelled"))
base_labels, sweep = None, []
for ni in (1, 2, 3, 5, 10, 20, 50):
    lab = BisectingKMeans(n_clusters=K, n_init=ni,
                          random_state=RANDOM_STATE).fit_predict(X)
    sizes = np.bincount(lab, minlength=K)
    if base_labels is None:
        base_labels, ari, moved = lab, 1.0, 0
    else:
        ari = adjusted_rand_score(base_labels, lab)
        # match the two labellings optimally, then count who actually moved
        tab = np.zeros((K, K), int)
        for i in range(K):
            for j in range(K):
                tab[i, j] = int(np.sum((base_labels == i) & (lab == j)))
        r, c = linear_sum_assignment(-tab)
        moved = n - int(tab[r, c].sum())
    print("  %-9d %12.4f %12.4f %-20s %10.4f %12s"
          % (ni, silhouette_score(X, lab), davies_bouldin_score(X, lab),
             "/".join(str(int(s)) for s in sizes), ari,
             "-" if ni == 1 else "%d customers" % moved))
    sweep.append(dict(n_init=ni, Silhouette=silhouette_score(X, lab),
                      DBI=davies_bouldin_score(X, lab),
                      Cluster_sizes="/".join(str(int(s)) for s in sizes),
                      ARI_vs_n_init_1=ari,
                      Customers_relabelled_vs_n_init_1=moved))
sw = pd.DataFrame(sweep)
sw.to_csv(os.path.join(R, "rev2_bisecting_n_init_sweep.csv"), index=False)

lo = float(sw[sw.n_init == 1].Silhouette.iloc[0])
hi = float(sw[sw.n_init == 10].Silhouette.iloc[0])
moved10 = int(sw[sw.n_init == 10].Customers_relabelled_vs_n_init_1.iloc[0])
print("""
  So: %.4f at the library default and %.4f from n_init=3 upward, a difference of
  %+.4f. BisectingKMeans runs its inner 2-means n_init times per split and keeps
  the best by inertia, so a larger n_init searches more and settles on a
  slightly different second split. The two answers are nearly the same
  partition - ARI %.4f, with %d of %d customers (%.2f%%) assigned differently -
  and the manuscript's 0.3567 is the n_init=10 one because n_init=10 is the
  project's fixed setting for every algorithm it runs, K-Means++ included.

  Neither number is wrong. What would be wrong is quoting one of them while
  describing the configuration as "scikit-learn defaults", which is why the
  table in section 1 lists every parameter and its origin.""" % (
    lo, hi, hi - lo, float(sw[sw.n_init == 10].ARI_vs_n_init_1.iloc[0]),
    moved10, n, 100.0 * moved10 / n))

# ------------------------------------------------------- the bootstrap procedure
print()
print("=" * 104)
print("3. THE BOOTSTRAP PROCEDURE FOR THIS COMPARISON")
print("=" * 104)
print("""  script        rfm_bootstrap_kmeanspp_vs_bisecting.py
  design        paired: both algorithms are refitted on the SAME resample, and
                the statistic is Silhouette(K-Means++) - Silhouette(Bisecting)
  B             100 resamples, drawn with replacement at full size (n = %d)
  indices       numpy default_rng(42) - deliberately the same index sequence as
                the K=3/K=4 study, so the two studies' resamples match
  seeds         random_state = 42 + b for BOTH models in resample b
  n_init        10 for both models, the project's fixed setting
  space         log1p + StandardScaler, 3D - NOT the PCA space, so the
                comparison is between algorithms and not between spaces
  scaling       fitted once on the full data; the resample draws rows of the
                already-scaled matrix, so the interval excludes the
                standardisation's own uncertainty
  evaluation    the resample itself, duplicates included
  interval      percentile: the 2.5th and 97.5th of the 100 differences
  result        mean +0.0023, median -0.0129, 95%% CI [-0.0314, +0.0602],
                K-Means++ ahead in 28%% of resamples
  caveat        that difference is BIMODAL, so neither the mean nor the
                percentile interval describes a typical resample: Bisecting
                holds its solution in 72 resamples (Silhouette 0.3564, smallest
                cluster 10.3%%, ARI 0.9721 with its own full-data partition) and
                collapses to a different split in 28 (0.2851, smallest cluster
                28.85%%, ARI 0.4826). Per-resample detail is in
                rev2_bisecting_bootstrap_detail.csv.""" % n)

print()
for f in ("rev2_bisecting_full_config.csv", "rev2_bisecting_n_init_sweep.csv"):
    print("written: %s" % os.path.join(R, f))
