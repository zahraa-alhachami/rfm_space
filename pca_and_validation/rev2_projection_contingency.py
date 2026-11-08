# -*- coding: utf-8 -*-
"""IJIES second review, item 1: contingency tables and optimal agreement.

For UCI, each two-variable projection (R-F, R-M, F-M) and the two-component
PCA is compared against the three-dimensional partition by:

  * the full 3x3 contingency table of customer counts, and
  * the optimal agreement rate: the cluster labelling of one partition is a
    naming, not an ordering, so the two partitions are matched by the
    assignment that maximises the diagonal (Hungarian algorithm on the
    contingency table) and the agreement is that diagonal over n.

The adjusted Rand index already published in projection2d_agreement.csv is
recomputed here as a cross-check: it answers a different question (agreement
corrected for chance, over all pairs of customers) and is the stricter number,
so both are reported rather than one standing for the other.

Protocol identical to rfm_projection2d_analysis.py, which produced the
published values: log1p -> StandardScaler (fitted on the columns the
projection uses) -> KMeans(n_clusters=3, init="k-means++", n_init=10,
random_state=42), on outputs/results/pca_official/rfm_raw.csv. The script
re-derives the manuscript's four reference values first and stops if any of
them fails to reproduce.

Writes only outputs/results/rev2_* files. Changes no existing result.

Usage:  python rev2_projection_contingency.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
K = 3
N_PERMUTATIONS = 1000
COLS = ["Recency", "Frequency", "Monetary"]
R = config.RESULTS_PATH
RAW = os.path.join(R, "pca_official", "rfm_raw.csv")
PROJECTIONS = [("R-F", ["Recency", "Frequency"]),
               ("R-M", ["Recency", "Monetary"]),
               ("F-M", ["Frequency", "Monetary"])]


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def scaled(df, cols):
    return StandardScaler().fit_transform(df[cols].apply(np.log1p))


def fit(X):
    km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    return km.fit_predict(X)


def contingency(a, b):
    """Rows: the 3D partition. Columns: the alternative partition."""
    tab = np.zeros((K, K), dtype=int)
    for i in range(K):
        for j in range(K):
            tab[i, j] = int(np.sum((a == i) & (b == j)))
    return tab


def optimal_agreement(tab):
    """Hungarian matching: maximise the diagonal, then read the agreement off it."""
    rows, cols = linear_sum_assignment(-tab)        # maximise
    matched = int(tab[rows, cols].sum())
    return matched, list(zip(rows.tolist(), cols.tolist()))


rfm = pd.read_csv(RAW)
n = len(rfm)
_section("STEP 0: RE-DERIVE THE MANUSCRIPT'S REFERENCE VALUES")
print("Loaded %s: %d customers" % (RAW, n))

X3 = scaled(rfm, COLS)
labels_3d = fit(X3)
X_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X3)
labels_pca = fit(X_pca)

checks = [("3D Silhouette", silhouette_score(X3, labels_3d), 0.3365),
          ("3D Davies-Bouldin", davies_bouldin_score(X3, labels_3d), 1.0483),
          ("PCA Silhouette", silhouette_score(X_pca, labels_pca), 0.3754),
          ("PCA Davies-Bouldin", davies_bouldin_score(X_pca, labels_pca), 0.9506),
          ("ARI(3D, PCA)", adjusted_rand_score(labels_3d, labels_pca), 0.9345)]
ok = True
for label, got, want in checks:
    good = abs(got - want) < 1e-4
    ok &= good
    print("  %-22s %.4f  (manuscript %.4f)  %s"
          % (label, got, want, "OK" if good else "MISMATCH"))
if not ok:
    raise SystemExit("reference values did not reproduce - stopping rather than "
                     "reporting agreement rates from a different fit")

_section("CONTINGENCY TABLES AND OPTIMAL AGREEMENT, EACH ALTERNATIVE vs 3D")
alternatives = [(name, fit(scaled(rfm, cols)), "+".join(cols))
                for name, cols in PROJECTIONS]
alternatives.append(("PCA", labels_pca, "PC1+PC2 of the 3D space"))

cells, summary = [], []
for name, labels, basis in alternatives:
    tab = contingency(labels_3d, labels)
    matched, pairs = optimal_agreement(tab)
    agreement = 100.0 * matched / n
    ari = adjusted_rand_score(labels_3d, labels)

    print("\n%s  (%s)" % (name, basis))
    print("        " + "".join("%10s" % ("alt %d" % j) for j in range(K)) + "%10s" % "row sum")
    for i in range(K):
        print("  3D %d " % i + "".join("%10d" % tab[i, j] for j in range(K))
              + "%10d" % tab[i].sum())
    print("  sum  " + "".join("%10d" % tab[:, j].sum() for j in range(K)) + "%10d" % n)
    print("  matched pairs (3D -> alt): %s"
          % ", ".join("%d->%d" % (i, j) for i, j in pairs))
    print("  optimal agreement: %d / %d = %.1f%%     ARI = %.4f"
          % (matched, n, agreement, ari))

    # What would this agreement be if the two partitions were unrelated?
    # An optimally matched agreement cannot go below 1/K and rises with
    # unbalanced cluster sizes, so the raw percentage means nothing without its
    # own chance level. The alternative's labels are shuffled, which keeps both
    # partitions' cluster sizes exactly and destroys only the correspondence.
    rng = np.random.default_rng(RANDOM_STATE)
    null = np.empty(N_PERMUTATIONS)
    for p in range(N_PERMUTATIONS):
        null[p] = optimal_agreement(contingency(labels_3d,
                                                rng.permutation(labels)))[0] / n
    null_mean = 100.0 * null.mean()
    null_hi = 100.0 * np.quantile(null, 0.95)
    excess = agreement - null_mean
    print("  chance level, labels shuffled %d times: mean %.1f%%, 95th pct %.1f%%"
          % (N_PERMUTATIONS, null_mean, null_hi))
    print("  so the agreement is %.1f points above chance, and the span from "
          "chance to perfect is %.1f points" % (excess, 100.0 - null_mean))

    for i in range(K):
        for j in range(K):
            cells.append(dict(Alternative=name, Basis=basis, Cluster_3D=i,
                              Cluster_alt=j, Customers=int(tab[i, j]),
                              Matched_by_Hungarian=bool((i, j) in pairs)))
    summary.append(dict(Alternative=name, Basis=basis, N=n,
                        Matched_customers=matched,
                        Optimal_agreement_pct=agreement,
                        Disagreeing_customers=n - matched,
                        Chance_agreement_pct=null_mean,
                        Chance_agreement_p95_pct=null_hi,
                        Excess_over_chance_points=excess,
                        Share_of_available_span_pct=100.0 * excess / (100.0 - null_mean),
                        ARI_vs_3D=ari,
                        Matching="; ".join("3D %d -> alt %d" % (i, j) for i, j in pairs)))

sm = pd.DataFrame(summary)
out_cells = os.path.join(R, "rev2_projection_contingency_cells.csv")
out_sum = os.path.join(R, "rev2_projection_agreement.csv")
pd.DataFrame(cells).to_csv(out_cells, index=False)
sm.to_csv(out_sum, index=False)

_section("SUMMARY")
print(sm[["Alternative", "Optimal_agreement_pct", "Chance_agreement_pct",
          "Excess_over_chance_points", "Share_of_available_span_pct",
          "Disagreeing_customers", "ARI_vs_3D"]].round(2).to_string(index=False))

# the published ARI file is the cross-check for the ARI column
pub = os.path.join(R, "projection2d_agreement.csv")
if os.path.exists(pub):
    old = pd.read_csv(pub)
    print()
    print("cross-check against projection2d_agreement.csv:")
    for _, r in old.iterrows():
        mine = sm[sm.Alternative == r.Projection]
        if len(mine):
            print("  %-4s published ARI %.4f   recomputed %.4f   |diff| %.2e"
                  % (r.Projection, r.ARI_vs_3D, mine.ARI_vs_3D.iloc[0],
                     abs(r.ARI_vs_3D - mine.ARI_vs_3D.iloc[0])))

print()
print("written: %s" % out_cells)
print("written: %s" % out_sum)
