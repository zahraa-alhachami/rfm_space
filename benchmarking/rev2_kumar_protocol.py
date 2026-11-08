# -*- coding: utf-8 -*-
"""IJIES second review, item 12: the Kumar protocol, and why it reports ~0.65.

Six cells: two algorithms (K-Means++ and Bisecting K-Means, K=3) across three
preprocessing pipelines on the SAME UCI RFM table:

    P1  raw RFM -> Min-Max                 (the Kumar protocol)
    P2  log1p   -> Min-Max
    P3  log1p   -> StandardScaler          (this project's pipeline)

Each cell is scored in its own space, which is what a paper prints. Then every
cell's labels are re-scored in ONE reference space (P3), which separates the two
things the printed number confuses: whether the partition is better, and whether
the ruler changed. That is the same decomposition the manuscript applies to PCA,
applied here to preprocessing.

Two diagnostics are reported beside the indices, because they are the mechanism:
the smallest cluster's share, and how much of the unit cube the Min-Max data
actually occupy. A Silhouette near 0.65 on three variables usually means a few
extreme customers have been given their own clusters while everyone else is
compressed into one corner - which is a statement about the preprocessing, not
about the segmentation.

Writes only outputs/results/rev2_* files. Changes no existing result.

Usage:  python rev2_kumar_protocol.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RANDOM_STATE = 42
K = 3
COLS = ["Recency", "Frequency", "Monetary"]
R = config.RESULTS_PATH
RAW = os.path.join(R, "pca_official", "rfm_raw.csv")
REFERENCE = "P3 log1p + StandardScaler"

rfm = pd.read_csv(RAW)
n = len(rfm)
V = rfm[COLS].to_numpy(float)

SPACES = {
    # The control: the same raw, unlogged variables as P1, but standardised
    # instead of Min-Max scaled. It separates the two ways a preprocessing
    # choice can inflate the index, and they are not the same failure.
    "P0 raw + StandardScaler (no log)": StandardScaler().fit_transform(V),
    "P1 raw + Min-Max": MinMaxScaler().fit_transform(V),
    "P2 log1p + Min-Max": MinMaxScaler().fit_transform(np.log1p(V)),
    REFERENCE: StandardScaler().fit_transform(np.log1p(V)),
}
METHODS = {
    "K-Means++": lambda: KMeans(n_clusters=K, init="k-means++", n_init=10,
                                random_state=RANDOM_STATE),
    "Bisecting K-Means": lambda: BisectingKMeans(n_clusters=K, n_init=10,
                                                 random_state=RANDOM_STATE),
}

print("=" * 112)
print("UCI, %d customers, K=%d, random_state=%d, n_init=10 for both algorithms"
      % (n, K, RANDOM_STATE))
print("=" * 112)

# ---------------------------------------------- what the three spaces look like
print("\nWHAT THE PREPROCESSING DOES TO THE DATA")
print("-" * 112)
print("  %-26s %-28s %10s %10s %10s" % ("space", "column", "median", "p99",
                                        "share below 0.1"))
shape_rows = []
for name, X in SPACES.items():
    for j, col in enumerate(COLS):
        x = X[:, j]
        if "Min-Max" in name:
            share = float((x < 0.1).mean())
        else:                      # standardised: the comparable statement is below
            share = float((x < np.quantile(x, 0.0) + 0.1 * np.ptp(x)).mean())
        print("  %-26s %-28s %10.4f %10.4f %9.1f%%"
              % (name, col, np.median(x), np.quantile(x, 0.99), 100 * share))
        shape_rows.append(dict(Space=name, Column=col, Median=float(np.median(x)),
                               P99=float(np.quantile(x, 0.99)),
                               Share_in_lowest_tenth_of_range=share))
    if "Min-Max" in name:
        corner = float((X < 0.1).all(axis=1).mean())
        print("  %-26s %-28s %9.1f%% of customers lie inside the [0,0.1] corner "
              "of all three axes" % (name, "ALL THREE", 100 * corner))

pd.DataFrame(shape_rows).to_csv(os.path.join(R, "rev2_kumar_space_shape.csv"),
                                index=False)

# ------------------------------------------------------------------ the six cells
labels = {}
rows = []
for space_name, X in SPACES.items():
    for method_name, make in METHODS.items():
        lab = make().fit_predict(X)
        labels[(space_name, method_name)] = (lab, X)
        sizes = np.bincount(lab, minlength=K)
        rows.append(dict(
            Space=space_name, Method=method_name, N=n,
            Silhouette_own_space=silhouette_score(X, lab),
            DBI_own_space=davies_bouldin_score(X, lab),
            Smallest_cluster_n=int(sizes.min()),
            Smallest_cluster_pct=100.0 * sizes.min() / n,
            Cluster_sizes=" / ".join(str(int(s)) for s in sizes)))

df = pd.DataFrame(rows)
print()
print("=" * 112)
print("WHAT EACH CELL PRINTS - scored in its own space, as a paper would report it")
print("=" * 112)
print("  %-26s %-19s %12s %12s %12s %22s"
      % ("space", "method", "Silhouette", "DBI", "smallest %", "cluster sizes"))
for _, r in df.iterrows():
    print("  %-26s %-19s %12.4f %12.4f %11.2f%% %22s"
          % (r.Space, r.Method, r.Silhouette_own_space, r.DBI_own_space,
             r.Smallest_cluster_pct, r.Cluster_sizes))

# ------------------------------------------- the same labels in one common space
Xref = SPACES[REFERENCE]
base_lab = labels[(REFERENCE, "K-Means++")][0]
base_sil = silhouette_score(Xref, base_lab)
base_dbi = davies_bouldin_score(Xref, base_lab)

dec = []
for (space_name, method_name), (lab, _X) in labels.items():
    sil_own = float(df[(df.Space == space_name)
                       & (df.Method == method_name)].Silhouette_own_space.iloc[0])
    sil_ref = silhouette_score(Xref, lab)
    dbi_ref = davies_bouldin_score(Xref, lab)
    dec.append(dict(
        Space=space_name, Method=method_name,
        Silhouette_own_space=sil_own, Silhouette_in_reference=sil_ref,
        DBI_in_reference=dbi_ref,
        Reported_gain=sil_own - base_sil,
        Partition_effect=sil_ref - base_sil,
        Space_effect=sil_own - sil_ref,
        Space_share_of_gain_pct=(100 * (sil_own - sil_ref) / (sil_own - base_sil))
        if abs(sil_own - base_sil) > 1e-12 else np.nan,
        ARI_vs_baseline=adjusted_rand_score(base_lab, lab),
        Survives_in_common_space=bool(sil_ref - base_sil > 0)))

dd = pd.DataFrame(dec)
print()
print("=" * 112)
print("THE SAME LABELS, ALL SCORED IN %s" % REFERENCE)
print("baseline = K-Means++ in that space: Silhouette %.4f, DBI %.4f"
      % (base_sil, base_dbi))
print("=" * 112)
print("  %-26s %-19s %11s %11s %11s %11s %9s %7s"
      % ("space", "method", "own space", "in ref", "gain", "partition",
         "space", "ARI"))
for _, r in dd.iterrows():
    print("  %-26s %-19s %11.4f %11.4f %+11.4f %+11.4f %+9.4f %7.4f"
          % (r.Space, r.Method, r.Silhouette_own_space, r.Silhouette_in_reference,
             r.Reported_gain, r.Partition_effect, r.Space_effect, r.ARI_vs_baseline))

df.merge(dd, on=["Space", "Method", "Silhouette_own_space"], how="outer").to_csv(
    os.path.join(R, "rev2_kumar_protocol.csv"), index=False)

# ------------------------------------------------ what P1 is actually clustering
# The first draft of this script explained P1's high Silhouette by degenerate
# clusters. That was wrong: its smallest cluster holds 626 customers, 14.4% of
# the file. The real mechanism is measured here instead.
print()
print("=" * 112)
print("WHAT P1 IS ACTUALLY CLUSTERING ON")
print("=" * 112)
X1 = SPACES["P1 raw + Min-Max"]
var = X1.var(axis=0, ddof=1)
print("  variance carried by each column after raw Min-Max:")
collapse = []
for col, v, share in zip(COLS, var, var / var.sum()):
    print("     %-10s %.6f   %5.1f%% of the total" % (col, v, 100 * share))
    collapse.append(dict(Quantity="variance share after raw Min-Max", Column=col,
                         Value=float(share)))
print("""
  Min-Max divides each column by its own maximum, and the RFM maxima are not
  comparable: Recency tops out in the hundreds of days while Frequency and
  Monetary have extreme upper tails. After the transform, Recency carries
  %.1f%% of the total variance and the other two carry %.1f%% between them. The
  three-variable space is, for distance purposes, one variable.""" % (
    100 * var[0] / var.sum(), 100 * (var[1] + var[2]) / var.sum()))

lab_p1 = labels[("P1 raw + Min-Max", "K-Means++")][0]
print("\n  the P1 K-Means++ centroids, in RAW units:")
for c in range(K):
    m = V[lab_p1 == c].mean(axis=0)
    print("     cluster %d  n=%4d   Recency %7.1f   Frequency %6.2f   Monetary %10.2f"
          % (c, int((lab_p1 == c).sum()), m[0], m[1], m[2]))

r_only = X1[:, [0]]
lab_r = KMeans(n_clusters=K, init="k-means++", n_init=10,
               random_state=RANDOM_STATE).fit_predict(r_only)
sil_r = silhouette_score(r_only, lab_r)
ari_r = adjusted_rand_score(lab_p1, lab_r)
print("""
  And the decisive test: cluster the min-max Recency column ALONE, one
  dimension, K=3. It scores Silhouette %.4f - higher than P1's own %.4f - and
  agrees with P1's three-dimensional partition at ARI %.4f. The "RFM
  segmentation" under this protocol IS a Recency segmentation."""
      % (sil_r, float(df[(df.Space == "P1 raw + Min-Max")
                         & (df.Method == "K-Means++")].Silhouette_own_space.iloc[0]),
         ari_r))
collapse.append(dict(Quantity="Silhouette of min-max Recency alone, 1-D, K=3",
                     Column="Recency", Value=float(sil_r)))
collapse.append(dict(Quantity="ARI between that 1-D partition and P1's 3-D partition",
                     Column="all three", Value=float(ari_r)))
pd.DataFrame(collapse).to_csv(os.path.join(R, "rev2_kumar_variance_collapse.csv"),
                              index=False)

# --------------------------------------------------------------------- the reading
p1_km = dd[(dd.Space == "P1 raw + Min-Max") & (dd.Method == "K-Means++")].iloc[0]
p1_bk = dd[(dd.Space == "P1 raw + Min-Max")
           & (dd.Method == "Bisecting K-Means")].iloc[0]
p1_km_row = df[(df.Space == "P1 raw + Min-Max") & (df.Method == "K-Means++")].iloc[0]
p1_bk_row = df[(df.Space == "P1 raw + Min-Max")
               & (df.Method == "Bisecting K-Means")].iloc[0]
p2_km = dd[(dd.Space == "P2 log1p + Min-Max") & (dd.Method == "K-Means++")].iloc[0]
P0 = "P0 raw + StandardScaler (no log)"
p0_km = dd[(dd.Space == P0) & (dd.Method == "K-Means++")].iloc[0]
p0_bk = dd[(dd.Space == P0) & (dd.Method == "Bisecting K-Means")].iloc[0]
p0_km_row = df[(df.Space == P0) & (df.Method == "K-Means++")].iloc[0]
# the word "extreme" needs a measurement behind it, and "the same partition"
# needs an ARI rather than two equal Silhouettes
_l0 = labels[(P0, "K-Means++")][0]
_l0b = labels[(P0, "Bisecting K-Means")][0]
p0_ari_between = adjusted_rand_score(_l0, _l0b)
_smallest = int(np.bincount(_l0).argmin())
p0_outlier_monetary = float(V[_l0 == _smallest][:, 2].mean())
p0_file_p99_monetary = float(np.quantile(V[:, 2], 0.99))

# The two figures the control-arm paragraph quotes were computed for the
# narrative and never written down. They are written here from the same two
# variables that paragraph uses, so the sentence has a file behind it. No
# computation changes: p0_outlier_monetary and p0_file_p99_monetary are used
# exactly as they already stand.
pd.DataFrame([
    dict(Quantity="mean Monetary of the smallest cluster under P0 "
                  "(raw + StandardScaler, no log)",
         Customers=int((_l0 == _smallest).sum()),
         Share_pct=100.0 * int((_l0 == _smallest).sum()) / len(_l0),
         Value_GBP=p0_outlier_monetary),
    dict(Quantity="99th percentile of Monetary over the whole file",
         Customers=len(_l0), Share_pct=100.0,
         Value_GBP=p0_file_p99_monetary),
]).to_csv(os.path.join(R, "rev2_kumar_outlier_cluster.csv"), index=False)

# How bad is that partition effect, relative to everything else measured? The
# comparison is read from the decomposition file rather than asserted: an
# earlier version of this script called it "the worst of any configuration
# tested anywhere in this project", which was wrong - Olist at K=2 is lower.
lower = []
dec_path = os.path.join(R, "rev2_decomposition_2x2.csv")
if os.path.exists(dec_path):
    dec2 = pd.read_csv(dec_path)
    for _, q in dec2.iterrows():
        if float(q.Partition_ref3D) < float(p1_km.Partition_effect):
            lower.append("%s at K=%d (partition effect %+.4f)"
                         % (q.Dataset, int(q.K), float(q.Partition_ref3D)))
if not lower:
    worst_phrase = ("the lowest partition effect measured anywhere in this "
                    "project")
else:
    worst_phrase = ("among the worst; only %s %s lower"
                    % (" and ".join(lower), "is" if len(lower) == 1 else "are"))

print()
print("=" * 112)
print("WHY THE FIRST PIPELINE REPORTS ABOUT 0.65")
print("=" * 112)
print("""  K-Means++ on raw Min-Max prints Silhouette %.4f against %.4f for this
  project's pipeline - a gain of %+.4f, larger than any gain in the manuscript.
  It is not a better partition. Scored on common ground its partition effect is
  %+.4f: %s, and
  its agreement with the baseline partition is ARI %.4f. So the number rises by
  %+.4f while the segmentation it describes gets substantially worse.

  The mechanism is NOT degenerate clusters - the smallest holds %d customers,
  %.2f%% of the file, and sizes are %s. It is the variance collapse measured
  above: after dividing each column by its own maximum, Recency carries %.1f%%
  of the variance and the other two carry %.1f%% between them, so the distances
  are Recency distances. A three-way split of one spread-out variable scores
  near 0.65 by construction - clustering min-max Recency alone, in one
  dimension, scores %.4f and reproduces this partition at ARI %.4f.

  Bisecting K-Means under the same protocol prints %.4f with partition effect
  %+.4f: the same artefact, slightly smaller because its split sequence lands
  differently on the same one-dimensional structure.

  The control arm separates that from a different failure. P0 keeps the same
  unlogged variables and only swaps the scaler: StandardScaler divides each
  column by its own SD, so all three carry equal variance and nothing collapses
  onto Recency. It still prints %.4f - and the reason is visible in the sizes,
  %s. Its smallest cluster holds %d customers, %.2f%% of the file: with the raw
  skew intact, K-Means spends a whole cluster on a handful of extreme customers
  - that cluster's mean Monetary is %s against a 99th percentile of %s for the
  file - and a Silhouette computed with them as a separate cluster is high
  because they really are far from everything. Partition effect %+.4f, ARI with
  the baseline %.4f. Both algorithms land on exactly the SAME partition here
  (ARI %.4f between them, %.4f Silhouette each), which is what happens when one
  structure in the data is strong enough to leave no choice.

  So the two unlogged pipelines inflate the index for two different reasons -
  P1 by collapsing three variables into one, P0 by turning outliers into
  clusters - and neither is a better segmentation. What removes BOTH is the LOG,
  not the choice of scaler. Taking logs before
  Min-Max (P2) brings the index to %.4f with a partition effect of %+.4f, and
  the remaining %+.4f over the standardised pipeline is itself still a ruler
  change, not a better partition. So "Min-Max versus StandardScaler" is the
  wrong axis to argue about: the decisive step is whether skewed monetary and
  frequency variables are logged before any scaler is applied at all.""" % (
    p1_km.Silhouette_own_space, base_sil, p1_km.Reported_gain,
    p1_km.Partition_effect, worst_phrase, p1_km.ARI_vs_baseline,
    p1_km.Reported_gain,
    p1_km_row.Smallest_cluster_n, p1_km_row.Smallest_cluster_pct,
    p1_km_row.Cluster_sizes,
    100 * var[0] / var.sum(), 100 * (var[1] + var[2]) / var.sum(),
    sil_r, ari_r,
    p1_bk.Silhouette_own_space, p1_bk.Partition_effect,
    p0_km.Silhouette_own_space, p0_km_row.Cluster_sizes,
    p0_km_row.Smallest_cluster_n, p0_km_row.Smallest_cluster_pct,
    "{:,.0f}".format(p0_outlier_monetary),
    "{:,.0f}".format(p0_file_p99_monetary),
    p0_km.Partition_effect, p0_km.ARI_vs_baseline, p0_ari_between,
    p0_bk.Silhouette_own_space,
    p2_km.Silhouette_own_space, p2_km.Partition_effect, p2_km.Reported_gain))

print()
for f in ("rev2_kumar_protocol.csv", "rev2_kumar_space_shape.csv",
          "rev2_kumar_variance_collapse.csv",
          "rev2_kumar_outlier_cluster.csv"):
    print("written: %s" % os.path.join(R, f))
