# -*- coding: utf-8 -*-
"""IJIES second review, item 6: the 2x2, and the decomposition with PCA as the reference.

The 2x2
-------
Two partitions (labels fitted in the 3D RFM space, labels fitted in the 2D PCA
space) scored in two spaces (3D, PCA) give four cells:

    A = Sil(3D labels,  scored in 3D)      B = Sil(3D labels,  scored in PCA)
    C = Sil(PCA labels, scored in 3D)      D = Sil(PCA labels, scored in PCA)

The reported gain a paper would print is D - A. It splits two ways, and both
are exact:

    reference 3D   :  partition = C - A,  space = D - C
    reference PCA  :  partition = D - B,  space = B - A

Each sums to D - A. The manuscript reports the first. A reviewer is entitled to
ask whether the conclusion is an artefact of choosing 3D as the reference, so
the second is computed here and the two are printed side by side.

Why all four cells are recomputed here rather than read from two files
----------------------------------------------------------------------
A and B are in proof_silhouette_sign.csv (exact, including Olist) and C and D
are in harness_decomposition.csv (Olist estimated on a 10,000-point subsample).
Stitching them would put an exact value and an estimate in one subtraction: the
two files' shared cell A agrees to the last digit on five datasets and differs
by 0.003625 on Olist for exactly that reason. So this script fits both
partitions itself, scores all four cells with the same estimator, and then
checks its own A and B against the proof file and its own C and D against the
harness.

Protocol: log1p -> StandardScaler fitted on the full dataset -> PCA(2) on that
scaled matrix -> KMeans(init="k-means++", n_init=10, random_state=42) in each
space, K in {2, 3, 4}. Silhouette exact everywhere, Olist included, because a
subsample draw can move a difference of this size.

Writes only outputs/results/rev2_* files. Changes no existing result.

Usage:  python rev2_decomposition_2x2.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import config
from rfm_scoring import silhouette

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
K_VALUES = [2, 3, 4]
COLS = ["Recency", "Frequency", "Monetary"]
R = config.RESULTS_PATH

DATASETS = [
    ("Online Retail II", os.path.join(R, "retail2", "rfm_raw_retail2.csv")),
    ("UCI", os.path.join(R, "pca_official", "rfm_raw.csv")),
    ("Superstore", os.path.join(R, "superstore", "rfm_raw_superstore.csv")),
    ("Bakery", os.path.join(R, "bakery", "rfm_raw_bakery.csv")),
    # Dunnhumby is deliberately absent: it is not one of the manuscript's
    # datasets, and a decomposition table in a reviewer response should carry
    # the manuscript's five and nothing else.
    ("Olist", os.path.join(R, "olist", "rfm_raw_olist.csv")),      # slowest last
]


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78, flush=True)


def fit(X, k):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    return km.fit_predict(X)


rows = []
for name, path in DATASETS:
    if not os.path.exists(path):
        print("MISSING %s - skipped" % path)
        continue
    rfm = pd.read_csv(path)
    n = len(rfm)
    X3 = StandardScaler().fit_transform(rfm[COLS].apply(np.log1p))
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X3)
    var = float(pca.explained_variance_ratio_.sum() * 100)
    _section("%s  (%d customers, PCA retains %.2f%%)" % (name, n, var))

    for k in K_VALUES:
        t0 = time.time()
        L3, Lp = fit(X3, k), fit(X2, k)
        A = silhouette(X3, L3, exact=True)
        B = silhouette(X2, L3, exact=True)
        C = silhouette(X3, Lp, exact=True)
        D = silhouette(X2, Lp, exact=True)
        rows.append(dict(
            Dataset=name, N=n, K=k, PCA_variance_pct=var,
            A_3Dlabels_in_3D=A, B_3Dlabels_in_PCA=B,
            C_PCAlabels_in_3D=C, D_PCAlabels_in_PCA=D,
            Reported_gain=D - A,
            Partition_ref3D=C - A, Space_ref3D=D - C,
            Partition_refPCA=D - B, Space_refPCA=B - A,
            Space_share_ref3D_pct=(100 * (D - C) / (D - A)) if (D - A) else np.nan,
            Space_share_refPCA_pct=(100 * (B - A) / (D - A)) if (D - A) else np.nan,
            ARI_3D_vs_PCA=adjusted_rand_score(L3, Lp),
            Seconds=time.time() - t0))
        r = rows[-1]
        print("  K=%d  A %.6f  B %.6f  C %.6f  D %.6f" % (k, A, B, C, D))
        print("        reported gain %+.6f" % r["Reported_gain"])
        print("        ref 3D : partition %+.6f  space %+.6f  (space is %.1f%% of the gain)"
              % (r["Partition_ref3D"], r["Space_ref3D"], r["Space_share_ref3D_pct"]))
        print("        ref PCA: partition %+.6f  space %+.6f  (space is %.1f%% of the gain)"
              % (r["Partition_refPCA"], r["Space_refPCA"], r["Space_share_refPCA_pct"]))
        print("        ARI(3D labels, PCA labels) %.4f   [%.1f s]"
              % (r["ARI_3D_vs_PCA"], r["Seconds"]), flush=True)

df = pd.DataFrame(rows)
out = os.path.join(R, "rev2_decomposition_2x2.csv")
df.to_csv(out, index=False)

# --------------------------------------------------- cross-checks against the files
_section("CROSS-CHECK: cells A and B against proof_silhouette_sign.csv")
p = os.path.join(R, "proof_silhouette_sign.csv")
if os.path.exists(p):
    pf = pd.read_csv(p)[["Dataset", "K", "Sil_3D", "Sil_PCA", "Scored_exactly"]]
    m = df.merge(pf, on=["Dataset", "K"], how="inner")
    m["dA"] = (m.A_3Dlabels_in_3D - m.Sil_3D).abs()
    m["dB"] = (m.B_3Dlabels_in_PCA - m.Sil_PCA).abs()
    print(m[["Dataset", "K", "A_3Dlabels_in_3D", "Sil_3D", "dA",
             "B_3Dlabels_in_PCA", "Sil_PCA", "dB"]].round(6).to_string(index=False))
    print("\n  worst |difference|: A %.3e   B %.3e" % (m.dA.max(), m.dB.max()))

_section("CROSS-CHECK: cells C and D against harness_decomposition.csv (K=3)")
h = os.path.join(R, "harness_decomposition.csv")
if os.path.exists(h):
    hf = pd.read_csv(h)
    hf = hf[hf.Method == "KMeans++ K=3 (PCA 2D)"][
        ["Dataset", "Sil_method_reference", "Sil_method_own", "Sil_baseline_own",
         "Partition_effect", "Space_effect"]]
    m = df[df.K == 3].merge(hf, on="Dataset", how="inner")
    m["dC"] = (m.C_PCAlabels_in_3D - m.Sil_method_reference).abs()
    m["dD"] = (m.D_PCAlabels_in_PCA - m.Sil_method_own).abs()
    m["dPartition"] = (m.Partition_ref3D - m.Partition_effect).abs()
    print(m[["Dataset", "N", "C_PCAlabels_in_3D", "Sil_method_reference", "dC",
             "D_PCAlabels_in_PCA", "Sil_method_own", "dD",
             "Partition_ref3D", "Partition_effect", "dPartition"]]
          .round(6).to_string(index=False))
    print("\n  Any dataset whose difference is not at rounding level is the one the")
    print("  harness estimates rather than scores exactly; that is the exact-versus-")
    print("  estimate gap already documented for Olist, not a disagreement.")

_section("SUMMARY: does the conclusion depend on which space is the reference?")
s = df[["Dataset", "K", "Reported_gain", "Partition_ref3D", "Partition_refPCA",
        "Space_share_ref3D_pct", "Space_share_refPCA_pct"]].copy()
s["Partition_positive_ref3D"] = s.Partition_ref3D > 0
s["Partition_positive_refPCA"] = s.Partition_refPCA > 0
print(s.round(4).to_string(index=False))
print()
print("  cases where the partition effect is positive under reference 3D : %d of %d"
      % (int(s.Partition_positive_ref3D.sum()), len(s)))
print("  cases where the partition effect is positive under reference PCA: %d of %d"
      % (int(s.Partition_positive_refPCA.sum()), len(s)))
print("  cases where the two references disagree on the SIGN              : %d of %d"
      % (int((s.Partition_positive_ref3D != s.Partition_positive_refPCA).sum()), len(s)))
print()
print("written: %s" % out)
