# -*- coding: utf-8 -*-
"""The cluster balance behind the hierarchical result on the PCA space.

Why this exists
---------------
rfm_hierarchical_pca_analysis.py reports a Silhouette and a DBI for each
linkage, and on UCI the best of them beats K-Means++ on the same space. What
makes that higher score inadmissible as a segmentation is the shape of the
partition, and the shape is the one thing that script does not write down. This
script writes it: the three cluster sizes, largest first, and their shares.

Only UCI and Online Retail II. Olist is deliberately absent - agglomerative
clustering there holds the full 93,358 x 93,358 distance matrix, which is
32.5 GiB, and that is why the exploratory script's third stage skips it.

Setup, identical to the script whose Silhouettes this reproduces
----------------------------------------------------------------
  * log1p on Recency, Frequency and Monetary, then StandardScaler, both fitted
    once on the whole dataset.
  * PCA(n_components=2, random_state=42) on that scaled matrix.
  * AgglomerativeClustering(n_clusters=3, linkage=L) for ward, complete and
    average, fitted in the 2D space.
  * Silhouette and Davies-Bouldin on the same 2D coordinates. The Silhouette
    goes through rfm_scoring, the project's single estimator; both datasets are
    below its 20,000-point threshold, so every value here is exact.

It checks itself: the three UCI Silhouettes must equal the ones in
hier_pca_uci_linkage_comparison.csv, which carries four decimals.

Writes outputs/results/hier_pca_cluster_balance.csv. Changes no existing result.

Usage:  python -m pca_and_validation.rev2_hier_pca_balance
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler

import config
from rfm_scoring import silhouette

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
K = 3
COLS = ["Recency", "Frequency", "Monetary"]
LINKAGES = ["ward", "complete", "average"]
R = config.RESULTS_PATH

DATASETS = [
    ("UCI", os.path.join(R, "rfm_raw.csv")),
    ("Online Retail II", os.path.join(R, "retail2", "rfm_raw_retail2.csv")),
]

rows = []
for name, path in DATASETS:
    if not os.path.exists(path):
        print("skipping %s: %s not found - run its preprocessing script first"
              % (name, path), flush=True)
        continue
    rfm = pd.read_csv(path)
    n = len(rfm)
    X = StandardScaler().fit_transform(rfm[COLS].apply(np.log1p))
    X2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)

    print()
    print("=" * 78)
    print("%s  (%d customers, K=%d in the PCA 2D space)" % (name, n, K))
    print("=" * 78)
    print("%-9s %11s %9s   %-22s %s"
          % ("Linkage", "Silhouette", "DBI", "sizes, largest first", "shares"))
    for linkage in LINKAGES:
        labels = AgglomerativeClustering(n_clusters=K,
                                         linkage=linkage).fit_predict(X2)
        sizes = sorted(np.bincount(labels, minlength=K).tolist(), reverse=True)
        pct = [100.0 * s / n for s in sizes]
        sil = silhouette(X2, labels)
        dbi = float(davies_bouldin_score(X2, labels))
        rows.append(dict(Dataset=name, N=n, K=K, Linkage=linkage, Silhouette=sil,
                         DBI=dbi, Size_1=sizes[0], Size_2=sizes[1],
                         Size_3=sizes[2], Pct_1=pct[0], Pct_2=pct[1],
                         Pct_3=pct[2]))
        print("%-9s %11.4f %9.4f   %-22s %s"
              % (linkage, sil, dbi, "/".join(str(s) for s in sizes),
                 "/".join("%.1f" % p for p in pct)))

out = pd.DataFrame(rows)
os.makedirs(R, exist_ok=True)
dest = os.path.join(R, "hier_pca_cluster_balance.csv")
out.to_csv(dest, index=False)

# ------------------------------------------- the check against the UCI table
print()
print("=" * 78)
print("CHECK: do the UCI Silhouettes equal hier_pca_uci_linkage_comparison.csv?")
print("=" * 78)
ref = os.path.join(R, "hier_pca_uci_linkage_comparison.csv")
if os.path.exists(ref) and (out.Dataset == "UCI").any():
    pub = pd.read_csv(ref).set_index("Linkage")
    uci = out[out.Dataset == "UCI"].set_index("Linkage")
    worst = 0.0
    for linkage in LINKAGES:
        a, b = float(pub.loc[linkage, "Silhouette"]), float(uci.loc[linkage, "Silhouette"])
        worst = max(worst, abs(a - round(b, 4)))
        print("  %-9s published %.4f   recomputed %.10f   |diff| %.1e"
              % (linkage, a, b, abs(a - round(b, 4))))
    print("  VERDICT: %s at the four decimals that file carries"
          % ("IDENTICAL" if worst == 0 else "DIFFERS (%.1e)" % worst))
else:
    print("  %s not found - run rfm_hierarchical_pca_analysis.py first" % ref)

print()
print("written: %s" % dest)
