# -*- coding: utf-8 -*-
"""The partition effect with a paired interval, on each of the five datasets.

Why this exists
---------------
The manuscript's per-dataset equivalence table needed a mean, an SD and an
interval for every dataset. Only UCI had them: rfm_commonspace_pca_test.py
bootstraps UCI and writes commonspace_bootstrap.csv. The other four rows had no
file behind them, so this script runs the same procedure on all five.

The procedure is the one rev2_bootstrap_provenance.csv documents for that UCI
study, followed here without a change:

  * B = 100 resamples of customers, drawn with replacement at full size;
    indices from numpy default_rng(42), one generator per dataset.
  * log1p and StandardScaler fitted ONCE on the full dataset; the resample draws
    rows of that already-scaled matrix.
  * inside each resample: K-Means++ (n_init=10, random_state=42+b) fitted in the
    resample's 3D space, and PCA(2) REFITTED on the resample with K-Means++
    fitted in that 2D space.
  * both partitions then scored in the SAME 3D space of that resample, with its
    duplicate rows, so the difference is a partition effect and not a change of
    ruler.
  * Silhouette through rfm_scoring, the project's single estimator. On Olist
    that means a 10,000-point subsample, and within one resample the SAME
    subsample serves both partitions, so the pairing is exact even where the
    level is an estimate.

Outputs (nothing existing is touched):
  rev2_partition_effect_per_iteration.csv   one row per dataset per resample
  rev2_partition_effect_summary.csv         mean 3D, mean PCA, mean difference,
                                            SD, the 2.5th and 97.5th percentiles,
                                            and the share of resamples where the
                                            PCA partition scores higher

It checks itself: the UCI row must reproduce commonspace_bootstrap.csv to 1e-9,
because the two are the same computation.

Usage:  python -m rev2_partition_effect_bootstrap
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config
from rfm_scoring import silhouette

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
K = 3
N_BOOTSTRAP = 100
COLS = ["Recency", "Frequency", "Monetary"]
R = config.RESULTS_PATH

DATASETS = [
    ("UCI", os.path.join(R, "pca_official", "rfm_raw.csv")),
    ("Online Retail II", os.path.join(R, "retail2", "rfm_raw_retail2.csv")),
    ("Superstore", os.path.join(R, "superstore", "rfm_raw_superstore.csv")),
    ("Bakery", os.path.join(R, "bakery", "rfm_raw_bakery.csv")),
    ("Olist", os.path.join(R, "olist", "rfm_raw_olist.csv")),
]


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78, flush=True)


def fit_km(X, seed):
    return KMeans(n_clusters=K, init="k-means++", n_init=10,
                  random_state=seed).fit_predict(X)


per_iter, summary = [], []

for name, path in DATASETS:
    if not os.path.exists(path):
        print("skipping %s: %s not found" % (name, path), flush=True)
        continue
    rfm = pd.read_csv(path)
    n = len(rfm)
    X3 = StandardScaler().fit_transform(rfm[COLS].apply(np.log1p))
    _section("%s  (%d customers)" % (name, n))

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        Xb = X3[idx]
        seed = RANDOM_STATE + b

        lab_3d = fit_km(Xb, seed)
        X2b = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xb)
        lab_pca = fit_km(X2b, seed)

        # both partitions in the resample's own 3D space; one subsample seed per
        # resample, so a paired difference is computed on the same points
        s_3d = silhouette(Xb, lab_3d, seed=seed)
        s_pca = silhouette(Xb, lab_pca, seed=seed)
        rows.append(dict(Dataset=name, N=n, Iteration=b + 1,
                         Sil_3Dlabels_in3D=s_3d, Sil_PCAlabels_in3D=s_pca,
                         Diff=s_pca - s_3d))
        if (b + 1) % 20 == 0:
            print("    ...%d/%d" % (b + 1, N_BOOTSTRAP), flush=True)

    df = pd.DataFrame(rows)
    per_iter.append(df)
    d = df.Diff
    summary.append(dict(
        Dataset=name, N=n, B=N_BOOTSTRAP,
        Mean_Sil_3D=float(df.Sil_3Dlabels_in3D.mean()),
        Mean_Sil_PCA_in_3D=float(df.Sil_PCAlabels_in3D.mean()),
        Mean_Diff=float(d.mean()), SD_Diff=float(d.std(ddof=1)),
        CI95_low=float(d.quantile(0.025)), CI95_high=float(d.quantile(0.975)),
        Pct_PCA_higher=float((d > 0).mean() * 100),
        Silhouette_estimator=("estimate (10,000-point subsample)" if n > 20000
                              else "exact")))
    s = summary[-1]
    print("  3D partition  in 3D: %.4f" % s["Mean_Sil_3D"])
    print("  PCA partition in 3D: %.4f" % s["Mean_Sil_PCA_in_3D"])
    print("  difference         : %+.4f   SD %.4f   95%% CI [%+.4f, %+.4f]"
          % (s["Mean_Diff"], s["SD_Diff"], s["CI95_low"], s["CI95_high"]))
    print("  resamples where the PCA partition scores higher: %.0f%%"
          % s["Pct_PCA_higher"], flush=True)

pi = pd.concat(per_iter, ignore_index=True)
out_iter = os.path.join(R, "rev2_partition_effect_per_iteration.csv")
out_sum = os.path.join(R, "rev2_partition_effect_summary.csv")
pi.to_csv(out_iter, index=False)
pd.DataFrame(summary).to_csv(out_sum, index=False)

# ------------------------------------------------- the check against UCI's own run
_section("CHECK: does the UCI row reproduce commonspace_bootstrap.csv?")
pub = os.path.join(R, "commonspace_bootstrap.csv")
if os.path.exists(pub):
    old = pd.read_csv(pub)
    new = pi[pi.Dataset == "UCI"]
    m = old.merge(new, on="Iteration", suffixes=("_pub", "_new"))
    worst = 0.0
    for col in ("Sil_3Dlabels_in3D", "Sil_PCAlabels_in3D", "Diff"):
        d = (m[col + "_pub"] - m[col + "_new"]).abs().max()
        worst = max(worst, float(d))
        print("  max |published - recomputed| on %-20s %.3e" % (col, d))
    print("  VERDICT: %s at 1e-9"
          % ("IDENTICAL" if worst <= 1e-9 else "DIFFERS (%.3e)" % worst))
else:
    print("  %s not found - run rfm_commonspace_pca_test.py first" % pub)

print()
print("written: %s" % out_iter)
print("written: %s" % out_sum)
