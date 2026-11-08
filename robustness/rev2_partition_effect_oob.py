# -*- coding: utf-8 -*-
"""The partition effect on UCI, measured on the customers a resample left out.

Why this exists
---------------
rev2_partition_effect_bootstrap.py evaluates both partitions inside the
resample that fitted them. A resample drawn with replacement holds duplicate
customers, and a Silhouette computed over duplicates is not an out-of-sample
quantity: a point sitting on top of itself has a zero intra-cluster distance.
The in-bag interval is therefore the right thing for a paired comparison of two
partitions and the wrong thing for the question "does either partition describe
customers it never saw". This script answers the second question on UCI, and
keeps the first arm alongside it so the two are read from one file.

Procedure, the same one rev2_bootstrap_provenance.csv documents for that study
-----------------------------------------------------------------------------
  * B = 100 resamples of customers, drawn with replacement at full size;
    indices from numpy default_rng(42).
  * log1p and StandardScaler fitted ONCE on the whole dataset; the resample
    draws rows of that already-scaled matrix.
  * inside each resample: K-Means++ (K=3, n_init=10, random_state=42+b) fitted
    in the resample's 3D space, and PCA(2, random_state=42) REFITTED on the
    resample with K-Means++ fitted in that 2D space.
  * in-bag arm: both partitions scored in the resample's own 3D space, exactly
    as the published UCI study does, so this arm is a check rather than a new
    number.
  * out-of-bag arm: the customers no draw of that resample touched,
    oob = setdiff1d(arange(n), unique(idx)). Each fitted model LABELS them -
    the 3D model with predict(Xo), the PCA model with predict(pca.transform(Xo))
    - and both label sets are scored in the out-of-bag 3D space Xo. Nothing is
    refitted out there: the partitions are the resample's, the points are not.
  * Silhouette through rfm_scoring, the project's single estimator. UCI has
    4,338 customers and an out-of-bag set near 1,598, both far below its
    20,000-point threshold, so every value here is exact.

Outputs (nothing existing is touched):
  rev2_partition_effect_oob_per_iteration.csv  both arms for each resample
  rev2_partition_effect_oob_summary.csv        one row per arm

It checks itself twice: the in-bag arm's difference must reproduce
commonspace_bootstrap.csv to 1e-9, and the out-of-bag arm must reproduce the
reference run quoted in the manuscript's Section 4.7 to four decimals.

Usage:  python -m robustness.rev2_partition_effect_oob
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
PATH = os.path.join(R, "pca_official", "rfm_raw.csv")

# the reference run, for the out-of-bag arm only
EXPECTED_OOB = dict(Mean_Sil_3D=0.3356, Mean_Sil_PCA_in_3D=0.3360,
                    Mean_Diff=0.0004, SD_Diff=0.0021, CI95_low=-0.0025,
                    CI95_high=0.0061, Pct_PCA_higher=47.0, Mean_N_oob=1598.0)


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78, flush=True)


def fit_km(X, seed):
    return KMeans(n_clusters=K, init="k-means++", n_init=10,
                  random_state=seed).fit(X)


if not os.path.exists(PATH):
    print("skipping: %s not found - run its preprocessing script first" % PATH,
          flush=True)
    sys.exit(0)

rfm = pd.read_csv(PATH)
n = len(rfm)
X3 = StandardScaler().fit_transform(rfm[COLS].apply(np.log1p))
_section("UCI  (%d customers, K=%d, B=%d)" % (n, K, N_BOOTSTRAP))

rng = np.random.default_rng(RANDOM_STATE)
rows = []
for b in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, size=n)
    Xb = X3[idx]
    seed = RANDOM_STATE + b

    km3 = fit_km(Xb, seed)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xb)
    km2 = fit_km(pca.transform(Xb), seed)

    # in-bag: both partitions in the resample's own 3D space
    in_3d = silhouette(Xb, km3.labels_, seed=seed)
    in_pca = silhouette(Xb, km2.labels_, seed=seed)

    # out-of-bag: the customers this resample never drew
    oob = np.setdiff1d(np.arange(n), np.unique(idx))
    Xo = X3[oob]
    oob_3d = silhouette(Xo, km3.predict(Xo), seed=seed)
    oob_pca = silhouette(Xo, km2.predict(pca.transform(Xo)), seed=seed)

    rows.append(dict(Iteration=b + 1, N=n, N_oob=len(oob),
                     InBag_Sil_3Dlabels_in3D=in_3d,
                     InBag_Sil_PCAlabels_in3D=in_pca,
                     InBag_Diff=in_pca - in_3d,
                     OOB_Sil_3Dlabels_in3D=oob_3d,
                     OOB_Sil_PCAlabels_in3D=oob_pca,
                     OOB_Diff=oob_pca - oob_3d))
    if (b + 1) % 20 == 0:
        print("    ...%d/%d" % (b + 1, N_BOOTSTRAP), flush=True)

df = pd.DataFrame(rows)

summary = []
for arm, prefix in (("in-bag (duplicates included)", "InBag"),
                    ("out-of-bag", "OOB")):
    d = df[prefix + "_Diff"]
    summary.append(dict(
        Dataset="UCI", Arm=arm, N=n, B=N_BOOTSTRAP,
        Mean_Sil_3D=float(df[prefix + "_Sil_3Dlabels_in3D"].mean()),
        Mean_Sil_PCA_in_3D=float(df[prefix + "_Sil_PCAlabels_in3D"].mean()),
        Mean_Diff=float(d.mean()), SD_Diff=float(d.std(ddof=1)),
        CI95_low=float(d.quantile(0.025)), CI95_high=float(d.quantile(0.975)),
        Pct_PCA_higher=float((d > 0).mean() * 100),
        Mean_N_oob=float(df.N_oob.mean()),
        Silhouette_estimator="exact"))
    s = summary[-1]
    print()
    print("  %s" % arm)
    print("    3D partition  in 3D: %.4f" % s["Mean_Sil_3D"])
    print("    PCA partition in 3D: %.4f" % s["Mean_Sil_PCA_in_3D"])
    print("    difference         : %+.4f   SD %.4f   95%% CI [%+.4f, %+.4f]"
          % (s["Mean_Diff"], s["SD_Diff"], s["CI95_low"], s["CI95_high"]))
    print("    resamples where the PCA partition scores higher: %.0f%%"
          % s["Pct_PCA_higher"])
    print("    customers scored: %s"
          % ("the %d in the resample, duplicates and all" % n if prefix == "InBag"
             else "%.1f left out on average, %d to %d"
                  % (df.N_oob.mean(), df.N_oob.min(), df.N_oob.max())))

out_iter = os.path.join(R, "rev2_partition_effect_oob_per_iteration.csv")
out_sum = os.path.join(R, "rev2_partition_effect_oob_summary.csv")
os.makedirs(R, exist_ok=True)
df.to_csv(out_iter, index=False)
pd.DataFrame(summary).to_csv(out_sum, index=False)

# ------------------------------------------- check 1: the in-bag arm is UCI's
_section("CHECK 1: does the in-bag arm reproduce commonspace_bootstrap.csv?")
pub = os.path.join(R, "commonspace_bootstrap.csv")
if os.path.exists(pub):
    old = pd.read_csv(pub)
    m = old.merge(df, on="Iteration")
    worst = float((m.Diff - m.InBag_Diff).abs().max())
    print("  max |published Diff - in-bag Diff| over %d resamples: %.3e"
          % (len(m), worst))
    print("  VERDICT: %s at 1e-9"
          % ("IDENTICAL" if worst <= 1e-9 else "DIFFERS (%.3e)" % worst))
else:
    print("  %s not found - run rfm_commonspace_pca_test.py first" % pub)

# ------------------------------- check 2: the out-of-bag arm vs the reference
_section("CHECK 2: does the out-of-bag arm reproduce the reference run?")
oob_row = summary[1]
print("  %-22s %10s %10s %10s" % ("quantity", "reference", "this run", "|diff|"))
print("  " + "-" * 56)
worst2 = 0.0
for key, want in EXPECTED_OOB.items():
    got = round(oob_row[key], 4 if key != "Mean_N_oob" else 0)
    d = abs(got - want)
    worst2 = max(worst2, d)
    print("  %-22s %10.4f %10.4f %10.4f" % (key, want, got, d))
print("  VERDICT: %s at four decimals"
      % ("IDENTICAL" if worst2 == 0 else "DIFFERS (%.4f)" % worst2))

print()
print("written: %s" % out_iter)
print("written: %s" % out_sum)
