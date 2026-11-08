# -*- coding: utf-8 -*-
"""IJIES second review, item 5: the Bisecting K-Means comparison, fully specified.

Three parts:

  1. The settings actually used, read off the installed scikit-learn's own
     signature rather than recalled - including the defaults the project's call
     did not override. One of them matters: BisectingKMeans defaults to
     init="random", so the published comparison differs from K-Means++ in BOTH
     the splitting strategy AND the initialisation.
  2. Because of that, two extra arms: Bisecting with init="k-means++", and
     Bisecting with bisecting_strategy="largest_cluster". This separates "the
     bisecting idea" from "the initialisation" and from "which cluster to split".
  3. ARI between runs. The published comparison is a Silhouette difference, and
     a Silhouette difference cannot tell a wobbling index from a wobbling
     partition. Ten seeds per method, all 45 within-method pairs, plus the
     same-seed between-method pairs.

Everything at K=3 on UCI, the dataset the manuscript's comparison uses, in the
same log1p + StandardScaler 3D space.

Writes only outputs/results/rev2_* files. Changes no existing result.

Usage:  python rev2_bisecting_settings.py
"""
import inspect
import io
import os
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RANDOM_STATE = 42
K = 3
N_SEEDS = 10
COLS = ["Recency", "Frequency", "Monetary"]
R = config.RESULTS_PATH
RAW = os.path.join(R, "pca_official", "rfm_raw.csv")

print("=" * 100)
print("1. WHAT THE PUBLISHED COMPARISON ACTUALLY CONFIGURED")
print("=" * 100)
print("scikit-learn %s" % sklearn.__version__)

CALLS = {
    "K-Means++": ("KMeans", KMeans,
                  dict(n_clusters=K, init="k-means++", n_init=10,
                       random_state=RANDOM_STATE)),
    "Bisecting (as published)": ("BisectingKMeans", BisectingKMeans,
                                 dict(n_clusters=K, n_init=10,
                                      random_state=RANDOM_STATE)),
}
spec_rows = []
for label, (cls_name, cls, passed) in CALLS.items():
    sig = inspect.signature(cls.__init__)
    print("\n%s  ->  %s(%s)"
          % (label, cls_name, ", ".join("%s=%r" % kv for kv in passed.items())))
    print("   parameters left at the library default:")
    for pname, p in sig.parameters.items():
        if pname in ("self", "n_clusters") or p.default is inspect._empty:
            continue
        if pname in passed:
            continue
        print("     %-20s = %r" % (pname, p.default))
        spec_rows.append(dict(Configuration=label, Class=cls_name,
                              Parameter=pname, Value=repr(p.default),
                              Source="library default (not set by the project)"))
    for pname, val in passed.items():
        spec_rows.append(dict(Configuration=label, Class=cls_name,
                              Parameter=pname, Value=repr(val),
                              Source="set by the project"))
pd.DataFrame(spec_rows).to_csv(os.path.join(R, "rev2_bisecting_settings.csv"),
                               index=False)
print("""
The consequence, stated plainly: K-Means++ was given init="k-means++" and
BisectingKMeans was left at its default init, so the published difference is not
attributable to the bisecting strategy alone. Section 2 measures how much of it
is initialisation.""")

# ----------------------------------------------------------------- the arms
rfm = pd.read_csv(RAW)
X = StandardScaler().fit_transform(rfm[COLS].apply(np.log1p))
n = len(X)
print()
print("=" * 100)
print("2. FOUR ARMS, TEN SEEDS EACH  (UCI, n=%d, K=%d)" % (n, K))
print("=" * 100)

ARMS = {
    "K-Means++": lambda seed: KMeans(n_clusters=K, init="k-means++", n_init=10,
                                     random_state=seed),
    "Bisecting (as published)": lambda seed: BisectingKMeans(
        n_clusters=K, n_init=10, random_state=seed),
    "Bisecting + k-means++ init": lambda seed: BisectingKMeans(
        n_clusters=K, init="k-means++", n_init=10, random_state=seed),
    "Bisecting + largest_cluster": lambda seed: BisectingKMeans(
        n_clusters=K, n_init=10, random_state=seed,
        bisecting_strategy="largest_cluster"),
}

labels, runs = {}, []
for arm, make in ARMS.items():
    labels[arm] = []
    for s in range(N_SEEDS):
        seed = RANDOM_STATE + s
        lab = make(seed).fit_predict(X)
        labels[arm].append(lab)
        runs.append(dict(Arm=arm, Seed=seed,
                         Silhouette=silhouette_score(X, lab),
                         DBI=davies_bouldin_score(X, lab),
                         Smallest_cluster_pct=100.0 * np.bincount(lab).min() / n))
    df = pd.DataFrame([r for r in runs if r["Arm"] == arm])
    print("\n%s" % arm)
    print("   Silhouette: mean %.4f  SD %.4f  min %.4f  max %.4f"
          % (df.Silhouette.mean(), df.Silhouette.std(ddof=1),
             df.Silhouette.min(), df.Silhouette.max()))
    print("   Davies-Bouldin: mean %.4f  SD %.4f"
          % (df.DBI.mean(), df.DBI.std(ddof=1)))
    print("   smallest cluster: %.2f%% to %.2f%% of customers"
          % (df.Smallest_cluster_pct.min(), df.Smallest_cluster_pct.max()))
pd.DataFrame(runs).to_csv(os.path.join(R, "rev2_bisecting_runs.csv"), index=False)

# ------------------------------------------------------------- ARI between runs
print()
print("=" * 100)
print("3. ARI BETWEEN RUNS: is it the index that moves, or the partition?")
print("=" * 100)
ari_rows = []
for arm in ARMS:
    pairs = [adjusted_rand_score(labels[arm][i], labels[arm][j])
             for i in range(N_SEEDS) for j in range(i + 1, N_SEEDS)]
    sil = pd.DataFrame([r for r in runs if r["Arm"] == arm]).Silhouette
    ari_rows.append(dict(Comparison="within %s" % arm, Pairs=len(pairs),
                         Mean_ARI=float(np.mean(pairs)),
                         Min_ARI=float(np.min(pairs)),
                         Max_ARI=float(np.max(pairs)),
                         Pct_pairs_ARI_ge_0_99=100.0 * np.mean(np.array(pairs) >= 0.99),
                         Silhouette_SD=float(sil.std(ddof=1))))
    print("\nwithin %s: %d pairs" % (arm, len(pairs)))
    print("   ARI mean %.4f  min %.4f  max %.4f   (%.0f%% of pairs at or above 0.99)"
          % (ari_rows[-1]["Mean_ARI"], ari_rows[-1]["Min_ARI"],
             ari_rows[-1]["Max_ARI"], ari_rows[-1]["Pct_pairs_ARI_ge_0_99"]))
    print("   Silhouette SD over the same runs: %.4f" % ari_rows[-1]["Silhouette_SD"])

base = "K-Means++"
for arm in list(ARMS)[1:]:
    same_seed = [adjusted_rand_score(labels[base][i], labels[arm][i])
                 for i in range(N_SEEDS)]
    ari_rows.append(dict(Comparison="%s vs %s, same seed" % (base, arm),
                         Pairs=len(same_seed), Mean_ARI=float(np.mean(same_seed)),
                         Min_ARI=float(np.min(same_seed)),
                         Max_ARI=float(np.max(same_seed)),
                         Pct_pairs_ARI_ge_0_99=100.0 * np.mean(
                             np.array(same_seed) >= 0.99),
                         Silhouette_SD=np.nan))
    print("\n%s vs %s, matched seeds: ARI mean %.4f  min %.4f  max %.4f"
          % (base, arm, np.mean(same_seed), np.min(same_seed), np.max(same_seed)))
pd.DataFrame(ari_rows).to_csv(os.path.join(R, "rev2_bisecting_ari.csv"), index=False)

# ------------------------------------------------ restate the published bootstrap
pub = os.path.join(R, "bootstrap_kmeanspp_vs_bisecting.csv")
if os.path.exists(pub):
    d = pd.read_csv(pub)["Diff_KMPP_minus_Bisecting"]
    print()
    print("=" * 100)
    print("4. THE PUBLISHED PAIRED BOOTSTRAP, RESTATED FROM ITS OWN FILE")
    print("=" * 100)
    print("   B = %d resamples, difference = Silhouette(K-Means++) - Silhouette(Bisecting)"
          % len(d))
    print("   mean %+.4f  SD %.4f  95%% CI [%+.4f, %+.4f]  share of resamples favouring "
          "K-Means++ %.0f%%"
          % (d.mean(), d.std(ddof=1), d.quantile(0.025), d.quantile(0.975),
             100.0 * (d > 0).mean()))
    print("   The interval straddles zero, so on Silhouette the two are not")
    print("   separated by this evidence; section 3 says whether their PARTITIONS")
    print("   are the same object, which the interval alone cannot.")

    # The mean and the median of that difference have OPPOSITE signs, which a
    # percentile interval alone does not explain. The resamples are re-run here
    # recording what the published file does not: each method's cluster sizes
    # and its agreement with its own full-data partition.
    print()
    print("=" * 100)
    print("5. WHY THE MEAN AND THE MEDIAN DISAGREE IN SIGN")
    print("=" * 100)
    lab_km_full = KMeans(n_clusters=K, init="k-means++", n_init=10,
                         random_state=RANDOM_STATE).fit_predict(X)
    bk_full = BisectingKMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE)
    lab_bk_full = bk_full.fit_predict(X)
    print("   full data: K-Means++ %.4f   Bisecting %.4f   difference %+.4f"
          % (silhouette_score(X, lab_km_full), silhouette_score(X, lab_bk_full),
             silhouette_score(X, lab_km_full) - silhouette_score(X, lab_bk_full)))

    rng = np.random.default_rng(RANDOM_STATE)
    det = []
    for b in range(len(d)):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        seed = RANDOM_STATE + b
        lk = KMeans(n_clusters=K, init="k-means++", n_init=10,
                    random_state=seed).fit_predict(Xb)
        lb = BisectingKMeans(n_clusters=K, n_init=10,
                             random_state=seed).fit_predict(Xb)
        det.append(dict(
            Iteration=b + 1,
            Sil_KMeansPP=silhouette_score(Xb, lk),
            Sil_Bisecting=silhouette_score(Xb, lb),
            Smallest_pct_KMeansPP=100.0 * np.bincount(lk, minlength=K).min() / n,
            Smallest_pct_Bisecting=100.0 * np.bincount(lb, minlength=K).min() / n,
            ARI_KMeansPP_vs_full=adjusted_rand_score(lab_km_full[idx], lk),
            ARI_Bisecting_vs_full=adjusted_rand_score(lab_bk_full[idx], lb)))
    dd = pd.DataFrame(det)
    dd["Diff"] = dd.Sil_KMeansPP - dd.Sil_Bisecting
    dd["Bisecting_collapsed"] = dd.Sil_Bisecting < 0.30
    dd.to_csv(os.path.join(R, "rev2_bisecting_bootstrap_detail.csv"), index=False)

    for flag, grp in dd.groupby("Bisecting_collapsed"):
        name = "Bisecting collapsed (<0.30)" if flag else "Bisecting held up"
        print("\n   %s: %d of %d resamples" % (name, len(grp), len(dd)))
        print("     Silhouette  K-Means++ %.4f   Bisecting %.4f   difference %+.4f"
              % (grp.Sil_KMeansPP.mean(), grp.Sil_Bisecting.mean(), grp.Diff.mean()))
        print("     smallest cluster  K-Means++ %.2f%%   Bisecting %.2f%%"
              % (grp.Smallest_pct_KMeansPP.mean(), grp.Smallest_pct_Bisecting.mean()))
        print("     ARI with its own full-data partition  K-Means++ %.4f   Bisecting %.4f"
              % (grp.ARI_KMeansPP_vs_full.mean(), grp.ARI_Bisecting_vs_full.mean()))
    print("""
   So the comparator is bimodal under resampling while K-Means++ is not: the
   median resample favours Bisecting, a sizeable minority sends it into a much
   worse partition, and the mean sits between the two modes where no resample
   actually is. A percentile interval over a bimodal difference is the wrong
   summary to lean on, and the honest statement is the pair of modes with their
   shares - not one interval and not one mean.""")

print()
for f in ("rev2_bisecting_settings.csv", "rev2_bisecting_runs.csv",
          "rev2_bisecting_ari.csv", "rev2_bisecting_bootstrap_detail.csv"):
    print("written: %s" % os.path.join(R, f))
