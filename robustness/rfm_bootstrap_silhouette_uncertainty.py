"""
IJIES Response follow-up, Tasks 1 and 3(a): quantify uncertainty in two
Silhouette comparisons via paired bootstrap resampling, using the same
100-resample procedure and seed policy already established in
rfm_bootstrap_stability_analysis.py.

Task 1 (Reviewer 1, Point 1 residual gap): is the K=3-vs-K=4 Silhouette
difference (0.3365 vs 0.3375, diff=0.0010) within resampling noise?
For each of 100 bootstrap resamples of the 3D baseline space, fit K=3 and
K=4 on the SAME resample and record Silhouette(K=4) - Silhouette(K=3).

Task 3a (Reviewer 2, Point 3): is the 11.6% PCA Silhouette gain (0.3365 ->
0.3754) a stable effect or within noise? For each of the SAME 100 bootstrap
resamples, fit K=3 on the 3D-scaled resample and (separately) fit PCA(2
components) + K=3 on that same resample, and record
Silhouette(PCA) - Silhouette(3D).

Design note: this is a PAIRED bootstrap (both configurations computed on
the identical resample each time), which is the statistically appropriate
design for a paired difference -- not two independent bootstrap
distributions compared post hoc.

Isolated script: reads only outputs/results/pca_official/rfm_raw.csv
(already-verified cleaned RFM table). Writes only
outputs/results/bootstrap_silhouette_k3_vs_k4.csv and
outputs/results/bootstrap_silhouette_pca_vs_3d.csv. Does not modify any
existing script or result.

Run with:
    python rfm_bootstrap_silhouette_uncertainty.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
N_BOOTSTRAP = 100
RESULTS_PATH = config.RESULTS_PATH
RAW_PATH = os.path.join(RESULTS_PATH, "pca_official", "rfm_raw.csv")


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


_section("BOOTSTRAP SILHOUETTE UNCERTAINTY -- K=3 vs K=4, and PCA vs 3D baseline")

rfm = pd.read_csv(RAW_PATH)
n = len(rfm)
print(f"Loaded {RAW_PATH}: {n} customers.")

X3_full = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p))

rng = np.random.default_rng(RANDOM_STATE)
rows_k3k4 = []
rows_pca = []

for b in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, size=n)
    X3_boot = X3_full[idx]

    # --- Task 1: K=3 vs K=4 on the 3D baseline space, same resample ---
    km3 = KMeans(n_clusters=3, init="k-means++", random_state=RANDOM_STATE + b, n_init=10)
    lab3 = km3.fit_predict(X3_boot)
    sil3 = silhouette_score(X3_boot, lab3)

    km4 = KMeans(n_clusters=4, init="k-means++", random_state=RANDOM_STATE + b, n_init=10)
    lab4 = km4.fit_predict(X3_boot)
    sil4 = silhouette_score(X3_boot, lab4)

    rows_k3k4.append({"Iteration": b + 1, "Silhouette_K3": sil3, "Silhouette_K4": sil4, "Diff_K4_minus_K3": sil4 - sil3})

    # --- Task 3a: PCA(2D) vs 3D baseline, K=3, same resample, PCA refit per resample ---
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2_boot = pca.fit_transform(X3_boot)
    km_pca = KMeans(n_clusters=3, init="k-means++", random_state=RANDOM_STATE + b, n_init=10)
    lab_pca = km_pca.fit_predict(X2_boot)
    sil_pca = silhouette_score(X2_boot, lab_pca)

    rows_pca.append({"Iteration": b + 1, "Silhouette_3D": sil3, "Silhouette_PCA": sil_pca,
                      "Diff_PCA_minus_3D": sil_pca - sil3})

    if (b + 1) % 20 == 0:
        print(f"  ...bootstrap {b + 1}/{N_BOOTSTRAP}")

df_k3k4 = pd.DataFrame(rows_k3k4)
df_pca = pd.DataFrame(rows_pca)

# ---------------------------------------------------------------------
_section("TASK 1 RESULT: K=3 vs K=4 (3D baseline space)")
for col, label in [("Silhouette_K3", "K=3"), ("Silhouette_K4", "K=4")]:
    s = df_k3k4[col]
    print(f"{label}: mean={s.mean():.4f}  SD={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}")

d = df_k3k4["Diff_K4_minus_K3"]
print(f"\nDiff (K4 - K3): mean={d.mean():.4f}  SD={d.std():.4f}  "
      f"2.5th pct={d.quantile(0.025):.4f}  97.5th pct={d.quantile(0.975):.4f}")
pct_k4_higher = (d > 0).mean() * 100
print(f"% of resamples where K=4 Silhouette > K=3 Silhouette: {pct_k4_higher:.1f}%")
ci_excludes_zero = d.quantile(0.025) > 0 or d.quantile(0.975) < 0
print(f"95% CI of the difference {'EXCLUDES' if ci_excludes_zero else 'INCLUDES'} zero.")
if ci_excludes_zero:
    print("=> The two configurations are NOT statistically indistinguishable at this resolution; "
          "the manuscript's phrasing should be revised (see recommendation in the report).")
else:
    print("=> The 95% CI of the paired difference includes zero and/or the sign flips across a "
          "substantial share of resamples: the point estimate's 0.0010 gap is within resampling "
          "noise, supporting (with actual evidence, not just a descriptive label) that K=3 and K=4 "
          "are not reliably distinguishable by Silhouette alone.")

df_k3k4.to_csv(os.path.join(RESULTS_PATH, "bootstrap_silhouette_k3_vs_k4.csv"), index=False)
print(f"\nSaved {os.path.join(RESULTS_PATH, 'bootstrap_silhouette_k3_vs_k4.csv')}")

# ---------------------------------------------------------------------
_section("TASK 3a RESULT: PCA(2D) vs 3D baseline (K=3)")
for col, label in [("Silhouette_3D", "3D baseline"), ("Silhouette_PCA", "PCA-enhanced")]:
    s = df_pca[col]
    print(f"{label}: mean={s.mean():.4f}  SD={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}")

d2 = df_pca["Diff_PCA_minus_3D"]
print(f"\nDiff (PCA - 3D): mean={d2.mean():.4f}  SD={d2.std():.4f}  "
      f"2.5th pct={d2.quantile(0.025):.4f}  97.5th pct={d2.quantile(0.975):.4f}")
pct_pca_higher = (d2 > 0).mean() * 100
print(f"% of resamples where PCA Silhouette > 3D baseline Silhouette: {pct_pca_higher:.1f}%")
ci2_excludes_zero = d2.quantile(0.025) > 0 or d2.quantile(0.975) < 0
print(f"95% CI of the difference {'EXCLUDES' if ci2_excludes_zero else 'INCLUDES'} zero.")
if ci2_excludes_zero and d2.mean() > 0:
    print("=> The PCA improvement is a stable, reproducible effect across resamples, not noise.")
elif ci2_excludes_zero and d2.mean() < 0:
    print("=> Unexpected: the paired bootstrap shows PCA reliably LOWER than 3D baseline -- "
          "this would contradict the manuscript's reported gain and must be reported honestly.")
else:
    print("=> The 95% CI includes zero: the PCA gain is not clearly distinguishable from noise "
          "under this resampled-refit design.")

df_pca.to_csv(os.path.join(RESULTS_PATH, "bootstrap_silhouette_pca_vs_3d.csv"), index=False)
print(f"\nSaved {os.path.join(RESULTS_PATH, 'bootstrap_silhouette_pca_vs_3d.csv')}")

_section("DONE")
