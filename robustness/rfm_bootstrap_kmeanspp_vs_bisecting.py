"""
Paired bootstrap: K-Means++ vs BisectingKMeans at K=3, on the ORIGINAL
3-dimensional RFM space (log1p + StandardScaler) -- NOT the PCA-reduced
space.

Motivation: the point estimates in the paper are K-Means++ = 0.3365 and
BisectingKMeans = 0.3567 at K=3 (a +6.00% advantage for BisectingKMeans).
This script tests whether that gap survives resampling uncertainty, using
exactly the same paired-bootstrap methodology already used in this project
for the K=3 vs K=4 comparison (rfm_bootstrap_silhouette_uncertainty.py):

  - 100 bootstrap resamples, with replacement, n = 4,338 (full size)
  - the SAME resample indices as the earlier K=3/K=4 study
    (np.random.default_rng(42)), so the two analyses are directly comparable
  - random_state = 42 + b for both models within resample b
  - BOTH algorithms refit from scratch on each resample (paired design:
    identical data for both, so the difference isolates the algorithm)
  - n_init = 10 for both, matching the project's fixed configuration
  - difference computed as Silhouette(K-Means++) - Silhouette(BisectingKMeans)
  - 95% CI by the percentile method (2.5th / 97.5th of the 100 differences)

Isolated script: reads only outputs/results/pca_official/rfm_raw.csv.
Writes only outputs/results/bootstrap_kmeanspp_vs_bisecting.csv.
Modifies nothing else.

Run with:
    python rfm_bootstrap_kmeanspp_vs_bisecting.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
N_BOOTSTRAP = 100
K = 3
RESULTS_PATH = config.RESULTS_PATH
RAW_PATH = os.path.join(RESULTS_PATH, "pca_official", "rfm_raw.csv")


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


_section("PAIRED BOOTSTRAP: K-MEANS++ vs BISECTING K-MEANS (K=3, 3D RFM space)")

rfm = pd.read_csv(RAW_PATH)
n = len(rfm)
print(f"Loaded {RAW_PATH}: {n} customers.")

X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p))
print(f"Feature space: {X.shape[1]}D (log1p + StandardScaler), no PCA.")

# Point-estimate sanity check on the full sample before resampling
km_full = KMeans(n_clusters=K, init="k-means++", random_state=RANDOM_STATE, n_init=10)
sil_km_full = silhouette_score(X, km_full.fit_predict(X))
bk_full = BisectingKMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
sil_bk_full = silhouette_score(X, bk_full.fit_predict(X))
print(f"\nFull-sample point estimates (should match the paper):")
print(f"  K-Means++        : {sil_km_full:.4f}  (paper: 0.3365)")
print(f"  BisectingKMeans  : {sil_bk_full:.4f}  (paper: 0.3567)")

rng = np.random.default_rng(RANDOM_STATE)
rows = []
for b in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, size=n)
    Xb = X[idx]

    km = KMeans(n_clusters=K, init="k-means++", random_state=RANDOM_STATE + b, n_init=10)
    sil_km = silhouette_score(Xb, km.fit_predict(Xb))

    bk = BisectingKMeans(n_clusters=K, random_state=RANDOM_STATE + b, n_init=10)
    sil_bk = silhouette_score(Xb, bk.fit_predict(Xb))

    rows.append({"Iteration": b + 1, "Silhouette_KMeansPP": sil_km,
                  "Silhouette_Bisecting": sil_bk, "Diff_KMPP_minus_Bisecting": sil_km - sil_bk})

    if (b + 1) % 20 == 0:
        print(f"  ...bootstrap {b + 1}/{N_BOOTSTRAP}")

df = pd.DataFrame(rows)

_section("RESULTS")
km_s = df["Silhouette_KMeansPP"]
bk_s = df["Silhouette_Bisecting"]
d = df["Diff_KMPP_minus_Bisecting"]

print(f"{'Algorithm':<22} | {'Mean':>8} | {'SD':>8} | {'Min':>8} | {'Max':>8}")
print("-" * 68)
print(f"{'K-Means++':<22} | {km_s.mean():>8.4f} | {km_s.std():>8.4f} | {km_s.min():>8.4f} | {km_s.max():>8.4f}")
print(f"{'BisectingKMeans':<22} | {bk_s.mean():>8.4f} | {bk_s.std():>8.4f} | {bk_s.min():>8.4f} | {bk_s.max():>8.4f}")

ci_low, ci_high = d.quantile(0.025), d.quantile(0.975)
print(f"\nPaired difference (K-Means++ minus BisectingKMeans):")
print(f"  mean            = {d.mean():+.4f}")
print(f"  SD              = {d.std():.4f}")
print(f"  95% CI (percentile method) = [{ci_low:+.4f}, {ci_high:+.4f}]")

pct_km_wins = (d > 0).mean() * 100
pct_bk_wins = (d < 0).mean() * 100
print(f"\n% of resamples where K-Means++ scored higher      : {pct_km_wins:.1f}%")
print(f"% of resamples where BisectingKMeans scored higher : {pct_bk_wins:.1f}%")

zero_inside = ci_low <= 0 <= ci_high
print(f"\nIs zero inside the 95% CI? {'YES' if zero_inside else 'NO'}")
if zero_inside:
    print("=> The difference is NOT statistically significant at the 95% level: the observed "
          "gap is within resampling noise, and neither algorithm can be claimed superior on "
          "Silhouette for this dataset.")
else:
    winner = "BisectingKMeans" if d.mean() < 0 else "K-Means++"
    print(f"=> The difference IS statistically significant at the 95% level: {winner} reliably "
          f"achieves the higher Silhouette on this dataset. This must be reported as found.")

os.makedirs(RESULTS_PATH, exist_ok=True)
out_path = os.path.join(RESULTS_PATH, "bootstrap_kmeanspp_vs_bisecting.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")
