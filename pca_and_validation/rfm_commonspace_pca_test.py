"""
Isolated test: does the reported +11.6% Silhouette gain from PCA survive
when both partitions are judged on common ground?

The paper's headline comparison is (3D labels, evaluated in 3D space) =
0.3365 versus (PCA labels, evaluated in 2D PCA space) = 0.3754. Two things
change at once there: the partition AND the space the metric is computed
in. Silhouette is a function of both, so this script separates them via a
2x2 matrix in which each partition is scored in each space, and then
isolates the partition effect alone through a paired bootstrap where BOTH
partitions are scored in the same 3D space.

A negative control (random labels with the 3D partition's exact cluster
sizes) establishes the chance baseline for the same metric.

Isolated script: reads only outputs/results/pca_official/rfm_raw.csv.
Writes only outputs/results/commonspace_*.csv. Modifies nothing else.

Run with:
    python rfm_commonspace_pca_test.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, adjusted_rand_score)

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
K = 3
N_BOOTSTRAP = 100
N_RANDOM = 100
COLS = ["Recency", "Frequency", "Monetary"]
RESULTS_PATH = config.RESULTS_PATH
RAW_PATH = os.path.join(RESULTS_PATH, "pca_official", "rfm_raw.csv")


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def fit_km(X, random_state=RANDOM_STATE):
    km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=random_state)
    return km.fit_predict(X)


def score_all(X, labels):
    return (silhouette_score(X, labels), davies_bouldin_score(X, labels),
            calinski_harabasz_score(X, labels))


# ===========================================================================
_section("STEP 1: BUILD BOTH PARTITIONS")

rfm = pd.read_csv(RAW_PATH)
n = len(rfm)
X3 = StandardScaler().fit_transform(rfm[COLS].apply(np.log1p))
labels_3d = fit_km(X3)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X2 = pca.fit_transform(X3)
labels_pca = fit_km(X2)

print(f"Customers: {n}  |  PCA variance retained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
print(f"ARI(3D labels, PCA labels) = {adjusted_rand_score(labels_3d, labels_pca):.4f}")

# ===========================================================================
_section("STEP 2: 2x2 MATRIX -- EACH PARTITION SCORED IN EACH SPACE")

cells = []
for lab_name, labels in [("3D labels", labels_3d), ("PCA labels", labels_pca)]:
    for sp_name, X in [("3D space", X3), ("PCA 2D space", X2)]:
        sil, dbi, ch = score_all(X, labels)
        cells.append({"Partition": lab_name, "Evaluation_space": sp_name,
                       "Silhouette": round(sil, 4), "Davies_Bouldin": round(dbi, 4),
                       "Calinski_Harabasz": round(ch, 1)})

matrix = pd.DataFrame(cells)
print(matrix.to_string(index=False))

sil_3d_in3d = matrix.query("Partition=='3D labels' and Evaluation_space=='3D space'")["Silhouette"].iloc[0]
sil_pca_in3d = matrix.query("Partition=='PCA labels' and Evaluation_space=='3D space'")["Silhouette"].iloc[0]
sil_3d_in2d = matrix.query("Partition=='3D labels' and Evaluation_space=='PCA 2D space'")["Silhouette"].iloc[0]
sil_pca_in2d = matrix.query("Partition=='PCA labels' and Evaluation_space=='PCA 2D space'")["Silhouette"].iloc[0]

print(f"\nAnchor check -- 3D labels in 3D space : {sil_3d_in3d:.4f} (paper: 0.3365) "
      f"{'OK' if abs(sil_3d_in3d-0.3365)<0.0001 else 'MISMATCH'}")
print(f"Anchor check -- PCA labels in PCA space: {sil_pca_in2d:.4f} (paper: 0.3754) "
      f"{'OK' if abs(sil_pca_in2d-0.3754)<0.0001 else 'MISMATCH'}")

_section("DECOMPOSING THE HEADLINE +0.0389 GAIN")
total = sil_pca_in2d - sil_3d_in3d
partition_effect = sil_pca_in3d - sil_3d_in3d
space_effect = sil_pca_in2d - sil_pca_in3d
print(f"Headline gain as reported (PCA-in-PCA minus 3D-in-3D) : {total:+.4f}  ({total/sil_3d_in3d*100:+.2f}%)")
print(f"  ... attributable to the PARTITION (both scored in 3D) : {partition_effect:+.4f}  "
      f"({partition_effect/sil_3d_in3d*100:+.2f}%)")
print(f"  ... attributable to the EVALUATION SPACE (same labels) : {space_effect:+.4f}  "
      f"({space_effect/sil_3d_in3d*100:+.2f}%)")
print(f"\nShare of the headline gain that is a space effect: {space_effect/total*100:.1f}%")

matrix.to_csv(os.path.join(RESULTS_PATH, "commonspace_matrix.csv"), index=False)
pd.DataFrame([{
    "Headline_gain": round(total, 4), "Partition_effect_in3D": round(partition_effect, 4),
    "Space_effect": round(space_effect, 4),
    "Space_effect_share_pct": round(space_effect / total * 100, 1),
}]).to_csv(os.path.join(RESULTS_PATH, "commonspace_decomposition.csv"), index=False)

# ===========================================================================
_section("STEP 3: PAIRED BOOTSTRAP -- BOTH PARTITIONS SCORED IN 3D SPACE")

rng = np.random.default_rng(RANDOM_STATE)
rows = []
for b in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, size=n)
    Xb = X3[idx]

    lab_3d_b = fit_km(Xb, RANDOM_STATE + b)
    pca_b = PCA(n_components=2, random_state=RANDOM_STATE)
    X2b = pca_b.fit_transform(Xb)
    lab_pca_b = fit_km(X2b, RANDOM_STATE + b)

    s_3d = silhouette_score(Xb, lab_3d_b)       # 3D partition, 3D space
    s_pca = silhouette_score(Xb, lab_pca_b)     # PCA partition, SAME 3D space
    rows.append({"Iteration": b + 1, "Sil_3Dlabels_in3D": s_3d,
                  "Sil_PCAlabels_in3D": s_pca, "Diff": s_pca - s_3d})
    if (b + 1) % 20 == 0:
        print(f"  ...bootstrap {b + 1}/{N_BOOTSTRAP}")

bs = pd.DataFrame(rows)
d = bs["Diff"]
ci_lo, ci_hi = d.quantile(0.025), d.quantile(0.975)
pct_pca_wins = (d > 0).mean() * 100

print(f"\n3D partition  (in 3D space): mean={bs['Sil_3Dlabels_in3D'].mean():.4f}  "
      f"SD={bs['Sil_3Dlabels_in3D'].std():.4f}")
print(f"PCA partition (in 3D space): mean={bs['Sil_PCAlabels_in3D'].mean():.4f}  "
      f"SD={bs['Sil_PCAlabels_in3D'].std():.4f}")
print(f"\nPaired difference (PCA minus 3D, both in 3D space):")
print(f"  mean   = {d.mean():+.4f}")
print(f"  SD     = {d.std():.4f}")
print(f"  95% CI = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
print(f"  PCA partition scored higher in {pct_pca_wins:.1f}% of resamples")
zero_in = ci_lo <= 0 <= ci_hi
print(f"  Zero inside the 95% CI? {'YES -- not significant' if zero_in else 'NO -- significant'}")

bs.to_csv(os.path.join(RESULTS_PATH, "commonspace_bootstrap.csv"), index=False)

# ===========================================================================
_section("STEP 4: NEGATIVE CONTROL -- RANDOM LABELS, SAME CLUSTER SIZES")

rng_ctrl = np.random.default_rng(RANDOM_STATE)
rand_scores = []
for _ in range(N_RANDOM):
    shuffled = rng_ctrl.permutation(labels_3d)   # preserves exact cluster sizes
    rand_scores.append(silhouette_score(X3, shuffled))
rand_scores = np.array(rand_scores)
print(f"Random labels (3D partition's exact sizes), scored in 3D space over {N_RANDOM} repeats:")
print(f"  mean={rand_scores.mean():+.4f}  SD={rand_scores.std():.4f}  "
      f"min={rand_scores.min():+.4f}  max={rand_scores.max():+.4f}")
print(f"\nFor scale: real 3D partition = {sil_3d_in3d:.4f}, real PCA partition (in 3D) = {sil_pca_in3d:.4f}")
print(f"Both real partitions sit far above chance; the chance baseline is ~{rand_scores.mean():.4f}.")

pd.DataFrame({"Iteration": range(1, N_RANDOM + 1), "Silhouette_random": rand_scores}).to_csv(
    os.path.join(RESULTS_PATH, "commonspace_negative_control.csv"), index=False)

# ===========================================================================
_section("STEP 5: VERDICT")
if zero_in:
    print(f"On common ground (both partitions scored in the full 3D space), the PCA partition's "
          f"advantage is {d.mean():+.4f} Silhouette with a 95% CI of [{ci_lo:+.4f}, {ci_hi:+.4f}] "
          f"that INCLUDES zero -- so the reported +11.6% gain does NOT survive as a genuine "
          f"clustering improvement: {space_effect/total*100:.1f}% of it is an artefact of "
          f"evaluating the metric in a lower-dimensional space rather than of better-separated "
          f"customers.")
else:
    direction = "remains in PCA's favour" if d.mean() > 0 else "REVERSES against PCA"
    print(f"On common ground (both partitions scored in the full 3D space), the difference "
          f"{direction}: {d.mean():+.4f} Silhouette, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], "
          f"excluding zero. However, only {partition_effect/total*100:.1f}% of the headline "
          f"+11.6% figure is attributable to the partition itself; the remaining "
          f"{space_effect/total*100:.1f}% is an evaluation-space effect.")

print(f"\nSaved four commonspace_*.csv files under {RESULTS_PATH}")
