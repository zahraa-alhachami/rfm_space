"""
Negative control on synthetic data: does the "space effect" appear even when
there is NO cluster structure at all?

Across all five real datasets, the apparent PCA Silhouette gain was shown to
be almost entirely an evaluation-space effect rather than a partition
improvement. If that effect is genuinely geometric -- a property of how
Silhouette behaves when distances are computed in fewer dimensions -- then it
must also appear on pure isotropic Gaussian noise, where K-Means can only
carve arbitrary regions out of a structureless cloud and no partition can be
"better" in any meaningful sense.

Data: standard 3D Gaussian, no planted clusters, at n = 1,000 / 5,000 /
20,000 (seed 42). Ten independent replicates per size are also reported so
the effect is not read off a single lucky draw.

Deviation from the paper's pipeline, stated explicitly: log1p is NOT applied
here. It is undefined for the negative values that Gaussian data contains,
and would in any case be pointless since this data is symmetric and unskewed
-- the transform exists in the real pipeline only to tame RFM's heavy right
skew. StandardScaler is still applied so the distance treatment matches.
Everything else is identical: KMeans(k-means++, K=3, n_init=10,
random_state=42), PCA(n_components=2, random_state=42).

Isolated script: generates its own data, reads nothing. Writes only
outputs/results/synthetic_commonspace_*.csv.

Run with:
    python rfm_synthetic_commonspace_control.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
K = 3
SIZES = [1000, 5000, 20000]
N_REPLICATES = 10
RESULTS_PATH = config.RESULTS_PATH


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def fit_km(X, random_state=RANDOM_STATE):
    return KMeans(n_clusters=K, init="k-means++", n_init=10,
                   random_state=random_state).fit_predict(X)


def run_one(n, seed):
    """One synthetic sample: returns the 2x2 cell values and the two effects."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(loc=0.0, scale=1.0, size=(n, 3))   # no cluster structure
    X3 = StandardScaler().fit_transform(raw)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(X3)
    var_ret = pca.explained_variance_ratio_.sum() * 100

    labels_3d = fit_km(X3)
    labels_pca = fit_km(X2)

    s_3d_3d = silhouette_score(X3, labels_3d)
    s_3d_2d = silhouette_score(X2, labels_3d)
    s_pca_3d = silhouette_score(X3, labels_pca)
    s_pca_2d = silhouette_score(X2, labels_pca)

    return {
        "N": n, "PCA_Variance_Retained_Pct": round(var_ret, 2),
        "ARI_3D_vs_PCA": round(adjusted_rand_score(labels_3d, labels_pca), 4),
        "Sil_3Dlabels_in3D": round(s_3d_3d, 4),
        "Sil_3Dlabels_inPCA": round(s_3d_2d, 4),
        "Sil_PCAlabels_in3D": round(s_pca_3d, 4),
        "Sil_PCAlabels_inPCA": round(s_pca_2d, 4),
        "DBI_3Dlabels_in3D": round(davies_bouldin_score(X3, labels_3d), 4),
        "DBI_PCAlabels_inPCA": round(davies_bouldin_score(X2, labels_pca), 4),
        "Apparent_Gain": round(s_pca_2d - s_3d_3d, 4),
        "Partition_Effect": round(s_pca_3d - s_3d_3d, 4),
        "Space_Effect": round(s_pca_2d - s_pca_3d, 4),
    }


_section("SYNTHETIC CONTROL: 3D GAUSSIAN NOISE, NO CLUSTER STRUCTURE")
print("If the space effect is geometric, it must appear here too -- on data where")
print("no partition can be genuinely better than any other.\n")

primary_rows = []
for n in SIZES:
    r = run_one(n, RANDOM_STATE)
    primary_rows.append(r)
    print(f"--- n={n:,} (PCA variance retained={r['PCA_Variance_Retained_Pct']:.2f}%, "
          f"ARI={r['ARI_3D_vs_PCA']:.4f}) ---")
    print(f"  3D labels  -> 3D space: {r['Sil_3Dlabels_in3D']:.4f}   "
          f"-> PCA space: {r['Sil_3Dlabels_inPCA']:.4f}")
    print(f"  PCA labels -> 3D space: {r['Sil_PCAlabels_in3D']:.4f}   "
          f"-> PCA space: {r['Sil_PCAlabels_inPCA']:.4f}")
    print(f"  apparent gain={r['Apparent_Gain']:+.4f}  |  partition effect="
          f"{r['Partition_Effect']:+.4f}  |  SPACE EFFECT={r['Space_Effect']:+.4f}\n")

primary = pd.DataFrame(primary_rows)

_section("PRIMARY RESULT -- THREE SYNTHETIC SAMPLES")
print(primary[["N", "PCA_Variance_Retained_Pct", "ARI_3D_vs_PCA", "Sil_3Dlabels_in3D",
                "Sil_PCAlabels_in3D", "Sil_PCAlabels_inPCA", "Apparent_Gain",
                "Partition_Effect", "Space_Effect"]].to_string(index=False))

# ---------------------------------------------------------------------
_section(f"REPLICATE CHECK -- {N_REPLICATES} INDEPENDENT DRAWS PER SIZE")
rep_rows = []
for n in SIZES:
    for rep in range(N_REPLICATES):
        r = run_one(n, RANDOM_STATE + 1000 * rep)
        r["Replicate"] = rep + 1
        rep_rows.append(r)
reps = pd.DataFrame(rep_rows)

print(f"{'n':>8} | {'space effect (mean +/- SD)':>28} | {'partition effect (mean +/- SD)':>32}")
print("-" * 78)
for n in SIZES:
    sub = reps[reps["N"] == n]
    print(f"{n:>8,} | {sub['Space_Effect'].mean():>+13.4f} +/- {sub['Space_Effect'].std():<10.4f} | "
          f"{sub['Partition_Effect'].mean():>+17.4f} +/- {sub['Partition_Effect'].std():<10.4f}")

os.makedirs(RESULTS_PATH, exist_ok=True)
primary.to_csv(os.path.join(RESULTS_PATH, "synthetic_commonspace_primary.csv"), index=False)
reps.to_csv(os.path.join(RESULTS_PATH, "synthetic_commonspace_replicates.csv"), index=False)

_section("VERDICT")
mean_space = reps["Space_Effect"].mean()
mean_part = reps["Partition_Effect"].mean()
all_positive = (reps["Space_Effect"] > 0).all()
print(f"Mean space effect across all {len(reps)} synthetic samples    : {mean_space:+.4f}")
print(f"Mean partition effect across all {len(reps)} synthetic samples: {mean_part:+.4f}")
print(f"Space effect positive in every single sample: {all_positive}")
print()
if all_positive and mean_space > 0.02:
    print("The space effect appears at full strength on data containing NO cluster structure")
    print("whatsoever. It therefore cannot be evidence of better-separated customers: it is a")
    print("geometric property of computing Silhouette in fewer dimensions. This is the")
    print("definitive control for the finding already observed across all five real datasets.")
else:
    print("The space effect does NOT reproduce on structureless data at the magnitude seen in")
    print("the real datasets -- the real-data finding would then need a different explanation.")

print(f"\nSaved synthetic_commonspace_primary.csv and synthetic_commonspace_replicates.csv")
