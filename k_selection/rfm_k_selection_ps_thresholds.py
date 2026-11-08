"""Supplementary analysis of prediction strength only: how much of the K it
selects is the criterion, and how much is the threshold?

This is an ADDITION requested after the main nine-criterion run. It changes
nothing in rfm_k_selection_null_corrected.py: the criterion list there stays
fixed, and the prediction-strength function below is copied from it verbatim
so the two are the same computation. The only differences are 40 train/test
splits instead of 20 (to match an independent check made at 40), and that the
per-split scores are kept rather than averaged away.

Tibshirani & Walther's rule is "the largest K whose prediction strength stays
above a threshold", and the threshold is a free parameter they set by
convention, not by derivation. So the selected K is reported here at four
thresholds rather than one, which is the honest way to present it.

Outputs (prefix ksel_, nothing existing is touched):
  ksel_ps_detail.csv       mean, SD and exceedance rates per dataset/space/K
  ksel_ps_thresholds.csv   selected K at each of 0.70, 0.80, 0.85, 0.90
  ksel_ps_per_split.csv    every individual split score
"""
import io
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RS = 42
K_RANGE = list(range(2, 11))
COLS = ["Recency", "Frequency", "Monetary"]
PS_REPS = 40
THRESHOLDS = [0.70, 0.80, 0.85, 0.90]
DETAIL_KS = [2, 3, 4]

# Same four datasets as the main run. Olist (n=93,358) is excluded there
# because Dunn and the gap statistic need full pairwise distances; it is kept
# out here too so the two tables describe the same set of datasets.
DATASETS = {
    "Online Retail II": "retail2/rfm_raw_retail2.csv",
    "UCI": "pca_official/rfm_raw.csv",
    "Superstore": "superstore/rfm_raw_superstore.csv",
    "Bakery": "bakery/rfm_raw_bakery.csv",
    "Dunnhumby": "dunnhumby/rfm_raw_dunnhumby.csv",
}


def fit(X, k, seed=RS):
    return KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit(X)


def prediction_strength_per_split(X, k, reps=PS_REPS, rng=None):
    """Identical to the main script's prediction_strength, except that the
    per-split scores are returned instead of their mean."""
    rng = rng or np.random.default_rng(RS)
    n = len(X)
    scores = []
    for r in range(reps):
        perm = rng.permutation(n)
        a, b = perm[:n // 2], perm[n // 2:]
        Xa, Xb = X[a], X[b]
        centroids = fit(Xa, k, RS + r).cluster_centers_
        lb_own = fit(Xb, k, RS + r).labels_
        lb_pred = np.argmin(cdist(Xb, centroids), axis=1)

        worst = 1.0
        for c in np.unique(lb_own):
            idx = np.flatnonzero(lb_own == c)
            m = len(idx)
            if m < 2:
                worst = 0.0
                continue
            _, counts = np.unique(lb_pred[idx], return_counts=True)
            pairs_same = (counts * (counts - 1)).sum()
            worst = min(worst, pairs_same / (m * (m - 1)))
        scores.append(worst)
    return np.asarray(scores, dtype=float)


def main():
    detail, thresh, per_split = [], [], []

    for name, rel in DATASETS.items():
        try:
            df = pd.read_csv(config.RESULTS_PATH + "/" + rel)
        except FileNotFoundError:
            print("skipping %s: %s not found - run its preprocessing script "
                  "first, or leave it out" % (name, rel), flush=True)
            continue
        X3 = StandardScaler().fit_transform(df[COLS].apply(np.log1p))
        X2 = PCA(n_components=2, random_state=RS).fit_transform(X3)

        for space, X in (("3D", X3), ("PCA 2D", X2)):
            print("\n--- %s / %s space (n=%d)" % (name, space, len(X)), flush=True)
            means = {}
            for k in K_RANGE:
                s = prediction_strength_per_split(X, k)
                means[k] = s.mean()
                for i, v in enumerate(s, 1):
                    per_split.append({"Dataset": name, "Space": space, "K": k,
                                      "Split": i, "Prediction_strength": v})
                if k in DETAIL_KS:
                    detail.append({
                        "Dataset": name, "Space": space, "K": k,
                        "Splits": len(s),
                        "PS_mean": s.mean(), "PS_sd": s.std(ddof=1),
                        "PS_min": s.min(), "PS_max": s.max(),
                        "Pct_splits_above_0.80": 100.0 * (s > 0.80).mean(),
                        "Pct_splits_above_0.90": 100.0 * (s > 0.90).mean(),
                    })
                print("  K=%2d  mean=%.4f  sd=%.4f  >0.80 in %5.1f%%  >0.90 in %5.1f%%"
                      % (k, s.mean(), s.std(ddof=1),
                         100.0 * (s > 0.80).mean(), 100.0 * (s > 0.90).mean()),
                      flush=True)

            row = {"Dataset": name, "Space": space}
            for t in THRESHOLDS:
                passing = [k for k in K_RANGE if means[k] >= t]
                row["Selected_K_at_%.2f" % t] = max(passing) if passing else None
            thresh.append(row)
            print("  selected K by threshold: " + ", ".join(
                "%.2f -> %s" % (t, row["Selected_K_at_%.2f" % t]) for t in THRESHOLDS),
                flush=True)

    out = config.RESULTS_PATH + "/"
    pd.DataFrame(detail).round(6).to_csv(out + "ksel_ps_detail.csv", index=False)
    pd.DataFrame(thresh).to_csv(out + "ksel_ps_thresholds.csv", index=False)
    pd.DataFrame(per_split).round(6).to_csv(out + "ksel_ps_per_split.csv", index=False)

    print("\n" + "=" * 78)
    print("SELECTED K BY THRESHOLD (largest K whose mean PS stays at or above it)")
    print("=" * 78)
    print(pd.DataFrame(thresh).to_string(index=False))
    print("\nSaved ksel_ps_detail.csv, ksel_ps_thresholds.csv, ksel_ps_per_split.csv")


if __name__ == "__main__":
    main()
