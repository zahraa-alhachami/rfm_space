"""Does any validity criterion select K=3 over K=2 on RFM data?

Context. Silhouette, Davies-Bouldin and Calinski-Harabasz all favour K=2 on
every dataset in this project. Those three share one blind spot: none of them
compares the observed clustering against a null distribution, so all three
reward fewer clusters almost by construction. This script adds the criteria
that do correct for a null, plus two that measure something else entirely.

The full list is fixed HERE, before the script is run, and every criterion is
reported whatever it says. No criterion is added or dropped afterwards.

  1  Silhouette                     (no null correction)   - known to favour K=2
  2  Davies-Bouldin                 (no null correction)   - known to favour K=2
  3  Calinski-Harabasz              (no null correction)   - known to favour K=2
  4  Gap statistic                  NULL-CORRECTED, Tibshirani et al. 2001,
                                    with the standard 1-SE rule
  5  Dunn index                     (no null correction, different geometry:
                                    worst-case separation, not average)
  6  Prediction strength            Tibshirani & Walther 2005, threshold 0.8
                                    - a predictive, not a geometric, criterion
  7  Bootstrap ARI stability        raw, and corrected against a random
                                    labelling with identical cluster sizes
  8  WCSS elbow                     maximum-curvature point

Everything runs in the log1p + StandardScaler 3D space, which is the space the
K-selection claim is made in, and is repeated in the 2-component PCA space
because that is the space the adopted model lives in.

Outputs (prefix ksel_, nothing existing is touched):
  ksel_criteria_by_k.csv      every criterion at every K, both spaces
  ksel_selected_k.csv         which K each criterion selects, and why
  ksel_gap_detail.csv         gap, its standard error, and the 1-SE decision
"""
import io
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

RS = 42
K_RANGE = list(range(2, 11))
COLS = ["Recency", "Frequency", "Monetary"]
GAP_B = 50          # reference datasets per K
PS_REPS = 20        # train/test splits for prediction strength
PS_THRESHOLD = 0.80
BOOT_N = 100

DATASETS = {
    "Online Retail II": "retail2/rfm_raw_retail2.csv",
    "UCI": "pca_official/rfm_raw.csv",
    "Superstore": "superstore/rfm_raw_superstore.csv",
    "Bakery": "bakery/rfm_raw_bakery.csv",
    "Dunnhumby": "dunnhumby/rfm_raw_dunnhumby.csv",
}
# Olist (n=93,358) is excluded: Dunn and the gap statistic both need full
# pairwise distances, which is impractical at that size. Noted, not hidden.


def fit(X, k, seed=RS):
    return KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit(X)


# --------------------------------------------------------------------------
# 4. Gap statistic (null-corrected)
# --------------------------------------------------------------------------
def log_wk(X, k):
    """log of the pooled within-cluster sum of squares."""
    if k == 1:
        return np.log(((X - X.mean(axis=0)) ** 2).sum())
    return np.log(fit(X, k).inertia_)


def gap_statistic(X, k_values, B=GAP_B, rng=None):
    """Tibshirani/Walther/Hastie (2001), reference drawn uniformly over a box
    aligned to the data's principal axes (the paper's recommended variant)."""
    rng = rng or np.random.default_rng(RS)
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xp = Xc @ Vt.T
    lo, hi = Xp.min(axis=0), Xp.max(axis=0)

    rows = []
    for k in k_values:
        obs = log_wk(X, k)
        ref = np.empty(B)
        for b in range(B):
            Zp = rng.uniform(lo, hi, size=Xp.shape)
            ref[b] = log_wk(Zp @ Vt, k)
        gap = ref.mean() - obs
        sk = ref.std(ddof=1) * np.sqrt(1.0 + 1.0 / B)
        rows.append({"K": k, "log_Wk": obs, "ref_mean": ref.mean(),
                     "Gap": gap, "s_k": sk})
    return pd.DataFrame(rows)


def gap_select(gap_df):
    """Smallest k with Gap(k) >= Gap(k+1) - s_(k+1)."""
    g, s, ks = gap_df.Gap.to_numpy(), gap_df.s_k.to_numpy(), gap_df.K.to_numpy()
    for i in range(len(ks) - 1):
        if g[i] >= g[i + 1] - s[i + 1]:
            return int(ks[i])
    return int(ks[-1])  # no k satisfies the rule inside the range


# --------------------------------------------------------------------------
# 5. Dunn index
# --------------------------------------------------------------------------
def dunn_index(D, labels):
    """min single-linkage distance between clusters / max cluster diameter."""
    uniq = np.unique(labels)
    diam = max(D[np.ix_(labels == c, labels == c)].max() for c in uniq)
    sep = min(D[np.ix_(labels == a, labels == b)].min()
              for i, a in enumerate(uniq) for b in uniq[i + 1:])
    return sep / diam if diam > 0 else np.nan


# --------------------------------------------------------------------------
# 6. Prediction strength
# --------------------------------------------------------------------------
def prediction_strength(X, k, reps=PS_REPS, rng=None):
    """Tibshirani & Walther (2005). Cluster the training half, use its
    centroids to assign the test half, and ask how well the test half's own
    clusters survive that assignment. The score is the worst cluster's
    co-membership rate, averaged over splits."""
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
            same = lb_pred[idx]
            # proportion of within-cluster pairs still co-assigned
            _, counts = np.unique(same, return_counts=True)
            pairs_same = (counts * (counts - 1)).sum()
            worst = min(worst, pairs_same / (m * (m - 1)))
        scores.append(worst)
    return float(np.mean(scores))


# --------------------------------------------------------------------------
# 7. Bootstrap ARI stability, raw and null-corrected
# --------------------------------------------------------------------------
def stability(X, k, reference_labels, n_boot=BOOT_N, rng=None):
    """Raw stability = mean ARI(refit-on-resample predicted on full data,
    reference). Null stability = the same statistic for random labellings
    holding the reference cluster sizes fixed. Corrected = (raw - null)/(1 - null),
    the same adjustment logic ARI itself applies to the Rand index."""
    rng = rng or np.random.default_rng(RS)
    n = len(X)
    sizes = np.bincount(reference_labels, minlength=k)

    raw, null = [], []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        km = fit(X[idx], k, RS + b)
        raw.append(adjusted_rand_score(reference_labels, km.predict(X)))

        shuffled = np.repeat(np.arange(k), sizes)
        rng.shuffle(shuffled)
        null.append(adjusted_rand_score(reference_labels, shuffled))

    raw_m, null_m = float(np.mean(raw)), float(np.mean(null))
    corrected = (raw_m - null_m) / (1.0 - null_m) if null_m < 1 else np.nan
    return raw_m, null_m, corrected


# --------------------------------------------------------------------------
# 8. WCSS elbow: point of maximum distance from the chord
# --------------------------------------------------------------------------
def elbow_select(ks, wcss):
    ks, wcss = np.asarray(ks, float), np.asarray(wcss, float)
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (wcss - wcss.min()) / (wcss.max() - wcss.min())
    # distance from each point to the line through the first and last point
    num = np.abs((x[-1] - x[0]) * (y[0] - y) - (x[0] - x) * (y[-1] - y[0]))
    den = np.hypot(x[-1] - x[0], y[-1] - y[0])
    return int(ks[np.argmax(num / den)])


# --------------------------------------------------------------------------
def analyse(name, X, space):
    print("\n--- %s / %s space (n=%d, d=%d)" % (name, space, X.shape[0], X.shape[1]),
          flush=True)
    D = squareform(pdist(X))

    rows = []
    for k in K_RANGE:
        km = fit(X, k)
        lb = km.labels_
        raw_s, null_s, corr_s = stability(X, k, lb)
        rows.append({
            "Dataset": name, "Space": space, "K": k,
            "WCSS": km.inertia_,
            "Silhouette": silhouette_score(X, lb),
            "Davies_Bouldin": davies_bouldin_score(X, lb),
            "Calinski_Harabasz": calinski_harabasz_score(X, lb),
            "Dunn": dunn_index(D, lb),
            "Prediction_Strength": prediction_strength(X, k),
            "Stability_raw": raw_s,
            "Stability_null": null_s,
            "Stability_corrected": corr_s,
            "Min_cluster_pct": 100.0 * np.bincount(lb).min() / len(lb),
        })
        print("  K=%2d  sil=%.4f  dbi=%.4f  ch=%7.1f  dunn=%.4f  ps=%.3f  stab*=%.4f"
              % (k, rows[-1]["Silhouette"], rows[-1]["Davies_Bouldin"],
                 rows[-1]["Calinski_Harabasz"], rows[-1]["Dunn"],
                 rows[-1]["Prediction_Strength"], rows[-1]["Stability_corrected"]),
              flush=True)

    by_k = pd.DataFrame(rows)

    gap_df = gap_statistic(X, [1] + K_RANGE)
    gap_df.insert(0, "Space", space)
    gap_df.insert(0, "Dataset", name)
    k_gap = gap_select(gap_df)
    print("  gap statistic selects K=%d" % k_gap, flush=True)

    ps = by_k.set_index("K")["Prediction_Strength"]
    passing = [k for k in K_RANGE if ps[k] >= PS_THRESHOLD]

    selected = [
        ("Silhouette", "maximum", int(by_k.loc[by_k.Silhouette.idxmax(), "K"]), "no"),
        ("Davies-Bouldin", "minimum", int(by_k.loc[by_k.Davies_Bouldin.idxmin(), "K"]), "no"),
        ("Calinski-Harabasz", "maximum", int(by_k.loc[by_k.Calinski_Harabasz.idxmax(), "K"]), "no"),
        ("Gap statistic", "smallest K passing the 1-SE rule", k_gap, "YES"),
        ("Dunn", "maximum", int(by_k.loc[by_k.Dunn.idxmax(), "K"]), "no"),
        ("Prediction strength", "largest K with PS >= %.2f" % PS_THRESHOLD,
         max(passing) if passing else None, "implicitly"),
        ("Bootstrap stability (raw)", "maximum", int(by_k.loc[by_k.Stability_raw.idxmax(), "K"]), "no"),
        ("Bootstrap stability (null-corrected)", "maximum",
         int(by_k.loc[by_k.Stability_corrected.idxmax(), "K"]), "YES"),
        ("WCSS elbow", "maximum curvature", elbow_select(by_k.K, by_k.WCSS), "no"),
    ]
    sel = pd.DataFrame([{"Dataset": name, "Space": space, "Criterion": c,
                         "Rule": r, "Selected_K": k, "Null_corrected": nc}
                        for c, r, k, nc in selected])
    return by_k, sel, gap_df


def main():
    all_by_k, all_sel, all_gap = [], [], []
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
            a, b, c = analyse(name, X, space)
            all_by_k.append(a)
            all_sel.append(b)
            all_gap.append(c)

    by_k = pd.concat(all_by_k, ignore_index=True)
    sel = pd.concat(all_sel, ignore_index=True)
    gap = pd.concat(all_gap, ignore_index=True)

    out = config.RESULTS_PATH + "/"
    by_k.round(6).to_csv(out + "ksel_criteria_by_k.csv", index=False)
    sel.to_csv(out + "ksel_selected_k.csv", index=False)
    gap.round(6).to_csv(out + "ksel_gap_detail.csv", index=False)

    print("\n" + "=" * 78)
    print("WHICH K DOES EACH CRITERION SELECT?")
    print("=" * 78)
    pivot = sel.pivot_table(index=["Criterion", "Null_corrected"],
                            columns=["Dataset", "Space"], values="Selected_K")
    print(pivot.to_string())

    print("\nCriteria selecting K=3 or more, by dataset/space:")
    for (ds, sp), grp in sel.groupby(["Dataset", "Space"]):
        hits = grp[grp.Selected_K >= 3].Criterion.tolist()
        print("  %-18s %-7s %s" % (ds, sp, ", ".join(hits) if hits else "NONE"))

    print("\nSaved ksel_criteria_by_k.csv, ksel_selected_k.csv, ksel_gap_detail.csv")


if __name__ == "__main__":
    main()
