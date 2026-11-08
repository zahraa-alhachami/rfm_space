# -*- coding: utf-8 -*-
"""IJIES second review, item 4: the nine criteria, their settings, and Table 13.

Three things, all derived from files and from the code that wrote them:

  1. Each criterion's definition and its exact settings, with the constant's
     name in rfm_k_selection_null_corrected.py so a reader can check it.
  2. Table 13 recounted and SPLIT: criteria that compare against a null
     distribution, criteria that do not, and the one that corrects implicitly.
     Counting all nine together is what makes the table look like a vote.
  3. The two elbow heuristics worked through on the actual WCSS numbers, because
     they disagree: the discrete second-difference rule returns K=3 and the
     maximum-distance-from-the-chord rule returns K=5 on the same curve. Both
     are "the elbow" in the literature; neither is wrong; the disagreement is
     the finding.

Writes only outputs/results/rev2_* files. Changes no existing result.

Usage:  python rev2_k_criteria_report.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd

import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

R = config.RESULTS_PATH
SRC = "rfm_k_selection_null_corrected.py"

# The manuscript's datasets. Dunnhumby is not one of them: it entered the wider
# project as a fifth independent source for the methodological stages, so a
# table in a reviewer response must not count it. Olist IS in the manuscript but
# never entered this stage - the Dunn index and the gap statistic both need the
# full pairwise distance matrix, impractical at 93,358 customers - and that
# absence is reported rather than left to be noticed.
MANUSCRIPT_DATASETS = ["UCI", "Online Retail II", "Superstore", "Bakery"]
MANUSCRIPT_ALL = MANUSCRIPT_DATASETS + ["Olist"]

# name, what it measures, direction, the null question, settings (constant names
# are the ones in SRC), and the citation the setting comes from
CRITERIA = [
    ("Silhouette", "mean per-point (b-a)/max(a,b) over all customers",
     "higher is better", "no null comparison",
     "sklearn silhouette_score, Euclidean, on the same scaled space; exact up to "
     "20,000 customers, 10,000-point subsample above (rfm_scoring.py)",
     "Rousseeuw 1987"),
    ("Davies-Bouldin", "mean over clusters of the worst within-to-between scatter ratio",
     "lower is better", "no null comparison",
     "sklearn davies_bouldin_score, no parameters", "Davies & Bouldin 1979"),
    ("Calinski-Harabasz", "between-cluster over within-cluster dispersion, scaled by df",
     "higher is better", "no null comparison",
     "sklearn calinski_harabasz_score, no parameters", "Calinski & Harabasz 1974"),
    ("Gap statistic", "log W_k against its expectation under a structureless reference",
     "largest gap, by the 1-SE rule", "YES - this is the null correction",
     "B = GAP_B = 50 reference datasets per K; reference drawn uniformly over a "
     "box ALIGNED TO THE DATA'S PRINCIPAL AXES (the paper's recommended variant, "
     "not a bounding box in the raw axes); s_k = sd(ref) * sqrt(1 + 1/B); rule: "
     "the smallest k with Gap(k) >= Gap(k+1) - s_(k+1); K range 2..10 plus the "
     "k=1 baseline, so K=1 is a possible answer and is reported when it wins",
     "Tibshirani, Walther & Hastie 2001"),
    ("Dunn", "minimum single-linkage separation divided by the maximum cluster diameter",
     "higher is better", "no null comparison",
     "Euclidean on the full pairwise distance matrix of the same scaled space; "
     "separation = min over cluster pairs of the closest pair of points; diameter "
     "= max over clusters of the farthest pair. Worst-case geometry, not average",
     "Dunn 1974"),
    ("Prediction strength", "worst cluster's co-assignment rate between halves",
     "largest K above the threshold", "implicitly - it asks whether structure replicates",
     "reps = PS_REPS = 20 random 50/50 splits; the training half is clustered and "
     "its centroids assign the test half, the test half is clustered on its own, "
     "and the score is the worst cluster's within-cluster pair co-assignment rate, "
     "averaged over splits; threshold PS_THRESHOLD = 0.80. The threshold is a "
     "convention, so ksel_ps_thresholds.csv reports 0.70/0.80/0.85/0.90 as well",
     "Tibshirani & Walther 2005"),
    ("Bootstrap stability (raw)", "mean ARI between a refit on a resample and the reference labels",
     "higher is better", "no - raw",
     "BOOT_N = 100 resamples with replacement; the model is refitted on each "
     "resample and used to predict ALL customers, then compared with the "
     "reference labels by ARI", "Hennig 2007"),
    ("Bootstrap stability (null-corrected)", "the same statistic, rescaled against a random labelling",
     "higher is better", "YES",
     "the null is a random labelling with the reference's cluster sizes held "
     "EXACTLY; corrected = (raw - null) / (1 - null), the adjustment ARI itself "
     "applies to the Rand index", "Hennig 2007, with the ARI correction logic"),
    ("WCSS elbow", "the knee of the within-cluster sum-of-squares curve",
     "the knee", "no null comparison",
     "maximum distance from the chord joining the first and last point of the "
     "min-max normalised curve over K = 2..10. A second, complementary heuristic "
     "- the discrete second difference - is in rfm_analysis.py and returns a "
     "different K on the same curve; both are shown in section 3 below",
     "Thorndike 1953; Satopaa et al. 2011 for the chord form"),
]

print("=" * 100)
print("1. THE NINE CRITERIA, AS CONFIGURED IN %s" % SRC)
print("=" * 100)
rows = []
for name, measures, direction, null, settings, cite in CRITERIA:
    rows.append(dict(Criterion=name, Measures=measures, Direction=direction,
                     Null_comparison=null, Settings=settings, Citation=cite,
                     Source_script=SRC))
    print("\n%s  [%s]" % (name, cite))
    print("   measures : %s" % measures)
    print("   direction: %s" % direction)
    print("   null     : %s" % null)
    print("   settings : %s" % settings)
pd.DataFrame(rows).to_csv(os.path.join(R, "rev2_k_criteria_settings.csv"), index=False)

# ----------------------------------------------------------------- Table 13
print()
print("=" * 100)
print("2. TABLE 13 RECOUNTED, SPLIT BY WHETHER THE CRITERION COMPARES AGAINST A NULL")
print("=" * 100)
sel_all = pd.read_csv(os.path.join(R, "ksel_selected_k.csv"))
dropped = sorted(set(sel_all.Dataset.unique()) - set(MANUSCRIPT_ALL))
sel = sel_all[sel_all.Dataset.isin(MANUSCRIPT_DATASETS)].copy()
print("cells: %d  =  %d criteria x %d datasets x %d spaces"
      % (len(sel), sel.Criterion.nunique(), sel.Dataset.nunique(), sel.Space.nunique()))
print("datasets: %s" % ", ".join(sorted(sel.Dataset.unique())))
print("spaces  : %s" % ", ".join(sorted(sel.Space.unique())))
print("excluded, not in the manuscript: %s (%d cells)"
      % (", ".join(dropped) if dropped else "none",
         len(sel_all) - len(sel)))
print("Olist is in the manuscript but absent from this stage by design: the Dunn "
      "index and the gap statistic both need the full pairwise distance matrix, "
      "impractical at 93,358 customers.")

GROUPS = {"no": "not null-corrected", "YES": "null-corrected",
          "implicitly": "implicitly null-corrected"}
sel["Group"] = sel.Null_corrected.map(GROUPS)

out = []
for group in ("not null-corrected", "implicitly null-corrected", "null-corrected"):
    g = sel[sel.Group == group]
    if not len(g):
        continue
    counts = g.Selected_K.value_counts().sort_index()
    print("\n%s  (%s)" % (group.upper(), ", ".join(sorted(g.Criterion.unique()))))
    print("   cells: %d" % len(g))
    for k, c in counts.items():
        print("     K=%-2d %3d cells  (%.1f%%)" % (k, c, 100.0 * c / len(g)))
    for k, c in counts.items():
        out.append(dict(Group=group,
                        Criteria=", ".join(sorted(g.Criterion.unique())),
                        Cells=len(g), Selected_K=int(k), Count=int(c),
                        Pct_of_group=100.0 * c / len(g)))

print("\nALL NINE TOGETHER (the count the manuscript's Table 13 prints)")
allc = sel.Selected_K.value_counts().sort_index()
for k, c in allc.items():
    print("     K=%-2d %3d cells  (%.1f%%)" % (k, c, 100.0 * c / len(sel)))
    out.append(dict(Group="all nine", Criteria="all", Cells=len(sel),
                    Selected_K=int(k), Count=int(c), Pct_of_group=100.0 * c / len(sel)))

# One row per criterion as well, with raw and null-corrected stability apart:
# the grouped counts answer "do the null-corrected criteria agree?", and these
# answer "what did THIS criterion choose?" - a reader needs both.
print()
print("-" * 100)
print("EACH CRITERION ON ITS OWN (8 cells each: %d datasets x 2 spaces)"
      % len(MANUSCRIPT_DATASETS))
print("-" * 100)
for crit in [c[0] for c in CRITERIA]:
    g = sel[sel.Criterion == crit]
    if not len(g):
        print("  %-38s no cells" % crit)
        continue
    counts = g.Selected_K.value_counts().sort_index()
    print("  %-38s %s" % (crit, ", ".join("K=%d in %d" % (k, c)
                                          for k, c in counts.items())))
    for k, c in counts.items():
        out.append(dict(Group="criterion", Criteria=crit, Cells=len(g),
                        Selected_K=int(k), Count=int(c),
                        Pct_of_group=100.0 * c / len(g)))
pd.DataFrame(out).to_csv(os.path.join(R, "rev2_table13_recount.csv"), index=False)

print()
gap_k1 = int(((sel.Criterion == "Gap statistic") & (sel.Selected_K == 1)).sum())
gap_n = int((sel.Criterion == "Gap statistic").sum())
print("Reading: the K=2 majority is carried by the criteria that never compare "
      "against a null. The gap statistic - one of the two that do - puts %d of "
      "its %d cells at K=1, i.e. no cluster structure at all, which a combined "
      "count hides." % (gap_k1, gap_n))

# ------------------------------------------------------- the two elbow rules
print()
print("=" * 100)
print("3. WHY ONE ELBOW RULE SAYS K=3 AND THE OTHER SAYS K=5, ON THE SAME CURVE")
print("=" * 100)


def second_difference(ks, wcss):
    """rfm_analysis.py's complementary heuristic: the largest drop in the drop."""
    d1 = np.diff(wcss)                       # first differences, all negative
    d2 = np.diff(d1)                         # second differences
    return int(ks[int(np.argmax(d2)) + 1]), d1, d2


def chord_distance(ks, wcss):
    """rfm_k_selection_null_corrected.py's elbow_select: furthest from the chord."""
    x = (np.asarray(ks, float) - min(ks)) / (max(ks) - min(ks))
    y = (np.asarray(wcss, float) - min(wcss)) / (max(wcss) - min(wcss))
    num = np.abs((x[-1] - x[0]) * (y[0] - y) - (x[0] - x) * (y[-1] - y[0]))
    den = np.hypot(x[-1] - x[0], y[-1] - y[0])
    dist = num / den
    return int(np.asarray(ks)[int(np.argmax(dist))]), dist


elbow_rows = []
for label, path in [("UCI, PCA space (the adopted model)",
                     os.path.join(R, "pca_official", "k_selection_table.csv")),
                    ("UCI, 3D space", os.path.join(R, "k_selection_table.csv"))]:
    if not os.path.exists(path):
        print("\n%s: %s not found" % (label, path))
        continue
    t = pd.read_csv(path).sort_values("K")
    ks, w = t.K.tolist(), t.WCSS.tolist()
    k_chord, dist = chord_distance(ks, w)
    k_2nd, d1, d2 = second_difference(ks, w)
    print("\n%s   [%s]" % (label, path))
    print("   %-4s %14s %14s %14s %12s" % ("K", "WCSS", "1st diff", "2nd diff",
                                           "chord dist"))
    for i, k in enumerate(ks):
        print("   %-4d %14.2f %14s %14s %12.4f"
              % (k, w[i],
                 "%+.2f" % d1[i - 1] if 0 < i else "",
                 "%+.2f" % d2[i - 2] if 1 < i else "",
                 dist[i]))
    print("   chord rule (max distance from the first-to-last line): K = %d" % k_chord)
    print("   second-difference rule (max curvature of the discrete curve): K = %d" % k_2nd)
    print("   The curve has no sharp corner: WCSS falls steeply from K=2 and keeps")
    print("   falling, so 'the elbow' is whatever the chosen functional calls one.")
    print("   The chord rule is a GLOBAL statistic - it depends on the endpoints, so")
    print("   extending the K range moves it - while the second difference is LOCAL")
    print("   and picks the first place the improvement per extra cluster stops")
    print("   shrinking as fast. Neither is a null-corrected test, and neither")
    print("   licenses the claim that the data contain that many clusters.")
    elbow_rows.append(dict(Curve=label, Source=path, K_min=min(ks), K_max=max(ks),
                           K_chord_rule=k_chord, K_second_difference_rule=k_2nd,
                           WCSS_at_K2=w[0], WCSS_at_K3=w[1] if len(w) > 1 else np.nan,
                           Note="the two rules disagree on the same curve"))
pd.DataFrame(elbow_rows).to_csv(os.path.join(R, "rev2_elbow_rules.csv"), index=False)

print()
for f in ("rev2_k_criteria_settings.csv", "rev2_table13_recount.csv",
          "rev2_elbow_rules.csv"):
    print("written: %s" % os.path.join(R, f))
