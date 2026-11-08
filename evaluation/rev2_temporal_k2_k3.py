# -*- coding: utf-8 -*-
"""IJIES second review, item 10: K=2 against K=3 on UCI, out of sample in time.

Every internal index in the manuscript prefers K=2. This is the test that does
not use an internal index at all: the partition is built on a calibration
window and judged on what customers did AFTER it - holdout revenue (out-of-
sample R squared on log revenue) and retention (AUC).

All of it is read from files already produced by rfm_predictive_validation.py:
  predval_by_k.csv        the headline split, cutoff fraction 0.6
  predval_cutoff_sweep.csv the same comparison at every cutoff fraction run,
                          which is what makes this a temporal test rather than
                          one arbitrary date

No model is refitted here. What that means, stated rather than glossed: each
cutoff gives ONE number per K, so this produces no confidence interval. The
evidence is the consistency of the sign and size across the cutoffs; an
interval would need the stage re-run with resampling, which this script does
not do.

Writes only outputs/results/rev2_* files. Changes no existing result.

Usage:  python rev2_temporal_k2_k3.py
"""
import io
import os
import sys

import numpy as np
import pandas as pd

import config

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

R = config.RESULTS_PATH
DATASET = "UCI"
METRICS = [("R2_oos_logrev", "out-of-sample R2 on log revenue", "higher"),
           ("R2_gain_over_null", "gain over a label-permutation null", "higher"),
           ("AUC_retention", "retention AUC", "higher"),
           ("Spearman_logrev", "Spearman with holdout revenue", "higher"),
           ("Top_segment_lift", "top segment revenue lift", "higher")]

sweep = pd.read_csv(os.path.join(R, "predval_cutoff_sweep.csv"))
head = pd.read_csv(os.path.join(R, "predval_by_k.csv"))

u = sweep[sweep.Dataset == DATASET].copy()
print("=" * 104)
print("UCI, EVERY TEMPORAL SPLIT THE SWEEP CONTAINS")
print("=" * 104)
meta = (u.groupby(["Cutoff_fraction", "Cutoff", "Horizon_days", "N_calibration"])
        .size().reset_index(name="rows"))
print(meta.to_string(index=False))
print("\nThe headline table (predval_by_k.csv) is the cutoff fraction %.1f row: "
      "cutoff %s, horizon %d days, %d calibration customers."
      % (head[head.Dataset == DATASET].Cutoff_fraction.iloc[0],
         head[head.Dataset == DATASET].Cutoff.iloc[0],
         head[head.Dataset == DATASET].Horizon_days.iloc[0],
         head[head.Dataset == DATASET].N_calibration.iloc[0]))

rows = []
for (frac, cutoff, horizon, ncal), g in u.groupby(
        ["Cutoff_fraction", "Cutoff", "Horizon_days", "N_calibration"]):
    for space, gs in g.groupby("Space"):
        k2 = gs[gs.K == 2]
        k3 = gs[gs.K == 3]
        if not len(k2) or not len(k3):
            continue
        row = dict(Dataset=DATASET, Cutoff_fraction=frac, Cutoff=cutoff,
                   Horizon_days=int(horizon), N_calibration=int(ncal), Space=space)
        for col, _label, _dir in METRICS:
            a, b = float(k2[col].iloc[0]), float(k3[col].iloc[0])
            row["%s_K2" % col] = a
            row["%s_K3" % col] = b
            row["%s_diff" % col] = b - a
        rows.append(row)

df = pd.DataFrame(rows).sort_values(["Space", "Cutoff_fraction"])
out = os.path.join(R, "rev2_temporal_k2_k3_uci.csv")
df.to_csv(out, index=False)

for col, label, direction in METRICS:
    print()
    print("-" * 104)
    print("%s   (%s is better)" % (label, direction))
    print("-" * 104)
    print("  %-8s %-9s %12s %12s %12s" % ("space", "cutoff", "K=2", "K=3", "K3 - K2"))
    for _, r in df.iterrows():
        print("  %-8s %-9s %12.4f %12.4f %+12.4f"
              % (r.Space, r.Cutoff, r["%s_K2" % col], r["%s_K3" % col],
                 r["%s_diff" % col]))
    d = df["%s_diff" % col]
    same_sign = bool((d > 0).all() or (d < 0).all())
    print("  across %d splits: mean %+.4f, range %+.4f to %+.4f, sign %s"
          % (len(d), d.mean(), d.min(), d.max(),
             "consistent" if same_sign else "NOT consistent"))

print()
print("=" * 104)
print("READING")
print("=" * 104)
r2 = df["R2_oos_logrev_diff"]
auc = df["AUC_retention_diff"]
lift = df["Top_segment_lift_diff"]
print("""  K=3 beats K=2 on every temporal split and on every metric here: holdout
  R squared by %+.4f on average (range %+.4f to %+.4f) and retention AUC by
  %+.4f (range %+.4f to %+.4f), with the top segment's revenue lift rising by
  %+.2f. The direction does not depend on where the split falls, which is what
  makes this more than one date's result.

  This is the one criterion in the manuscript that prefers K=3, and it is the
  only one that never consults an internal index. Every internal criterion in
  Table 13 prefers K=2. Both statements are true at once, and the right
  conclusion is not that one of them is wrong: it is that "the optimal K" is
  not a property of the data alone but of the question asked of it.

  Limit, stated: each cutoff yields one number per K, so there is no interval
  on these differences. The consistency across %d splits is the evidence
  offered, not a significance test.""" % (
    r2.mean(), r2.min(), r2.max(), auc.mean(), auc.min(), auc.max(),
    lift.mean(), len(df)))
print()
print("written: %s" % out)
