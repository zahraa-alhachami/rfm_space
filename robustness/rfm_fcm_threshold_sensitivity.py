"""
IJIES Response, Task A.4: fuzzy c-means ambiguity-threshold sensitivity
sweep. Exact configuration as specified in the Response document and as
originally used in rfm_fcm_analysis.py: scikit-fuzzy cmeans, m=2.0,
error=0.005, maxiter=1000, seed=42, on the UCI 3D standardized RFM space.

The original membership matrix was not persisted to disk in any prior
session, so this script refits FCM ONCE (identical config, therefore
identical result to the original run) and saves the membership matrix,
then recomputes the ambiguous-customer percentage and the most-ambiguous
segment pair at five candidate top-two-membership-gap thresholds from
that single fit -- not five separate clusterings.

Isolated script: reads only outputs/results/rfm_raw.csv. Writes only
outputs/results/fcm_threshold_sensitivity.csv and
outputs/results/fcm_membership_matrix_uci.csv. Does not modify
rfm_fcm_analysis.py or any existing result.

Run with:
    python rfm_fcm_threshold_sensitivity.py
"""
import os
import sys

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
C = 3
FCM_M = 2.0
FCM_ERROR = 0.005
FCM_MAXITER = 1000
RAW_PATH = os.path.join(config.RESULTS_PATH, "rfm_raw.csv")
SEGMENT_ORDER = ["Champions", "At-Risk Customers", "Lost Customers"]


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


_section("TASK 3: FUZZY C-MEANS THRESHOLD SENSITIVITY SWEEP (UCI)")

rfm = pd.read_csv(RAW_PATH)
print(f"Loaded {RAW_PATH}: {len(rfm)} customers.")

X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p))

print("\nRefitting FCM once (membership matrix was not previously saved to disk):")
print(f"  m={FCM_M}, error={FCM_ERROR}, maxiter={FCM_MAXITER}, seed={RANDOM_STATE}, c={C}")
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X.T, c=C, m=FCM_M, error=FCM_ERROR, maxiter=FCM_MAXITER, seed=RANDOM_STATE
)
hard_labels = np.argmax(u, axis=0)
sil = silhouette_score(X, hard_labels)
print(f"Fuzzy Partition Coefficient: {fpc:.4f}")
print(f"Silhouette after defuzzification: {sil:.4f} (expect ~0.3241, matching the original run, to confirm "
      f"this refit reproduces the manuscript's reported number)")

# Rank fuzzy clusters by composite score (same rule as the hard K-Means++ segments)
# to translate cluster indices 0/1/2 into Champions/At-Risk/Lost consistently.
rfm_log = rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p)
scaler = StandardScaler().fit(rfm_log)
centers_raw = np.expm1(scaler.inverse_transform(cntr))
composite = (1 / np.clip(centers_raw[:, 0], 1e-9, None)) * centers_raw[:, 1] * centers_raw[:, 2]
order = np.argsort(-composite)
cluster_to_segment = {int(idx): SEGMENT_ORDER[rank] for rank, idx in enumerate(order)}
print("Cluster -> Segment mapping (by composite score):", cluster_to_segment)

membership_df = pd.DataFrame(u.T, columns=[f"membership_{cluster_to_segment[i]}" for i in range(C)])
membership_df.insert(0, "CustomerID", rfm["CustomerID"].values)
out_u_path = os.path.join(config.RESULTS_PATH, "fcm_membership_matrix_uci.csv")
membership_df.to_csv(out_u_path, index=False)
print(f"Saved membership matrix to {out_u_path}")

sorted_u = np.sort(u, axis=0)[::-1, :]
top1, top2 = sorted_u[0, :], sorted_u[1, :]
gap = top1 - top2

sorted_idx = np.argsort(-u, axis=0)
top1_cluster = sorted_idx[0, :]
top2_cluster = sorted_idx[1, :]

_section("THRESHOLD SWEEP (single fit, five candidate thresholds)")
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
sweep_rows = []
for thr in thresholds:
    ambiguous_mask = gap < thr
    pct_ambiguous = ambiguous_mask.mean() * 100

    if ambiguous_mask.sum() > 0:
        pairs = [
            tuple(sorted((cluster_to_segment[a], cluster_to_segment[b])))
            for a, b in zip(top1_cluster[ambiguous_mask], top2_cluster[ambiguous_mask])
        ]
        pair_counts = pd.Series(pairs).value_counts()
        top_pair = pair_counts.index[0]
        top_pair_pct = pair_counts.iloc[0] / ambiguous_mask.sum() * 100
        top_pair_str = f"{top_pair[0]} <-> {top_pair[1]} ({top_pair_pct:.1f}% of ambiguous customers)"
    else:
        top_pair_str = "N/A (no ambiguous customers at this threshold)"

    print(f"threshold={thr:.2f} | ambiguous={pct_ambiguous:5.2f}% | most-ambiguous boundary: {top_pair_str}")
    sweep_rows.append({"Threshold": thr, "Pct_Ambiguous": round(pct_ambiguous, 2),
                        "Most_Ambiguous_Boundary": top_pair_str})

sweep_table = pd.DataFrame(sweep_rows)
out_path = os.path.join(config.RESULTS_PATH, "fcm_threshold_sensitivity.csv")
sweep_table.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")

_section("VERDICT: DOES THE CORE CONCLUSION CHANGE ACROSS THIS RANGE?")
boundaries = [r["Most_Ambiguous_Boundary"].split(" (")[0] for r in sweep_rows]
if len(set(boundaries)) == 1:
    print(f"STABLE: the At-Risk/Lost boundary (or whichever is reported above) remains the most ambiguous "
          f"pairwise boundary at every tested threshold (0.10 through 0.30). The manuscript's core finding "
          f"does not depend on the specific 0.20 cutoff.")
else:
    print("NOT STABLE: the identity of the most-ambiguous boundary changes somewhere in this threshold "
          "range -- see table above for exactly where. This qualifies the manuscript's claim and should "
          "be reported honestly rather than omitted.")
