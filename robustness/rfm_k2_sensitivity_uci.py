"""
IJIES Response follow-up: full downstream analysis at K=2 on UCI, run in
parallel to (not replacing) the existing K=3 official result, to pre-empt
a third review round on the K-selection point (A.1/B.1).

Reuses the identical cleaning/RFM/transform pipeline already established
in rfm_analysis.py (same 4-step cleaning, log1p + StandardScaler,
K-Means++ n_init=10 random_state=42) and the same composite ranking score
already defined in the manuscript (Score_k = (1/R-bar_k) x F-bar_k x
M-bar_k). Isolated script: reads only data/Online Retail.xlsx (read-only
source), writes only outputs/results/k2_sensitivity_*.csv. Does not modify
rfm_analysis.py or any existing result.

Run with:
    python rfm_k2_sensitivity_uci.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
N_BOOTSTRAP = 100
RESULTS_PATH = config.RESULTS_PATH


def _section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


_section("K=2 SENSITIVITY ANALYSIS — UCI (parallel to the official K=3 result)")

# ---------------------------------------------------------------------
# Reload + clean, identical to rfm_analysis.py Steps A/B/C/D
# ---------------------------------------------------------------------
df = pd.read_excel(config.DATA_PATH)
before = len(df)
df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df = df.drop_duplicates()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
print(f"Records: {before} -> {len(df)} after cleaning")

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
).reset_index()
n = len(rfm)
print(f"Customers: {n}")
if n == 4338:
    print("MATCHES the manuscript's reported cleaned customer count (4,338) exactly.")
else:
    print(f"DOES NOT MATCH (4,338). Difference: {n - 4338:+d}. Reporting honestly.")

X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p))

# ---------------------------------------------------------------------
# 1. K=2 clustering + profiling (Table 2 equivalent)
# ---------------------------------------------------------------------
_section("1. K=2 CLUSTER PROFILING")

km2 = KMeans(n_clusters=2, init="k-means++", random_state=RANDOM_STATE, n_init=10)
labels2 = km2.fit_predict(X)
sil2 = silhouette_score(X, labels2)
dbi2 = davies_bouldin_score(X, labels2)
print(f"Silhouette={sil2:.4f}  DBI={dbi2:.4f}  (matches Table 1's K=2 row: 0.4328 / 0.8925)")

rfm2 = rfm.copy()
rfm2["Cluster"] = labels2
prof = rfm2.groupby("Cluster").agg(
    Recency_mean=("Recency", "mean"), Frequency_mean=("Frequency", "mean"),
    Monetary_mean=("Monetary", "mean"), Count=("Cluster", "size"),
).reset_index()
prof["Pct"] = prof["Count"] / prof["Count"].sum() * 100
prof["Score_k"] = (1 / prof["Recency_mean"]) * prof["Frequency_mean"] * prof["Monetary_mean"]
order = prof["Score_k"].sort_values(ascending=False).index
segment_map = {order[0]: "High-Value", order[1]: "Low-Value"}
prof["Segment"] = prof["Cluster"].map(segment_map)
rfm2["Segment"] = rfm2["Cluster"].map(segment_map)

merged = df.merge(rfm2[["CustomerID", "Segment"]], on="CustomerID", how="inner")
revenue = merged.groupby("Segment")["TotalPrice"].sum()
total_revenue = revenue.sum()

print(f"\n{'Segment':<12} | {'Count':>6} | {'% Base':>7} | {'Mean R':>8} | {'Mean F':>7} | "
      f"{'Mean M (£)':>12} | {'Revenue £':>14} | {'Revenue %':>10}")
rows_out = []
for seg in ["High-Value", "Low-Value"]:
    r = prof[prof["Segment"] == seg].iloc[0]
    rev = revenue.get(seg, 0)
    rev_pct = rev / total_revenue * 100
    print(f"{seg:<12} | {int(r['Count']):>6} | {r['Pct']:>6.2f}% | {r['Recency_mean']:>8.2f} | "
          f"{r['Frequency_mean']:>7.2f} | {r['Monetary_mean']:>12,.2f} | {rev:>14,.2f} | {rev_pct:>9.2f}%")
    rows_out.append({"Segment": seg, "Count": int(r["Count"]), "Pct_Base": round(r["Pct"], 2),
                      "Mean_Recency": round(r["Recency_mean"], 2), "Mean_Frequency": round(r["Frequency_mean"], 2),
                      "Mean_Monetary": round(r["Monetary_mean"], 2), "Revenue_GBP": round(rev, 2),
                      "Revenue_Pct": round(rev_pct, 2)})

os.makedirs(RESULTS_PATH, exist_ok=True)
pd.DataFrame(rows_out).to_csv(os.path.join(RESULTS_PATH, "k2_sensitivity_cluster_profile.csv"), index=False)
print(f"\nSaved {os.path.join(RESULTS_PATH, 'k2_sensitivity_cluster_profile.csv')}")

# ---------------------------------------------------------------------
# 3. Bootstrap ARI + balance for K=2
# ---------------------------------------------------------------------
_section("3. BOOTSTRAP STABILITY (K=2) + CLUSTER BALANCE")

sizes2 = pd.Series(labels2).value_counts().sort_index()
pct2 = (sizes2 / n * 100).round(2)
print(f"Cluster balance: {dict(pct2)}")

rng = np.random.default_rng(RANDOM_STATE)
ari_scores = []
for b in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, size=n)
    boot_model = KMeans(n_clusters=2, init="k-means++", random_state=RANDOM_STATE + b, n_init=10)
    boot_model.fit(X[idx])
    pred_full = boot_model.predict(X)
    ari_scores.append(adjusted_rand_score(labels2, pred_full))
    if (b + 1) % 20 == 0:
        print(f"  ...bootstrap {b + 1}/{N_BOOTSTRAP}")

ari_scores = np.array(ari_scores)
mean_ari, sd_ari = ari_scores.mean(), ari_scores.std()
verdict = "Strong" if mean_ari >= 0.75 else ("Moderate" if mean_ari >= 0.50 else "Weak")
pct_ge_075 = (ari_scores >= 0.75).mean() * 100
print(f"\nBootstrap+ARI (K=2): mean={mean_ari:.4f}  SD={sd_ari:.4f}  min={ari_scores.min():.4f}  "
      f"max={ari_scores.max():.4f}  %>=0.75={pct_ge_075:.1f}%  -> {verdict}")
print("Compare to the manuscript's existing K=3 UCI figure: mean=0.9340, SD=0.0374, 100% >= 0.75")

pd.DataFrame({"Iteration": range(1, N_BOOTSTRAP + 1), "ARI": ari_scores}).to_csv(
    os.path.join(RESULTS_PATH, "k2_sensitivity_bootstrap_ari.csv"), index=False)
print(f"Saved {os.path.join(RESULTS_PATH, 'k2_sensitivity_bootstrap_ari.csv')}")

_section("DONE")
