# -*- coding: utf-8 -*-
"""Five reference tables for numbers the manuscript quotes from no shipped file.

Why this exists
---------------
Tables 4, 9 and 11, the cleaning calendar of Section 3.1 and the K=2/K=3
cross-tabulation were each computed somewhere in the project's history, and each
was quoted in the paper, but none of them had a small file a reader could open
and check. This script derives all five from the same RFM tables the rest of the
package reads, and from the raw workbook for the cleaning counts.

Nothing here is new analysis. Every number is already in the manuscript, and the
script refuses to write anything unless each one reproduces: the expected values
are listed below and checked to the precision the paper prints them at. A single
mismatch stops the run before any file is written, because a reference file that
disagrees with the paper is worse than no file.

The configuration is the project's throughout
---------------------------------------------
  * log1p on Recency, Frequency and Monetary, then StandardScaler, both fitted
    once on the dataset being described. A projection is standardized on its own
    two columns only, which is how such studies report a projection.
  * K-Means++ with n_init=10 and random_state=42, K=3 except where a K=2
    partition is named.
  * PCA(n_components=2, random_state=42) where a PCA space is named.
  * segments are the project's: a cluster's composite score is
    (1 / mean Recency) * mean Frequency * mean Monetary, and the three clusters
    in descending score are Champions, At-Risk Customers, Lost Customers.
  * Table 4 describes UCI through its PCA partition and the other four datasets
    through their 3D partition, which is what the manuscript reports.

A note on Bakery
----------------
Table 4's Bakery row is the 3D partition, 25.08 / 22.17 / 52.75. The package
also ships cluster_profiles_bakery.csv, whose split is 29.87 / 26.67 / 43.47:
that file comes from the PCA pipeline, so the two describe different partitions
of the same customers. The same two-scan distinction is already flagged for
Table 5 in the README.

Outputs, all aggregate - no customer appears in any of them:
  rev2_table4_balance.csv          5 datasets x 3 segments
  rev2_table9_common_space.csv     2 partitions x 2 spaces
  rev2_table11_projections.csv     3D, PCA and the three 2D projections
  rev2_uci_cleaning_calendar.csv   the cleaning steps, revenue and the calendar
  rev2_k2_vs_k3_crosstab.csv       the K=3 PCA partition against the K=2 3D one

Usage:  python -m pca_and_validation.rev2_manuscript_reference_numbers
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
K = 3
COLS = ["Recency", "Frequency", "Monetary"]
SEGMENT_ORDER = ["Champions", "At-Risk Customers", "Lost Customers"]
R = config.RESULTS_PATH

DATASETS = [
    ("UCI", os.path.join("pca_official", "rfm_raw.csv"), "PCA"),
    ("Online Retail II", os.path.join("retail2", "rfm_raw_retail2.csv"), "3D"),
    ("Olist", os.path.join("olist", "rfm_raw_olist.csv"), "3D"),
    ("Superstore", os.path.join("superstore", "rfm_raw_superstore.csv"), "3D"),
    ("Bakery", os.path.join("bakery", "rfm_raw_bakery.csv"), "3D"),
]
PROJECTIONS = [("R-M", ["Recency", "Monetary"]),
               ("F-M", ["Frequency", "Monetary"]),
               ("R-F", ["Recency", "Frequency"])]

# --------------------------------------------------------------- what the paper says
EXP_T4 = {
    "UCI": (17.31, 38.27, 44.42),
    "Online Retail II": (20.69, 38.45, 40.86),
    "Olist": (20.74, 3.00, 76.26),
    "Superstore": (36.57, 39.34, 24.09),
    "Bakery": (25.08, 22.17, 52.75),
}
EXP_T4_REVENUE = {"Online Retail II": 74.66}          # Champions' share
EXP_T9 = {("3D", "3D"): (0.3365, 1.0483, 3625.3),
          ("3D", "PCA"): (0.3726, 0.9512, 4272.6),
          ("PCA", "3D"): (0.3379, 1.0498, 3618.6),
          ("PCA", "PCA"): (0.3754, 0.9506, 4282.2)}
EXP_T11 = {"R-M": (0.3403, 0.2893, 0.5540, 21.42, 70.74, 5.55),
           "F-M": (0.4588, 0.2972, 0.5425, 17.06, 69.32, 6.14),
           "R-F": (0.3919, 0.3023, 0.5781, 18.90, 63.16, 13.04)}
EXP_CLEAN = [("rows in the workbook", 541909),
             ("removed: missing CustomerID", 135080),
             ("removed: Quantity <= 0", 8905),
             ("removed: UnitPrice <= 0", 40),
             ("removed: duplicate rows", 5192),
             ("rows after cleaning", 392692),
             ("customers after aggregation", 4338)]
EXP_REVENUE = 8887208.89
EXP_TOP_MONTH = ("2011-11", 1156205.61)
EXP_SUNDAY = 785490.32
EXP_ABSENT_DAY = "Saturday"
EXP_CT = [[1927, 0], [745, 915], [0, 751]]
EXP_ARI_K2 = 0.412

failures = []


def check(label, want, got, decimals=4):
    """Equality at the precision the paper prints. Records, never raises."""
    ok = round(float(got), decimals) == round(float(want), decimals)
    if not ok:
        failures.append((label, want, got, decimals))
    print("    %-54s paper %12.*f   computed %12.*f   %s"
          % (label, decimals, want, decimals, got, "ok" if ok else "MISMATCH"))
    return ok


def _section(t):
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92, flush=True)


def scaled(df, cols):
    return StandardScaler().fit_transform(df[cols].apply(np.log1p))


def fit_km(X, k=K, seed=RANDOM_STATE):
    return KMeans(n_clusters=k, init="k-means++", n_init=10,
                  random_state=seed).fit_predict(X)


def profile(df, labels):
    """The project's cluster profile, ordered Champions, At-Risk, Lost."""
    t = df.copy()
    t["Cluster"] = labels
    p = t.groupby("Cluster").agg(
        Mean_Recency=("Recency", "mean"), Mean_Frequency=("Frequency", "mean"),
        Mean_Monetary=("Monetary", "mean"), Customers=("Recency", "size"),
        Revenue=("Monetary", "sum"))
    p["Pct_customers"] = p.Customers / p.Customers.sum() * 100.0
    p["Revenue_pct"] = p.Revenue / p.Revenue.sum() * 100.0
    p["Composite_Score"] = (1.0 / p.Mean_Recency) * p.Mean_Frequency * p.Mean_Monetary
    order = list(p.Composite_Score.sort_values(ascending=False).index)
    p["Segment"] = [SEGMENT_ORDER[order.index(i)] for i in p.index]
    return p.loc[order]


missing = [rel for _n, rel, _s in DATASETS
           if not os.path.exists(os.path.join(R, rel))]
if missing:
    print("skipping: these RFM tables are not there yet - %s" % ", ".join(missing))
    print("run their preprocessing scripts first")
    sys.exit(0)

# =====================================================================  TABLE 4
_section("TABLE 4: cluster balance and revenue share, by dataset")
t4_rows = []
uci = None
for name, rel, space in DATASETS:
    df = pd.read_csv(os.path.join(R, rel))
    X3 = scaled(df, COLS)
    X = (PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X3)
         if space == "PCA" else X3)
    labels = fit_km(X)
    if name == "UCI":
        uci = dict(df=df, X3=X3, X2=X, labels_pca=labels)
    p = profile(df, labels)
    print("\n  %s  (%d customers, %s partition, K=%d)"
          % (name, len(df), space, K))
    for cl in p.index:
        r = p.loc[cl]
        print("    %-20s n=%6d  %6.2f%%   revenue %6.2f%%"
              % (r.Segment, int(r.Customers), r.Pct_customers, r.Revenue_pct))
        t4_rows.append(dict(
            Dataset=name, Space=space, N=len(df), K=K, Segment=r.Segment,
            Raw_cluster_label=int(cl), Customers=int(r.Customers),
            Pct_customers=float(r.Pct_customers), Revenue=float(r.Revenue),
            Revenue_pct=float(r.Revenue_pct),
            Mean_Recency=float(r.Mean_Recency),
            Mean_Frequency=float(r.Mean_Frequency),
            Mean_Monetary=float(r.Mean_Monetary),
            Composite_Score=float(r.Composite_Score)))
    print("    against the paper:")
    for seg, want in zip(SEGMENT_ORDER, EXP_T4[name]):
        got = float(p[p.Segment == seg].Pct_customers.iloc[0])
        check("%s / %s, share of customers" % (name, seg), want, got, 2)
    if name in EXP_T4_REVENUE:
        got = float(p[p.Segment == "Champions"].Revenue_pct.iloc[0])
        check("%s / Champions, share of revenue" % name,
              EXP_T4_REVENUE[name], got, 2)

# =====================================================================  TABLE 9
_section("TABLE 9: UCI's two partitions, each scored in both spaces")
labels_3d = fit_km(uci["X3"])
labels_pca = uci["labels_pca"]
spaces = {"3D": uci["X3"], "PCA": uci["X2"]}
parts = {"3D": labels_3d, "PCA": labels_pca}
t9_rows = []
print("  %-10s %-7s %10s %10s %12s" % ("partition", "space", "Silhouette",
                                       "DBI", "CH"))
for pname, labels in parts.items():
    for sname, X in spaces.items():
        sil = float(silhouette_score(X, labels))
        dbi = float(davies_bouldin_score(X, labels))
        ch = float(calinski_harabasz_score(X, labels))
        print("  %-10s %-7s %10.4f %10.4f %12.1f" % (pname, sname, sil, dbi, ch))
        t9_rows.append(dict(Partition=pname, Scored_in_space=sname, N=len(uci["df"]),
                            K=K, Silhouette=sil, Davies_Bouldin=dbi,
                            Calinski_Harabasz=ch))
print("\n  against the paper:")
for (pname, sname), (w_sil, w_dbi, w_ch) in EXP_T9.items():
    row = [r for r in t9_rows
           if r["Partition"] == pname and r["Scored_in_space"] == sname][0]
    tag = "%s labels in %s" % (pname, sname)
    check("%s, Silhouette" % tag, w_sil, row["Silhouette"], 4)
    check("%s, Davies-Bouldin" % tag, w_dbi, row["Davies_Bouldin"], 4)
    check("%s, Calinski-Harabasz" % tag, w_ch, row["Calinski_Harabasz"], 1)

# ====================================================================  TABLE 11
_section("TABLE 11: the full spaces against the three 2D projections")
t11_rows = []


def projection_row(name, cols, X_own, labels):
    p = profile(uci["df"], labels)
    sil_own = float(silhouette_score(X_own, labels))
    sil_3d = float(silhouette_score(uci["X3"], labels))
    ari = float(adjusted_rand_score(labels_3d, labels))
    return dict(Row=name, Columns_used=" + ".join(cols), N=len(uci["df"]), K=K,
                Silhouette_own_space=sil_own, Silhouette_in_3D=sil_3d,
                ARI_vs_3D=ari,
                Smallest_segment_pct=float(p.Pct_customers.min()),
                Champions_revenue_pct=float(
                    p[p.Segment == "Champions"].Revenue_pct.iloc[0]),
                Lost_revenue_pct=float(
                    p[p.Segment == "Lost Customers"].Revenue_pct.iloc[0]))


t11_rows.append(projection_row("3D", COLS, uci["X3"], labels_3d))
t11_rows.append(projection_row("PCA", ["PC1", "PC2"], uci["X2"], labels_pca))
for name, cols in PROJECTIONS:
    X2 = scaled(uci["df"], cols)
    t11_rows.append(projection_row(name, cols, X2, fit_km(X2)))

print("  %-5s %11s %11s %9s %9s %11s %9s"
      % ("row", "Sil own", "Sil in 3D", "ARI v 3D", "min seg%", "Champ rev%",
         "Lost rev%"))
for r in t11_rows:
    print("  %-5s %11.4f %11.4f %9.4f %9.2f %11.2f %9.2f"
          % (r["Row"], r["Silhouette_own_space"], r["Silhouette_in_3D"],
             r["ARI_vs_3D"], r["Smallest_segment_pct"],
             r["Champions_revenue_pct"], r["Lost_revenue_pct"]))
print("\n  against the paper (the 3D and PCA rows are quoted in Tables 4 and 9):")
for name, want in EXP_T11.items():
    r = [x for x in t11_rows if x["Row"] == name][0]
    check("%s, Silhouette in its own space" % name, want[0],
          r["Silhouette_own_space"], 4)
    check("%s, Silhouette in 3D" % name, want[1], r["Silhouette_in_3D"], 4)
    check("%s, ARI against the 3D partition" % name, want[2], r["ARI_vs_3D"], 4)
    check("%s, smallest segment" % name, want[3], r["Smallest_segment_pct"], 2)
    check("%s, Champions' revenue" % name, want[4],
          r["Champions_revenue_pct"], 2)
    check("%s, Lost customers' revenue" % name, want[5],
          r["Lost_revenue_pct"], 2)

# ============================================================  THE CALENDAR
_section("THE CLEANING CALENDAR: %s" % config.DATA_PATH)
raw = pd.read_excel(config.DATA_PATH)
clean_rows = [dict(Section="cleaning", Item=EXP_CLEAN[0][0], Records=len(raw))]
df = raw
step_counts = []
for item, column, keep in (("removed: missing CustomerID", "CustomerID", None),
                           ("removed: Quantity <= 0", "Quantity", "pos"),
                           ("removed: UnitPrice <= 0", "UnitPrice", "pos"),
                           ("removed: duplicate rows", None, "dup")):
    if keep is None:
        n_removed = int(df[column].isnull().sum())
        df = df.dropna(subset=[column])
    elif keep == "pos":
        n_removed = int((df[column] <= 0).sum())
        df = df[df[column] > 0]
    else:
        n_removed = int(df.duplicated().sum())
        df = df.drop_duplicates()
    step_counts.append((item, n_removed))
    clean_rows.append(dict(Section="cleaning", Item=item, Records=n_removed))
df = df.copy()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
clean_rows.append(dict(Section="cleaning", Item="rows after cleaning",
                       Records=len(df)))
n_customers = int(df["CustomerID"].nunique())
clean_rows.append(dict(Section="cleaning", Item="customers after aggregation",
                       Records=n_customers))

total_revenue = float(df["TotalPrice"].sum())
clean_rows.append(dict(Section="revenue", Item="total revenue over the clean rows",
                       Value_GBP=total_revenue))
by_month = df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalPrice"].sum()
top_month = str(by_month.idxmax())
clean_rows.append(dict(Section="revenue", Item="highest month: %s" % top_month,
                       Value_GBP=float(by_month.max())))
by_day = df.groupby(df["InvoiceDate"].dt.day_name())["TotalPrice"].sum()
WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
        "Sunday"]
for day in WEEK:
    present = day in by_day.index
    clean_rows.append(dict(Section="calendar", Item="revenue on %s" % day,
                           Present=present,
                           Value_GBP=float(by_day[day]) if present else np.nan))

print("  step                                         records   remaining")
remaining = len(raw)
print("  %-42s %9d" % (EXP_CLEAN[0][0], remaining))
for item, n_removed in step_counts:
    remaining -= n_removed
    print("  %-42s %9d %11d" % (item, n_removed, remaining))
print("  %-42s %9d" % ("rows after cleaning", len(df)))
print("  %-42s %9d" % ("customers after aggregation", n_customers))
print("  total revenue %.2f, highest month %s at %.2f"
      % (total_revenue, top_month, by_month.max()))
print("  weekdays: %s"
      % ", ".join("%s %s" % (d, "%.2f" % by_day[d] if d in by_day.index
                             else "absent") for d in WEEK))
print("\n  against the paper:")
for (item, want), got in zip(EXP_CLEAN,
                             [len(raw)] + [n for _i, n in step_counts]
                             + [len(df), n_customers]):
    check(item, want, got, 0)
check("total revenue", EXP_REVENUE, total_revenue, 2)
check("highest month's revenue (%s)" % EXP_TOP_MONTH[0], EXP_TOP_MONTH[1],
      float(by_month.max()), 2)
check("revenue on Sunday", EXP_SUNDAY, float(by_day["Sunday"]), 2)
absent_ok = EXP_ABSENT_DAY not in by_day.index
if not absent_ok:
    failures.append(("%s is absent from the data" % EXP_ABSENT_DAY, 0, 1, 0))
print("    %-54s paper %12s   computed %12s   %s"
      % ("%s absent from the data" % EXP_ABSENT_DAY, "absent",
         "absent" if absent_ok else "present", "ok" if absent_ok else "MISMATCH"))
if top_month != EXP_TOP_MONTH[0]:
    failures.append(("highest month is %s" % EXP_TOP_MONTH[0], 0, 1, 0))

# =========================================================  K=2 AGAINST K=3
_section("THE K=2 PARTITION AGAINST THE K=3 ONE (UCI)")
labels_k2 = fit_km(uci["X3"], k=2)
ct = pd.crosstab(labels_pca, labels_k2)
ari_k2 = float(adjusted_rand_score(labels_pca, labels_k2))
seg_of = {int(cl): profile(uci["df"], labels_pca).loc[cl, "Segment"]
          for cl in sorted(set(labels_pca))}
print("  rows: the K=3 PCA partition in raw K-Means label order")
print("  %-6s %-20s %8s %10s %10s" % ("label", "segment", "size", "K2 c0", "K2 c1"))
ct_rows = []
for cl in ct.index:
    print("  %-6d %-20s %8d %10d %10d"
          % (cl, seg_of[int(cl)], int(ct.loc[cl].sum()), int(ct.loc[cl, 0]),
             int(ct.loc[cl, 1])))
    ct_rows.append(dict(K3_raw_label=int(cl), K3_segment=seg_of[int(cl)],
                        K3_size=int(ct.loc[cl].sum()),
                        In_K2_cluster_0=int(ct.loc[cl, 0]),
                        In_K2_cluster_1=int(ct.loc[cl, 1]),
                        ARI_K3PCA_vs_K2_3D=ari_k2))
print("  K=2 cluster sizes: %s" % np.bincount(labels_k2).tolist())
print("  ARI between the two partitions: %.4f" % ari_k2)
print("\n  against the paper:")
for i, (want_0, want_1) in enumerate(EXP_CT):
    check("crosstab row %d, customers in K=2 cluster 0" % i, want_0,
          ct_rows[i]["In_K2_cluster_0"], 0)
    check("crosstab row %d, customers in K=2 cluster 1" % i, want_1,
          ct_rows[i]["In_K2_cluster_1"], 0)
check("ARI between the K=3 PCA and K=2 3D partitions", EXP_ARI_K2, ari_k2, 3)

# ==============================================================  WRITE OR STOP
_section("VERDICT")
if failures:
    print("  STOP: %d quantity(ies) do not reproduce. Nothing was written."
          % len(failures))
    for label, want, got, dec in failures:
        print("    %-54s paper %.*f   computed %.*f" % (label, dec, want, dec, got))
    sys.exit(1)

print("  every checked quantity reproduces. Writing the five reference files.")
os.makedirs(R, exist_ok=True)
written = [
    ("rev2_table4_balance.csv", pd.DataFrame(t4_rows)),
    ("rev2_table9_common_space.csv", pd.DataFrame(t9_rows)),
    ("rev2_table11_projections.csv", pd.DataFrame(t11_rows)),
    ("rev2_uci_cleaning_calendar.csv",
     pd.DataFrame(clean_rows, columns=["Section", "Item", "Records", "Value_GBP",
                                       "Present"])),
    ("rev2_k2_vs_k3_crosstab.csv", pd.DataFrame(ct_rows)),
]
for fname, frame in written:
    path = os.path.join(R, fname)
    frame.to_csv(path, index=False)
    print("  written: %-38s %3d rows x %2d cols" % (path, len(frame),
                                                    len(frame.columns)))
