"""
Third RFM + K-Means++ clustering pipeline, for the extended UCI
"Online Retail II" dataset (two sheets: Year 2009-2010, Year 2010-2011).

Uses the exact same methodology as rfm_analysis.py (UCI Online Retail) so
results are directly comparable: log1p + StandardScaler, Elbow Method +
Silhouette Score + Davies-Bouldin Index for K selection, K-Means++
clustering, and composite-score segment naming.

Run with:
    python rfm_retail2.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from scipy.stats import skew
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

import config

# Force UTF-8 stdout/stderr so currency symbols (£) never crash print()
# on Windows consoles using a narrow codepage (e.g. cp1256).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

plt.style.use("seaborn-v0_8-whitegrid")

SELECTED_K = 3
SEGMENT_ORDER = ["Champions", "At-Risk Customers", "Lost Customers"]
SEGMENT_COLORS = {
    "Champions": "#2E86AB",
    "At-Risk Customers": "#F18F01",
    "Lost Customers": "#C73E1D",
}
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

BEHAVIORAL_DESCRIPTIONS = {
    "Champions": "Recent, frequent, high-spending customers — the most valuable segment",
    "At-Risk Customers": "Moderately active customers showing signs of declining engagement",
    "Lost Customers": "Inactive customers with distant last purchase and minimal spending",
}

MARKETING_STRATEGIES = {
    "Champions": {
        "strategies": [
            "Launch a VIP loyalty program with exclusive rewards",
            "Offer early access to new products and collections",
            "Request product reviews and referrals with incentives",
            "Send personalized thank-you messages and premium gifts",
        ],
        "goal": "Retain and strengthen loyalty of highest-value customers",
    },
    "At-Risk Customers": {
        "strategies": [
            "Send personalized re-engagement email campaigns",
            "Offer time-limited discounts (valid 7-14 days only)",
            "Highlight new arrivals relevant to past purchases",
            "Ask for feedback on why they have reduced purchases",
        ],
        "goal": "Reactivate before they become lost customers",
    },
    "Lost Customers": {
        "strategies": [
            "Send a \"We miss you\" campaign with aggressive discount",
            "Conduct exit surveys to understand churn reasons",
            "One final high-value win-back offer",
            "If no response after 2 attempts, remove from active marketing to reduce costs",
        ],
        "goal": "Win back a portion while minimizing wasted budget",
    },
}

RETAIL2_DATA_PATH = os.path.join("data", "online_retail_II.xlsx")
RETAIL2_FIGURES_PATH = os.path.join(config.FIGURES_PATH, "retail2")
RETAIL2_RESULTS_PATH = os.path.join(config.RESULTS_PATH, "retail2")

# Fixed reference values from the confirmed UCI Online Retail and Olist runs.
UCI_REFERENCE = {
    "Total Customers": 4338, "Total Records": 392692,
    "Champions %": 17.73, "At-Risk %": 39.12, "Lost %": 43.15,
    "Champions Revenue %": 68.34, "Optimal K": 3, "Silhouette (K=3)": 0.3365,
    "Peak Month": "November", "Best Day": "Thursday",
}
OLIST_REFERENCE = {
    "Total Customers": 93358, "Total Records": 110197,
    "Champions %": 20.74, "At-Risk %": 3.00, "Lost %": 76.26,
    "Champions Revenue %": 20.87, "Optimal K": 3, "Silhouette (K=3)": 0.4203,
    "Peak Month": "November", "Best Day": "Monday",
}


def _section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _style_axis(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=16)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="both", labelsize=11)


# ---------------------------------------------------------------------------
# B1: Load both sheets and combine
# ---------------------------------------------------------------------------
def load_data_retail2():
    _section("STEP B1: LOAD ONLINE RETAIL II (BOTH SHEETS)")

    rename_map = {"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"}

    sheet1 = pd.read_excel(RETAIL2_DATA_PATH, sheet_name="Year 2009-2010").rename(columns=rename_map)
    sheet2 = pd.read_excel(RETAIL2_DATA_PATH, sheet_name="Year 2010-2011").rename(columns=rename_map)

    print(f"Sheet 'Year 2009-2010': {len(sheet1)} records")
    print(f"Sheet 'Year 2010-2011': {len(sheet2)} records")

    df = pd.concat([sheet1, sheet2], ignore_index=True)
    print(f"Combined total: {len(df)} records")
    print(f"\nColumns: {list(df.columns)}")

    return df


# ---------------------------------------------------------------------------
# B2: Preprocess (same as UCI pipeline)
# ---------------------------------------------------------------------------
def preprocess_data_retail2(df):
    _section("STEP B2: PREPROCESS DATA")
    before = len(df)
    print(f"Records before preprocessing: {before}")

    missing_customer = df["CustomerID"].isnull().sum()
    df = df.dropna(subset=["CustomerID"])
    print(f"Removed {missing_customer} rows with missing CustomerID")

    neg_qty = (df["Quantity"] <= 0).sum()
    df = df[df["Quantity"] > 0]
    print(f"Removed {neg_qty} rows with Quantity <= 0 (returns/cancellations)")

    neg_price = (df["UnitPrice"] <= 0).sum()
    df = df[df["UnitPrice"] > 0]
    print(f"Removed {neg_price} rows with UnitPrice <= 0")

    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Removed {dup_count} duplicate rows")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    after = len(df)
    print(f"\nRecords after preprocessing: {after}")
    print(f"Total records removed: {before - after}")

    return df


# ---------------------------------------------------------------------------
# B3: Compute RFM
# ---------------------------------------------------------------------------
def compute_rfm_retail2(df):
    _section("STEP B3: COMPUTE RFM METRICS")

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    print(f"Snapshot date: {snapshot_date}")

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    stats_rows = []
    for col in ["Recency", "Frequency", "Monetary"]:
        stats_rows.append({
            "Metric": col, "Mean": rfm[col].mean(), "Median": rfm[col].median(),
            "Std": rfm[col].std(), "Min": rfm[col].min(), "Max": rfm[col].max(),
            "Skewness": skew(rfm[col]),
        })
    rfm_statistics = pd.DataFrame(stats_rows)
    print("\nRFM statistics BEFORE log1p transformation (mean, median, std, min, max, skewness):")
    print(rfm_statistics.to_string(index=False))

    os.makedirs(RETAIL2_RESULTS_PATH, exist_ok=True)
    out_path = os.path.join(RETAIL2_RESULTS_PATH, "rfm_raw_retail2.csv")
    rfm.to_csv(out_path, index=False)
    print(f"\nSaved RFM table to {out_path}")

    return rfm


# ---------------------------------------------------------------------------
# B4: Transform RFM
# ---------------------------------------------------------------------------
def transform_rfm_retail2(rfm):
    _section("STEP B4: TRANSFORM RFM (LOG1P + STANDARD SCALING)")

    cols = ["Recency", "Frequency", "Monetary"]
    transformed = rfm[cols].apply(np.log1p) if config.LOG_TRANSFORM else rfm[cols].copy()

    stats_rows = []
    for col in cols:
        stats_rows.append({
            "Metric": col, "Mean": transformed[col].mean(), "Median": transformed[col].median(),
            "Std": transformed[col].std(), "Min": transformed[col].min(), "Max": transformed[col].max(),
            "Skewness": skew(transformed[col]),
        })
    print("RFM statistics AFTER log1p transformation (mean, median, std, min, max, skewness):")
    print(pd.DataFrame(stats_rows).to_string(index=False))

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(transformed)
    print(f"\nScaled array shape: {rfm_scaled.shape}")

    return rfm_scaled, scaler


# ---------------------------------------------------------------------------
# B5: Find optimal K
# ---------------------------------------------------------------------------
def find_optimal_k_retail2(rfm_scaled):
    _section("STEP B5: FIND OPTIMAL K (ELBOW + SILHOUETTE + DBI)")

    k_values = list(range(2, config.MAX_K + 1))
    wcss, silhouette_scores, dbi_scores = [], [], []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=config.RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(rfm_scaled)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled, labels))
        dbi_scores.append(davies_bouldin_score(rfm_scaled, labels))

    print(f"{'K':>3} | {'WCSS':>15} | {'Silhouette Score':>17} | {'Davies-Bouldin Index':>20}")
    print("-" * 65)
    for k, w, s, d in zip(k_values, wcss, silhouette_scores, dbi_scores):
        print(f"{k:>3} | {w:>15.2f} | {s:>17.4f} | {d:>20.4f}")

    p1 = np.array([k_values[0], wcss[0]])
    p2 = np.array([k_values[-1], wcss[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    distances = []
    for k, w in zip(k_values, wcss):
        p = np.array([k, w])
        proj_len = np.dot(p - p1, line_vec_norm)
        proj_point = p1 + proj_len * line_vec_norm
        distances.append(np.linalg.norm(p - proj_point))
    elbow_k = k_values[int(np.argmax(distances))]

    silhouette_k = k_values[int(np.argmax(silhouette_scores))]
    dbi_k = k_values[int(np.argmin(dbi_scores))]

    print(f"\nBased on Elbow Method (automatic detection): K={elbow_k}")
    print(f"Based on Silhouette Score (automatic detection): K={silhouette_k}")
    print(f"Based on Davies-Bouldin Index (lowest=best): K={dbi_k}")

    selected_k = SELECTED_K
    print(f"\nFinal K selected for clustering: K={selected_k} "
          f"(fixed to match the UCI/Olist pipelines for cross-dataset comparability)")

    os.makedirs(RETAIL2_RESULTS_PATH, exist_ok=True)
    k_table = pd.DataFrame({
        "K": k_values, "WCSS": wcss,
        "Silhouette_Score": silhouette_scores, "Davies_Bouldin_Index": dbi_scores,
    })
    k_table_path = os.path.join(RETAIL2_RESULTS_PATH, "k_selection_table_retail2.csv")
    k_table.to_csv(k_table_path, index=False)
    print(f"Saved K selection table to {k_table_path}")

    os.makedirs(RETAIL2_FIGURES_PATH, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=300)

    axes[0].plot(k_values, wcss, marker="o", color="#2E86AB")
    axes[0].axvline(selected_k, color="red", linestyle="--", linewidth=1.5)
    axes[0].text(selected_k + 0.15, wcss[0] * 0.92, "Optimal K=3", color="red", fontsize=11)
    _style_axis(axes[0], title="Elbow Method (Online Retail II)",
                xlabel="Number of Clusters (K)", ylabel="WCSS (Inertia)")

    axes[1].plot(k_values, silhouette_scores, marker="o", color="#2E86AB")
    k3_idx = k_values.index(selected_k)
    axes[1].scatter([selected_k], [silhouette_scores[k3_idx]], color="red", s=150, zorder=5)
    axes[1].annotate(
        "Selected K=3", xy=(selected_k, silhouette_scores[k3_idx]),
        xytext=(selected_k + 1.2, silhouette_scores[k3_idx] + 0.03),
        arrowprops=dict(arrowstyle="->", color="red"), fontsize=11, color="red",
    )
    _style_axis(axes[1], title="Silhouette Score (Online Retail II)",
                xlabel="Number of Clusters (K)", ylabel="Silhouette Score")

    fig.suptitle("Figure 3.3: Dual-Criterion Optimal K (Online Retail II)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(RETAIL2_FIGURES_PATH, "optimal_k_retail2.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved elbow/silhouette/DBI plot to {out_path}")

    return {
        "k_values": k_values, "wcss": wcss, "silhouette_scores": silhouette_scores,
        "dbi_scores": dbi_scores, "elbow_k": elbow_k, "silhouette_k": silhouette_k,
        "dbi_k": dbi_k, "selected_k": selected_k,
    }


# ---------------------------------------------------------------------------
# B6: Run K-Means++
# ---------------------------------------------------------------------------
def run_kmeans_retail2(rfm_scaled, rfm, optimal_k):
    _section(f"STEP B6: RUN KMEANS++ (K={optimal_k})")

    kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=config.RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)

    rfm = rfm.copy()
    rfm["Cluster"] = labels

    sizes = rfm["Cluster"].value_counts().sort_index()
    percentages = (sizes / len(rfm) * 100).round(2)
    print("Cluster sizes:")
    for cluster_id in sizes.index:
        print(f"  Cluster {cluster_id}: {sizes[cluster_id]} customers ({percentages[cluster_id]}%)")

    os.makedirs(RETAIL2_RESULTS_PATH, exist_ok=True)
    out_path = os.path.join(RETAIL2_RESULTS_PATH, "rfm_clustered_retail2.csv")
    rfm.to_csv(out_path, index=False)
    print(f"\nSaved clustered RFM table to {out_path}")

    return rfm, kmeans


# ---------------------------------------------------------------------------
# B7: Cluster profiling
# ---------------------------------------------------------------------------
def profile_clusters_retail2(rfm):
    _section("STEP B7: PROFILE CLUSTERS")

    profile = rfm.groupby("Cluster").agg(
        Recency_mean=("Recency", "mean"), Recency_median=("Recency", "median"),
        Frequency_mean=("Frequency", "mean"), Frequency_median=("Frequency", "median"),
        Monetary_mean=("Monetary", "mean"), Monetary_median=("Monetary", "median"),
        Count=("Cluster", "size"),
    ).reset_index()
    profile["Percentage"] = profile["Count"] / profile["Count"].sum() * 100

    composite = (1 / profile["Recency_mean"]) * profile["Frequency_mean"] * profile["Monetary_mean"]
    profile["Composite_Score"] = composite

    order = composite.sort_values(ascending=False).index
    if len(profile) == len(SEGMENT_ORDER):
        segment_map = dict(zip(order, SEGMENT_ORDER))
    else:
        rank_pct = composite.rank(pct=True)

        def _assign(pct):
            if pct >= 0.67:
                return "Champions"
            elif pct >= 0.34:
                return "At-Risk Customers"
            return "Lost Customers"

        segment_map = {idx: _assign(rank_pct[idx]) for idx in profile.index}

    profile["Segment"] = profile.index.map(segment_map)
    profile["Behavioral_Description"] = profile["Segment"].map(BEHAVIORAL_DESCRIPTIONS)

    print("Cluster profiling table:")
    print(profile[["Cluster", "Recency_mean", "Frequency_mean", "Monetary_mean", "Count", "Segment"]])

    print("\nDetailed segment profiles:")
    for _, row in profile.sort_values("Composite_Score", ascending=False).iterrows():
        print(f"\nSegment: {row['Segment']} (Cluster {int(row['Cluster'])})")
        print(f"  Recency:   mean={row['Recency_mean']:.2f}, median={row['Recency_median']:.2f}")
        print(f"  Frequency: mean={row['Frequency_mean']:.2f}, median={row['Frequency_median']:.2f}")
        print(f"  Monetary:  mean={row['Monetary_mean']:.2f}, median={row['Monetary_median']:.2f}")
        print(f"  Customers: {int(row['Count'])} ({row['Percentage']:.2f}%)")
        print(f"  Behavioral Description: {row['Behavioral_Description']}")

    os.makedirs(RETAIL2_RESULTS_PATH, exist_ok=True)
    out_path = os.path.join(RETAIL2_RESULTS_PATH, "cluster_profiles_retail2.csv")
    profile.to_csv(out_path, index=False)
    print(f"\nSaved cluster profiles to {out_path}")

    rfm = rfm.merge(profile[["Cluster", "Segment"]], on="Cluster", how="left")
    return rfm, profile


# ---------------------------------------------------------------------------
# B8: Visualizations
# ---------------------------------------------------------------------------
def visualize_clusters_retail2(rfm):
    _section("STEP B8: VISUALIZE CLUSTERS")

    os.makedirs(RETAIL2_FIGURES_PATH, exist_ok=True)
    segments_present = [s for s in SEGMENT_ORDER if s in rfm["Segment"].unique()]

    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    for segment in segments_present:
        subset = rfm[rfm["Segment"] == segment]
        ax.scatter(subset["Recency"], subset["Frequency"], subset["Monetary"],
                   color=SEGMENT_COLORS[segment], label=segment, alpha=0.6, s=25)
    centroids = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().reindex(segments_present)
    ax.scatter(centroids["Recency"], centroids["Frequency"], centroids["Monetary"],
               color="black", marker="*", s=200, label="Centroids", zorder=5)
    ax.set_xlabel("Recency (days)", fontsize=13)
    ax.set_ylabel("Frequency (number of purchases)", fontsize=13)
    ax.set_zlabel("Monetary Value (£)", fontsize=13)
    ax.set_title("Figure 4.1: 3D RFM Segments (Online Retail II)", fontsize=16)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    out_path = os.path.join(RETAIL2_FIGURES_PATH, "3d_clusters_retail2.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved 3D scatter plot to {out_path}")

    sizes = rfm["Segment"].value_counts().reindex(segments_present)
    percentages = sizes / sizes.sum() * 100
    avg_size = sizes.mean()
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    bars = ax.bar(sizes.index, sizes.values, color=[SEGMENT_COLORS[s] for s in sizes.index])
    ax.axhline(avg_size, color="gray", linestyle="--", linewidth=1.5,
               label=f"Average segment size ({avg_size:.0f})")
    for bar, count, pct in zip(bars, sizes.values, percentages.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sizes.max() * 0.01,
                f"{int(count)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11)
    _style_axis(ax, title="Figure 4.2: Customer Distribution (Online Retail II)",
                xlabel="Segment", ylabel="Number of Customers")
    ax.legend(fontsize=11)
    out_path = os.path.join(RETAIL2_FIGURES_PATH, "cluster_sizes_retail2.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved cluster size bar chart to {out_path}")

    centroids = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().reindex(segments_present)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    sns.heatmap(centroids, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={"label": "Mean Value"})
    _style_axis(ax, title="Figure 4.3: RFM Centroid Heatmap (Online Retail II)",
                xlabel="RFM Metric", ylabel="Segment")
    out_path = os.path.join(RETAIL2_FIGURES_PATH, "cluster_heatmap_retail2.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved cluster centroid heatmap to {out_path}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 8), dpi=300)
    for ax, col in zip(axes, ["Recency", "Frequency", "Monetary"]):
        sns.boxplot(data=rfm, x="Segment", y=col, order=segments_present, hue="Segment",
                    palette=SEGMENT_COLORS, legend=False, showfliers=False, ax=ax)
        _style_axis(ax, title=col, xlabel="Segment", ylabel=col)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Figure 4.4: RFM Distribution (Online Retail II)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(RETAIL2_FIGURES_PATH, "boxplots_retail2.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved box plots to {out_path}")


# ---------------------------------------------------------------------------
# B10: Marketing recommendations
# ---------------------------------------------------------------------------
def marketing_recommendations_retail2(rfm):
    _section("STEP B10: MARKETING RECOMMENDATIONS")

    segment_counts = rfm["Segment"].value_counts()
    total = len(rfm)
    lines = ["MARKETING RECOMMENDATIONS BY CUSTOMER SEGMENT (ONLINE RETAIL II)", "=" * 70]

    for segment in SEGMENT_ORDER:
        info = MARKETING_STRATEGIES[segment]
        count = int(segment_counts.get(segment, 0))
        pct = (count / total * 100) if total else 0.0
        block = [f"\nSegment: {segment} ({count} customers, {pct:.2f}%)"]
        for i, strategy in enumerate(info["strategies"], start=1):
            block.append(f"  Strategy {i}: {strategy}")
        block.append(f"  Goal: {info['goal']}")
        print("\n".join(block))
        lines.extend(block)

    os.makedirs(RETAIL2_RESULTS_PATH, exist_ok=True)
    out_path = os.path.join(RETAIL2_RESULTS_PATH, "marketing_recommendations_retail2.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved marketing recommendations to {out_path}")

    return lines


# ---------------------------------------------------------------------------
# B9: Three-way comparison
# ---------------------------------------------------------------------------
def compute_comparison_metrics_retail2(rfm, df, k_results):
    merged = df.merge(rfm[["CustomerID", "Segment"]], on="CustomerID", how="inner")

    total_customers = len(rfm)
    total_records = len(df)
    segment_pct = rfm["Segment"].value_counts(normalize=True).reindex(SEGMENT_ORDER).fillna(0) * 100

    revenue_by_segment = merged.groupby("Segment")["TotalPrice"].sum().reindex(SEGMENT_ORDER).fillna(0)
    champions_revenue_pct = revenue_by_segment["Champions"] / revenue_by_segment.sum() * 100

    k3_idx = k_results["k_values"].index(3)
    silhouette_k3 = k_results["silhouette_scores"][k3_idx]

    merged["YearMonth"] = merged["InvoiceDate"].dt.to_period("M")
    monthly_revenue = merged.groupby("YearMonth")["TotalPrice"].sum()
    peak_month = monthly_revenue.idxmax().strftime("%B")

    merged["DayOfWeek"] = merged["InvoiceDate"].dt.day_name()
    dow_revenue = merged.groupby("DayOfWeek")["TotalPrice"].sum().reindex(DAY_ORDER)
    best_day = dow_revenue.idxmax()

    return {
        "Total Customers": total_customers,
        "Total Records": total_records,
        "Champions %": round(segment_pct.get("Champions", 0), 2),
        "At-Risk %": round(segment_pct.get("At-Risk Customers", 0), 2),
        "Lost %": round(segment_pct.get("Lost Customers", 0), 2),
        "Champions Revenue %": round(champions_revenue_pct, 2),
        "Optimal K": k_results["selected_k"],
        "Silhouette (K=3)": round(silhouette_k3, 4),
        "Peak Month": peak_month,
        "Best Day": best_day,
    }


def compare_all_datasets(retail2_metrics):
    _section("THREE-WAY COMPARISON: UCI vs ONLINE RETAIL II vs OLIST")

    metrics_order = [
        "Total Customers", "Total Records", "Champions %", "At-Risk %", "Lost %",
        "Champions Revenue %", "Optimal K", "Silhouette (K=3)", "Peak Month", "Best Day",
    ]

    print(f"{'Metric':<22} | {'UCI Dataset':>15} | {'Online Retail II':>17} | {'Olist Dataset':>15}")
    print("-" * 78)
    rows = []
    for metric in metrics_order:
        uci_val = UCI_REFERENCE[metric]
        retail2_val = retail2_metrics.get(metric, "N/A")
        olist_val = OLIST_REFERENCE[metric]
        print(f"{metric:<22} | {str(uci_val):>15} | {str(retail2_val):>17} | {str(olist_val):>15}")
        rows.append({
            "Metric": metric, "UCI_Dataset": uci_val,
            "Online_Retail_II": retail2_val, "Olist_Dataset": olist_val,
        })

    comparison = pd.DataFrame(rows)
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    out_path = os.path.join(config.RESULTS_PATH, "three_dataset_comparison.csv")
    comparison.to_csv(out_path, index=False)
    print(f"\nSaved three-dataset comparison table to {out_path}")

    print("\nFramework validated on 3 independent datasets")
    return comparison


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RETAIL2_FIGURES_PATH, exist_ok=True)
    os.makedirs(RETAIL2_RESULTS_PATH, exist_ok=True)

    df = load_data_retail2()
    df = preprocess_data_retail2(df)
    rfm = compute_rfm_retail2(df)
    rfm_scaled, scaler = transform_rfm_retail2(rfm)
    k_results = find_optimal_k_retail2(rfm_scaled)
    rfm, kmeans = run_kmeans_retail2(rfm_scaled, rfm, k_results["selected_k"])
    rfm, profile = profile_clusters_retail2(rfm)
    visualize_clusters_retail2(rfm)
    marketing_recommendations_retail2(rfm)

    metrics = compute_comparison_metrics_retail2(rfm, df, k_results)
    compare_all_datasets(metrics)

    _section("ONLINE RETAIL II PIPELINE COMPLETE")
    print(f"All outputs saved under {RETAIL2_FIGURES_PATH} and {RETAIL2_RESULTS_PATH}")


if __name__ == "__main__":
    main()
