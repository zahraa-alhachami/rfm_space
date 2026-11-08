"""
Fourth RFM + K-Means++ clustering pipeline, for the public Superstore Sales
dataset (US retail, ~10k order-line records) — used as an additional
generalization test on a dataset genuinely distinct from UCI/Retail
II/Olist. Same methodology as rfm_analysis.py.

Run with:
    python rfm_superstore.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import skew
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

plt.style.use("seaborn-v0_8-whitegrid")

SELECTED_K = 3
SEGMENT_ORDER = ["Champions", "At-Risk Customers", "Lost Customers"]
SEGMENT_COLORS = {"Champions": "#2E86AB", "At-Risk Customers": "#F18F01", "Lost Customers": "#C73E1D"}

DATA_PATH = os.path.join("data", "superstore", "superstore.csv")
FIGURES_PATH = os.path.join(config.FIGURES_PATH, "superstore")
RESULTS_PATH = os.path.join(config.RESULTS_PATH, "superstore")


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


def load_and_preprocess():
    _section("STEP 1: LOAD + PREPROCESS SUPERSTORE DATA")
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    before = len(df)
    print(f"Raw records: {before}")

    df = df.dropna(subset=["Customer ID"])
    df = df[df["Sales"] > 0]
    df = df[df["Quantity"] > 0]
    df = df.drop_duplicates()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date"])

    after = len(df)
    print(f"Clean records: {after} (removed {before - after})")
    return df


def compute_rfm(df):
    _section("STEP 2: COMPUTE RFM")
    snapshot_date = df["Order Date"].max() + pd.Timedelta(days=1)
    print(f"Snapshot date: {snapshot_date}")

    rfm = df.groupby("Customer ID").agg(
        Recency=("Order Date", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Order ID", "nunique"),
        Monetary=("Sales", "sum"),
    ).reset_index().rename(columns={"Customer ID": "CustomerID"})

    print(rfm[["Recency", "Frequency", "Monetary"]].agg(["mean", "median", "std", "min", "max"]))
    os.makedirs(RESULTS_PATH, exist_ok=True)
    rfm.to_csv(os.path.join(RESULTS_PATH, "rfm_raw_superstore.csv"), index=False)
    return rfm


def transform_rfm(rfm):
    _section("STEP 3: TRANSFORM (LOG1P + STANDARDSCALER)")
    cols = ["Recency", "Frequency", "Monetary"]
    for c in cols:
        print(f"  {c} skew before: {skew(rfm[c]):.4f}")
    transformed = rfm[cols].apply(np.log1p)
    for c in cols:
        print(f"  {c} skew after: {skew(transformed[c]):.4f}")
    scaled = StandardScaler().fit_transform(transformed)
    print(f"Scaled shape: {scaled.shape}")
    return scaled


def find_optimal_k(scaled):
    _section("STEP 4: FIND OPTIMAL K")
    k_values = list(range(2, config.MAX_K + 1))
    wcss, sils, dbis = [], [], []
    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", random_state=config.RANDOM_STATE, n_init=10)
        labels = km.fit_predict(scaled)
        wcss.append(km.inertia_)
        sils.append(silhouette_score(scaled, labels))
        dbis.append(davies_bouldin_score(scaled, labels))
        print(f"  K={k}: WCSS={km.inertia_:.2f}, Silhouette={sils[-1]:.4f}, DBI={dbis[-1]:.4f}")

    os.makedirs(RESULTS_PATH, exist_ok=True)
    pd.DataFrame({"K": k_values, "WCSS": wcss, "Silhouette_Score": sils, "Davies_Bouldin_Index": dbis}).to_csv(
        os.path.join(RESULTS_PATH, "k_selection_table_superstore.csv"), index=False
    )

    os.makedirs(FIGURES_PATH, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=300)
    axes[0].plot(k_values, wcss, marker="o", color="#2E86AB")
    axes[0].axvline(SELECTED_K, color="red", linestyle="--")
    _style_axis(axes[0], "Elbow Method (Superstore)", "K", "WCSS")
    axes[1].plot(k_values, sils, marker="o", color="#2E86AB")
    axes[1].axvline(SELECTED_K, color="red", linestyle="--")
    _style_axis(axes[1], "Silhouette Score (Superstore)", "K", "Silhouette")
    fig.suptitle("Optimal K Determination (Superstore Dataset)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(FIGURES_PATH, "optimal_k_superstore.png"), dpi=300)
    plt.close(fig)

    return {"k_values": k_values, "wcss": wcss, "silhouette_scores": sils, "dbi_scores": dbis}


def run_kmeans_and_profile(scaled, rfm):
    _section(f"STEP 5: RUN K-MEANS++ (K={SELECTED_K}) + PROFILE")
    km = KMeans(n_clusters=SELECTED_K, init="k-means++", random_state=config.RANDOM_STATE, n_init=10)
    labels = km.fit_predict(scaled)
    print(f"n_iter_: {km.n_iter_}")

    rfm = rfm.copy()
    rfm["Cluster"] = labels
    sizes = rfm["Cluster"].value_counts().sort_index()
    for cid in sizes.index:
        print(f"  Cluster {cid}: {sizes[cid]} ({sizes[cid] / len(rfm) * 100:.2f}%)")

    profile = rfm.groupby("Cluster").agg(
        Recency_mean=("Recency", "mean"), Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"), Count=("Cluster", "size"),
    ).reset_index()
    profile["Percentage"] = profile["Count"] / profile["Count"].sum() * 100
    composite = (1 / profile["Recency_mean"]) * profile["Frequency_mean"] * profile["Monetary_mean"]
    profile["Composite_Score"] = composite
    order = composite.sort_values(ascending=False).index
    segment_map = dict(zip(order, SEGMENT_ORDER))
    profile["Segment"] = profile.index.map(segment_map)

    print(profile[["Cluster", "Recency_mean", "Frequency_mean", "Monetary_mean", "Count", "Percentage", "Segment"]])

    os.makedirs(RESULTS_PATH, exist_ok=True)
    profile.to_csv(os.path.join(RESULTS_PATH, "cluster_profiles_superstore.csv"), index=False)
    rfm = rfm.merge(profile[["Cluster", "Segment"]], on="Cluster", how="left")
    rfm.to_csv(os.path.join(RESULTS_PATH, "rfm_clustered_superstore.csv"), index=False)

    return rfm, profile, km


def visualize(rfm):
    _section("STEP 6: VISUALIZE")
    os.makedirs(FIGURES_PATH, exist_ok=True)
    segs = [s for s in SEGMENT_ORDER if s in rfm["Segment"].unique()]

    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    for s in segs:
        sub = rfm[rfm["Segment"] == s]
        ax.scatter(sub["Recency"], sub["Frequency"], sub["Monetary"], color=SEGMENT_COLORS[s], label=s, alpha=0.6, s=25)
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("Monetary ($)")
    ax.set_title("3D RFM Segments (Superstore Dataset)", fontsize=16)
    ax.legend()
    plt.savefig(os.path.join(FIGURES_PATH, "3d_clusters_superstore.png"), dpi=300)
    plt.close(fig)

    sizes = rfm["Segment"].value_counts().reindex(segs)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.bar(sizes.index, sizes.values, color=[SEGMENT_COLORS[s] for s in sizes.index])
    _style_axis(ax, "Customer Distribution (Superstore Dataset)", "Segment", "Number of Customers")
    plt.savefig(os.path.join(FIGURES_PATH, "cluster_sizes_superstore.png"), dpi=300)
    plt.close(fig)
    print(f"Saved figures to {FIGURES_PATH}")


def main():
    os.makedirs(FIGURES_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    df = load_and_preprocess()
    rfm = compute_rfm(df)
    scaled = transform_rfm(rfm)
    k_results = find_optimal_k(scaled)
    rfm, profile, km = run_kmeans_and_profile(scaled, rfm)
    visualize(rfm)

    k3_idx = k_results["k_values"].index(3)
    _section("SUPERSTORE SUMMARY")
    print(f"Total customers: {len(rfm)}")
    print(f"Total clean records: {len(df)}")
    print(f"Mean Frequency: {rfm['Frequency'].mean():.2f}")
    print(f"Silhouette (K=3): {k_results['silhouette_scores'][k3_idx]:.4f}")
    for _, row in profile.iterrows():
        print(f"{row['Segment']}: {int(row['Count'])} ({row['Percentage']:.2f}%)")


if __name__ == "__main__":
    main()
