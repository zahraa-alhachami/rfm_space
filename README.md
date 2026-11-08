# Reproducibility Repository — IJIES Paper 20267142

**"Multi-Criterion Validation of K-Means++ RFM Customer Segmentation and the
Space Dependence of Silhouette across Five Retail Datasets"**

This repository contains the complete implementation used to produce every
number, table, and figure reported in the manuscript, plus the additional
analyses added during the 2nd review round (Bisecting K-Means comparator,
fuzzy c-means threshold sensitivity sweep). All results are reproducible
exactly: every script uses `random_state=42` / `seed=42`.

## Environment

```
pip install -r requirements.txt
```
Every result in the manuscript and in the response was produced with:

| Package | Version |
|---|---|
| Python | 3.14.6 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| numpy | 2.5.0 |
| scipy | 1.18.0 |

`requirements.txt` carries the full pinned list and agrees with this table. `config.py` holds the shared constants
(`RANDOM_STATE=42`, `MAX_K=10`, data paths).

Two of these matter for exact reproduction rather than merely for running the
code: `scikit-learn` fixes the `BisectingKMeans` defaults the comparator relies
on (`init='random'`, `bisecting_strategy='biggest_inertia'`, `n_init=1` unless
set), and `numpy` fixes `default_rng`'s bit stream, which is what makes every
bootstrap resample reproducible.

## Data availability

Raw third-party datasets are **not redistributed** in this repository
(licensing and size). No Olist records and no customer-level tables are
included. The repository contains only aggregate statistics computed from Olist,
which carry the same CC BY-NC-SA 4.0 attribution. Every RFM table is rebuilt
from the downloads by the scripts in `preprocessing/`.

**Fingerprints.** The md5 and byte size of each file this work read are listed
below so a reader can confirm they downloaded the same thing. They were computed
from the working copies, not transcribed.

| File | md5 | Bytes |
|---|---|---|
| `data/Online Retail.xlsx` | `8f8e6d94ba88f976f4d8290cb2dea7fd` | 23,715,344 |
| `data/online_retail_II.xlsx` | `ed54ccfc5d358481c399cc11d0a244be` | 45,622,278 |
| `data/superstore/superstore.csv` | `badd588d3d65617f3201f4b110590cfd` | 2,304,566 |
| `data/bakery/bakery.csv` | `1e1c79c72c7260d95d7cfc2ab9940880` | 16,492,305 |

**Olist** is nine files, fingerprinted separately. Its licence is
**CC BY-NC-SA 4.0: non-commercial, share-alike, attribution required.** This
repository therefore links to it and redistributes none of it, and any reuse of
these results that is commercial in nature would need a different data source.

| Olist file | md5 | Bytes |
|---|---|---|
| `olist_customers_dataset.csv` | `8a2c4244856aab4bde3b8ed81f8ca251` | 9,033,957 |
| `olist_geolocation_dataset.csv` | `6d8464e41c8e2013955e437b6b4fafbd` | 61,273,883 |
| `olist_order_items_dataset.csv` | `f4fa76976662e9cba0633f9e49ed2645` | 15,438,671 |
| `olist_order_payments_dataset.csv` | `75ce0c041d18e5f250c5cea9a8042944` | 5,777,138 |
| `olist_order_reviews_dataset.csv` | `914066be00d2db0c3d857f071cff1688` | 14,451,670 |
| `olist_orders_dataset.csv` | `8bd60e55c1ca229d9f70b62f3e72f22c` | 17,654,914 |
| `olist_products_dataset.csv` | `e935401104a3669223b83c9a01fdf5ca` | 2,379,446 |
| `olist_sellers_dataset.csv` | `5b22c02facdf842d6c0aea36e1849018` | 174,703 |
| `product_category_name_translation.csv` | `4196d142e8f2b9697521fc50c97f626b` | 2,613 |

Download instructions:

| Dataset | Source | Expected location |
|---|---|---|
| UCI Online Retail | https://archive.ics.uci.edu/dataset/352/online+retail | `data/Online Retail.xlsx` |
| Online Retail II | https://archive.ics.uci.edu/dataset/502/online+retail+ii | `data/online_retail_II.xlsx` |
| Olist Brazilian E-Commerce (**CC BY-NC-SA 4.0, non-commercial**) | https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce | `data/archive/olist_*.csv` |
| Superstore | Tableau Sample - Superstore (fictional teaching data). **Verified as the same classic release with dates shifted forward by one calendar year**, by comparing this copy against `https://raw.githubusercontent.com/Ayon-coder/FUTURE_ML_01/main/Sample%20-%20Superstore.csv` (2,277,811 bytes, md5 `3324c5214bceb630ceeb9df72d0852eb`). Orders sheet: 9,994 rows, 793 customers, 5,009 orders in both; sums identical to the last digit - Sales 2,297,200.8603, Quantity 37,873, Discount 1,561.09, Profit 286,397.0217. Keyed on Row ID, the offset is **+365 days on 7,843 rows and +366 on 2,151** (Ship Date +365/7,840 and +366/2,154). `Order ID` differs on 9,987 rows only because the year is inside the identifier. `Postal Code` differs on 11 rows, all Burlington VT, where the public copy has 05408 and this one is empty; nothing in the analysis reads that column. The remaining text differences are encoding, not data: this copy is **UTF-8** and the public one **cp1252**, so read each in its own encoding and Customer Name agrees on every row while Product Name comes down to 16 rows of trailing whitespace. RFM table: Frequency and Monetary identical for all **793** customers; Recency identical for **770**, differing by exactly **one day for 23** - the leap day inside the shift. **The copy stored here is the 2015-2018 one, so byte-level reproduction needs it** (md5 `badd588d3d65617f3201f4b110590cfd`); a 2014-2017 file yields an RFM that differs by one day of Recency on about 23 customers and is otherwise identical. File composition: Orders 9,994 rows, Returns 800, People 6 | `data/superstore/superstore.csv` |
| Bakery (Carvalho & Maciel, 2026) | Open-access supplementary material, https://doi.org/10.63963/001c.157562 | `data/bakery/bakery.csv` |

## Seeds

One constant and three rules. Everything else follows from them, and nothing in
this work draws a random number any other way.

| Where | Seed | Note |
|---|---|---|
| `config.RANDOM_STATE` | 42 | every estimator's `random_state` unless a loop overrides it |
| K-Means and Bisecting K-Means, single fits | 42 | with `n_init=10`, which is set explicitly for both |
| bootstrap resample indices | `numpy.random.default_rng(42)` | one generator per analysis, drawn `B=100` times |
| the model inside resample `b` | `42 + b` | both models in a paired comparison get the same seed |
| the Silhouette subsample, where used | `42 + b` | only above 20,000 customers; the same subsample serves both arms of a paired difference |
| the gap statistic's reference draws | `default_rng(42)` | `B=50` reference datasets per K |
| prediction strength splits | `default_rng(42)`, model seed `42 + r` | 20 random 50/50 splits |
| the synthetic control's replicates | `42 + 1000 * replicate` | 10 replicates per configuration |

## Folder structure and manuscript mapping

```
ijies_reproducibility_repo/
├── config.py                              Shared constants (seed=42, MAX_K=10, paths)
├── requirements.txt                       Pinned dependencies
├── rfm_scoring.py                         The project's single Silhouette estimator: exact up
│                                            to 20,000 customers, a 10,000-point subsample above.
│                                            Imported by the two scripts that score Olist, so that
│                                            one definition decides every Silhouette in the package
├── preprocessing/
│   ├── rfm_analysis.py                    UCI: cleaning -> RFM -> K-scan -> K-Means++ ->
│   │                                        cluster labelling -> robustness -> revenue/temporal/
│   │                                        geographic/product modules -> K-Means vs K-Means++
│   │                                        -> Reproduces: Table 1, Sec. 4.1, Sec. 4.4 (run-to-run
│   │                                        stability), Sec. 4.7, the Score_k labelling (Eq. 1)
│   ├── rfm_retail2.py                     Online Retail II generalisation -> Table 4, Online Retail II column
│   ├── rfm_olist.py                       Olist generalisation (customer_unique_id grouping,
│   │                                        documented reason) -> Table 4, Olist column, Sec. 4.5
│   ├── rfm_superstore.py                  Superstore generalisation -> Table 4, Superstore column, Sec. 4.5
│   ├── rfm_bakery_pipeline_official.py    The bakery dashboard artifacts under the adopted
│   │                                        model: profiles, K table and figures. Imports
│   │                                        rfm_analysis.py's functions unmodified
│   └── rfm_bakery.py                      Carvalho & Maciel reproduction (Stage B: exact replica
│                                            of their published code, confirms 0.3383/0.3872 to 4
│                                            decimals) + Stage A (framework applied to their data)
│                                            -> Sec. 4.5 cross-sector paragraph, Table 3, Table 4 (Bakery column)
├── pca_and_validation/
│   ├── rfm_analysis_pca_official.py       PCA(2 components) + Varimax + K-Means++, the official
│   │                                        adopted UCI model -> Table 2, Sec. 4.3, Table 2b row
│   │                                        "2 PCs, no rotation" / "2 PCs, Varimax-rotated"
│   ├── rfm_extended_k_pca_official.py     Extended K=2-30 scan in the PCA clustering space ->
│   │                                        supports the K scan from 2 to 30 in Section 3.3
│   ├── rfm_pca_analysis.py                The PCA and Varimax cross-dataset summary
│   └── rfm_commonspace_pca_test.py        Scores the 3D and PCA partitions in ONE space, with a
│                                            100-resample paired interval on the partition effect
├── robustness/
│   ├── rfm_bootstrap_stability_analysis.py  100-resample bootstrap + Adjusted Rand Index (exact
│   │                                          procedure specified in the Response to Reviewers,
│   │                                          Part A.3: fresh K-Means++ refit per resample,
│   │                                          predict on the FULL original dataset, ARI is
│   │                                          permutation-invariant so no label alignment needed)
│   │                                          -> Sec. 4.4 bootstrap paragraph (mean/SD/min/max
│   │                                          per dataset)
│   ├── rfm_fcm_analysis.py                Fuzzy c-means (m=2.0, error=0.005, maxiter=1000,
│   │                                        seed=42) + top-two-membership-gap ambiguity analysis
│   │                                        at the original 0.20 threshold -> Sec. 4.4 fuzzy
│   │                                        paragraph (18.88% / 17.52% / 16.45%)
│   ├── rfm_synthetic_commonspace_control.py  The control with no clusters in it: structureless
│   │                                        Gaussian data, where a correct decomposition must put
│   │                                        the whole apparent gain on the space
│   └── rfm_fcm_threshold_sensitivity.py   Refits FCM
│                                            once with the identical configuration, then sweeps
│                                            the ambiguity threshold across 0.10/0.15/0.20/0.25/
│                                            0.30 from that single membership matrix -> confirms
│                                            the At-Risk/Lost boundary remains the most ambiguous
│                                            pairwise boundary at every tested threshold
├── benchmarking/
│   └── rfm_bisecting_kmeans_uci.py        Same
│                                            cleaned 4,338-customer UCI data, same log1p+
│                                            StandardScaler pipeline, same K=2-10 range, substitutes
│                                            sklearn.cluster.BisectingKMeans for K-Means++ ->
│                                            contemporary same-data comparator result: Silhouette=
│                                            0.3567 at matched K=3 (vs. 0.3365 for K-Means++, a
│                                            +6.00% difference -- reported honestly as an
│                                            outperformance, not adjusted)
├── cluster_profiles_reference/            Pre-computed output CSVs for every number above, so
│                                            results can be checked without re-running the full
│                                            pipeline (re-running reproduces these exactly, seed=42)
├── k_selection/                           The nine validity criteria and their settings:
│   ├── rfm_k_selection_null_corrected.py    Silhouette, Davies-Bouldin, Calinski-Harabasz, the
│   │                                        null-corrected gap statistic (B=50, principal-axis
│   │                                        reference box, 1-SE rule), Dunn, prediction strength
│   │                                        (20 splits, threshold 0.80), bootstrap ARI stability
│   │                                        raw and null-corrected, and the WCSS elbow
│   ├── rfm_k_selection_ps_thresholds.py     prediction strength at 0.70/0.80/0.85/0.90, because
│   │                                        the selected K follows the threshold convention
│   └── rev2_k_criteria_report.py            each criterion's definition, settings and citation;
│                                            the Table 13 recount split by whether the criterion
│                                            compares against a null; and the two elbow rules
│                                            worked through on the same WCSS curve
├── k_scan/                                K=2-10 tables per dataset, plus the nine-criterion
│                                            tables (ksel_criteria_by_k.csv, ksel_selected_k.csv,
│                                            ksel_ps_thresholds.csv). Dataset-level statistics
│                                            only; no customer appears in any of them
├── evaluation/                            The out-of-sample criterion, the only one here that
│   ├── rfm_predictive_validation.py         uses no internal index: a temporal calibration/
│   │                                        holdout split, the segmentation refitted inside every
│   │                                        CV fold, a label-permutation null, and BG/NBD +
│   │                                        Gamma-Gamma implemented from the published likelihoods
│   ├── rev2_temporal_k2_k3.py               K=2 against K=3 at every temporal cutoff
│   └── predval_*.csv                        their outputs, dataset-level
├── per_iteration/                         One row per bootstrap resample, so a reader can
│                                            recompute an interval instead of taking the published
│                                            one on trust - and can see that the K-Means++ versus
│                                            Bisecting difference is bimodal rather than noisy
└── review_round_2/                        Every table behind the second-round response: the
                                             contingency tables and optimally matched agreement,
                                             the K=3/K=4 paired differences with their intervals
                                             and equivalence verdicts, the recount, the comparator's
                                             full configuration and n_init sweep, the 2x2
                                             decomposition under either reference space, the
                                             temporal test, the preprocessing comparison, the
                                             synthetic control's sensitivity arms, the bootstrap
                                             provenance table, and the data fingerprints
```

Three files sit at the root beside `config.py` and `requirements.txt`:
`LICENSE` (MIT, covering the code only), `rfm_scoring.py` - the single
Silhouette estimator described in the tree above - and `rev2_package_audit.py`,
which recomputes the fingerprints in the Data availability section.

## Which output backs which table (v67 numbering)

| Manuscript or response item | Output file |
|---|---|
| Table 1, UCI K=2-10 | a run writes `k_selection_table.csv` (3D) and `pca_official/k_selection_table.csv` (PCA space); shipped here as `cluster_profiles_reference/k_selection_table_uci.csv` |
| Table 2, official cluster profile | a run writes `pca_official/cluster_profiles.csv`; shipped here as `cluster_profiles_reference/cluster_profiles_uci_official.csv` |
| Section 4.5, the profile, size and revenue of each additional dataset | a run writes each dataset's `cluster_profiles*.csv`; shipped here as `cluster_profiles_reference/cluster_profiles_retail2.csv`, `cluster_profiles_reference/cluster_profiles_olist.csv`, `cluster_profiles_reference/cluster_profiles_superstore.csv` and `cluster_profiles_reference/cluster_profiles_bakery.csv`, with their K scans as `k_scan/k_selection_table_retail2.csv`, `k_scan/k_selection_table_olist.csv` and `k_scan/k_selection_table_superstore.csv` |
| Section 3.3, the K scan from 2 to 30 | `k_extended_2_to_30.csv` |
| **Table 5**, the K scan across four datasets | the 3D-space scans: `cluster_profiles_reference/k_selection_table_uci_3d.csv`, `k_scan/k_selection_table_retail2.csv`, `k_scan/k_selection_table_olist.csv`, `k_scan/k_selection_table_superstore.csv` and **`k_scan/k_selection_table_bakery_3d.csv`**. Bakery has two scans in this package and they are different tables: the 3D one above is what Table 5 reports, `k_scan/k_selection_table_bakery.csv` is the PCA space |
| **Table 6**, K=3 against K=4 with intervals and the equivalence margin | **`review_round_2/rev2_paired_bootstrap_summary.csv`** (draws in `per_iteration/rev2_paired_bootstrap_per_iteration.csv`) |
| **Table 8**, the three preprocessing pipelines and why one reports about 0.65 | **`review_round_2/rev2_kumar_protocol.csv`** (with `rev2_kumar_space_shape.csv`, `rev2_kumar_variance_collapse.csv`) |
| **Table 13**, the 2x2 decomposition under either reference space | **`review_round_2/rev2_decomposition_2x2.csv`** |
| **Table 10**, partition and space effects at K=3 on the five datasets | `review_round_2/rev2_decomposition_2x2.csv` (the K=3 rows) |
| The nine criteria: definitions and settings (Section 3.3) | **`review_round_2/rev2_k_criteria_settings.csv`** |
| **Table 14**, what each criterion selects, and the recount | **`k_scan/ksel_selected_k.csv`** and **`review_round_2/rev2_table13_recount.csv`** |
| K-Means against K-Means++, run to run | `cluster_profiles_reference/algorithm_comparison_uci.csv` (a run writes `pca_official/algorithm_comparison.csv`) |
| hierarchical clustering on the PCA space, and why a higher Silhouette is not admissible | `cluster_profiles_reference/hier_pca_uci_linkage_comparison.csv` for the three linkages, and `cluster_profiles_reference/hier_pca_cluster_balance.csv` for the cluster sizes and shares behind each of them |
| the pipeline without the log step, and with RobustScaler | `cluster_profiles_reference/alternative_preprocessing_uci.csv` |
| what K=2 gives: the two-segment split and its revenue share | `cluster_profiles_reference/k2_sensitivity_cluster_profile.csv`, stability draws in `per_iteration/k2_sensitivity_bootstrap_ari.csv` |
| the descriptive statistics, the 3D profile, revenue and the algorithm comparison for UCI | `cluster_profiles_reference/rfm_statistics_uci_3d.csv`, `cluster_profiles_reference/cluster_profiles_uci_3d.csv`, `cluster_profiles_reference/revenue_analysis_uci_3d.csv`, `cluster_profiles_reference/algorithm_comparison_uci_3d.csv` |
| the outlier cluster behind the 0.65 preprocessing result | `review_round_2/rev2_kumar_outlier_cluster.csv` |
| the two elbow rules on one curve | `review_round_2/rev2_elbow_rules.csv` |
| optimal agreement of the 2D projections | `review_round_2/rev2_projection_agreement.csv`, cells in `rev2_projection_contingency_cells.csv` |
| K=2 against K=3 out of sample in time | `review_round_2/rev2_temporal_k2_k3_uci.csv`, from `evaluation/predval_cutoff_sweep.csv` |
| **Table 12**, the partition effect with a paired interval, per dataset | **`review_round_2/rev2_partition_effect_summary.csv`**, draws in `per_iteration/rev2_partition_effect_per_iteration.csv`. UCI's row is also in `per_iteration/commonspace_bootstrap.csv`, which the newer script reproduces to 1e-9 as its own check |
| **Table 12, out-of-bag arm (Section 4.7)** | **`review_round_2/rev2_partition_effect_oob_summary.csv`**, draws in `per_iteration/rev2_partition_effect_oob_per_iteration.csv`. Both arms are in those two files: the in-bag arm is there as a check and reproduces `per_iteration/commonspace_bootstrap.csv` to 1e-9, and the out-of-bag arm scores each resample's two partitions on the customers it never drew, about 1,598 of UCI's 4,338 |
| **Table 3**, bootstrap stability per dataset | `cluster_profiles_reference/bootstrap_ari_summary.csv`, draws in `per_iteration/bootstrap_ari_raw.csv` |
| the fuzzy-boundary paragraph | `cluster_profiles_reference/fcm_threshold_sensitivity.csv` |
| the Bisecting K-Means comparator | `cluster_profiles_reference/bisecting_kmeans_uci_kscan.csv`, every parameter in `review_round_2/rev2_bisecting_full_config.csv`, the n_init pair in `rev2_bisecting_n_init_sweep.csv`, the two modes in `per_iteration/rev2_bisecting_bootstrap_detail.csv` |
| Bakery's own K scan | `k_scan/k_selection_table_bakery.csv` (a run writes it as `bakery/k_selection_table.csv`; the dataset is in the shipped name because the bare one read as the generic table) |
| PCA with and without Varimax, three datasets | `cluster_profiles_reference/pca_varimax_final_summary.csv` |
| the synthetic control, replicates and sensitivity | `review_round_2/rev2_gauss_replicate_spread.csv`, `rev2_gauss_sensitivity_summary.csv` |
| every bootstrap's estimand and interval | `review_round_2/rev2_bootstrap_provenance.csv` |
| the data fingerprints in this README | `review_round_2/rev2_data_fingerprints.csv` |
| **Table 4**, the cluster balance and the revenue share of each segment, every dataset | **`review_round_2/rev2_table4_balance.csv`**. UCI is described through its PCA partition and the other four through their 3D partition, which is what the table reports; segments are ordered by the project's composite score |
| **Table 9**, UCI's two partitions scored in both spaces, with Davies-Bouldin and Calinski-Harabasz | **`review_round_2/rev2_table9_common_space.csv`** |
| **Table 11**, the three 2D projections against the full 3D and PCA spaces | **`review_round_2/rev2_table11_projections.csv`**. Each projection is standardized on its own two columns, and its partition is then scored both in that plane and in the 3D space |
| the cleaning calendar, the revenue total, the highest month and the revenue by weekday (Section 3.1) | **`review_round_2/rev2_uci_cleaning_calendar.csv`** |
| the K=3 partition against the K=2 one, cell by cell | **`review_round_2/rev2_k2_vs_k3_crosstab.csv`** |

## Reproducing from scratch

**Run every script as a module, from the package root:**

```
python -m preprocessing.rfm_analysis
```

Not `python preprocessing/rfm_analysis.py`. That form puts the *script's*
folder on `sys.path` instead of the package root, so `import config` fails with
`ModuleNotFoundError: No module named 'config'`. The `-m` form puts the root on
the path, which is where `config.py` lives.

### Order, dependencies and time

Times are wall clock from one full cold run of all 39 steps on a single
machine (Windows, Python 3.14.6, no GPU, empty PREDVAL_CACHE): **99m 17s in total**,
every step finishing. Five of them are most of that time - step 36 (17m 30s), step 5 (13m 53s), step 35 (13m 33s), step 23 (12m 23s), step 25 (9m 31s) -
and the other 34 steps together take about 32m 26s.

| # | Command | Reads | Writes | Time |
|---|---|---|---|---|
| 1 | `python -m preprocessing.rfm_analysis` | `data/Online Retail.xlsx` | `outputs/results/` UCI 3D: `rfm_raw.csv`, `rfm_clustered.csv`, `cluster_profiles.csv`, `k_selection_table.csv`, the revenue, temporal, geographic and product tables, 19 figures | 1m 07s |
| 2 | `python -m pca_and_validation.rfm_analysis_pca_official` | `data/Online Retail.xlsx`, **imports** `preprocessing/rfm_analysis.py` | `outputs/results/pca_official/` — including **`pca_official/rfm_raw.csv`, the UCI RFM table most later scripts read** | 1m 08s |
| 3 | `python -m pca_and_validation.rfm_extended_k_pca_official` | `pca_official/rfm_raw.csv` (step 2) | `k_extended_2_to_30.csv` | 15s |
| 4 | `python -m preprocessing.rfm_retail2` | `data/online_retail_II.xlsx` | `retail2/rfm_raw_retail2.csv`, profiles, K table | 1m 30s |
| 5 | `python -m preprocessing.rfm_olist` | `data/archive/olist_*.csv` | `olist/rfm_raw_olist.csv`, profiles, K table | **13m 53s** |
| 6 | `python -m preprocessing.rfm_superstore` | `data/superstore/superstore.csv` | `superstore/rfm_raw_superstore.csv`, profiles, K table | 8s |
| 7 | `python -m preprocessing.rfm_bakery` | `data/bakery/bakery.csv` | `bakery/rfm_raw_bakery.csv`, `bakery_comparison_summary.csv` | 9s |
| 8 | `python -m preprocessing.rfm_bakery_pipeline_official` | `data/bakery/bakery.csv`, **imports** `rfm_analysis.py` | `bakery/cluster_profiles.csv`, `bakery/k_selection_table.csv`, figures | 20s |
| 9 | `python -m k_selection.rfm_bakery_3d_kscan` | `bakery/rfm_raw_bakery.csv` (step 7) | `bakery/k_selection_table_3d_baseline.csv` - Bakery's K scan in the plain 3D space, shipped as `k_scan/k_selection_table_bakery_3d.csv` and **not** the same table as `k_scan/k_selection_table_bakery.csv`, which is the PCA space | 16s |
| 10 | `python -m pca_and_validation.rfm_pca_analysis` | `rfm_raw.csv` (step 1), `retail2/` and `olist/` RFM tables (steps 4, 5) | `pca_varimax_final_summary.csv` | 4m 11s |
| 11 | `python -m pca_and_validation.rfm_hierarchical_pca_analysis` | `rfm_raw.csv` (step 1), `retail2/` and `olist/` RFM tables (steps 4, 5) | `hier_pca_uci_linkage_comparison.csv`, `hier_pca_final_comprehensive_table.csv`. **Olist is skipped**: agglomerative clustering there holds the full 93,358 x 93,358 distance matrix, 32.5 GiB, so the third stage prints `skipping Olist` and the step finishes | 8s |
| 12 | `python -m pca_and_validation.rev2_hier_pca_balance` | `rfm_raw.csv` (step 1), `retail2/rfm_raw_retail2.csv` (step 4), **imports** `rfm_scoring.py` | `hier_pca_cluster_balance.csv` - each linkage's Silhouette, its DBI and the three cluster sizes and shares behind it, for UCI and Online Retail II. This is what makes the higher hierarchical Silhouette inadmissible: its clusters are a 96.1 / 3.4 / 0.5 split | 8s |
| 13 | `python -m pca_and_validation.rfm_commonspace_pca_test` | `pca_official/rfm_raw.csv` (step 2) | `commonspace_matrix.csv`, `commonspace_decomposition.csv`, `commonspace_bootstrap.csv`, `commonspace_negative_control.csv` | 1m 20s |
| 14 | `python -m robustness.rfm_synthetic_commonspace_control` | nothing - it simulates | `synthetic_commonspace_primary.csv`, `synthetic_commonspace_replicates.csv` | 3m 05s |
| 15 | `python -m robustness.rfm_alternative_preprocessing_uci` | `pca_official/rfm_raw.csv` (step 2) | `alternative_preprocessing_uci.csv` - the pipeline with the log step dropped, and with RobustScaler | 5s |
| 16 | `python -m robustness.rfm_k2_sensitivity_uci` | `pca_official/rfm_raw.csv` (step 2) | `k2_sensitivity_cluster_profile.csv`, `k2_sensitivity_bootstrap_ari.csv` | 41s |
| 17 | `python -m robustness.rfm_bootstrap_stability_analysis` | the RFM tables from steps 2, 4, 5 | `bootstrap_ari_raw.csv`, `bootstrap_ari_summary.csv` | 2m 13s |
| 18 | `python -m robustness.rfm_fcm_analysis` | `pca_official/rfm_raw.csv` | `fcm_*.csv` | 2m 43s |
| 19 | `python -m robustness.rfm_fcm_threshold_sensitivity` | `pca_official/rfm_raw.csv` | `fcm_threshold_sensitivity.csv` | 3s |
| 20 | `python -m benchmarking.rfm_bisecting_kmeans_uci` | `pca_official/rfm_raw.csv` | `bisecting_kmeans_uci_kscan.csv` | 6s |
| 21 | `python -m robustness.rfm_bootstrap_silhouette_uncertainty` | `pca_official/rfm_raw.csv` | `bootstrap_silhouette_k3_vs_k4.csv`, `bootstrap_silhouette_pca_vs_3d.csv` | 1m 23s |
| 22 | `python -m robustness.rfm_bootstrap_kmeanspp_vs_bisecting` | `pca_official/rfm_raw.csv` | `bootstrap_kmeanspp_vs_bisecting.csv` | 56s |
| 23 | `python -m robustness.rev2_partition_effect_bootstrap` | the five RFM tables, **imports** `rfm_scoring.py`, and `commonspace_bootstrap.csv` (step 13) for its own check | `rev2_partition_effect_per_iteration.csv`, `rev2_partition_effect_summary.csv` - the partition effect with a paired interval on every dataset, which only UCI had before | **12m 23s** |
| 24 | `python -m robustness.rev2_partition_effect_oob` | `pca_official/rfm_raw.csv` (step 2), **imports** `rfm_scoring.py`, and `commonspace_bootstrap.csv` (step 13) for its first check | `rev2_partition_effect_oob_per_iteration.csv`, `rev2_partition_effect_oob_summary.csv` - the same partition effect measured on the customers each resample left out, with the in-bag arm beside it in one file. UCI only | 1m 05s |
| 25 | `python -m k_selection.rfm_k_selection_null_corrected` | the RFM tables from steps 2, 4, 6, 7 | `ksel_criteria_by_k.csv`, `ksel_selected_k.csv`, `ksel_gap_detail.csv` | 9m 31s |
| 26 | `python -m k_selection.rfm_k_selection_ps_thresholds` | the same RFM tables | `ksel_ps_thresholds.csv`, `ksel_ps_detail.csv`, `ksel_ps_per_split.csv` | 2m 22s |
| 27 | `python -m evaluation.rfm_predictive_validation` | the **raw** data files directly, plus its cache | `predval_by_k.csv`, `predval_reference.csv`, `predval_cutoff_sweep.csv`, `predval_segment_detail.csv` | 3m 02s cold |
| 28 | `python -m evaluation.rev2_temporal_k2_k3` | `predval_cutoff_sweep.csv`, `predval_by_k.csv` (step 27) | `rev2_temporal_k2_k3_uci.csv` | 1s |
| 29 | `python -m pca_and_validation.rev2_projection_contingency` | `pca_official/rfm_raw.csv` | `rev2_projection_agreement.csv`, `rev2_projection_contingency_cells.csv` | 5s |
| 30 | `python -m benchmarking.rev2_bisecting_settings` | `pca_official/rfm_raw.csv`, and `bootstrap_kmeanspp_vs_bisecting.csv` (step 22) for its per-resample detail | `rev2_bisecting_settings.csv`, `rev2_bisecting_runs.csv`, `rev2_bisecting_ari.csv`, `rev2_bisecting_bootstrap_detail.csv` | 1m 11s |
| 31 | `python -m benchmarking.rev2_bisecting_config` | `pca_official/rfm_raw.csv` | `rev2_bisecting_full_config.csv`, `rev2_bisecting_n_init_sweep.csv` | 8s |
| 32 | `python -m benchmarking.rev2_kumar_protocol` | `pca_official/rfm_raw.csv`, and `rev2_decomposition_2x2.csv` (step 36) for one comparison | `rev2_kumar_*.csv` | 9s |
| 33 | `python -m k_selection.rev2_k_criteria_report` | `ksel_selected_k.csv` (step 25), the K tables from steps 1 and 2 | `rev2_k_criteria_settings.csv`, `rev2_table13_recount.csv`, `rev2_elbow_rules.csv` | 1s |
| 34 | `python -m robustness.rev2_gauss_sensitivity` | `synthetic_commonspace_replicates.csv` (step 14), and its own simulation | `rev2_gauss_replicate_spread.csv`, `rev2_gauss_sensitivity_detail.csv`, `rev2_gauss_sensitivity_summary.csv` | 1m 37s |
| 35 | `python -m robustness.rev2_paired_bootstrap` | the five RFM tables, **imports** `rfm_scoring.py` | `rev2_paired_bootstrap_summary.csv`, `rev2_paired_bootstrap_per_iteration.csv` | **13m 33s** |
| 36 | `python -m pca_and_validation.rev2_decomposition_2x2` | the five RFM tables, **imports** `rfm_scoring.py` | `rev2_decomposition_2x2.csv` | **17m 30s** |
| 37 | `python -m pca_and_validation.rev2_manuscript_reference_numbers` | the five RFM tables, and `data/Online Retail.xlsx` itself for the cleaning counts | `rev2_table4_balance.csv`, `rev2_table9_common_space.csv`, `rev2_table11_projections.csv`, `rev2_uci_cleaning_calendar.csv`, `rev2_k2_vs_k3_crosstab.csv` - five tables the paper quotes that had no file of their own. The step checks each number against the manuscript's and **writes nothing if one disagrees** | 49s |
| 38 | `python -m robustness.rev2_bootstrap_provenance` | every bootstrap output above - **run it last, since it records which of them exist** | `rev2_bootstrap_provenance.csv` | 1s |
| 39 | `python -m rev2_package_audit` | the files under `data/` | `rev2_data_fingerprints.csv` | 1s |

**The dependency that matters most:** steps 1, 2, 4, 5, 6 and 7 are what produce
the `rfm_raw_*.csv` tables. Steps 17 to 26 and 29 to 36 read those tables and do
not rebuild them, so running any of them first fails with
`FileNotFoundError: outputs/results/.../rfm_raw_*.csv`. Step 27 is the
exception: it reads the raw data files itself.

Steps 21, 22, 23, 24, 35 and 36 are the resampling and exact-Silhouette runs, and their
time is governed by Olist's size; together with Olist's own 13 minutes they are
most of why a full cold reproduction is about an hour and a quarter rather than
half an hour.

### Output directories

Most scripts create the directories they write to. If you run a single script
out of order, create the output tree first:

```
mkdir outputs\\results   outputs\\figures          (Windows)
mkdir -p outputs/results outputs/figures          (Linux, macOS)
```

### PREDVAL_CACHE

`evaluation/rfm_predictive_validation.py` caches each dataset's cleaned
transactions so a re-run does not re-read the spreadsheets. By default the
cache goes to `%TEMP%/predval_clean_cache` (`$TMPDIR` on Unix) — **outside this
folder**, which means a second run can read state a first run left behind, and
a "clean" test is not clean unless you redirect it:

```
set PREDVAL_CACHE=C:\\path\\to\\an\\empty\\folder      (Windows)
export PREDVAL_CACHE=/path/to/an/empty/folder        (Linux, macOS)
```

Delete that folder to force a cold run. The cached files are cleaned
transaction tables, so they are customer-level: keep them out of any
repository.

### Datasets you did not download

Every script that loops over datasets skips one whose file is absent and says
so (`skipping <name>: <path> not found`), then finishes with the rest. A reader
who downloads only some of the five still gets a complete run over those.
