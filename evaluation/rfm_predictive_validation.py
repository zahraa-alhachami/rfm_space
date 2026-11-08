"""Out-of-sample predictive validation of RFM segments.

WHY THIS EXISTS
---------------
Every validity measure used in this project so far is internal: it asks whether
the clusters are geometrically tidy in some space. None of them asks the
question a reviewer of a segmentation paper actually cares about:

    does knowing a customer's segment today tell you anything about what that
    customer does next?

Both IJIES reviewers asked for exactly this. Reviewer 2: "add an external
criterion such as future-period value/retention behavior or a clearly separated
downstream evaluation that tests whether the proposed segments have predictive
or operational utility." Fader, Hardie & Lee (2005, Journal of Marketing
Research) make the same point about the CLV literature generally: holdout
validation should come before any value calculation, and it is usually missing.

This also turns the K question from an unanswerable internal one into an
answerable external one. Internally, every index in this project prefers K=2
(9 criteria over five datasets in two spaces, 90 cells, K=3 chosen in 4; on the four datasets the IJIES manuscript reports, 72 cells with K=3 chosen in 2). Externally, the question is whether
splitting At-Risk from Lost buys any predictive power. Whatever the answer, it
is reported.

DESIGN
------
1. Transactions are split in time at a cutoff into a calibration period and a
   holdout period. Nothing from the holdout touches the segmentation.
2. RFM is built from the calibration period only.
3. For each K, the segmentation is refit INSIDE each cross-validation fold
   (scaler, PCA and KMeans all fit on training customers only), so there is no
   leakage of any kind - not even unsupervised leakage.
4. The segment -> outcome map (mean holdout revenue, retention rate) is
   estimated on training customers and evaluated on held-out customers.
5. A label-permutation null gives the score an uninformative segmentation earns,
   so a positive number can be read against something.
6. BG/NBD + Gamma-Gamma, fitted on the same calibration data, is the reference:
   an established individual-level model of the same outcomes. It is implemented
   here directly from the published likelihoods rather than taken from a
   package, so every step is auditable.

Outcomes predicted, per calibration customer:
    holdout revenue, whether the customer returned at all, holdout invoice count

Outputs (prefix predval_, inside outputs/results/):
  predval_by_k.csv          predictive power at each K, both spaces, with null
  predval_reference.csv     BG/NBD + Gamma-Gamma on the same split
  predval_cutoff_sweep.csv  sensitivity of the chosen K to the split date
  predval_segment_detail.csv per-segment holdout behaviour at each K

Run:
    python rfm_predictive_validation.py
"""

import io
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, betaln, hyp2f1, expit
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import config

def _force_utf8_stdout():
    """Called only when this file is run directly.

    A module must not rewrap sys.stdout on import. Other scripts import the
    loaders and period helpers from here and wrap stdout themselves; wrapping at
    module level put two TextIOWrappers over the same buffer, the first was
    garbage-collected, it closed the shared buffer, and every later print raised
    "I/O operation on closed file". Same bug was fixed in rfm_dec.py.
    """
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RS = config.RANDOM_STATE
K_RANGE = [2, 3, 4, 5, 6]
N_FOLDS = 5
CUTOFF_FRACTIONS = [0.50, 0.60, 0.70]      # share of the time span used for calibration
PRIMARY_FRACTION = 0.60
RFM_COLS = ["Recency", "Frequency", "Monetary"]

CACHE = os.environ.get("PREDVAL_CACHE") or os.path.join(
    os.environ.get("TEMP", "."), "predval_clean_cache")

# Every loader returns the same four columns - CustomerID, InvoiceNo,
# InvoiceDate, TotalPrice - using each dataset's own cleaning rules exactly as
# the project's pipeline scripts define them, so the RFM built here matches the
# RFM built elsewhere. The per-dataset notes record the rules that matter.
def _uci():
    df = pd.read_excel("data/Online Retail.xlsx")
    df = df.dropna(subset=["CustomerID"])
    df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]
    df["TotalPrice"] = df.Quantity * df.UnitPrice
    df["CustomerID"] = df.CustomerID.astype(np.int64).astype(str)
    return df[["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"]]


def _retail2():
    df = pd.concat(pd.read_excel("data/online_retail_II.xlsx", sheet_name=None).values(),
                   ignore_index=True)
    df = df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice",
                            "Customer ID": "CustomerID"})
    df = df.dropna(subset=["CustomerID"])
    df = df[(df.Quantity > 0) & (df.UnitPrice > 0)]
    df["TotalPrice"] = df.Quantity * df.UnitPrice
    df["CustomerID"] = df.CustomerID.astype(np.int64).astype(str)
    return df[["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"]]


def _olist():
    # customer_unique_id is the stable buyer key; customer_id is per-order, so
    # grouping on it would force every Frequency to 1. Same correction as
    # rfm_olist.py.
    orders = pd.read_csv("data/archive/olist_orders_dataset.csv")
    items = pd.read_csv("data/archive/olist_order_items_dataset.csv")
    custs = pd.read_csv("data/archive/olist_customers_dataset.csv")
    df = items.merge(orders, on="order_id", how="inner").merge(
        custs, on="customer_id", how="inner")
    df["TotalPrice"] = df.price + df.freight_value
    df = df.rename(columns={"customer_unique_id": "CustomerID",
                            "order_id": "InvoiceNo",
                            "order_purchase_timestamp": "InvoiceDate"})
    df = df[df.TotalPrice > 0]
    return df[["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"]]


def _superstore():
    df = pd.read_csv("data/superstore/superstore.csv", encoding="latin1")
    df = df.dropna(subset=["Customer ID"])
    df = df[df.Sales > 0]
    df = df.rename(columns={"Customer ID": "CustomerID", "Order ID": "InvoiceNo",
                            "Order Date": "Order_Date", "Sales": "TotalPrice"})
    df["InvoiceDate"] = pd.to_datetime(df.Order_Date, errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    return df[["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"]]


def _bakery():
    # CustomerID 0 is the walk-in code, not a trackable buyer; duplicates are
    # removed on SaleID+CustomerID+ProductID. Same rules as rfm_bakery.py.
    df = pd.read_csv("data/bakery/bakery.csv")
    df = df.drop_duplicates(subset=["SaleID", "CustomerID", "ProductID"])
    df = df.dropna(subset=["CustomerID", "SaleDate", "ItemValue"])
    df = df[(df.CustomerID != 0) & (df.Quantity > 0) & (df.ItemValue > 0)]
    df["TotalPrice"] = df.Quantity * df.ItemValue
    df = df.rename(columns={"SaleID": "InvoiceNo", "SaleDate": "InvoiceDate"})
    df["CustomerID"] = df.CustomerID.astype(np.int64).astype(str)
    return df[["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"]]


def _dunnhumby():
    # Prepared and verified by rfm_dunnhumby_prepare.py, which checks the
    # download against the publisher's stated row and variable counts before
    # writing anything. The customer unit is a HOUSEHOLD, and TotalPrice is
    # revenue net of loyalty and coupon discounts rather than shelf price -
    # both disclosed wherever this dataset is reported.
    df = pd.read_csv(os.path.join("data", "dunnhumby", "transactions_clean.csv"),
                     dtype={"CustomerID": str})
    return df[["CustomerID", "InvoiceNo", "InvoiceDate", "TotalPrice"]]


DATASETS = {
    "UCI": _uci,
    "Online Retail II": _retail2,
    "Superstore": _superstore,
    "Bakery": _bakery,
    "Olist": _olist,
    "Dunnhumby": _dunnhumby,
}


# ---------------------------------------------------------------------------
# Loading and cleaning - identical rules to rfm_analysis.py / rfm_retail2.py
# ---------------------------------------------------------------------------
def load_clean(name, loader):
    os.makedirs(CACHE, exist_ok=True)
    cached = os.path.join(CACHE, name.replace(" ", "_") + "_txn.csv")
    if os.path.exists(cached):
        df = pd.read_csv(cached, parse_dates=["InvoiceDate"], dtype={"CustomerID": str})
        print("  loaded %d transactions from cache" % len(df))
    else:
        print("  loading %s ..." % name, flush=True)
        df = loader()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df = df.dropna(subset=["InvoiceDate", "CustomerID"])
        df.to_csv(cached, index=False)
    print("  %d rows, %d customers, %s to %s (%d days)"
          % (len(df), df.CustomerID.nunique(), df.InvoiceDate.min().date(),
             df.InvoiceDate.max().date(),
             (df.InvoiceDate.max() - df.InvoiceDate.min()).days))
    return df


def split_periods(df, fraction):
    start, end = df.InvoiceDate.min(), df.InvoiceDate.max()
    cutoff = start + (end - start) * fraction
    cal = df[df.InvoiceDate < cutoff]
    hold = df[df.InvoiceDate >= cutoff]
    return cal, hold, cutoff


def calibration_rfm(cal):
    """RFM as the rest of the project defines it, on calibration data only."""
    snapshot = cal.InvoiceDate.max() + pd.Timedelta(days=1)
    rfm = cal.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()
    return rfm


def holdout_outcomes(rfm, hold):
    """What each calibration customer went on to do. Absent customers get zeros."""
    agg = hold.groupby("CustomerID").agg(
        H_Revenue=("TotalPrice", "sum"),
        H_Invoices=("InvoiceNo", "nunique"),
    ).reset_index()
    out = rfm[["CustomerID"]].merge(agg, on="CustomerID", how="left").fillna(0.0)
    out["H_Active"] = (out.H_Invoices > 0).astype(int)
    return out


# ---------------------------------------------------------------------------
# Segment-based prediction, fully nested
# ---------------------------------------------------------------------------
def fit_segments(rfm_train, k, space, seed=RS):
    """Fit the project's pipeline on training customers only.

    Returns a function mapping a raw RFM frame to labels, so the identical
    transform chain can be applied to held-out customers.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(rfm_train[RFM_COLS].apply(np.log1p))
    pca = None
    if space == "PCA 2D":
        pca = PCA(n_components=2, random_state=seed)
        Xs = pca.fit_transform(Xs)
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed).fit(Xs)

    def assign(rfm_any):
        X = scaler.transform(rfm_any[RFM_COLS].apply(np.log1p))
        if pca is not None:
            X = pca.transform(X)
        return kmeans.predict(X)

    return assign, kmeans.labels_


def oos_scores(y_true, y_pred, active_true, active_pred):
    """Out-of-sample quality of a segment-level prediction.

    R2 can be negative, which is informative: it means segment means predict
    worse than the overall mean. Spearman is reported because for targeting the
    ranking matters more than the level.
    """
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    rho = np.nan
    if len(np.unique(y_pred)) > 1:
        rho = float(spearmanr(y_true, y_pred).statistic)

    auc = np.nan
    if len(np.unique(active_true)) > 1 and len(np.unique(active_pred)) > 1:
        auc = float(roc_auc_score(active_true, active_pred))

    return r2, rho, auc


def evaluate_k(rfm, out, k, space, rng):
    """Nested CV: the segmentation itself is refit on each training fold."""
    y = out.H_Revenue.to_numpy(float)
    y_log = np.log1p(y)
    active = out.H_Active.to_numpy(int)

    pred_log = np.full(len(rfm), np.nan)
    pred_act = np.full(len(rfm), np.nan)
    pred_log_null = np.full(len(rfm), np.nan)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RS)
    for tr, te in kf.split(rfm):
        assign, _ = fit_segments(rfm.iloc[tr], k, space)
        lb_tr = assign(rfm.iloc[tr])
        lb_te = assign(rfm.iloc[te])

        # segment -> outcome map, estimated on the training fold only
        rev_map = pd.Series(y_log[tr]).groupby(lb_tr).mean()
        act_map = pd.Series(active[tr]).groupby(lb_tr).mean()
        overall_rev, overall_act = y_log[tr].mean(), active[tr].mean()

        pred_log[te] = [rev_map.get(l, overall_rev) for l in lb_te]
        pred_act[te] = [act_map.get(l, overall_act) for l in lb_te]

        # null: same segment sizes, labels shuffled, so the map carries no signal
        lb_tr_perm = rng.permutation(lb_tr)
        rev_map_null = pd.Series(y_log[tr]).groupby(lb_tr_perm).mean()
        pred_log_null[te] = [rev_map_null.get(l, overall_rev) for l in lb_te]

    r2, rho, auc = oos_scores(y_log, pred_log, active, pred_act)
    r2_null, rho_null, _ = oos_scores(y_log, pred_log_null, active, pred_act)

    # top-segment concentration: share of holdout revenue captured by the
    # segment with the highest predicted value, against its share of customers
    assign_full, _ = fit_segments(rfm, k, space)
    lb_full = assign_full(rfm)
    seg_rev = pd.Series(y).groupby(lb_full).mean()
    top = seg_rev.idxmax()
    in_top = lb_full == top
    capture = float(y[in_top].sum() / y.sum()) if y.sum() > 0 else np.nan
    share = float(in_top.mean())

    return dict(
        K=k, Space=space,
        R2_oos_logrev=r2, R2_null_logrev=r2_null, R2_gain_over_null=r2 - r2_null,
        Spearman_logrev=rho, Spearman_null=rho_null,
        AUC_retention=auc,
        Top_segment_revenue_capture=capture,
        Top_segment_customer_share=share,
        Top_segment_lift=(capture / share) if share > 0 else np.nan,
    )


# ---------------------------------------------------------------------------
# BG/NBD + Gamma-Gamma, implemented from the published likelihoods
# ---------------------------------------------------------------------------
def bgnbd_summary(cal):
    """Per-customer calibration summary in BG/NBD terms.

    x     = number of REPEAT transactions (invoices - 1)
    t_x   = days from first to last transaction
    T_cal = days from first transaction to the end of the calibration window

    Note this recency is NOT the project's RFM Recency (days since last
    purchase); BG/NBD defines it as the age at last purchase.

    The observation-window column is called T_cal, not T, on purpose: a pandas
    DataFrame already has a .T attribute (the transpose), so s.T would silently
    return the transposed frame instead of this column.
    """
    end = cal.InvoiceDate.max()
    inv = cal.groupby(["CustomerID", "InvoiceNo"]).agg(
        d=("InvoiceDate", "min"), v=("TotalPrice", "sum")).reset_index()
    g = inv.groupby("CustomerID")
    s = pd.DataFrame({
        "x": g.size() - 1,
        "t_x": (g.d.max() - g.d.min()).dt.days.astype(float),
        "T_cal": (end - g.d.min()).dt.days.astype(float),
        "mx": g.v.mean(),
    }).reset_index()
    return s


def _unpack(params):
    """r, alpha, b > 0 and a > 1, enforced by reparameterisation.

    a > 1 is not cosmetic: the BG/NBD conditional expectation carries a factor
    (a+b+x-1)/(a-1) and a Gaussian hypergeometric whose third argument is
    a+b+x-1. At a <= 1 the factor flips sign and that third argument can reach
    zero, where 2F1 is singular, so the expectation collapses. Constraining
    a > 1 keeps the expectation well defined at every customer.
    """
    r, alpha, b = np.exp(params[0]), np.exp(params[1]), np.exp(params[3])
    a = 1.0 + np.exp(params[2])
    return r, alpha, a, b


def _bgnbd_nll(params, x, t_x, T):
    r, alpha, a, b = _unpack(params)
    ln_a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    t1 = ln_a1 - (r + x) * np.log(alpha + T) + betaln(a, b + x) - betaln(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        t2 = np.where(
            x > 0,
            ln_a1 - (r + x) * np.log(alpha + t_x)
            + betaln(a + 1, np.maximum(b + x - 1, 1e-9)) - betaln(a, b),
            -np.inf,
        )
    ll = np.logaddexp(t1, t2)
    if not np.all(np.isfinite(ll)):
        return 1e12
    return -float(ll.sum())


def fit_bgnbd(s):
    args = (s["x"].to_numpy(float), s["t_x"].to_numpy(float), s["T_cal"].to_numpy(float))
    best = None
    for start in ([0.0, 0.0, 0.0, 0.0], [np.log(0.5), np.log(10.0), np.log(0.5), np.log(2.0)]):
        res = minimize(_bgnbd_nll, start, args=args,
                       method="Nelder-Mead",
                       options=dict(maxiter=20000, maxfev=20000, xatol=1e-6, fatol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    r, alpha, a, b = _unpack(best.x)
    return dict(bg_r=r, bg_alpha=alpha, bg_a=a, bg_b=b, bg_nll=float(best.fun))


def bgnbd_p_alive(s, p):
    """P(still active | x, t_x, T). Reported separately because a per-customer
    churn probability is directly useful for the segment profiling, and because
    it is well conditioned even where the full expectation is delicate."""
    r, alpha, a, b = p["bg_r"], p["bg_alpha"], p["bg_a"], p["bg_b"]
    x = s["x"].to_numpy(float)
    t_x = s["t_x"].to_numpy(float)
    T = s["T_cal"].to_numpy(float)

    # Computed in log space. The direct form overflows on a large base: numpy
    # evaluates both branches of a where() before selecting, so the x = 0
    # customers - whose (alpha+T)/(alpha+t_x) can be enormous since t_x = 0 -
    # overflow to inf even though their branch is discarded.
    log_odds = (np.log(a) - np.log(np.maximum(b + x - 1.0, 1e-12))
                + (r + x) * (np.log(alpha + T) - np.log(alpha + t_x)))
    log_odds = np.where(x > 0, log_odds, -np.inf)
    return expit(-log_odds)


def bgnbd_expected(s, p, t_future):
    """E[Y(t) | x, t_x, T] - expected transactions in the next t days."""
    r, alpha, a, b = p["bg_r"], p["bg_alpha"], p["bg_a"], p["bg_b"]
    x = s["x"].to_numpy(float)
    T = s["T_cal"].to_numpy(float)

    z = t_future / (alpha + T + t_future)
    f = hyp2f1(r + x, b + x, a + b + x - 1.0, z)
    bracket = 1.0 - ((alpha + T) / (alpha + T + t_future)) ** (r + x) * f
    expected = ((a + b + x - 1.0) / (a - 1.0)) * bracket * bgnbd_p_alive(s, p)

    # A non-finite value means 2F1 failed for that customer; fall back to the
    # well-conditioned P(alive) x rate x horizon, and say how often that happened.
    bad = ~np.isfinite(expected)
    fallback_rate = float(bad.mean())
    if bad.any():
        fallback = bgnbd_p_alive(s, p) * ((r + x) / (alpha + T)) * t_future
        expected = np.where(bad, fallback, expected)
        print("    note: 2F1 fallback used for %d of %d customers (%.2f%%)"
              % (bad.sum(), len(bad), 100 * fallback_rate), flush=True)
    expected = np.nan_to_num(np.clip(expected, 0.0, None), nan=0.0,
                             posinf=0.0, neginf=0.0)
    return expected, fallback_rate


def _gg_nll(params, x, mx):
    p, q, gam = np.exp(params)
    ll = (gammaln(p * x + q) - gammaln(p * x) - gammaln(q)
          + q * np.log(gam) + (p * x - 1.0) * np.log(mx)
          + p * x * np.log(x) - (p * x + q) * np.log(gam + mx * x))
    if not np.all(np.isfinite(ll)):
        return 1e12
    return -float(ll.sum())


def fit_gamma_gamma(s):
    m = s[(s["x"] > 0) & (s["mx"] > 0)]
    if len(m) < 20:
        return None
    best = None
    for start in ([0.0, 0.0, np.log(m["mx"].mean())],
                  [np.log(2.0), np.log(2.0), np.log(m["mx"].median())]):
        res = minimize(_gg_nll, start, args=(m["x"].to_numpy(float), m["mx"].to_numpy(float)),
                       method="Nelder-Mead",
                       options=dict(maxiter=20000, maxfev=20000, xatol=1e-6, fatol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    p, q, gam = np.exp(best.x)
    return dict(gg_p=p, gg_q=q, gg_gamma=gam, gg_nll=float(best.fun))


def gg_expected(s, g):
    """E[M | x, mx] - a shrinkage of the customer's own mean toward the
    population mean, the shrinkage being stronger the fewer transactions."""
    p, q, gam = g["gg_p"], g["gg_q"], g["gg_gamma"]
    x = s["x"].to_numpy(float)
    mx = s["mx"].to_numpy(float)
    pop = gam * p / (q - 1.0) if q > 1.0 else gam * p / 1e-6
    w = (p * x) / (p * x + q - 1.0)
    return w * mx + (1.0 - w) * pop


# ---------------------------------------------------------------------------
def run_dataset(name, loader):
    print("\n" + "=" * 92)
    print("DATASET: %s" % name)
    print("=" * 92, flush=True)
    df = load_clean(name, loader)

    by_k_rows, ref_rows, sweep_rows, seg_rows = [], [], [], []

    for frac in CUTOFF_FRACTIONS:
        cal, hold, cutoff = split_periods(df, frac)
        rfm = calibration_rfm(cal)
        out = holdout_outcomes(rfm, hold)
        horizon = (df.InvoiceDate.max() - cutoff).days
        primary = abs(frac - PRIMARY_FRACTION) < 1e-9

        n_returned = int(out.H_Active.sum())
        print("\n--- cutoff %.0f%% = %s | calibration %d customers | holdout %d days | "
              "returned %d (%.1f%%)"
              % (100 * frac, cutoff.date(), len(rfm), horizon, n_returned,
                 100 * out.H_Active.mean()), flush=True)

        # A dataset whose customers almost never return cannot be externally
        # validated: there is no holdout behaviour to predict. Olist, where
        # roughly 97% of buyers purchase once, is the case this guard exists
        # for. The diagnostic is recorded rather than the dataset being
        # silently dropped, because the absence is itself a reportable boundary
        # condition on the external criterion.
        if n_returned < 200 or out.H_Active.mean() < 0.02:
            sweep_rows.append(dict(
                Dataset=name, Cutoff_fraction=frac, Cutoff=str(cutoff.date()),
                Horizon_days=horizon, N_calibration=len(rfm),
                N_returned=n_returned, Return_rate=float(out.H_Active.mean()),
                Skipped_reason="too few returning customers for external validation"))
            print("    SKIPPED: only %d customers return, so there is no holdout "
                  "behaviour to predict" % n_returned, flush=True)
            continue

        rng = np.random.default_rng(RS)
        for space in ("3D", "PCA 2D"):
            for k in K_RANGE:
                row = evaluate_k(rfm, out, k, space, rng)
                row.update(Dataset=name, Cutoff_fraction=frac,
                           Cutoff=str(cutoff.date()), Horizon_days=horizon,
                           N_calibration=len(rfm))
                sweep_rows.append(row)
                if primary:
                    by_k_rows.append(row)
                    print("  %-7s K=%d  R2=%+.4f (null %+.4f, gain %+.4f)  rho=%.4f  "
                          "AUC=%.4f  top-lift=%.2fx"
                          % (space, k, row["R2_oos_logrev"], row["R2_null_logrev"],
                             row["R2_gain_over_null"], row["Spearman_logrev"],
                             row["AUC_retention"], row["Top_segment_lift"]), flush=True)

        if primary:
            # per-segment holdout behaviour, for the thesis's profiling tables
            for space in ("3D", "PCA 2D"):
                for k in K_RANGE:
                    assign, _ = fit_segments(rfm, k, space)
                    lb = assign(rfm)
                    gdf = pd.DataFrame(dict(seg=lb, rev=out.H_Revenue, act=out.H_Active,
                                            inv=out.H_Invoices))
                    agg = gdf.groupby("seg").agg(
                        N=("rev", "size"), Mean_holdout_revenue=("rev", "mean"),
                        Retention_rate=("act", "mean"),
                        Mean_holdout_invoices=("inv", "mean")).reset_index()
                    agg.insert(0, "K", k)
                    agg.insert(0, "Space", space)
                    agg.insert(0, "Dataset", name)
                    seg_rows.append(agg)

            # --- reference model on the same split
            print("  fitting BG/NBD and Gamma-Gamma ...", flush=True)
            s = bgnbd_summary(cal)
            bg = fit_bgnbd(s)
            exp_tx, fallback_rate = bgnbd_expected(s, bg, float(horizon))
            gg = fit_gamma_gamma(s)
            pred_val = exp_tx * gg_expected(s, gg) if gg else exp_tx

            merged = s[["CustomerID"]].copy()
            merged["pred_tx"] = exp_tx
            merged["pred_val"] = pred_val
            merged["p_alive"] = bgnbd_p_alive(s, bg)
            merged = merged.merge(out, on="CustomerID", how="left").fillna(0.0)
            merged.to_csv(os.path.join(config.RESULTS_PATH,
                          "predval_bgnbd_%s.csv" % name.replace(" ", "_")), index=False)

            def rho(a_, b_):
                a_, b_ = np.asarray(a_, float), np.asarray(b_, float)
                if len(np.unique(a_)) < 2 or len(np.unique(b_)) < 2:
                    return np.nan
                return float(spearmanr(a_, b_).statistic)

            y = merged.H_Revenue.to_numpy(float)
            act = merged.H_Active.to_numpy(int)
            rho_val = rho(y, merged.pred_val)
            rho_tx = rho(merged.H_Invoices, merged.pred_tx)

            def safe_auc(score):
                score = np.asarray(score, float)
                if (not np.all(np.isfinite(score)) or len(np.unique(act)) < 2
                        or len(np.unique(score)) < 2):
                    return np.nan
                return float(roc_auc_score(act, score))

            # Expected transactions is the retention score. P(alive) is NOT:
            # BG/NBD assigns P(alive) = 1 to every customer with zero repeat
            # transactions, because death can only follow a transaction. In a
            # base dominated by one-time buyers that inverts the ranking, so
            # its AUC is reported only as a diagnostic of that structure.
            auc = safe_auc(merged.pred_tx)
            auc_palive = safe_auc(merged.p_alive)
            zero_repeat = float((s["x"] == 0).mean())
            order = np.argsort(-merged.pred_val.to_numpy())
            top10 = order[: max(1, len(order) // 10)]
            ref_rows.append(dict(
                Dataset=name, Cutoff=str(cutoff.date()), Horizon_days=horizon,
                N=len(merged), **bg, **(gg or {}),
                Spearman_value=rho_val, Spearman_transactions=rho_tx,
                AUC_retention=auc, AUC_from_p_alive=auc_palive,
                Mean_p_alive=float(merged.p_alive.mean()),
                Zero_repeat_share=zero_repeat,
                F2_fallback_rate=fallback_rate,
                Reference_model_reliable=bool(fallback_rate < 0.5),
                A_at_lower_bound=bool(abs(bg["bg_a"] - 1.0) < 1e-4),
                Top_decile_revenue_capture=float(y[top10].sum() / y.sum()) if y.sum() else np.nan,
            ))
            print("    BG/NBD r=%.4f alpha=%.4f a=%.4f b=%.4f%s"
                  % (bg["bg_r"], bg["bg_alpha"], bg["bg_a"], bg["bg_b"],
                     "   [a at its lower bound: the unconstrained MLE wants a<1]"
                     if ref_rows[-1]["A_at_lower_bound"] else ""), flush=True)
            print("    rho(value)=%.4f  rho(tx)=%.4f  AUC(expected tx)=%.4f  "
                  "AUC(p_alive)=%.4f  top-decile capture=%.1f%%  zero-repeat share=%.1f%%"
                  % (rho_val, rho_tx, auc, auc_palive,
                     100 * ref_rows[-1]["Top_decile_revenue_capture"],
                     100 * zero_repeat), flush=True)
            if fallback_rate >= 0.5:
                print("    REFERENCE MODEL NOT RELIABLE on this dataset: the exact "
                      "conditional expectation failed for %.0f%% of customers, so its "
                      "numbers above are not reported as a benchmark."
                      % (100 * fallback_rate), flush=True)

    return by_k_rows, ref_rows, sweep_rows, seg_rows


def main():
    by_k, ref, sweep, seg = [], [], [], []
    for name, loader in DATASETS.items():
        try:
            a, b, c, d = run_dataset(name, loader)
        except FileNotFoundError as missing:
            print("skipping %s: %s" % (name, missing), flush=True)
            continue
        by_k += a
        ref += b
        sweep += c
        seg += d

    out = config.RESULTS_PATH
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(by_k).round(6).to_csv(os.path.join(out, "predval_by_k.csv"), index=False)
    pd.DataFrame(ref).round(6).to_csv(os.path.join(out, "predval_reference.csv"), index=False)
    pd.DataFrame(sweep).round(6).to_csv(os.path.join(out, "predval_cutoff_sweep.csv"), index=False)
    if seg:
        pd.concat(seg, ignore_index=True).round(4).to_csv(
            os.path.join(out, "predval_segment_detail.csv"), index=False)

    d = pd.DataFrame(by_k)
    print("\n" + "=" * 92)
    print("WHICH K MAXIMISES OUT-OF-SAMPLE PREDICTIVE POWER?")
    print("=" * 92)
    for (ds, sp), g in d.groupby(["Dataset", "Space"]):
        best_r2 = int(g.loc[g.R2_gain_over_null.idxmax(), "K"])
        best_auc = int(g.loc[g.AUC_retention.idxmax(), "K"])
        best_rho = int(g.loc[g.Spearman_logrev.idxmax(), "K"])
        print("  %-18s %-7s  R2 gain -> K=%d | retention AUC -> K=%d | Spearman -> K=%d"
              % (ds, sp, best_r2, best_auc, best_rho))

    sw = pd.DataFrame(sweep)
    print("\nSensitivity of the best K to the split date (by R2 gain over null):")
    for (ds, sp), g in sw.groupby(["Dataset", "Space"]):
        picks = [(f, int(gg.loc[gg.R2_gain_over_null.idxmax(), "K"]))
                 for f, gg in g.groupby("Cutoff_fraction")]
        print("  %-18s %-7s  %s" % (ds, sp,
              "  ".join("%.0f%%->K=%d" % (100 * f, k) for f, k in picks)))

    print("\nSaved predval_by_k.csv, predval_reference.csv, predval_cutoff_sweep.csv, "
          "predval_segment_detail.csv")


if __name__ == "__main__":
    _force_utf8_stdout()
    main()
