"""
Rolling PCA Factor Dashboard
==============================
Answers: "What latent macro forces are driving my target,
          and how have those exposures shifted over time?"

Pipeline:
  1. Download macro inputs + target
  2. Compute log returns → vol-standardize
  3. For each rolling window:
       a. PCA on basket → K latent Factors
       b. Sign-align eigenvectors to first window (not just previous)
       c. OLS: target ~ factors → rolling betas
       d. Record correlation of each Factor with each input (rolling labels)
  4. Visualize: betas, labels, R², residuals, current snapshot

What changed from v1:
  - Output is factor betas, not back-transformed ticker weights.
    Ticker weights collapse the factor structure back to asset space
    and throw away the information PCA surfaced.
  - Sign alignment anchors to first window, not just previous window,
    preventing sign drift over long samples.
  - Rolling label tracking in every window, not just the last.
    This is the key diagnostic: does Factor 2 mean the same thing in 2019 as 2023?
  - R² gap (OLS vs PCR) retained as a truncation diagnostic.
  - Residual (idiosyncratic) return series added as its own panel.
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Ticker remapping (Yahoo Finance special tickers)
# ─────────────────────────────────────────────────────────────────────────────
TICKER_REMAP = {
    "VIX"  : "^VIX",
    "DXY"  : "DX-Y.NYB",
    "TNX"  : "^TNX",
    "TYX"  : "^TYX",
    "IRX"  : "^IRX",
    "GSPC" : "^GSPC",
    "DJI"  : "^DJI",
    "IXIC" : "^IXIC",
    "RUT"  : "^RUT",
}
TICKER_DISPLAY = {v: k for k, v in TICKER_REMAP.items()}

def remap(t):   return TICKER_REMAP.get(t, t)
def unmap(t):   return TICKER_DISPLAY.get(t, t)


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rolling PCA Factor Dashboard", layout="wide")

st.markdown("""
<style>
  .metric-label  { font-size: 0.75rem; color: #888; }
  .block-container { padding-top: 1.5rem; }
  div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("Rolling PCA Factor Dashboard")
st.caption(
    "Answers: *what latent macro forces are driving my target, "
    "and how have those exposures shifted over time?*"
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")

st.sidebar.subheader("Macro Inputs")
st.sidebar.caption(
    "These are the macro risk factors PCA runs on. "
    "Each should proxy a distinct, economically meaningful dimension. "
    "Avoid loading this with correlated equity names — you'll just get "
    "the market factor plus noise."
)

DEFAULT_BASKET = "SPY, TLT, HYG, VIXY, UUP, GSG, IWM, IWF, IWD, EEM"
ticker_input = st.sidebar.text_area(
    "Macro Inputs (comma-separated)",
    value=DEFAULT_BASKET,
    help=(
        "SPY=equity  TLT=rates  HYG=credit  VIXY=vol  "
        "UUP=dollar  GSG=commodity  IWM=size  IWF=growth  IWD=value  EEM=EM"
    )
)
input_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

target_ticker = st.sidebar.text_input(
    "Target Ticker",
    value="ARKK",
    help="The asset you want to decompose into macro factor exposures."
).strip().upper()

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

freq_label = st.sidebar.radio(
    "Data Frequency",
    options=["Daily", "Weekly", "Monthly"],
    index=0,
    horizontal=True,
    help=(
        "Daily = most granular, noisiest. "
        "Weekly = smoother, reduces microstructure noise. "
        "Monthly = regime-level only, fewest observations."
    )
)
FREQ_MAP = {"Daily": "B", "Weekly": "W-FRI", "Monthly": "ME"}
data_freq = FREQ_MAP[freq_label]

lookback_years = st.sidebar.select_slider(
    "Data Lookback (years)",
    options=[1, 2, 3, 5, 7, 10],
    value=5,
    help="1 year gives ~252 daily / ~52 weekly / ~12 monthly observations."
)

# Window labels adapt to frequency
FREQ_WINDOW_LABEL = {"Daily": "trading days", "Weekly": "weeks", "Monthly": "months"}
win_label = FREQ_WINDOW_LABEL[freq_label]

WINDOW_DEFAULTS  = {"Daily": 60,  "Weekly": 26,  "Monthly": 12}
WINDOW_MINS      = {"Daily": 20,  "Weekly": 8,   "Monthly": 6}
WINDOW_MAXS      = {"Daily": 120, "Weekly": 52,  "Monthly": 24}
WINDOW_STEPS     = {"Daily": 5,   "Weekly": 2,   "Monthly": 1}

pca_window = st.sidebar.slider(
    f"Rolling Window ({win_label})",
    min_value=WINDOW_MINS[freq_label],
    max_value=WINDOW_MAXS[freq_label],
    value=WINDOW_DEFAULTS[freq_label],
    step=WINDOW_STEPS[freq_label],
    help=(
        "How many periods per rolling window. "
        "Shorter = more regime-sensitive. Longer = smoother, more stable."
    )
)

VOL_WIN_OPTIONS  = {"Daily": [21, 63, 126, 252], "Weekly": [4, 13, 26, 52], "Monthly": [3, 6, 12]}
VOL_WIN_DEFAULTS = {"Daily": 1, "Weekly": 1, "Monthly": 1}

vol_window = st.sidebar.selectbox(
    f"Vol-Standardization Window ({win_label})",
    options=VOL_WIN_OPTIONS[freq_label],
    index=VOL_WIN_DEFAULTS[freq_label],
    help="Window for computing rolling std used to normalize returns before PCA."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Factor Truncation")
st.sidebar.caption(
    "Truncation is where the signal lives. Using ALL PCs is algebraically "
    "identical to OLS — the rotation cancels out. Keep only the PCs that "
    "reflect systematic co-movement."
)

truncation_method = st.sidebar.radio(
    "Method",
    ["Fixed N", "Variance Threshold"],
    index=0,
    help=(
        "Fixed N is more principled for factor analysis. "
        "Variance threshold adapts but risks including too many PCs."
    )
)

if truncation_method == "Fixed N":
    n_fixed = st.sidebar.slider(
        "Number of Factors to keep",
        min_value=1, max_value=min(len(input_tickers), 8),
        value=min(3, len(input_tickers)), step=1
    )
    var_thresh = None
else:
    var_thresh = st.sidebar.slider(
        "Variance threshold", min_value=50, max_value=95,
        value=80, step=5, format="%d%%"
    ) / 100.0
    n_fixed = None

st.sidebar.markdown("---")
st.sidebar.subheader("Dendrogram (diagnostic)")
n_clusters = st.sidebar.slider(
    "Number of clusters",
    min_value=2, max_value=min(8, len(input_tickers)),
    value=min(4, len(input_tickers)), step=1
)

run = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(basket, target, years, _vol_win):
    all_orig  = list(dict.fromkeys(basket + [target]))
    all_yf    = [remap(t) for t in all_orig]
    end       = datetime.today()
    start     = end - timedelta(days=years * 365 + _vol_win + 90)

    for attempt in range(3):
        try:
            raw = yf.download(all_yf, start=start, end=end,
                              auto_adjust=True, progress=False)
            prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            break
        except Exception:
            if attempt < 2: time.sleep(2 ** attempt)
            else: return pd.DataFrame()

    if prices is None or prices.empty:
        return pd.DataFrame()

    # Rename yf tickers back to user tickers
    rename = {yf_t: orig for orig, yf_t in zip(all_orig, all_yf)}
    prices = prices.rename(columns=rename)
    return prices.dropna(how="all")


# ─────────────────────────────────────────────────────────────────────────────
# Core transforms
# ─────────────────────────────────────────────────────────────────────────────
def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def vol_standardize(rets, win):
    """Divide each return by its rolling std. Removes heteroskedasticity."""
    std = rets.rolling(win).std()
    return (rets / std).dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Rolling PCA — the correct approach
# ─────────────────────────────────────────────────────────────────────────────
def rolling_pca(std_basket: pd.DataFrame,
                std_target: pd.Series,
                window: int,
                n_fixed: int | None,
                var_thresh: float | None) -> dict:
    """
    For each rolling window of `window` days:

      1. PCA on the basket → K latent Factors
      2. Sign-align eigenvectors to the FIRST window (not just prev window)
         so betas are comparable across all time.
      3. OLS: target ~ factors → rolling betas
      4. Record correlation of each Factor with each input (rolling labels).
         This answers: "what does Factor 2 mean in June 2022 vs June 2019?"
      5. Record R², residual, variance explained per PC.

    Output is a dict of time-indexed series / dataframes.
    """
    T           = len(std_basket)
    n_assets    = std_basket.shape[1]
    asset_cols  = std_basket.columns.tolist()

    # ── determine max PCs we'll ever need ──
    max_k = n_assets - 1  # never use all (that's OLS)

    # Storage
    out_dates    = []
    out_betas    = []          # (K,) beta of target on each PC
    out_r2       = []
    out_r2_ols   = []
    out_resid    = []          # last-period residual in each window
    out_var_exp  = []          # (K,) variance explained by each retained Factor
    out_pc_corr  = []          # (K, N) correlation of each Factor with each asset
    out_n_pcs    = []
    out_contribs     = []      # (K,) = beta_k * F_k(t) for last obs in window
    out_target       = []      # actual target return at time t (for overlay)
    out_pc1_loadings = []      # (N,) eigenvector weights for PC1 in each window
    out_loadings     = []      # (N, k) eigenvector weights for ALL factors each window

    reference_loadings = None  # anchored to first window for sign alignment

    for t in range(window, T):
        # ── slice window ──
        X = std_basket.values[t - window : t]    # (window, N)
        y = std_target.values[t - window : t]    # (window,)

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y  = X[valid], y[valid]

        if len(X) < n_assets + 5:
            continue

        # ── PCA via eigendecomposition of sample covariance ──
        X_dm = X - X.mean(axis=0)
        cov  = np.cov(X_dm, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)      # eigh for symmetric
        idx      = np.argsort(eigvals)[::-1]
        eigvals  = eigvals[idx]
        eigvecs  = eigvecs[:, idx]                   # (N, N)

        # ── Sign alignment: anchor to first window ──
        if reference_loadings is None:
            reference_loadings = eigvecs.copy()
        else:
            for k in range(eigvecs.shape[1]):
                if np.dot(eigvecs[:, k], reference_loadings[:, k]) < 0:
                    eigvecs[:, k] *= -1

        # ── Determine how many PCs to retain ──
        total_var = eigvals.sum()
        cum_var   = np.cumsum(eigvals) / total_var

        if n_fixed is not None:
            k = min(n_fixed, max_k)
        else:
            k = int(np.searchsorted(cum_var, var_thresh) + 1)
            k = min(k, max_k)
        k = max(k, 1)

        # ── Factor realizations: (window, k) ──
        factors = X_dm @ eigvecs[:, :k]

        # ── OLS: target ~ factors ──
        y_dm  = y - y.mean()
        X_reg = np.column_stack([np.ones(len(factors)), factors])
        betas, _, _, _ = np.linalg.lstsq(X_reg, y_dm, rcond=None)

        y_hat = X_reg @ betas
        resid = y_dm - y_hat
        ss_res = (resid**2).sum()
        ss_tot = (y_dm**2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # ── Naive OLS R² for comparison ──
        X_ols = np.column_stack([np.ones(len(X)), X])
        betas_ols, _, _, _ = np.linalg.lstsq(X_ols, y_dm, rcond=None)
        y_hat_ols = X_ols @ betas_ols
        ss_res_ols = ((y_dm - y_hat_ols)**2).sum()
        r2_ols = 1 - ss_res_ols / ss_tot if ss_tot > 0 else np.nan

        # ── Rolling PC-to-asset correlation ──
        # corrcoef returns (k + N, k + N); top-left k×N block is what we want
        corr_block = np.corrcoef(factors.T, X.T)[:k, k:]   # (k, N)

        # ── Store ──
        # contributions: beta_k * F_k at last obs in window
        # sum = fitted value for that period — what the stacked area shows
        last_factors  = factors[-1, :]
        pc_betas      = betas[1:k+1]
        contributions = pc_betas * last_factors     # (k,) element-wise

        out_dates.append(std_basket.index[t])
        out_betas.append(betas[1:k+1])
        out_r2.append(r2)
        out_r2_ols.append(r2_ols)
        out_resid.append(resid[-1])
        out_var_exp.append(eigvals[:k] / total_var)
        out_pc_corr.append(corr_block)
        out_n_pcs.append(k)
        out_contribs.append(contributions)
        out_target.append(y_dm[-1])                 # demeaned target return
        out_pc1_loadings.append(eigvecs[:, 0].copy())  # PC1 eigenvector (N,)
        out_loadings.append(eigvecs[:, :k].copy())     # (N, k) all factor eigenvectors

    return {
        "dates"    : out_dates,
        "betas"    : out_betas,
        "r2"       : out_r2,
        "r2_ols"   : out_r2_ols,
        "resid"    : out_resid,
        "var_exp"  : out_var_exp,
        "pc_corr"  : out_pc_corr,
        "n_pcs"    : out_n_pcs,
        "assets"   : asset_cols,
        "contribs"      : out_contribs,       # list of (k,) arrays
        "target"        : out_target,         # list of floats (demeaned target return)
        "pc1_loadings"  : out_pc1_loadings,   # list of (N,) eigenvectors for PC1
        "loadings"      : out_loadings,         # list of (N, k) eigenvectors all factors
    }


def pad_series(results, key, n_pcs_max, n_assets=None):
    """
    Pad ragged arrays to uniform shape for DataFrame construction.
    Used when var_thresh causes k to vary across windows.
    """
    rows = []
    for i, arr in enumerate(results[key]):
        if key in ("betas", "var_exp"):
            padded = np.full(n_pcs_max, np.nan)
            padded[:len(arr)] = arr
        elif key == "pc_corr":
            padded = np.full((n_pcs_max, n_assets), np.nan)
            padded[:arr.shape[0], :arr.shape[1]] = arr
        rows.append(padded)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Dendrogram
# ─────────────────────────────────────────────────────────────────────────────
def build_dendrogram(corr_matrix, labels, n_clusters):
    dist = 1 - corr_matrix.abs()
    np.fill_diagonal(dist.values, 0)
    dist = (dist + dist.T) / 2
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="ward")
    return Z


# ─────────────────────────────────────────────────────────────────────────────
# Color palette for PCs
# ─────────────────────────────────────────────────────────────────────────────
PC_COLORS = [
    "#2196F3", "#FF5722", "#4CAF50",
    "#9C27B0", "#FF9800", "#00BCD4", "#E91E63", "#8BC34A"
]

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,alpha)' — Plotly-safe."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# Main flow
# ─────────────────────────────────────────────────────────────────────────────
if not run:
    st.info("👈 Configure parameters and click **Run Analysis**.")

    st.markdown("""
    ### What this tool answers

    **"What latent macro forces are driving my target, and how have those exposures shifted over time?"**

    ---

    ### Pipeline

    1. **Macro Inputs** — PCA runs on a set of economically distinct assets
       (equity, rates, credit, vol, dollar, commodities, size, growth, value, EM).
       These span genuinely different risk dimensions. Loading the inputs with
       correlated sector ETFs collapses everything into one equity factor.

    2. **Vol-standardize returns** — divide each return by its rolling std before PCA.
       This prevents high-vol inputs from dominating the first Factor simply by being noisier.

    3. **Rolling PCA** — in each window, eigendecompose the sample covariance matrix
       of the basket. Extract K latent Factors.

    4. **Sign alignment** — eigenvectors are only defined up to a sign flip.
       We anchor signs to the first window, so betas are comparable across all time.
       Without this, a beta that flips sign might just be a sign flip in the eigenvector.

    5. **Regress target on factors** — OLS in each window gives rolling betas.
       These answer: "how much does ARKK load on the volatility factor right now?"

    6. **Rolling label tracking** — in each window, correlate each Factor with each
       input. This tracks whether Factor 2 means the same thing in 2019 as in 2023.
       Factor interpretation is not stable. This makes the instability visible.

    ---

    ### Key outputs

    - **Rolling betas** — how the target's sensitivity to each latent factor changes over time
    - **Rolling Factor labels** — what each Factor currently represents economically
    - **R² vs OLS gap** — how much the truncation is filtering vs. a naive regression
    - **Residual** — the idiosyncratic component PCA doesn't explain (potential alpha)
    """)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Downloading price data..."):
    prices = load_data(input_tickers, target_ticker, lookback_years, vol_window)

if prices is None or prices.empty:
    st.error("No price data returned. Check tickers or try again.")
    st.stop()

avail_inputs = [t for t in input_tickers if t in prices.columns]
missing      = [t for t in input_tickers if t not in prices.columns]
if target_ticker not in prices.columns:
    st.error(f"Target '{target_ticker}' not found in downloaded data.")
    st.stop()
if missing:
    st.warning(f"Tickers not found: {', '.join(missing)}")
if len(avail_inputs) < 3:
    st.error("Need at least 3 inputs with data.")
    st.stop()

with st.spinner("Computing standardized returns..."):
    lr_daily = log_returns(prices)
    # Resample to chosen frequency before PCA.
    # For weekly/monthly we sum log returns (additive), then vol-standardize.
    if data_freq == "B":
        lr = lr_daily
    else:
        lr = lr_daily.resample(data_freq).sum().dropna(how="all")
    std  = vol_standardize(lr, vol_window)
    std_basket = std[avail_inputs].dropna(axis=1, how="all").dropna()
    std_target = std[target_ticker].dropna()
    common     = std_basket.index.intersection(std_target.index)
    std_basket = std_basket.loc[common]
    std_target = std_target.loc[common]

n_assets = std_basket.shape[1]

with st.spinner("Running rolling PCA..."):
    res = rolling_pca(std_basket, std_target, pca_window, n_fixed, var_thresh)

if not res["dates"]:
    st.error("No valid rolling windows computed. Reduce window size or add more data.")
    st.stop()

# ── Build uniform dataframes ──
n_windows  = len(res["dates"])
n_pcs_max  = max(res["n_pcs"])
dates      = res["dates"]

# Betas: (windows, n_pcs_max)
betas_arr  = np.full((n_windows, n_pcs_max), np.nan)
var_arr    = np.full((n_windows, n_pcs_max), np.nan)
for i, (b, v) in enumerate(zip(res["betas"], res["var_exp"])):
    betas_arr[i, :len(b)] = b
    var_arr[i,   :len(v)] = v

# PC-to-asset correlations: (windows, n_pcs_max, n_assets)
corr_arr = np.full((n_windows, n_pcs_max, n_assets), np.nan)
for i, c in enumerate(res["pc_corr"]):
    corr_arr[i, :c.shape[0], :c.shape[1]] = c

# Eigenvector loadings array: (windows, n_pcs_max, n_assets)
loadings_arr = np.full((n_windows, n_pcs_max, n_assets), np.nan)
for i, L in enumerate(res["loadings"]):       # L is (N, k)
    k = L.shape[1]
    loadings_arr[i, :k, :L.shape[0]] = L.T   # store as (k, N)

# Contributions: beta_k * F_k(t) per PC per window (windows, n_pcs_max)
contrib_arr = np.full((n_windows, n_pcs_max), np.nan)
for i, c in enumerate(res["contribs"]):
    contrib_arr[i, :len(c)] = c

pc_cols    = [f"Factor {k+1}" for k in range(n_pcs_max)]
beta_df    = pd.DataFrame(betas_arr,   index=dates, columns=pc_cols)
var_df     = pd.DataFrame(var_arr,     index=dates, columns=pc_cols)
contrib_df = pd.DataFrame(contrib_arr, index=dates, columns=pc_cols)
target_ser = pd.Series(res["target"],  index=dates, name="Target")

# Factor decomposition: β_k × w_kj × r_j(t) per factor k, per input j, per window t
# Sum across j = β_k × F_k(t) = the black line for factor k
# loadings_arr: (windows, n_pcs_max, n_assets)
# betas_arr:    (windows, n_pcs_max)
std_inputs_at_dates = std_basket.reindex(dates).values   # (windows, N)

# Build dict of DataFrames: factor_decomp[k] = (windows, N) contributions
factor_decomp = {}
for k in range(n_pcs_max):
    beta_k   = betas_arr[:, k]                    # (windows,)
    w_k      = loadings_arr[:, k, :]              # (windows, N)
    r        = std_inputs_at_dates                 # (windows, N)
    # contribution of input j to target via factor k: β_k * w_kj * r_j(t)
    contrib_k = (beta_k[:, None] * w_k * r)       # (windows, N)
    factor_decomp[k] = pd.DataFrame(contrib_k, index=dates, columns=avail_inputs)

# Keep pc1_ticker_contrib_df alias for backward compat with any other references
pc1_load_arr = np.vstack(res["pc1_loadings"])
std_basket_at_dates = std_inputs_at_dates
pc1_ticker_contrib    = pc1_load_arr * std_basket_at_dates
pc1_ticker_contrib_df = pd.DataFrame(pc1_ticker_contrib, index=dates, columns=avail_inputs)
pc1_realized = pc1_ticker_contrib_df.sum(axis=1)
r2_ser     = pd.Series(res["r2"],     index=dates, name="R\u00b2")
r2_ols_ser = pd.Series(res["r2_ols"], index=dates, name="OLS R\u00b2")
resid_ser  = pd.Series(res["resid"],  index=dates, name="Residual")
n_pcs_ser  = pd.Series(res["n_pcs"], index=dates, name="N Factors")


# ─────────────────────────────────────────────────────────────────────────────
# Summary header metrics
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Target",          target_ticker)
m2.metric("Avg R²",          f"{r2_ser.mean():.1%}")
m3.metric("Latest R²",       f"{r2_ser.iloc[-1]:.1%}")
m4.metric("Avg Factors used",    f"{n_pcs_ser.mean():.1f}")
m5.metric("Windows computed",f"{n_windows}")
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab3, tab2, tab6, tab4, tab5 = st.tabs([
    "① Inputs",
    "② Factor Labels",
    "③ Factor Betas",
    "④ Factor Decomposition",
    "⑤ Fit & Residuals",
    "⑥ Current Regime",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Input Diagnostic (correlation + dendrogram)
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Input Diagnostic: Correlation Structure")
    st.caption(
        "This tab is purely diagnostic. It shows how the macro inputs relate to each other. "
        "If everything is highly correlated (>0.8), PCA will be dominated by a single "
        "market factor and the remaining Factors will be weak. Aim for diverse inputs."
    )

    recent_lr = lr[avail_inputs].iloc[-vol_window:].dropna(axis=1, how="all")
    good_cols = [c for c in avail_inputs if c in recent_lr.columns and recent_lr[c].notna().sum() > 10]

    if len(good_cols) >= 3:
        corr_m = recent_lr[good_cols].corr()

        # Correlation heatmap
        fig_corr = px.imshow(
            corr_m, text_auto=".2f",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title=f"Input Correlation Matrix (last {vol_window} days)"
        )
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Dendrogram
        st.subheader("Dendrogram (Ward Linkage)")
        Z = build_dendrogram(corr_m, good_cols, n_clusters)
        fig_d, ax = plt.subplots(figsize=(12, 4))
        eff_clusters = min(n_clusters, len(good_cols))
        thresh = Z[-(eff_clusters - 1), 2] if eff_clusters <= len(Z) else 0
        dendrogram(Z, labels=good_cols, ax=ax, color_threshold=thresh,
                   above_threshold_color="grey", leaf_rotation=45, leaf_font_size=10)
        ax.set_ylabel("Distance")
        ax.set_title(f"Ward Linkage  |  {eff_clusters} clusters")
        plt.tight_layout()
        st.pyplot(fig_d)

        # Cluster table
        from scipy.cluster.hierarchy import fcluster
        clust = fcluster(Z, t=eff_clusters, criterion="maxclust")
        rows  = []
        for c in sorted(set(clust)):
            members  = [good_cols[i] for i in range(len(good_cols)) if clust[i] == c]
            sub_corr = corr_m.loc[members, members]
            avg      = sub_corr.values[np.triu_indices(len(members), k=1)]
            rows.append({
                "Cluster"              : c,
                "Members"              : ", ".join(members),
                "Avg Intra-Cluster Corr": f"{avg.mean():.3f}" if len(avg) > 0 else "—"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Diagnosis
        all_corrs = corr_m.values[np.triu_indices(len(good_cols), k=1)]
        mean_corr = all_corrs.mean()
        if mean_corr > 0.7:
            st.warning(
                f"⚠️ Mean pairwise correlation across inputs is {mean_corr:.2f}. The inputs are highly collinear. "
                "Factor 1 will dominate. Consider replacing some inputs with more orthogonal macro factors."
            )
        elif mean_corr > 0.4:
            st.info(f"ℹ️ Mean pairwise correlation: {mean_corr:.2f}. Moderate collinearity — reasonable set of inputs.")
        else:
            st.success(f"✅ Mean pairwise correlation: {mean_corr:.2f}. Good spread across risk dimensions.")
    else:
        st.warning("Not enough tickers for correlation analysis.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Rolling Factor Betas
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"{target_ticker} — Sensitivity to Each Latent Factor")
    st.caption(
        "Each line is a rolling beta from regressing the Target on one latent Factor. "
        "A persistently negative beta means the Target moves opposite to that Factor. "
        "Cross-reference with the Factor Labels tab to interpret each beta economically."
    )

    fig_betas = go.Figure()
    for k in range(n_pcs_max):
        col = f"Factor {k+1}"
        if beta_df[col].notna().any():
            fig_betas.add_trace(go.Scatter(
                x=beta_df.index, y=beta_df[col],
                name=col, mode="lines",
                line=dict(color=PC_COLORS[k % len(PC_COLORS)], width=2),
                hovertemplate=f"{col}: %{{y:.3f}}<extra></extra>"
            ))
    fig_betas.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.5)")
    fig_betas.update_layout(
        title=f"{target_ticker} — Rolling Betas on Latent Macro Factors ({pca_window}d window)",
        yaxis_title="Beta",
        hovermode="x unified",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_betas, use_container_width=True)

    # Individual beta panels for detail
    st.subheader("Individual Factor Beta Detail")
    st.caption("Shaded region shows ±1 rolling std of each beta.")

    n_cols = min(2, n_pcs_max)
    cols_per_row = [st.columns(n_cols) for _ in range((n_pcs_max + n_cols - 1) // n_cols)]
    flat_cols = [c for row in cols_per_row for c in row]

    for k in range(n_pcs_max):
        col  = f"Factor {k+1}"
        bser = beta_df[col].dropna()
        if bser.empty:
            continue
        roll_std = bser.rolling(20).std().fillna(0)
        upper    = bser + roll_std
        lower    = bser - roll_std

        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(
            x=list(bser.index) + list(bser.index[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself", fillcolor=hex_to_rgba(PC_COLORS[k % len(PC_COLORS)], 0.12),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, name="±1σ band"
        ))
        fig_k.add_trace(go.Scatter(
            x=bser.index, y=bser,
            name=col, mode="lines",
            line=dict(color=PC_COLORS[k % len(PC_COLORS)], width=2)
        ))
        fig_k.add_hline(y=0, line_dash="dot", line_color="grey")
        fig_k.update_layout(
            title=f"{col} Beta  (mean: {bser.mean():+.3f})",
            height=260, showlegend=False,
            margin=dict(t=40, b=20, l=40, r=10),
            yaxis_title="Beta"
        )
        flat_cols[k].plotly_chart(fig_k, use_container_width=True)

    # Variance explained
    st.subheader("Variance Explained by Retained Factors (Input Structure)")
    st.caption(
        "Shows how much of the inputs' total variance each retained Factor explains. "
        "Factor 1 dominance (>60%) means a one-factor world. "
        "A flatter distribution means multiple orthogonal forces matter."
    )
    fig_var = go.Figure()
    for k in range(n_pcs_max):
        col = f"Factor {k+1}"
        if var_df[col].notna().any():
            fig_var.add_trace(go.Scatter(
                x=var_df.index, y=var_df[col],
                name=col, stackgroup="one", mode="lines",
                fillcolor=hex_to_rgba(PC_COLORS[k % len(PC_COLORS)], 0.67),
                line=dict(color=PC_COLORS[k % len(PC_COLORS)])
            ))
    fig_var.update_layout(
        title="Rolling Variance Explained (Stacked)",
        yaxis_title="Variance Explained",
        yaxis_tickformat=".0%",
        hovermode="x unified", height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_var, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Rolling Factor Labels
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("What Is Each Factor Capturing Over Time?")
    st.caption(
        "Lines show the rolling correlation between each Factor and each input. "
        "Sign matters: a strong negative correlation means the Factor moves opposite to that input. "
        "The heatmap below shows regime shifts — color transitions indicate a change in Factor character."
    )

    from collections import Counter

    asset_color_list = [
        "#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800",
        "#00BCD4", "#E91E63", "#8BC34A", "#795548", "#607D8B"
    ]
    asset_color_map = {
        asset: asset_color_list[j % len(asset_color_list)]
        for j, asset in enumerate(avail_inputs)
    }

    for k in range(n_pcs_max):
        pc_label   = f"Factor {k+1}"
        load_k     = loadings_arr[:, k, :]              # (windows, N) eigenvector weights
        valid_rows = ~np.all(np.isnan(load_k), axis=1)

        if not valid_rows.any():
            continue

        valid_dates  = [d for d, v in zip(dates, valid_rows) if v]
        valid_loads  = load_k[valid_rows]               # (windows, N)

        # ── Current and early loadings ──
        n_valid      = len(valid_loads)
        early_n      = max(1, n_valid // 5)
        early_load   = np.nanmean(valid_loads[:early_n], axis=0)   # (N,)
        current_load = valid_loads[-1]                              # (N,)

        # Sort by current loading magnitude (largest absolute weight first)
        sort_idx = np.argsort(np.abs(current_load))[::-1]

        # ── Detect significant sign flips in eigenvector weights ──
        FLIP_THRESH = 0.15   # eigenvector weights are smaller than correlations
        flipped = []
        for j, asset in enumerate(avail_inputs):
            e, c = early_load[j], current_load[j]
            if np.isnan(e) or np.isnan(c):
                continue
            if np.sign(e) != np.sign(c) and (abs(e) > FLIP_THRESH or abs(c) > FLIP_THRESH):
                direction = "negative → positive" if c > 0 else "positive → negative"
                flipped.append((asset, e, c, direction))
        flipped.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)

        # ── Interpretation card ──
        pos_now = [(avail_inputs[i], current_load[i]) for i in sort_idx if current_load[i] >  FLIP_THRESH]
        neg_now = [(avail_inputs[i], current_load[i]) for i in sort_idx if current_load[i] < -FLIP_THRESH]

        top_pos = pos_now[0][0] if pos_now else None
        top_neg = neg_now[0][0] if neg_now else None   # already sorted by |load| desc

        if top_pos and abs(current_load[avail_inputs.index(top_pos)]) >= 0.3:
            direction_label = f"Primarily loads on <b>{top_pos}</b> (positive)"
        elif top_neg and abs(current_load[avail_inputs.index(top_neg)]) >= 0.3:
            direction_label = f"Primarily loads on <b>{top_neg}</b> (negative)"
        else:
            direction_label = "No single input dominates this Factor right now"

        pos_str = ", ".join(f"<b>{a}</b> ({v:+.3f})" for a, v in pos_now) or "none"
        neg_str = ", ".join(f"<b>{a}</b> ({v:+.3f})" for a, v in neg_now) or "none"

        if flipped:
            flip_lines = "".join(
                f"<li><b>{a}</b>: {d} &nbsp;<span style='color:#888;'>({e:+.3f} → {c:+.3f})</span></li>"
                for a, e, c, d in flipped[:4]
            )
            flip_html = f"""
<div style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.08);">
  <span style="color:#FFB74D; font-size:0.82rem;">⚡ Loading sign flips since start of sample:</span>
  <ul style="margin:4px 0 0 0; padding-left:16px; color:#aaa; font-size:0.82rem; line-height:1.7;">
    {flip_lines}
  </ul>
</div>"""
        else:
            flip_html = '<div style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.08);"><span style="color:#888; font-size:0.82rem;">No significant loading flips — Factor structure is stable.</span></div>'

        st.markdown(f"### {pc_label}")
        st.markdown(f"""
<div style="background:rgba(33,150,243,0.07); border-left:3px solid #2196F3;
            border-radius:4px; padding:12px 16px; margin-bottom:12px; font-size:0.88rem;">
  <div style="margin-bottom:6px;">{direction_label}
    <span style="color:#888; font-size:0.78rem; margin-left:8px;">(current eigenvector)</span>
  </div>
  <div style="color:#aaa; line-height:1.7;">
    Positive weights: {pos_str}<br>
    Negative weights: {neg_str}
  </div>
  {flip_html}
</div>
""", unsafe_allow_html=True)

        # ── Rolling eigenvector weight chart ──
        # Each line = one input's weight in the factor definition over time.
        # This is the eigenvector directly — not a derived correlation.
        # A sign flip here means the factor's composition changed structurally.
        fig_ev = go.Figure()
        for j, asset in enumerate(avail_inputs):
            w_col = valid_loads[:, j]
            if np.isnan(w_col).all():
                continue
            fig_ev.add_trace(go.Scatter(
                x=valid_dates, y=w_col,
                name=asset, mode="lines",
                line=dict(color=asset_color_map[asset], width=1.8),
                hovertemplate=f"{asset}: %{{y:+.3f}}<extra></extra>"
            ))

        fig_ev.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.5)")
        fig_ev.update_layout(
            title=f"{pc_label} — Eigenvector Weights Over Time",
            yaxis=dict(title="Eigenvector weight"),
            hovermode="x unified",
            height=380,
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.15,
                xanchor="left", x=0,
                font_size=10
            ),
            margin=dict(t=40, b=80)
        )
        st.plotly_chart(fig_ev, use_container_width=True)

        # ── Eigenvector heatmap ──
        # Same layout as before but now showing actual loadings, not correlations.
        # Sort rows by current loading (most positive at top, most negative at bottom).
        heat_sort = np.argsort(current_load)[::-1]
        heatmap_z = valid_loads[:, heat_sort].T          # (N, windows)
        heatmap_y = [avail_inputs[i] for i in heat_sort]

        # Symmetric color range based on max absolute loading
        zmax = max(0.2, np.nanmax(np.abs(heatmap_z)))
        zmax = round(zmax + 0.05, 1)

        fig_heat = go.Figure()
        fig_heat.add_trace(go.Heatmap(
            z=heatmap_z,
            x=valid_dates,
            y=heatmap_y,
            zmin=-zmax, zmax=zmax,
            colorscale=[
                [0.0,  "#d32f2f"],
                [0.35, "#ffcdd2"],
                [0.5,  "#f5f5f5"],
                [0.65, "#c8e6c9"],
                [1.0,  "#2e7d32"],
            ],
            colorbar=dict(
                title="Weight",
                thickness=12, len=0.8,
                tickfont=dict(size=10)
            ),
            hovertemplate="<b>%{y}</b><br>%{x|%Y-%m-%d}<br>Weight: %{z:+.3f}<extra></extra>",
            xgap=0.5, ygap=1,
        ))

        # Annotate current value on rightmost column
        for r, asset in enumerate(heatmap_y):
            val = heatmap_z[r, -1]
            if not np.isnan(val):
                fig_heat.add_annotation(
                    x=valid_dates[-1], y=asset,
                    text=f" {val:+.3f}",
                    showarrow=False,
                    font=dict(size=9, color="black" if abs(val) < zmax * 0.6 else "white"),
                    xanchor="left",
                )

        fig_heat.update_layout(
            title=f"{pc_label} — Eigenvector Loadings Heatmap  (green = positive weight, red = negative)",
            height=max(220, 28 * len(heatmap_y) + 60),
            xaxis=dict(showgrid=False),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            margin=dict(t=45, b=20, l=60, r=80),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — R² and Residuals
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model Fit: R² and Residual Analysis")

    # Rolling R²
    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Scatter(
        x=r2_ser.index, y=r2_ser.values,
        name=f"Factor R² ({n_pcs_max} PCs max)",
        mode="lines", fill="tozeroy",
        line=dict(color="#2196F3", width=2),
        fillcolor="rgba(33,150,243,0.1)"
    ))
    fig_r2.add_trace(go.Scatter(
        x=r2_ols_ser.index, y=r2_ols_ser.values,
        name="OLS R² (all inputs)",
        mode="lines", line=dict(color="grey", dash="dash", width=1.5)
    ))
    fig_r2.add_hline(y=r2_ser.mean(), line_dash="dot", line_color="#2196F3",
                     annotation_text=f"Factor model mean: {r2_ser.mean():.1%}")
    fig_r2.update_layout(
        title=f"{target_ticker} — Rolling R²: PCR vs Naive OLS ({pca_window}d window)",
        yaxis=dict(range=[0, 1], title="R²", tickformat=".0%"),
        hovermode="x unified", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    st.caption(
        "OLS R² is always ≥ Factor R² because OLS uses all degrees of freedom. "
        "The gap (shaded below) shows how much variance is coming from noisy higher-order PCs "
        "that PCR deliberately discards. A larger gap = more filtering happening."
    )

    # R² gap
    gap = r2_ols_ser - r2_ser
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(
        x=gap.index, y=gap.values,
        name="R² Gap (OLS − PCR)",
        mode="lines", fill="tozeroy",
        line=dict(color="#FF9800", width=1.5),
        fillcolor="rgba(255,152,0,0.15)"
    ))
    fig_gap.add_hline(y=gap.mean(), line_dash="dot", line_color="#FF9800",
                      annotation_text=f"Mean gap: {gap.mean():.3f}")
    fig_gap.update_layout(
        title="R² Gap — Variance Attributed to Noisy PCs (Discarded by Truncation)",
        yaxis_title="R² Gap", hovermode="x unified", height=300
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    # PCs used over time
    fig_npcs = go.Figure()
    fig_npcs.add_trace(go.Scatter(
        x=n_pcs_ser.index, y=n_pcs_ser.values,
        name="Factors retained", mode="lines",
        line=dict(color="#4CAF50", shape="hv", width=2)
    ))
    fig_npcs.add_hline(
        y=n_assets, line_dash="dash", line_color="red",
        annotation_text=f"All {n_assets} tickers (= OLS)"
    )
    fig_npcs.update_layout(
        title="Number of Factors Retained Per Window",
        yaxis=dict(title="N Factors", dtick=1),
        hovermode="x unified", height=280
    )
    st.plotly_chart(fig_npcs, use_container_width=True)

    st.markdown("---")

    # Residual analysis
    st.subheader("Residual Returns — What Macro Factors Don't Explain")
    st.caption(
        "The residual is the portion of the target's return not explained by the retained Factors. "
        "**Persistent positive drift = either alpha or a missing factor.** "
        "Mean-reverting residuals = idiosyncratic noise. "
        "Large, one-directional moves (2020, 2022) signal regime shifts the model missed."
    )

    fig_resid = go.Figure()
    pos = resid_ser.clip(lower=0)
    neg = resid_ser.clip(upper=0)
    fig_resid.add_trace(go.Bar(
        x=resid_ser.index, y=pos,
        name="Positive residual", marker_color="#4CAF50", opacity=0.75
    ))
    fig_resid.add_trace(go.Bar(
        x=resid_ser.index, y=neg,
        name="Negative residual", marker_color="#F44336", opacity=0.75
    ))
    fig_resid.update_layout(
        barmode="overlay",
        title=f"{target_ticker} — Monthly Residual Return",
        yaxis_title="Standardized return",
        hovermode="x unified", height=350
    )
    st.plotly_chart(fig_resid, use_container_width=True)

    # Cumulative residual
    cum_resid = resid_ser.cumsum()
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=cum_resid.index, y=cum_resid.values,
        name="Cumulative residual", mode="lines", fill="tozeroy",
        line=dict(color="#9C27B0", width=2),
        fillcolor="rgba(156,39,176,0.1)"
    ))
    fig_cum.add_hline(y=0, line_dash="dot", line_color="grey")
    fig_cum.update_layout(
        title="Cumulative Residual — Drift Signals Alpha or Missing Factor",
        yaxis_title="Cumulative standardized return",
        hovermode="x unified", height=350
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Residual stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean residual",       f"{resid_ser.mean():+.4f}")
    c2.metric("Residual std",         f"{resid_ser.std():.4f}")
    c3.metric("Cumulative residual",  f"{cum_resid.iloc[-1]:+.4f}")
    c4.metric("% positive months",   f"{(resid_ser > 0).mean():.1%}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Current Snapshot
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader(f"Current Regime — {dates[-1].strftime('%Y-%m-%d')}")
    st.caption(
        "What each Factor currently represents, the Target's live exposure to each, "
        "and whether that exposure is elevated or depressed relative to its own history."
    )

    latest_corr = corr_arr[-1]    # (n_pcs_max, n_assets)
    latest_betas = betas_arr[-1]  # (n_pcs_max,)

    for k in range(n_pcs_max):
        if np.isnan(latest_betas[k]):
            continue

        pc_label   = f"Factor {k+1}"
        beta_now   = latest_betas[k]
        beta_mean  = beta_df[pc_label].mean()
        beta_zscore = (beta_now - beta_mean) / beta_df[pc_label].std() if beta_df[pc_label].std() > 0 else 0

        # Top 3 assets this Factor correlates with right now
        corr_row   = latest_corr[k]                       # (n_assets,)
        valid_idx  = [i for i in range(n_assets) if not np.isnan(corr_row[i])]
        top3_idx   = sorted(valid_idx, key=lambda i: abs(corr_row[i]), reverse=True)[:3]
        top3_str   = ", ".join(
            f"**{avail_inputs[i]}** ({corr_row[i]:+.2f})"
            for i in top3_idx
        )

        # Var explained now
        var_now = var_arr[-1, k]

        with st.container():
            col_label, col_beta, col_var = st.columns([3, 2, 2])

            col_label.markdown(f"### {pc_label}")
            col_label.markdown(f"Currently: {top3_str}")

            delta_str = f"{beta_now - beta_mean:+.3f} vs avg"
            col_beta.metric(
                label="Current Beta",
                value=f"{beta_now:+.3f}",
                delta=delta_str,
                delta_color="normal"
            )

            col_var.metric(
                label="Variance Explained",
                value=f"{var_now:.1%}" if not np.isnan(var_now) else "—"
            )

            # Beta history for this Factor
            bser_full = beta_df[pc_label].dropna()
            fig_snap = go.Figure()
            fig_snap.add_trace(go.Scatter(
                x=bser_full.index, y=bser_full,
                mode="lines",
                line=dict(color=PC_COLORS[k % len(PC_COLORS)], width=1.5),
                name=pc_label
            ))
            fig_snap.add_hline(y=beta_mean, line_dash="dot", line_color="grey",
                               annotation_text=f"avg: {beta_mean:.3f}")
            # Highlight last point
            fig_snap.add_trace(go.Scatter(
                x=[bser_full.index[-1]], y=[beta_now],
                mode="markers",
                marker=dict(color=PC_COLORS[k % len(PC_COLORS)], size=10, symbol="circle"),
                name="Now", showlegend=False
            ))
            fig_snap.update_layout(
                height=220, showlegend=False,
                margin=dict(t=10, b=20, l=40, r=10),
                yaxis_title="Beta"
            )
            st.plotly_chart(fig_snap, use_container_width=True)

        st.markdown("---")

    # Config summary
    with st.expander("Configuration used for this run"):
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Macro inputs | {', '.join(avail_inputs)} |
| Target | {target_ticker} |
| Data frequency | {freq_label} |
| Rolling window | {pca_window} {win_label} |
| Vol-standardization window | {vol_window} {win_label} |
| Truncation | {'Fixed: ' + str(n_fixed) + ' PCs' if n_fixed else f'Variance ≥ {var_thresh:.0%}'} |
| Data lookback | {lookback_years} years |
| Windows computed | {n_windows} |
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — Factor Decomposition
# ═══════════════════════════════════════════════════════════════════════════
with tab6:

    st.subheader(f"{target_ticker} — Factor Decomposition")
    st.caption(
        "Each chart shows how one Factor contributed to the Target's return over time, "
        "decomposed into the inputs that drove that Factor. "
        "Filled areas = β_k × w_kj × r_j(t) per input j — stacked so they sum to the black line. "
        "The black line = β_k × F_k(t), the Factor's total contribution to the Target. "
        "Red dashed verticals = periods where that Factor's contribution exceeded 2σ."
    )

    # ── Shared color palette ──
    ticker_palette = [
        "#4FC3F7", "#FF7043", "#66BB6A", "#CE93D8", "#FFB74D",
        "#4DD0E1", "#F48FB1", "#AED581", "#A1887F", "#90A4AE"
    ]
    ticker_color_map = {
        asset: ticker_palette[j % len(ticker_palette)]
        for j, asset in enumerate(avail_inputs)
    }

    PLOT_BG   = "rgba(18,18,30,1)"
    PAPER_BG  = "rgba(12,12,20,1)"
    GRID_COL  = "rgba(255,255,255,0.05)"
    ZERO_COL  = "rgba(255,255,255,0.18)"

    for k in range(n_pcs_max):
        pc_label   = f"Factor {k+1}"
        decomp_df  = factor_decomp[k].dropna(how="all")
        if decomp_df.empty:
            continue

        fitted_k   = contrib_df[pc_label].reindex(decomp_df.index).dropna()
        if fitted_k.empty:
            continue

        decomp_df  = decomp_df.reindex(fitted_k.index)

        # ── Regime events for this factor ──
        roll_m  = fitted_k.rolling(60, min_periods=20).mean()
        roll_s  = fitted_k.rolling(60, min_periods=20).std()
        thresh  = roll_m.abs() + 2 * roll_s
        ev_mask = (np.abs(fitted_k.values) > thresh.values) & ~np.isnan(thresh.values)
        ev_dates = fitted_k.index[ev_mask]

        # ── Dominant input this factor ──
        dominant = decomp_df.abs().mean().idxmax()
        var_exp_k = var_df[pc_label].mean()

        # ── Header row ──
        h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
        h1.markdown(f"#### {pc_label}")
        h2.metric("Avg var explained", f"{var_exp_k:.1%}" if not np.isnan(var_exp_k) else "—")
        h3.metric("Dominant input", dominant)
        h4.metric("Regime events", int(ev_mask.sum()))

        # ── Stacked area chart ──
        # Use two stackgroups (positive / negative) so areas diverge from zero
        # correctly, matching the reference image style.
        fig = go.Figure()

        for j, asset in enumerate(avail_inputs):
            col   = decomp_df[asset]
            color = ticker_color_map[asset]
            # hex → rgba helper for fill
            r_int = int(color[1:3], 16)
            g_int = int(color[3:5], 16)
            b_int = int(color[5:7], 16)
            fill_color = f"rgba({r_int},{g_int},{b_int},0.55)"

            show_leg = (k == 0)   # only show legend on first chart; same colors throughout

            # Positive stack
            pos_vals = col.clip(lower=0)
            fig.add_trace(go.Scatter(
                x=decomp_df.index, y=pos_vals,
                name=asset,
                stackgroup="pos",
                mode="none",
                fillcolor=fill_color,
                legendgroup=asset,
                showlegend=show_leg,
                hovertemplate=f"<b>{asset}</b>: %{{y:+.3f}}<extra></extra>"
            ))
            # Negative stack
            neg_vals = col.clip(upper=0)
            fig.add_trace(go.Scatter(
                x=decomp_df.index, y=neg_vals,
                name=asset,
                stackgroup="neg",
                mode="none",
                fillcolor=fill_color,
                legendgroup=asset,
                showlegend=False,
                hoverinfo="skip"
            ))

        # Black line = β_k × F_k(t) — total Factor contribution to Target
        fig.add_trace(go.Scatter(
            x=fitted_k.index, y=fitted_k.values,
            name=f"β_{k+1}·F_{k+1}(t)",
            mode="lines",
            line=dict(color="rgba(255,255,255,0.92)", width=2),
            hovertemplate=f"<b>β·F = %{{y:+.3f}}</b><extra></extra>"
        ))

        # Zero line
        fig.add_hline(y=0, line_color=ZERO_COL, line_width=1)

        # Regime verticals
        for ev in ev_dates:
            fig.add_vline(x=ev, line_dash="dash",
                          line_color="rgba(220,80,80,0.5)", line_width=1)

        # Annotate peak event
        if len(ev_dates) > 0:
            peak = fitted_k.abs().idxmax()
            fig.add_annotation(
                x=peak, y=fitted_k[peak],
                text=f"  {peak.strftime('%b %Y')}",
                showarrow=False,
                font=dict(color="rgba(220,100,100,0.9)", size=10),
                xanchor="left", yanchor="middle"
            )

        fig.update_layout(
            paper_bgcolor=PAPER_BG,
            plot_bgcolor=PLOT_BG,
            font=dict(color="rgba(255,255,255,0.72)", size=11),
            yaxis=dict(
                title="β_k · w_kj · r_j(t)",
                gridcolor=GRID_COL,
                zerolinecolor=ZERO_COL,
                tickfont=dict(size=10),
            ),
            xaxis=dict(gridcolor=GRID_COL, tickfont=dict(size=10)),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="rgba(15,15,26,0.95)",
                bordercolor="rgba(255,255,255,0.15)",
                font_size=12
            ),
            height=380,
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.12,
                xanchor="left", x=0,
                font=dict(size=10, color="rgba(255,255,255,0.65)"),
                bgcolor="rgba(0,0,0,0)",
            ) if k == 0 else dict(visible=False),
            margin=dict(t=10, b=80 if k == 0 else 30, l=60, r=20),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
