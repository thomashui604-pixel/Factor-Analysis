"""
Rolling PCA Factor Dashboard
==============================
Answers: "What latent macro forces are driving my target,
          and how have those exposures shifted over time?"

Pipeline:
  1. Download macro universe + target
  2. Compute log returns → vol-standardize
  3. For each rolling window:
       a. PCA on basket → K latent factors
       b. Sign-align eigenvectors to first window (not just previous)
       c. OLS: target ~ factors → rolling betas
       d. Record correlation of each PC with each basket asset (rolling labels)
  4. Visualize: betas, labels, R², residuals, current snapshot

What changed from v1:
  - Output is factor betas, not back-transformed ticker weights.
    Ticker weights collapse the factor structure back to asset space
    and throw away the information PCA surfaced.
  - Sign alignment anchors to first window, not just previous window,
    preventing sign drift over long samples.
  - Rolling label tracking in every window, not just the last.
    This is the key diagnostic: does PC2 mean the same thing in 2019 as 2023?
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

st.sidebar.subheader("Universe")
st.sidebar.caption(
    "These are the macro risk factors PCA runs on. "
    "Each should proxy a distinct, economically meaningful dimension. "
    "Avoid loading this with correlated equity names — you'll just get "
    "the market factor plus noise."
)

DEFAULT_BASKET = "SPY, TLT, HYG, VIXY, UUP, GSG, IWM, IWF, IWD, EEM"
ticker_input = st.sidebar.text_area(
    "Macro Basket (comma-separated)",
    value=DEFAULT_BASKET,
    help=(
        "SPY=equity  TLT=rates  HYG=credit  VIXY=vol  "
        "UUP=dollar  GSG=commodity  IWM=size  IWF=growth  IWD=value  EEM=EM"
    )
)
basket_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

target_ticker = st.sidebar.text_input(
    "Target Ticker",
    value="ARKK",
    help="The asset you want to decompose into macro factor exposures."
).strip().upper()

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

lookback_years = st.sidebar.slider(
    "Data Lookback (years)", min_value=2, max_value=10, value=5, step=1
)

pca_window = st.sidebar.slider(
    "Rolling Window (trading days)",
    min_value=20, max_value=120, value=60, step=5,
    help=(
        "How many days per rolling window. "
        "60d ≈ 3 months (regime-sensitive). "
        "120d ≈ 6 months (smoother, slower)."
    )
)

vol_window = st.sidebar.selectbox(
    "Vol-Standardization Window (days)",
    options=[21, 63, 126, 252], index=1,
    help="Window for computing rolling std used to normalize returns before PCA."
)

st.sidebar.markdown("---")
st.sidebar.subheader("PC Truncation")
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
        "Number of PCs to keep",
        min_value=1, max_value=min(len(basket_tickers), 8),
        value=min(3, len(basket_tickers)), step=1
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
    min_value=2, max_value=min(8, len(basket_tickers)),
    value=min(4, len(basket_tickers)), step=1
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

      1. PCA on the basket → K latent factors
      2. Sign-align eigenvectors to the FIRST window (not just prev window)
         so betas are comparable across all time.
      3. OLS: target ~ factors → rolling betas
      4. Record correlation of each PC with each basket asset (rolling labels).
         This answers: "what does PC2 mean in June 2022 vs June 2019?"
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
    out_var_exp  = []          # (K,) variance explained by each retained PC
    out_pc_corr  = []          # (K, N) correlation of each PC with each asset
    out_n_pcs    = []

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
        out_dates.append(std_basket.index[t])
        out_betas.append(betas[1:k+1])           # drop intercept
        out_r2.append(r2)
        out_r2_ols.append(r2_ols)
        out_resid.append(resid[-1])
        out_var_exp.append(eigvals[:k] / total_var)
        out_pc_corr.append(corr_block)
        out_n_pcs.append(k)

    return {
        "dates"    : out_dates,
        "betas"    : out_betas,        # list of arrays, lengths vary if var_thresh
        "r2"       : out_r2,
        "r2_ols"   : out_r2_ols,
        "resid"    : out_resid,
        "var_exp"  : out_var_exp,
        "pc_corr"  : out_pc_corr,      # list of (k, N) arrays
        "n_pcs"    : out_n_pcs,
        "assets"   : asset_cols,
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

    1. **Macro basket** — PCA runs on a set of economically distinct assets
       (equity, rates, credit, vol, dollar, commodities, size, growth, value, EM).
       These span genuinely different risk dimensions. Loading the basket with
       correlated sector ETFs collapses everything into one equity factor.

    2. **Vol-standardize returns** — divide each return by its rolling std before PCA.
       This prevents high-vol assets from dominating the first PC simply by being noisier.

    3. **Rolling PCA** — in each window, eigendecompose the sample covariance matrix
       of the basket. Extract K latent factors.

    4. **Sign alignment** — eigenvectors are only defined up to a sign flip.
       We anchor signs to the first window, so betas are comparable across all time.
       Without this, a beta that flips sign might just be a sign flip in the eigenvector.

    5. **Regress target on factors** — OLS in each window gives rolling betas.
       These answer: "how much does ARKK load on the volatility factor right now?"

    6. **Rolling label tracking** — in each window, correlate each PC with each
       basket asset. This tracks whether PC2 means the same thing in 2019 as in 2023.
       Factor interpretation is not stable. This makes the instability visible.

    ---

    ### Key outputs

    - **Rolling betas** — how the target's sensitivity to each latent factor changes over time
    - **Rolling PC labels** — what each factor currently represents economically
    - **R² vs OLS gap** — how much the truncation is filtering vs. a naive regression
    - **Residual** — the idiosyncratic component PCA doesn't explain (potential alpha)
    """)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Downloading price data..."):
    prices = load_data(basket_tickers, target_ticker, lookback_years, vol_window)

if prices is None or prices.empty:
    st.error("No price data returned. Check tickers or try again.")
    st.stop()

avail_basket = [t for t in basket_tickers if t in prices.columns]
missing      = [t for t in basket_tickers if t not in prices.columns]
if target_ticker not in prices.columns:
    st.error(f"Target '{target_ticker}' not found in downloaded data.")
    st.stop()
if missing:
    st.warning(f"Tickers not found: {', '.join(missing)}")
if len(avail_basket) < 3:
    st.error("Need at least 3 basket tickers with data.")
    st.stop()

with st.spinner("Computing standardized returns..."):
    lr   = log_returns(prices)
    std  = vol_standardize(lr, vol_window)
    std_basket = std[avail_basket].dropna(axis=1, how="all").dropna()
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

pc_cols    = [f"PC{k+1}" for k in range(n_pcs_max)]
beta_df    = pd.DataFrame(betas_arr, index=dates, columns=pc_cols)
var_df     = pd.DataFrame(var_arr,   index=dates, columns=pc_cols)
r2_ser     = pd.Series(res["r2"],     index=dates, name="R²")
r2_ols_ser = pd.Series(res["r2_ols"], index=dates, name="OLS R²")
resid_ser  = pd.Series(res["resid"],  index=dates, name="Residual")
n_pcs_ser  = pd.Series(res["n_pcs"], index=dates, name="N PCs")


# ─────────────────────────────────────────────────────────────────────────────
# Summary header metrics
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Target",          target_ticker)
m2.metric("Avg R²",          f"{r2_ser.mean():.1%}")
m3.metric("Latest R²",       f"{r2_ser.iloc[-1]:.1%}")
m4.metric("Avg PCs used",    f"{n_pcs_ser.mean():.1f}")
m5.metric("Windows computed",f"{n_windows}")
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Basket Diagnostic",
    "📊 Rolling Factor Betas",
    "🏷️ Rolling PC Labels",
    "📈 R² & Residuals",
    "🔍 Current Snapshot",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Basket diagnostic (correlation + dendrogram)
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Basket Diagnostic: Correlation Structure")
    st.caption(
        "This tab is purely diagnostic. It shows how the basket assets relate to each other. "
        "If everything is highly correlated (>0.8), your PCA will be dominated by a single "
        "market factor and the remaining PCs will be weak. Aim for a diverse basket."
    )

    recent_lr = lr[avail_basket].iloc[-vol_window:].dropna(axis=1, how="all")
    good_cols = [c for c in avail_basket if c in recent_lr.columns and recent_lr[c].notna().sum() > 10]

    if len(good_cols) >= 3:
        corr_m = recent_lr[good_cols].corr()

        # Correlation heatmap
        fig_corr = px.imshow(
            corr_m, text_auto=".2f",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title=f"Basket Correlation Matrix (last {vol_window} days)"
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
                f"⚠️ Mean pairwise correlation is {mean_corr:.2f}. The basket is highly collinear. "
                "PC1 will dominate. Consider replacing some assets with more orthogonal macro factors."
            )
        elif mean_corr > 0.4:
            st.info(f"ℹ️ Mean pairwise correlation: {mean_corr:.2f}. Moderate collinearity — reasonable basket.")
        else:
            st.success(f"✅ Mean pairwise correlation: {mean_corr:.2f}. Good spread across risk dimensions.")
    else:
        st.warning("Not enough tickers for correlation analysis.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Rolling Factor Betas
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Rolling Factor Betas: {target_ticker}'s Sensitivity to Each Latent Factor")
    st.caption(
        "Each line is a beta from regressing the target on one latent PC. "
        "**This is the primary output.** "
        "A rising PC1 beta means the target is becoming more sensitive to whatever PC1 represents. "
        "Check Tab 3 to see what each PC represents at any given time."
    )

    fig_betas = go.Figure()
    for k in range(n_pcs_max):
        col = f"PC{k+1}"
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
        col  = f"PC{k+1}"
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
            fill="toself", fillcolor=PC_COLORS[k % len(PC_COLORS)].replace(")", ",0.12)").replace("rgb", "rgba") if "rgb" in PC_COLORS[k%len(PC_COLORS)] else PC_COLORS[k % len(PC_COLORS)] + "20",
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
    st.subheader("Variance Explained by Retained PCs (Universe Structure)")
    st.caption(
        "Shows how much of the basket's total variance each retained PC explains. "
        "PC1 dominance (>60%) means a one-factor world. "
        "A flatter distribution means multiple orthogonal forces matter."
    )
    fig_var = go.Figure()
    for k in range(n_pcs_max):
        col = f"PC{k+1}"
        if var_df[col].notna().any():
            fig_var.add_trace(go.Scatter(
                x=var_df.index, y=var_df[col],
                name=col, stackgroup="one", mode="lines",
                fillcolor=PC_COLORS[k % len(PC_COLORS)] + "AA",
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
# TAB 3 — Rolling PC Labels
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Rolling PC Label Tracking")
    st.caption(
        "In each rolling window, we correlate each PC's realized time series "
        "with each basket asset's returns. "
        "**This tracks whether the economic interpretation of each PC is stable over time.** "
        "If PC2's label shifts from 'Credit' to 'Dollar' midway through the sample, "
        "any model treating PC2 as a fixed factor is wrong."
    )

    for k in range(n_pcs_max):
        pc_label = f"PC{k+1}"

        # Build time series of correlation with each asset
        corr_ts = pd.DataFrame(
            corr_arr[:, k, :],
            index=dates,
            columns=avail_basket
        ).dropna(how="all")

        if corr_ts.empty:
            continue

        st.subheader(f"{pc_label} — Rolling Correlation with Each Basket Asset")

        fig_label = go.Figure()
        asset_colors = px.colors.qualitative.Set2
        for j, asset in enumerate(avail_basket):
            if asset in corr_ts.columns and corr_ts[asset].notna().any():
                fig_label.add_trace(go.Scatter(
                    x=corr_ts.index,
                    y=corr_ts[asset],
                    name=asset, mode="lines",
                    line=dict(color=asset_colors[j % len(asset_colors)], width=1.8),
                    hovertemplate=f"{asset}: %{{y:.2f}}<extra></extra>"
                ))
        fig_label.add_hline(y=0,    line_dash="dot",  line_color="rgba(150,150,150,0.5)")
        fig_label.add_hline(y=0.5,  line_dash="dash", line_color="rgba(150,150,150,0.3)")
        fig_label.add_hline(y=-0.5, line_dash="dash", line_color="rgba(150,150,150,0.3)")
        fig_label.update_layout(
            yaxis=dict(range=[-1, 1], title="Correlation with PC"),
            hovermode="x unified",
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_label, use_container_width=True)

        # Label stability summary
        # For each window, what asset has the highest |corr|?
        dominant_idx  = np.nanargmax(np.abs(corr_arr[:, k, :]), axis=1)
        dominant_name = [avail_basket[i] for i in dominant_idx]
        from collections import Counter
        counts = Counter(dominant_name)
        total  = len(dominant_name)

        stability_rows = [
            {"Asset": a, "Dominant in % of windows": f"{c/total:.0%}", "Count": c}
            for a, c in counts.most_common()
        ]
        with st.expander(f"{pc_label} — Label Stability (what asset dominates, and how often)"):
            st.dataframe(pd.DataFrame(stability_rows), use_container_width=True, hide_index=True)

            # Interpret stability
            top_pct = counts.most_common(1)[0][1] / total
            top_asset = counts.most_common(1)[0][0]
            if top_pct > 0.8:
                st.success(
                    f"✅ **{pc_label} is stable**: dominated by **{top_asset}** in "
                    f"{top_pct:.0%} of windows. This factor has a consistent economic interpretation."
                )
            elif top_pct > 0.5:
                st.info(
                    f"ℹ️ **{pc_label} is moderately stable**: {top_asset} leads ({top_pct:.0%}) "
                    f"but other assets take over at times. Interpret with some caution."
                )
            else:
                st.warning(
                    f"⚠️ **{pc_label} is unstable**: no single asset dominates. "
                    f"The economic interpretation of this PC shifts across regimes. "
                    f"Treat betas on this factor as regime-dependent."
                )

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
        name=f"PCR R² ({n_pcs_max} PCs max)",
        mode="lines", fill="tozeroy",
        line=dict(color="#2196F3", width=2),
        fillcolor="rgba(33,150,243,0.1)"
    ))
    fig_r2.add_trace(go.Scatter(
        x=r2_ols_ser.index, y=r2_ols_ser.values,
        name="OLS R² (all basket tickers)",
        mode="lines", line=dict(color="grey", dash="dash", width=1.5)
    ))
    fig_r2.add_hline(y=r2_ser.mean(), line_dash="dot", line_color="#2196F3",
                     annotation_text=f"PCR mean: {r2_ser.mean():.1%}")
    fig_r2.update_layout(
        title=f"{target_ticker} — Rolling R²: PCR vs Naive OLS ({pca_window}d window)",
        yaxis=dict(range=[0, 1], title="R²", tickformat=".0%"),
        hovermode="x unified", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    st.caption(
        "OLS R² is always ≥ PCR R² because OLS uses all degrees of freedom. "
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
        name="PCs retained", mode="lines",
        line=dict(color="#4CAF50", shape="hv", width=2)
    ))
    fig_npcs.add_hline(
        y=n_assets, line_dash="dash", line_color="red",
        annotation_text=f"All {n_assets} tickers (= OLS)"
    )
    fig_npcs.update_layout(
        title="Number of PCs Retained Per Window",
        yaxis=dict(title="N PCs", dtick=1),
        hovermode="x unified", height=280
    )
    st.plotly_chart(fig_npcs, use_container_width=True)

    st.markdown("---")

    # Residual analysis
    st.subheader("Residual Returns — What Macro Factors Don't Explain")
    st.caption(
        "The residual is the portion of the target's return not explained by the retained PCs. "
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
    st.subheader(f"Current Snapshot — {dates[-1].strftime('%Y-%m-%d')}")
    st.caption(
        "The regime right now: what each PC currently represents, "
        "what the target's exposure is, and how that compares to the full-sample average."
    )

    latest_corr = corr_arr[-1]    # (n_pcs_max, n_assets)
    latest_betas = betas_arr[-1]  # (n_pcs_max,)

    for k in range(n_pcs_max):
        if np.isnan(latest_betas[k]):
            continue

        pc_label   = f"PC{k+1}"
        beta_now   = latest_betas[k]
        beta_mean  = beta_df[pc_label].mean()
        beta_zscore = (beta_now - beta_mean) / beta_df[pc_label].std() if beta_df[pc_label].std() > 0 else 0

        # Top 3 assets this PC correlates with right now
        corr_row   = latest_corr[k]                       # (n_assets,)
        valid_idx  = [i for i in range(n_assets) if not np.isnan(corr_row[i])]
        top3_idx   = sorted(valid_idx, key=lambda i: abs(corr_row[i]), reverse=True)[:3]
        top3_str   = ", ".join(
            f"**{avail_basket[i]}** ({corr_row[i]:+.2f})"
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

            # Beta history for this PC
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
| Basket tickers | {', '.join(avail_basket)} |
| Target | {target_ticker} |
| Rolling window | {pca_window} days |
| Vol-standardization window | {vol_window} days |
| Truncation | {'Fixed: ' + str(n_fixed) + ' PCs' if n_fixed else f'Variance ≥ {var_thresh:.0%}'} |
| Data lookback | {lookback_years} years |
| Windows computed | {n_windows} |
        """)
