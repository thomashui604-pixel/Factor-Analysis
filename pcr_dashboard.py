import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from numpy.linalg import lstsq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from io import BytesIO
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
TICKER_REMAP = {
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "TNX": "^TNX",
    "TYX": "^TYX",
    "IRX": "^IRX",
    "GSPC": "^GSPC",
    "DJI": "^DJI",
    "IXIC": "^IXIC",
    "RUT": "^RUT",
}

TICKER_DISPLAY = {v: k for k, v in TICKER_REMAP.items()}


def remap_tickers(tickers):
    return [TICKER_REMAP.get(t, t) for t in tickers]


def display_name(ticker):
    return TICKER_DISPLAY.get(ticker, ticker)


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="PCR Dashboard", layout="wide")
st.title("PCA / PCR Rolling Analysis Dashboard")

# ─────────────────────────────────────────────
# Sidebar: Inputs
# ─────────────────────────────────────────────
st.sidebar.header("Configuration")

default_tickers = "SPY, QQQ, IWM, TLT, HYG, GLD, USO, UUP, VIX, EEM"
ticker_input = st.sidebar.text_area(
    "Basket Tickers (comma-separated)",
    value=default_tickers,
    help="All basket tickers feed into PCA directly. The dendrogram visualizes their correlation structure separately."
)
basket_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

target_ticker = st.sidebar.text_input(
    "Target Ticker",
    value="IGV",
    help="The ticker you want to regress against the PCs."
).strip().upper()

st.sidebar.markdown("---")
st.sidebar.subheader("Windows & Parameters")

vol_window = st.sidebar.selectbox(
    "Vol Standardization Window (days)",
    options=[63, 126, 252],
    index=2,
    help="Window for computing std deviation used to normalize returns."
)

pca_window = st.sidebar.slider(
    "Rolling PCA Window (days)",
    min_value=20, max_value=60, value=30, step=5,
    help="Number of trading days for each rolling PCA computation."
)

st.sidebar.markdown("---")
st.sidebar.subheader("PC Truncation")
st.sidebar.caption(
    "This is the key control. Using ALL PCs makes PCA identical to OLS — "
    "the rotation cancels out. Truncation is the signal."
)

truncation_method = st.sidebar.radio(
    "Truncation Method",
    options=["Fixed Number of PCs", "Variance Threshold"],
    index=0,
    help="Fixed number is more principled. Variance threshold adapts but risks including all PCs."
)

if truncation_method == "Fixed Number of PCs":
    max_pcs = min(len(basket_tickers), 10)
    n_fixed_pcs = st.sidebar.slider(
        "Number of PCs to Keep",
        min_value=1, max_value=max_pcs,
        value=min(3, max_pcs), step=1,
        help="Regress target on only this many PCs. Remaining PCs are treated as noise."
    )
    variance_threshold = None
else:
    variance_threshold = st.sidebar.slider(
        "Variance Explained Threshold",
        min_value=50, max_value=95, value=80, step=5,
        format="%d%%",
        help="Include PCs until this % of variance is explained. Capped at 95% to prevent using all PCs."
    )
    variance_threshold = variance_threshold / 100.0
    n_fixed_pcs = None

lookback_years = st.sidebar.slider(
    "Data Lookback (years)",
    min_value=1, max_value=10, value=3, step=1,
    help="How many years of historical data to download."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dendrogram Settings")
st.sidebar.caption("Dendrogram is a diagnostic visualization — it does not affect PCA input.")

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2, max_value=min(10, len(basket_tickers)),
    value=min(5, len(basket_tickers)),
    step=1,
    help="How many clusters to cut the dendrogram into (visual only)."
)


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers, target, years, _vol_window):
    all_tickers = list(set(tickers + [target]))
    yf_tickers = remap_tickers(all_tickers)
    end = datetime.today()
    start = end - timedelta(days=years * 365 + _vol_window + 60)

    prices = None
    for attempt in range(3):
        try:
            data = yf.download(yf_tickers, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"]
            else:
                prices = data[["Close"]]
                prices.columns = yf_tickers
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise e

    if prices is None or prices.empty:
        return pd.DataFrame()

    rename_map = {yf: orig for orig, yf in zip(all_tickers, yf_tickers)}
    prices = prices.rename(columns=rename_map)
    prices = prices.dropna(how="all")
    return prices


# ─────────────────────────────────────────────
# Core computations
# ─────────────────────────────────────────────
def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def standardize_returns(log_returns, vol_win):
    """Standardize log returns by rolling vol (std dev)."""
    rolling_std = log_returns.rolling(window=vol_win).std()
    standardized = log_returns / rolling_std
    return standardized.dropna()


def run_dendrogram_analysis(corr_matrix, tickers):
    """Compute linkage for dendrogram from correlation matrix."""
    dist_matrix = 1 - corr_matrix.abs()
    np.fill_diagonal(dist_matrix.values, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    condensed = squareform(dist_matrix.values, checks=False)
    Z = linkage(condensed, method="ward")
    return Z


def rolling_pca_regression(std_returns_basket, std_returns_target, pca_win,
                           n_fixed_pcs=None, var_thresh=None):
    """
    Rolling PCA on full basket → truncated regression on target → back-transform
    to effective ticker weights.

    Key: n_pcs is ALWAYS less than n_tickers. Using all PCs is algebraically
    identical to OLS and defeats the purpose.
    """
    n_obs = len(std_returns_basket)
    tickers = std_returns_basket.columns.tolist()
    n_tickers = len(tickers)

    # Storage
    dates = []
    all_loadings = []
    all_var_explained = []
    all_n_pcs = []
    all_ticker_weights = []
    all_ticker_weights_ols = []  # naive OLS comparison
    all_r2 = []
    all_r2_ols = []
    all_pc_betas = []

    min_start = max(0, pca_win)
    prev_components = None

    for i in range(min_start, n_obs):
        window_start = i - pca_win
        X_window = std_returns_basket.iloc[window_start:i].values
        y_window = std_returns_target.iloc[window_start:i].values

        valid_mask = ~(np.isnan(X_window).any(axis=1) | np.isnan(y_window))
        X_clean = X_window[valid_mask]
        y_clean = y_window[valid_mask]

        if len(X_clean) < n_tickers + 2:
            continue

        # ── PCA on full basket ──
        pca = PCA()
        try:
            pc_scores = pca.fit_transform(X_clean)
        except Exception:
            continue

        # ── Sign consistency ──
        if prev_components is not None:
            n_compare = min(pca.components_.shape[0], prev_components.shape[0])
            for pc_idx in range(n_compare):
                dot = np.dot(pca.components_[pc_idx], prev_components[pc_idx])
                if dot < 0:
                    pca.components_[pc_idx] *= -1
                    pc_scores[:, pc_idx] *= -1
        prev_components = pca.components_.copy()

        # ── Determine number of PCs to keep (TRUNCATION) ──
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        if n_fixed_pcs is not None:
            n_pcs = min(n_fixed_pcs, len(cum_var))
        else:
            n_pcs = int(np.searchsorted(cum_var, var_thresh) + 1)
            n_pcs = min(n_pcs, len(cum_var))

        # Hard cap: never use all PCs (that's just OLS)
        if n_pcs >= n_tickers:
            n_pcs = max(1, n_tickers - 1)

        # ── Stage 2: Regress target on truncated PCs ──
        X_pcs = pc_scores[:, :n_pcs]
        reg = LinearRegression().fit(X_pcs, y_clean)
        r2 = reg.score(X_pcs, y_clean)

        # ── Back-transform to ticker weights ──
        loadings = pca.components_[:n_pcs, :].T  # (n_tickers, n_pcs)
        pc_betas = reg.coef_
        ticker_weights = loadings @ pc_betas

        # ── Naive OLS for comparison ──
        reg_ols = LinearRegression().fit(X_clean, y_clean)
        r2_ols = reg_ols.score(X_clean, y_clean)
        ticker_weights_ols = reg_ols.coef_

        date = std_returns_basket.index[i]
        dates.append(date)
        all_loadings.append(pca.components_[:n_pcs, :])
        all_var_explained.append(pca.explained_variance_ratio_[:n_pcs])
        all_n_pcs.append(n_pcs)
        all_ticker_weights.append(ticker_weights)
        all_ticker_weights_ols.append(ticker_weights_ols)
        all_r2.append(r2)
        all_r2_ols.append(r2_ols)
        all_pc_betas.append(pc_betas)

    results = {
        "dates": dates,
        "loadings": all_loadings,
        "var_explained": all_var_explained,
        "n_pcs": all_n_pcs,
        "ticker_weights": all_ticker_weights,
        "ticker_weights_ols": all_ticker_weights_ols,
        "r2": all_r2,
        "r2_ols": all_r2_ols,
        "pc_betas": all_pc_betas,
        "tickers": tickers,
    }
    return results


# ─────────────────────────────────────────────
# Run the pipeline
# ─────────────────────────────────────────────
run_button = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

if run_button:
    with st.spinner("Downloading data..."):
        prices = load_data(basket_tickers, target_ticker, lookback_years, vol_window)

    if prices.empty:
        st.error("Failed to download price data. Try again in a moment (Yahoo Finance may be rate limiting).")
        st.stop()

    available_basket = [t for t in basket_tickers if t in prices.columns]
    missing = [t for t in basket_tickers if t not in prices.columns]
    if target_ticker not in prices.columns:
        st.error(f"Target ticker '{target_ticker}' not found.")
        st.stop()
    if missing:
        st.warning(f"Missing tickers (not found or rate limited): {', '.join(missing)}")
    if len(available_basket) < 3:
        st.error("Need at least 3 basket tickers with data.")
        st.stop()

    with st.spinner("Computing returns and standardizing..."):
        log_ret = compute_log_returns(prices)
        std_ret = standardize_returns(log_ret, vol_window)
        # PCA runs on ALL basket tickers
        std_basket = std_ret[[t for t in available_basket if t in std_ret.columns]].dropna(axis=1, how="all")
        std_target = std_ret[target_ticker].dropna()
        common_idx = std_basket.dropna().index.intersection(std_target.dropna().index)
        std_basket = std_basket.loc[common_idx]
        std_target = std_target.loc[common_idx]

    available_basket = std_basket.columns.tolist()

    # ─── Tab Layout ───
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dendrogram (Diagnostic)",
        "🔄 Rolling PCA Loadings",
        "🎯 Ticker Weight Decomposition",
        "📈 Model Diagnostics",
        "📋 Summary Table"
    ])

    # ═══════════════════════════════════════════
    # TAB 1: Dendrogram (diagnostic only — does NOT gate PCA)
    # ═══════════════════════════════════════════
    with tab1:
        st.subheader("Dendrogram & Correlation Structure")
        st.caption(
            "This is a diagnostic view of the basket's correlation structure. "
            "It does not affect PCA — all basket tickers feed into PCA directly."
        )

        recent_ret = log_ret[available_basket].iloc[-vol_window:].dropna(axis=1, how="all")
        available_basket_dendro = [t for t in available_basket if t in recent_ret.columns and recent_ret[t].notna().sum() > 10]

        if len(available_basket_dendro) >= 3:
            recent_ret = recent_ret[available_basket_dendro]
            corr_matrix = recent_ret.corr().dropna(axis=0, how="all").dropna(axis=1, how="all")

            Z = run_dendrogram_analysis(corr_matrix, available_basket_dendro)

            fig_dendro, ax = plt.subplots(figsize=(12, 5))
            effective_clusters = min(n_clusters, len(available_basket_dendro))
            dendro_result = dendrogram(
                Z,
                labels=available_basket_dendro,
                ax=ax,
                color_threshold=Z[-(effective_clusters - 1), 2] if effective_clusters <= len(Z) else 0,
                above_threshold_color="grey",
                leaf_rotation=45,
                leaf_font_size=10
            )
            ax.set_title(f"Dendrogram (Ward Linkage, {vol_window}d Correlation Distance)")
            ax.set_ylabel("Distance")
            plt.tight_layout()
            st.pyplot(fig_dendro)

            # Cluster assignments (informational)
            clusters = fcluster(Z, t=effective_clusters, criterion="maxclust")
            cluster_df_rows = []
            for c in sorted(set(clusters)):
                members = [available_basket_dendro[i] for i in range(len(available_basket_dendro)) if clusters[i] == c]
                sub_corr = corr_matrix.loc[members, members]
                avg_corr = sub_corr.mean().mean()
                cluster_df_rows.append({
                    "Cluster": c,
                    "Members": ", ".join(members),
                    "Avg Intra-Cluster Corr": f"{avg_corr:.3f}"
                })
            st.dataframe(pd.DataFrame(cluster_df_rows), use_container_width=True, hide_index=True)

            # Correlation heatmap
            st.subheader("Correlation Matrix (Full Basket)")
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title=f"Correlation Matrix ({vol_window}d)"
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Not enough tickers for dendrogram (need 3+).")

    # ═══════════════════════════════════════════
    # Run PCA/PCR on FULL basket
    # ═══════════════════════════════════════════
    with st.spinner("Running rolling PCA regression on full basket..."):
        results = rolling_pca_regression(
            std_basket, std_target, pca_window,
            n_fixed_pcs=n_fixed_pcs, var_thresh=variance_threshold
        )

    if len(results["dates"]) == 0:
        st.error("No valid rolling windows. Try reducing PCA window or adding more data.")
        st.stop()

    # ═══════════════════════════════════════════
    # TAB 2: Rolling PCA Loadings
    # ═══════════════════════════════════════════
    with tab2:
        st.subheader("Rolling PC1 Loadings Over Time")
        st.caption("Shows how each ticker's contribution to PC1 evolves across all basket tickers.")

        pc1_loadings_over_time = []
        for i, date in enumerate(results["dates"]):
            loadings_i = results["loadings"][i]
            row = {"Date": date}
            for j, ticker in enumerate(results["tickers"]):
                row[ticker] = loadings_i[0, j]
            pc1_loadings_over_time.append(row)

        pc1_df = pd.DataFrame(pc1_loadings_over_time).set_index("Date")

        fig_loadings = go.Figure()
        for col in pc1_df.columns:
            fig_loadings.add_trace(go.Scatter(
                x=pc1_df.index, y=pc1_df[col],
                name=col, mode="lines"
            ))
        fig_loadings.update_layout(
            title=f"Rolling PC1 Loadings ({pca_window}d window, {len(available_basket)} tickers)",
            yaxis_title="Loading",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_loadings, use_container_width=True)

        # PC2 loadings
        has_pc2 = any(n >= 2 for n in results["n_pcs"])
        if has_pc2:
            st.subheader("Rolling PC2 Loadings Over Time")
            pc2_loadings_over_time = []
            for i, date in enumerate(results["dates"]):
                loadings_i = results["loadings"][i]
                row = {"Date": date}
                if loadings_i.shape[0] >= 2:
                    for j, ticker in enumerate(results["tickers"]):
                        row[ticker] = loadings_i[1, j]
                else:
                    for ticker in results["tickers"]:
                        row[ticker] = np.nan
                pc2_loadings_over_time.append(row)

            pc2_df = pd.DataFrame(pc2_loadings_over_time).set_index("Date")
            fig_loadings2 = go.Figure()
            for col in pc2_df.columns:
                fig_loadings2.add_trace(go.Scatter(
                    x=pc2_df.index, y=pc2_df[col],
                    name=col, mode="lines"
                ))
            fig_loadings2.update_layout(
                title=f"Rolling PC2 Loadings ({pca_window}d window)",
                yaxis_title="Loading",
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig_loadings2, use_container_width=True)

        # Variance explained over time
        st.subheader("Rolling Variance Explained")
        var_data = []
        for i, date in enumerate(results["dates"]):
            ve = results["var_explained"][i]
            row = {"Date": date}
            for pc_idx in range(len(ve)):
                row[f"PC{pc_idx+1}"] = ve[pc_idx]
            row["Cumulative"] = sum(ve)
            var_data.append(row)
        var_df = pd.DataFrame(var_data).set_index("Date")

        fig_var = go.Figure()
        pc_cols = [c for c in var_df.columns if c.startswith("PC")]
        for col in pc_cols:
            fig_var.add_trace(go.Scatter(
                x=var_df.index, y=var_df[col],
                name=col, stackgroup="one",
                mode="lines"
            ))
        if variance_threshold is not None:
            fig_var.add_hline(
                y=variance_threshold, line_dash="dash", line_color="red",
                annotation_text=f"{variance_threshold:.0%} threshold"
            )
        fig_var.update_layout(
            title="Rolling Variance Explained by Retained PCs (Stacked)",
            yaxis_title="Variance Explained",
            yaxis_tickformat=".0%",
            hovermode="x unified",
            height=450
        )
        st.plotly_chart(fig_var, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 3: Ticker Weight Decomposition
    # ═══════════════════════════════════════════
    with tab3:
        st.subheader(f"Effective Ticker Weights on {target_ticker}")
        st.caption(
            "Back-transformed weights: how much each basket ticker contributes to "
            "explaining the target through the truncated PCs. Because PCs are truncated, "
            "these differ from naive OLS weights — that's the point."
        )

        weights_data = []
        for i, date in enumerate(results["dates"]):
            row = {"Date": date}
            tw = results["ticker_weights"][i]
            for j, ticker in enumerate(results["tickers"]):
                row[ticker] = tw[j]
            weights_data.append(row)
        weights_df = pd.DataFrame(weights_data).set_index("Date")

        fig_weights = go.Figure()
        for col in weights_df.columns:
            fig_weights.add_trace(go.Scatter(
                x=weights_df.index, y=weights_df[col],
                name=col, mode="lines"
            ))
        fig_weights.add_hline(y=0, line_dash="dot", line_color="grey")
        fig_weights.update_layout(
            title=f"Rolling PCR Ticker Weights on {target_ticker}",
            yaxis_title="Effective Weight",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_weights, use_container_width=True)

        # Current snapshot: PCA vs OLS side by side
        st.subheader("Current Snapshot — PCR vs Naive OLS")
        st.caption(
            "If these are identical, truncation isn't working (too many PCs retained). "
            "Differences show where PCA is filtering noise from the regression."
        )
        latest_pca_w = results["ticker_weights"][-1]
        latest_ols_w = results["ticker_weights_ols"][-1]
        latest_tickers = results["tickers"]

        snap_df = pd.DataFrame({
            "Ticker": latest_tickers,
            "PCR Weight": latest_pca_w,
            "OLS Weight": latest_ols_w,
            "Difference": latest_pca_w - latest_ols_w
        }).sort_values("PCR Weight", ascending=True)

        fig_compare = make_subplots(rows=1, cols=2,
                                    subplot_titles=["PCR (Truncated PCA)", "Naive OLS"],
                                    shared_yaxes=True)
        fig_compare.add_trace(go.Bar(
            x=snap_df["PCR Weight"], y=snap_df["Ticker"],
            orientation="h", name="PCR",
            marker_color=snap_df["PCR Weight"].apply(
                lambda x: "steelblue" if x >= 0 else "indianred"
            )
        ), row=1, col=1)
        fig_compare.add_trace(go.Bar(
            x=snap_df["OLS Weight"], y=snap_df["Ticker"],
            orientation="h", name="OLS",
            marker_color=snap_df["OLS Weight"].apply(
                lambda x: "steelblue" if x >= 0 else "indianred"
            )
        ), row=1, col=2)
        fig_compare.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_compare, use_container_width=True)

        # Weight difference table
        diff_df = snap_df.copy()
        diff_df["PCR Weight"] = diff_df["PCR Weight"].map(lambda x: f"{x:.4f}")
        diff_df["OLS Weight"] = diff_df["OLS Weight"].map(lambda x: f"{x:.4f}")
        diff_df["Difference"] = diff_df["Difference"].map(lambda x: f"{x:+.4f}")
        st.dataframe(diff_df, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════
    # TAB 4: Model Diagnostics
    # ═══════════════════════════════════════════
    with tab4:
        st.subheader("Rolling Model Diagnostics")

        # R² over time: PCR vs OLS
        r2_df = pd.DataFrame({
            "Date": results["dates"],
            "PCR R²": results["r2"],
            "OLS R²": results["r2_ols"]
        }).set_index("Date")

        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Scatter(
            x=r2_df.index, y=r2_df["PCR R²"],
            name="PCR R² (truncated)", mode="lines",
            line=dict(color="steelblue")
        ))
        fig_r2.add_trace(go.Scatter(
            x=r2_df.index, y=r2_df["OLS R²"],
            name="OLS R² (all variables)", mode="lines",
            line=dict(color="grey", dash="dash")
        ))
        fig_r2.update_layout(
            title=f"Rolling R² — PCR vs Naive OLS ({pca_window}d window)",
            yaxis_title="R²",
            yaxis_range=[0, 1],
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_r2, use_container_width=True)

        st.caption(
            "OLS R² is always ≥ PCR R² because OLS uses all degrees of freedom. "
            "The gap shows how much R² comes from noisy higher-order PCs — "
            "that's variance you're intentionally discarding."
        )

        # R² gap (how much the truncation is filtering)
        r2_df["R² Gap"] = r2_df["OLS R²"] - r2_df["PCR R²"]
        fig_gap = go.Figure()
        fig_gap.add_trace(go.Scatter(
            x=r2_df.index, y=r2_df["R² Gap"],
            name="R² Gap (OLS - PCR)", mode="lines",
            fill="tozeroy",
            line=dict(color="darkorange")
        ))
        fig_gap.update_layout(
            title="R² Gap: Variance Attributed to Noisy PCs (Discarded)",
            yaxis_title="R² Gap",
            hovermode="x unified",
            height=300
        )
        st.plotly_chart(fig_gap, use_container_width=True)

        # Number of PCs used over time
        npcs_df = pd.DataFrame({
            "Date": results["dates"],
            "Num PCs": results["n_pcs"]
        }).set_index("Date")

        fig_npcs = go.Figure()
        fig_npcs.add_trace(go.Scatter(
            x=npcs_df.index, y=npcs_df["Num PCs"],
            name="# PCs", mode="lines",
            line=dict(color="darkorange", shape="hv")
        ))
        fig_npcs.add_hline(
            y=len(available_basket), line_dash="dash", line_color="red",
            annotation_text=f"Total tickers = {len(available_basket)} (OLS equivalent)"
        )
        fig_npcs.update_layout(
            title="Number of PCs Retained",
            yaxis_title="Number of PCs",
            yaxis_dtick=1,
            hovermode="x unified",
            height=350
        )
        st.plotly_chart(fig_npcs, use_container_width=True)

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg PCR R²", f"{np.mean(results['r2']):.3f}")
        col2.metric("Current PCR R²", f"{results['r2'][-1]:.3f}")
        col3.metric("Avg # PCs", f"{np.mean(results['n_pcs']):.1f}")
        col4.metric("Avg R² Gap", f"{np.mean(r2_df['R² Gap']):.3f}")

    # ═══════════════════════════════════════════
    # TAB 5: Summary Table
    # ═══════════════════════════════════════════
    with tab5:
        st.subheader("Configuration Summary")

        trunc_desc = (
            f"Fixed: {n_fixed_pcs} PCs" if n_fixed_pcs is not None
            else f"Variance threshold: {variance_threshold:.0%}"
        )

        config_data = {
            "Parameter": [
                "Basket Tickers (all used in PCA)",
                "Target",
                "Vol Standardization Window",
                "PCA Rolling Window",
                "PC Truncation",
                "Data Lookback",
            ],
            "Value": [
                ", ".join(available_basket),
                target_ticker,
                f"{vol_window} days",
                f"{pca_window} days",
                trunc_desc,
                f"{lookback_years} years",
            ]
        }
        st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)

        st.subheader("Latest Window Detail")
        st.markdown(f"**Date:** {results['dates'][-1].strftime('%Y-%m-%d')}")
        st.markdown(f"**PCR R²:** {results['r2'][-1]:.4f}")
        st.markdown(f"**OLS R²:** {results['r2_ols'][-1]:.4f}")
        st.markdown(f"**PCs used:** {results['n_pcs'][-1]} / {len(available_basket)}")

        # Variance explained breakdown
        ve_latest = results["var_explained"][-1]
        ve_rows = []
        for k, v in enumerate(ve_latest):
            ve_rows.append({
                "PC": f"PC{k+1}",
                "Var Explained": f"{v:.2%}",
                "Cumulative": f"{sum(ve_latest[:k+1]):.2%}"
            })
        st.dataframe(pd.DataFrame(ve_rows), use_container_width=True, hide_index=True)

        # Ticker weights table: PCR vs OLS
        st.subheader("Latest Ticker Weights — PCR vs OLS")
        tw_pca = results["ticker_weights"][-1]
        tw_ols = results["ticker_weights_ols"][-1]
        tw_df = pd.DataFrame({
            "Ticker": results["tickers"],
            "PCR Weight": tw_pca,
            "OLS Weight": tw_ols,
            "Abs PCR": np.abs(tw_pca)
        }).sort_values("Abs PCR", ascending=False)
        tw_df["PCR Weight"] = tw_df["PCR Weight"].map(lambda x: f"{x:.4f}")
        tw_df["OLS Weight"] = tw_df["OLS Weight"].map(lambda x: f"{x:.4f}")
        tw_df["Abs PCR"] = tw_df["Abs PCR"].map(lambda x: f"{x:.4f}")
        st.dataframe(tw_df, use_container_width=True, hide_index=True)

else:
    st.info("👈 Configure your parameters in the sidebar and click **Run Analysis** to start.")
    st.markdown("""
    ### How This Tool Works

    **Pipeline:**
    1. **Download data** from Yahoo Finance for your basket + target tickers
    2. **Compute standardized log returns** (normalized by rolling volatility)
    3. **PCA on the full basket** — extracts principal components from all tickers
    4. **Truncate** — keep only the top K PCs, discard the rest as noise. This is the key step — without truncation, PCA regression is algebraically identical to OLS
    5. **Regress target on truncated PCs**, then back-transform to effective ticker weights
    6. **Compare to naive OLS** to verify the truncation is actually doing something

    **Tabs:**
    - **Dendrogram** — diagnostic view of correlation structure (does not affect PCA)
    - **Rolling PCA Loadings** — how PC compositions change over time
    - **Ticker Weight Decomposition** — the key output, with PCR vs OLS comparison
    - **Model Diagnostics** — R² comparison, number of PCs, R² gap analysis
    - **Summary Table** — configuration and latest snapshot
    """)
