import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from io import BytesIO
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

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
    help="These are the tickers that go into the dendrogram and PCA."
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

variance_threshold = st.sidebar.slider(
    "Variance Explained Threshold",
    min_value=0.70, max_value=0.99, value=0.90, step=0.01,
    format="%.0f%%",
    help="Fixed threshold: include PCs until this % of variance is explained."
)

lookback_years = st.sidebar.slider(
    "Data Lookback (years)",
    min_value=1, max_value=10, value=3, step=1,
    help="How many years of historical data to download."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dendrogram Settings")

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2, max_value=min(10, len(basket_tickers)),
    value=min(5, len(basket_tickers)),
    step=1,
    help="How many clusters to cut the dendrogram into."
)

representative_method = st.sidebar.radio(
    "Cluster Representative Selection",
    options=["Highest Avg Correlation", "Most Liquid (by volume)", "Both (weighted)"],
    index=0,
    help="How to pick the representative ticker from each cluster."
)


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers, target, years):
    all_tickers = list(set(tickers + [target]))
    end = datetime.today()
    start = end - timedelta(days=years * 365 + vol_window + 60)  # extra buffer
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Handle both MultiIndex and single-ticker case
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = all_tickers

    prices = prices.dropna(how="all")
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def load_volume(tickers, years):
    end = datetime.today()
    start = end - timedelta(days=years * 365 + 60)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        vol = data["Volume"]
    else:
        vol = data[["Volume"]]
        vol.columns = tickers
    return vol


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
    # Make symmetric and handle floating point
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    condensed = squareform(dist_matrix.values, checks=False)
    Z = linkage(condensed, method="ward")
    return Z


def select_representatives(corr_matrix, clusters, tickers, volume_data, method):
    """Select one representative per cluster."""
    representatives = []
    cluster_map = {}

    unique_clusters = sorted(set(clusters))
    for c in unique_clusters:
        members = [tickers[i] for i in range(len(tickers)) if clusters[i] == c]
        if len(members) == 1:
            representatives.append(members[0])
            cluster_map[members[0]] = members
            continue

        # Avg correlation within cluster
        sub_corr = corr_matrix.loc[members, members]
        avg_corr = sub_corr.mean(axis=1)

        if method == "Highest Avg Correlation":
            rep = avg_corr.idxmax()
        elif method == "Most Liquid (by volume)":
            avg_vol = volume_data[members].mean()
            rep = avg_vol.idxmax()
        else:  # Both (weighted)
            # Normalize both metrics to [0,1] and average
            corr_score = (avg_corr - avg_corr.min()) / (avg_corr.max() - avg_corr.min() + 1e-10)
            avg_vol = volume_data[members].mean()
            vol_score = (avg_vol - avg_vol.min()) / (avg_vol.max() - avg_vol.min() + 1e-10)
            combined = 0.5 * corr_score + 0.5 * vol_score
            rep = combined.idxmax()

        representatives.append(rep)
        cluster_map[rep] = members

    return representatives, cluster_map


def rolling_pca_regression(std_returns_basket, std_returns_target, pca_win, var_thresh):
    """
    Run rolling PCA on basket, regress target on PCs, back out ticker weights.
    Returns a dict of rolling results.
    """
    n_obs = len(std_returns_basket)
    tickers = std_returns_basket.columns.tolist()
    n_tickers = len(tickers)

    # Storage
    dates = []
    all_loadings = []      # n_tickers x n_pcs per window
    all_var_explained = []  # variance explained per PC
    all_n_pcs = []          # number of PCs used
    all_ticker_weights = [] # effective ticker weights on target
    all_r2 = []             # regression R²
    all_pc_betas = []       # regression betas on PCs

    min_start = max(0, pca_win)  # need at least pca_win observations

    for i in range(min_start, n_obs):
        window_start = i - pca_win
        X_window = std_returns_basket.iloc[window_start:i].values
        y_window = std_returns_target.iloc[window_start:i].values

        # Check for NaN
        valid_mask = ~(np.isnan(X_window).any(axis=1) | np.isnan(y_window))
        X_clean = X_window[valid_mask]
        y_clean = y_window[valid_mask]

        if len(X_clean) < n_tickers + 2:
            continue

        # PCA
        pca = PCA()
        try:
            pc_scores = pca.fit_transform(X_clean)
        except Exception:
            continue

        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_pcs = int(np.searchsorted(cum_var, var_thresh) + 1)
        n_pcs = min(n_pcs, len(cum_var))

        # Regression: target ~ PC1 + PC2 + ...
        X_pcs = pc_scores[:, :n_pcs]
        reg = LinearRegression().fit(X_pcs, y_clean)
        r2 = reg.score(X_pcs, y_clean)

        # Back out ticker weights
        # loadings: (n_tickers, n_pcs) = pca.components_.T[:, :n_pcs]
        loadings = pca.components_[:n_pcs, :].T  # shape: (n_tickers, n_pcs)
        pc_betas = reg.coef_  # shape: (n_pcs,)
        ticker_weights = loadings @ pc_betas  # shape: (n_tickers,)

        date = std_returns_basket.index[i]
        dates.append(date)
        all_loadings.append(pca.components_[:n_pcs, :])  # (n_pcs, n_tickers)
        all_var_explained.append(pca.explained_variance_ratio_[:n_pcs])
        all_n_pcs.append(n_pcs)
        all_ticker_weights.append(ticker_weights)
        all_r2.append(r2)
        all_pc_betas.append(pc_betas)

    results = {
        "dates": dates,
        "loadings": all_loadings,
        "var_explained": all_var_explained,
        "n_pcs": all_n_pcs,
        "ticker_weights": all_ticker_weights,
        "r2": all_r2,
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
        prices = load_data(basket_tickers, target_ticker, lookback_years)
        volume = load_volume(basket_tickers, lookback_years)

    # Check which tickers we actually got
    available_basket = [t for t in basket_tickers if t in prices.columns]
    missing = [t for t in basket_tickers if t not in prices.columns]
    if target_ticker not in prices.columns:
        st.error(f"Target ticker '{target_ticker}' not found in downloaded data.")
        st.stop()
    if missing:
        st.warning(f"Missing tickers (not found): {', '.join(missing)}")
    if len(available_basket) < 3:
        st.error("Need at least 3 basket tickers with data.")
        st.stop()

    with st.spinner("Computing returns and standardizing..."):
        log_ret = compute_log_returns(prices)
        std_ret = standardize_returns(log_ret, vol_window)
        # Separate basket and target
        std_basket = std_ret[[t for t in available_basket if t in std_ret.columns]].dropna(axis=1, how="all")
        std_target = std_ret[target_ticker].dropna()
        # Align
        common_idx = std_basket.dropna().index.intersection(std_target.dropna().index)
        std_basket = std_basket.loc[common_idx]
        std_target = std_target.loc[common_idx]

    available_basket = std_basket.columns.tolist()

    # ─── Tab Layout ───
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dendrogram & Clusters",
        "🔄 Rolling PCA Loadings",
        "🎯 Ticker Weight Decomposition",
        "📈 Model Diagnostics",
        "📋 Summary Table"
    ])

    # ═══════════════════════════════════════════
    # TAB 1: Dendrogram
    # ═══════════════════════════════════════════
    with tab1:
        st.subheader("Dendrogram & Cluster Analysis")

        # Use recent data for dendrogram (last vol_window days)
        recent_ret = log_ret[available_basket].iloc[-vol_window:]
        corr_matrix = recent_ret.corr()

        Z = run_dendrogram_analysis(corr_matrix, available_basket)

        # Plot dendrogram with matplotlib (scipy requires it)
        fig_dendro, ax = plt.subplots(figsize=(12, 5))
        dendro_result = dendrogram(
            Z,
            labels=available_basket,
            ax=ax,
            color_threshold=Z[-(n_clusters - 1), 2] if n_clusters <= len(Z) else 0,
            above_threshold_color="grey",
            leaf_rotation=45,
            leaf_font_size=10
        )
        ax.set_title(f"Dendrogram (Ward Linkage, {vol_window}d Correlation Distance)")
        ax.set_ylabel("Distance")
        plt.tight_layout()
        st.pyplot(fig_dendro)

        # Cluster assignments
        clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

        # Select representatives
        reps, cluster_map = select_representatives(
            corr_matrix, clusters, available_basket, volume[available_basket].iloc[-vol_window:],
            representative_method
        )

        st.subheader("Cluster Assignments & Representatives")
        cluster_df_rows = []
        for i, rep in enumerate(reps):
            members = cluster_map[rep]
            cluster_df_rows.append({
                "Cluster": i + 1,
                "Representative": rep,
                "Members": ", ".join(members),
                "Avg Intra-Cluster Corr": f"{corr_matrix.loc[members, members].mean().mean():.3f}"
            })
        cluster_df = pd.DataFrame(cluster_df_rows)
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

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

    # ═══════════════════════════════════════════
    # TAB 2-5: Run PCA/PCR on representatives
    # ═══════════════════════════════════════════
    with st.spinner("Running rolling PCA regression..."):
        # Use representatives for PCA
        pca_tickers = [t for t in reps if t in std_basket.columns]
        if len(pca_tickers) < 2:
            st.error("Need at least 2 representative tickers for PCA.")
            st.stop()

        std_basket_pca = std_basket[pca_tickers]
        results = rolling_pca_regression(
            std_basket_pca, std_target, pca_window, variance_threshold
        )

    if len(results["dates"]) == 0:
        st.error("No valid rolling windows. Try reducing PCA window or adding more data.")
        st.stop()

    # ═══════════════════════════════════════════
    # TAB 2: Rolling PCA Loadings
    # ═══════════════════════════════════════════
    with tab2:
        st.subheader("Rolling PC1 Loadings Over Time")
        st.caption("Shows how each ticker's contribution to PC1 evolves.")

        # Extract PC1 loadings over time
        pc1_loadings_over_time = []
        for i, date in enumerate(results["dates"]):
            loadings_i = results["loadings"][i]  # (n_pcs, n_tickers)
            row = {"Date": date}
            for j, ticker in enumerate(results["tickers"]):
                row[ticker] = loadings_i[0, j]  # PC1
            pc1_loadings_over_time.append(row)

        pc1_df = pd.DataFrame(pc1_loadings_over_time).set_index("Date")

        fig_loadings = go.Figure()
        for col in pc1_df.columns:
            fig_loadings.add_trace(go.Scatter(
                x=pc1_df.index, y=pc1_df[col],
                name=col, mode="lines"
            ))
        fig_loadings.update_layout(
            title=f"Rolling PC1 Loadings ({pca_window}d window)",
            yaxis_title="Loading",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_loadings, use_container_width=True)

        # PC2 loadings if typically present
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
        # Stacked area for individual PCs
        pc_cols = [c for c in var_df.columns if c.startswith("PC")]
        for col in pc_cols:
            fig_var.add_trace(go.Scatter(
                x=var_df.index, y=var_df[col],
                name=col, stackgroup="one",
                mode="lines"
            ))
        fig_var.add_hline(
            y=variance_threshold, line_dash="dash", line_color="red",
            annotation_text=f"{variance_threshold:.0%} threshold"
        )
        fig_var.update_layout(
            title="Rolling Variance Explained by PC (Stacked)",
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
            "These are the backed-out weights: how much each basket ticker "
            "contributes to explaining the target, through the PCs."
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
            title=f"Rolling Ticker Weights on {target_ticker}",
            yaxis_title="Effective Weight",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig_weights, use_container_width=True)

        # Current snapshot
        st.subheader("Current Snapshot (Most Recent Window)")
        latest_weights = results["ticker_weights"][-1]
        latest_tickers = results["tickers"]
        snap_df = pd.DataFrame({
            "Ticker": latest_tickers,
            "Weight": latest_weights
        }).sort_values("Weight", ascending=True)

        fig_bar = px.bar(
            snap_df, x="Weight", y="Ticker", orientation="h",
            title=f"Current Ticker Weights on {target_ticker}",
            color="Weight",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 4: Model Diagnostics
    # ═══════════════════════════════════════════
    with tab4:
        st.subheader("Rolling Model Diagnostics")

        # R² over time
        r2_df = pd.DataFrame({
            "Date": results["dates"],
            "R²": results["r2"]
        }).set_index("Date")

        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Scatter(
            x=r2_df.index, y=r2_df["R²"],
            name="R²", mode="lines",
            line=dict(color="steelblue")
        ))
        fig_r2.update_layout(
            title=f"Rolling R² ({pca_window}d window) — {target_ticker} ~ PCs",
            yaxis_title="R²",
            yaxis_range=[0, 1],
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_r2, use_container_width=True)

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
        fig_npcs.update_layout(
            title=f"Number of PCs Used (to reach {variance_threshold:.0%} threshold)",
            yaxis_title="Number of PCs",
            yaxis_dtick=1,
            hovermode="x unified",
            height=350
        )
        st.plotly_chart(fig_npcs, use_container_width=True)

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg R²", f"{np.mean(results['r2']):.3f}")
        col2.metric("Current R²", f"{results['r2'][-1]:.3f}")
        col3.metric("Avg # PCs", f"{np.mean(results['n_pcs']):.1f}")
        col4.metric("Current # PCs", f"{results['n_pcs'][-1]}")

    # ═══════════════════════════════════════════
    # TAB 5: Summary Table
    # ═══════════════════════════════════════════
    with tab5:
        st.subheader("Configuration Summary")
        config_data = {
            "Parameter": [
                "Basket Tickers (input)",
                "PCA Tickers (representatives)",
                "Target",
                "Vol Standardization Window",
                "PCA Rolling Window",
                "Variance Threshold",
                "Data Lookback",
                "Dendrogram Clusters",
                "Representative Method"
            ],
            "Value": [
                ", ".join(available_basket),
                ", ".join(pca_tickers),
                target_ticker,
                f"{vol_window} days",
                f"{pca_window} days",
                f"{variance_threshold:.0%}",
                f"{lookback_years} years",
                str(n_clusters),
                representative_method
            ]
        }
        st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)

        st.subheader("Latest Window Detail")
        st.markdown(f"**Date:** {results['dates'][-1].strftime('%Y-%m-%d')}")
        st.markdown(f"**R²:** {results['r2'][-1]:.4f}")
        st.markdown(f"**PCs used:** {results['n_pcs'][-1]}")

        # Variance explained breakdown
        ve_latest = results["var_explained"][-1]
        ve_rows = []
        for k, v in enumerate(ve_latest):
            ve_rows.append({"PC": f"PC{k+1}", "Var Explained": f"{v:.2%}", "Cumulative": f"{sum(ve_latest[:k+1]):.2%}"})
        st.dataframe(pd.DataFrame(ve_rows), use_container_width=True, hide_index=True)

        # Ticker weights table
        st.subheader("Latest Ticker Weights")
        tw_latest = results["ticker_weights"][-1]
        tw_df = pd.DataFrame({
            "Ticker": results["tickers"],
            "Weight": tw_latest,
            "Abs Weight": np.abs(tw_latest)
        }).sort_values("Abs Weight", ascending=False)
        tw_df["Weight"] = tw_df["Weight"].map(lambda x: f"{x:.4f}")
        tw_df["Abs Weight"] = tw_df["Abs Weight"].map(lambda x: f"{x:.4f}")
        st.dataframe(tw_df, use_container_width=True, hide_index=True)

else:
    st.info("👈 Configure your parameters in the sidebar and click **Run Analysis** to start.")
    st.markdown("""
    ### How This Tool Works

    **Pipeline:**
    1. **Download data** from Yahoo Finance for your basket + target tickers
    2. **Compute standardized log returns** (normalized by rolling volatility)
    3. **Dendrogram clustering** groups correlated tickers; one representative per cluster is selected
    4. **Rolling PCA** on representative tickers extracts principal components
    5. **Rolling regression** of target on PCs, then **back out effective ticker weights**

    **Tabs:**
    - **Dendrogram & Clusters** — visualize correlation structure and cluster assignments
    - **Rolling PCA Loadings** — see how PC compositions change over time
    - **Ticker Weight Decomposition** — the key output: how much each ticker drives your target
    - **Model Diagnostics** — R², number of PCs, model stability over time
    - **Summary Table** — current configuration and latest snapshot
    """)
