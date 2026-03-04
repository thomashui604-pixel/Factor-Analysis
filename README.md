# PCR Rolling Analysis Dashboard

A Streamlit-based tool for running Principal Component Regression (PCR) analysis on financial instruments with rolling windows.

## Pipeline

1. **Data Download** — Pulls price & volume data from Yahoo Finance
2. **Standardized Log Returns** — Daily log returns normalized by rolling volatility (252d default)
3. **Dendrogram Clustering** — Groups correlated tickers using Ward linkage on correlation distance
4. **Representative Selection** — Picks one ticker per cluster (by avg correlation, liquidity, or both)
5. **Rolling PCA** — Extracts principal components on a rolling window (30d default)
6. **Rolling PCR** — Regresses target on selected PCs (90% variance threshold)
7. **Weight Decomposition** — Backs out effective ticker weights on the target through PC loadings × regression betas

## Setup

```bash
pip install -r requirements.txt
streamlit run pcr_dashboard.py
```

## Usage

1. Enter your basket tickers (comma-separated) in the sidebar
2. Enter your target ticker
3. Adjust windows, thresholds, and clustering parameters
4. Click **Run Analysis**
5. Explore the five tabs:
   - **Dendrogram & Clusters** — correlation structure and cluster assignments
   - **Rolling PCA Loadings** — how PC compositions rotate over time
   - **Ticker Weight Decomposition** — effective impact of each ticker on the target
   - **Model Diagnostics** — R², number of PCs, stability metrics
   - **Summary Table** — configuration and latest snapshot

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Vol Window | 252d | Std deviation window for return normalization |
| PCA Window | 30d | Rolling window for PCA computation |
| Variance Threshold | 90% | Include PCs until this % of variance is explained |
| Clusters | 5 | Number of dendrogram clusters |
| Representative Method | Highest Avg Correlation | How to pick cluster representatives |

## Notes

- Ticker count going into PCA should stay at 5-10 for stability with a 30d window
- VIX requires `^VIX` as the Yahoo Finance symbol — the app uses the ticker as-is, so enter `^VIX` if needed
- The dendrogram uses the most recent `vol_window` days of data for clustering
- All rolling computations are temporally aligned (same window for PCA and regression)
