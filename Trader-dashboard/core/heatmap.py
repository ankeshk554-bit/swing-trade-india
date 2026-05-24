"""
Market Heat Map - Visual representation of Nifty stock performance.
Uses Plotly treemap and grid heat map for at-a-glance market view.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from core.utils import load_data
from core.indicators import compute_indicators


def build_heatmap_data(symbols, max_stocks=50):
    """
    Build performance data for heat map visualization.

    Parameters
    ----------
    symbols : list
        Stock symbols (e.g., NIFTY50 list)
    max_stocks : int
        Maximum stocks to include

    Returns
    -------
    pd.DataFrame
        Columns: Symbol, Close, Change%, RSI, RVOL, Volume, Sector, Label
    """
    rows = []
    for symbol in symbols[:max_stocks]:
        try:
            df = load_data(symbol, interval="1d", period="5d")
            if df is None or df.empty or len(df) < 2:
                continue
            df = compute_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            close = latest.get("Close", 0)
            prev_close = prev.get("Close", close)
            change_pct = ((close - prev_close) / prev_close * 100) if prev_close > 0 else 0

            rows.append({
                "Symbol": symbol.replace(".NS", ""),
                "FullSymbol": symbol,
                "Close": round(close, 2),
                "Change%": round(change_pct, 2),
                "RSI": round(latest.get("RSI", 50), 1),
                "RVOL": round(latest.get("RVOL", 1.0), 2),
                "Volume": int(latest.get("Volume", 0)),
                "Label": f"{symbol.replace('.NS','')}<br>{change_pct:+.2f}%",
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def plot_treemap_heatmap(df):
    """
    Create a Plotly treemap heat map of stocks by performance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from build_heatmap_data()

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return None

    # Color: green for gainers, red for losers, intensity by magnitude
    colors = []
    for val in df["Change%"]:
        if val > 0:
            intensity = min(1.0, val / 5.0)
            r = int(30 * (1 - intensity))
            g = int(200 * intensity + 50 * (1 - intensity))
            colors.append(f"rgb({r},{g},{30})")
        else:
            intensity = min(1.0, abs(val) / 5.0)
            r = int(200 * intensity + 50 * (1 - intensity))
            g = int(30 * (1 - intensity))
            colors.append(f"rgb({r},{g},{30})")

    fig = go.Figure()

    # Use scatter for a grid-like treemap
    # Actually, let's use a proper treemap
    n = len(df)
    if n == 0:
        return None

    # Group by approximate sector (first part of symbol as proxy)
    df["Sector"] = df["Symbol"].apply(lambda x: _guess_sector(x))

    fig = px.treemap(
        df,
        path=["Sector", "Symbol"],
        values=df["Volume"].abs(),
        color="Change%",
        color_continuous_scale=[
            (0.0, "rgb(220, 50, 50)"),
            (0.3, "rgb(180, 30, 30)"),
            (0.48, "rgb(50, 50, 50)"),
            (0.5, "rgb(30, 30, 30)"),
            (0.52, "rgb(50, 50, 50)"),
            (0.7, "rgb(30, 150, 30)"),
            (1.0, "rgb(30, 220, 30)"),
        ],
        range_color=[-5, 5],
        hover_data={"Close": True, "Change%": ":.2f", "RSI": True, "RVOL": True},
        labels={"Change%": "Change %", "Symbol": "Stock"},
        title="",
    )

    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[1]:.2f}%",
        textfont={"size": 11, "color": "white"},
        hovertemplate="<b>%{label}</b><br>Change: %{customdata[1]:+.2f}%<br>Close: ₹%{customdata[0]:,.2f}<br>RSI: %{customdata[2]:.1f}<br>RVOL: %{customdata[3]:.2f}x<extra></extra>",
    )

    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
        coloraxis_showscale=False,
    )

    return fig


def plot_grid_heatmap(df):
    """
    Create a grid-style heat map (like Finviz) using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from build_heatmap_data()

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df.empty:
        return None

    # Sort by Change% descending
    df = df.sort_values("Change%", ascending=False).reset_index(drop=True)

    # Create a grid: ~8 columns × N rows
    n_cols = 8
    n_rows = (len(df) + n_cols - 1) // n_cols

    # Build grid data
    grid_labels = []
    grid_changes = []
    grid_colors = []
    hover_texts = []

    for i in range(n_rows * n_cols):
        if i < len(df):
            row = df.iloc[i]
            grid_labels.append(row["Symbol"])
            grid_changes.append(row["Change%"])
            hover_texts.append(
                f"{row['Symbol']}<br>₹{row['Close']:,.2f}<br>{row['Change%']:+.2f}%<br>RSI: {row['RSI']}"
            )
            val = row["Change%"]
            if val > 0:
                intensity = min(1.0, val / 4.0)
                grid_colors.append(f"rgba(0, {int(180*intensity+40)}, 0, {0.5+0.5*intensity})")
            else:
                intensity = min(1.0, abs(val) / 4.0)
                grid_colors.append(f"rgba({int(180*intensity+40)}, 0, 0, {0.5+0.5*intensity})")
        else:
            grid_labels.append("")
            grid_changes.append(0)
            grid_colors.append("rgba(0,0,0,0)")
            hover_texts.append("")

    # Reshape to grid
    grid_data = np.array(grid_changes).reshape(n_rows, n_cols)
    grid_labels_arr = np.array(grid_labels).reshape(n_rows, n_cols)
    grid_colors_arr = np.array(grid_colors).reshape(n_rows, n_cols)
    hover_arr = np.array(hover_texts).reshape(n_rows, n_cols)

    fig = go.Figure()

    for r in range(n_rows):
        for c in range(n_cols):
            label = grid_labels_arr[r, c]
            if label:
                fig.add_annotation(
                    x=c,
                    y=n_rows - 1 - r,
                    text=f"<b>{label}</b><br>{grid_data[r,c]:+.2f}%",
                    showarrow=False,
                    font=dict(size=10, color="white"),
                    bgcolor=grid_colors_arr[r, c],
                    borderpad=4,
                    width=110,
                    height=55,
                )

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, n_cols - 0.5]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, n_rows - 0.5]),
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(400, n_rows * 60),
        hovermode="closest",
    )

    return fig


def _guess_sector(symbol):
    """Rough sector assignment based on stock name."""
    sym = symbol.lower()
    bank_stocks = ["hdfcbank", "icicibank", "sbIn", "axisbank", "kotakbank",
                    "indusindbk", "bandhanbnk", "federalbnk", "pnb", "bankbaroda"]
    it_stocks = ["tcs", "infosys", "wipro", "hcltech", "techm", "ltim",
                 "mindtree", "persistent", "cyient", "coforge", "mphasis"]
    auto_stocks = ["maruti", "tatamotors", "m&m", "bajajauto", "eichermot",
                   "hero moto", "ashokley", "tvsmotor"]
    pharma_stocks = ["sunpharma", "drreddy", "cipla", "divislab", "biocon",
                     "lupin", "alatent", "glenmark", "torrentpharma"]
    metal_stocks = ["tatasteel", "jswsteel", "hindalco", "vedanta", "coalindia",
                    "nationalum", "hdfcmetal", "apollometal"]
    energy_stocks = ["reliance", "ongc", "ntpc", "powergrid", "adani green",
                     "adani power", "tata power", "cairn", "oil india", "ioc",
                     "bpcl", "hpcl", "gail"]

    for s in bank_stocks:
        if s in sym:
            return "Banking"
    for s in it_stocks:
        if s in sym:
            return "IT"
    for s in auto_stocks:
        if s in sym:
            return "Auto"
    for s in pharma_stocks:
        if s in sym:
            return "Pharma"
    for s in metal_stocks:
        if s in sym:
            return "Metal"
    for s in energy_stocks:
        if s in sym:
            return "Energy"

    return "Others"
