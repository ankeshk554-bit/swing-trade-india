import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from core.indicators import detect_vcp, compute_indicators
from core.volume_profile import volume_profile, vwap_with_bands


def plot_chart(df, symbol, trades_df=None, show_vwap=True, show_bb=True, show_macd=True):
    """
    Professional trading chart with multiple panels.

    Features:
      - Candlestick + EMA50/EMA200 overlay
      - Volume bars (colored green/red)
      - Bollinger Bands (optional)
      - Anchored VWAP (optional)
      - MACD subplot (optional)
      - VCP pivot markers
      - Backtest trade markers
    """
    # Ensure all indicators computed
    if "EMA50" not in df.columns:
        df = compute_indicators(df)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.15, 0.15],
        subplot_titles=(f"{symbol} — Price", "Volume", "MACD")
    )

    # ===== ROW 1: Candlesticks + Overlays =====
    # Color candles based on close vs open
    colors = ["#00ff9f" if c >= o else "#ff4d6d" for c, o in zip(df["Close"], df["Open"])]

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#00ff9f",
        decreasing_line_color="#ff4d6d",
        showlegend=True
    ), row=1, col=1)

    # EMA50
    if "EMA50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA50"],
            mode="lines",
            line=dict(color="#f5c542", width=1.5, dash="dash"),
            name="EMA50"
        ), row=1, col=1)

    # EMA200
    if "EMA200" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA200"],
            mode="lines",
            line=dict(color="#ff8c42", width=2),
            name="EMA200"
        ), row=1, col=1)

    # Bollinger Bands
    if show_bb and "BB_UPPER" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_UPPER"],
            mode="lines",
            line=dict(color="rgba(100, 149, 237, 0.3)", width=1),
            name="BB Upper",
            showlegend=True
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_LOWER"],
            mode="lines",
            line=dict(color="rgba(100, 149, 237, 0.3)", width=1),
            fill="tonexty",
            fillcolor="rgba(100, 149, 237, 0.05)",
            name="BB Lower",
            showlegend=True
        ), row=1, col=1)

    # Supertrend overlay
    if "SUPERTREND" in df.columns:
        supertrend_colors = ["#00ff9f" if d == 1 else "#ff4d6d"
                             for d in df["SUPERTREND_DIR"]]
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SUPERTREND"],
            mode="markers",
            marker=dict(size=3, color=supertrend_colors),
            name="Supertrend"
        ), row=1, col=1)

    # VCP Pivot Marker
    vcp = detect_vcp(df)
    if vcp["VCP_Flag"]:
        pivot_idx = df.index[-1]
        pivot_price = vcp["Pivot"]
        marker_color = "#00ff9f" if vcp["VolumeDryUp"] else "#ff4d6d"
        marker_text = f"VCP Stage {vcp['Stage']}"

        fig.add_trace(go.Scatter(
            x=[pivot_idx],
            y=[pivot_price],
            mode="markers+text",
            marker=dict(symbol="diamond", size=14, color=marker_color,
                        line=dict(color="white", width=1)),
            text=[marker_text],
            textposition="top center",
            textfont=dict(color="white", size=10),
            name="VCP Pivot"
        ), row=1, col=1)

    # ===== ROW 2: Volume =====
    vol_colors = ["#00ff9f" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ff4d6d"
                  for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume",
        marker_color=vol_colors,
        marker_line_width=0,
        opacity=0.8
    ), row=2, col=1)

    # Volume MA overlay
    if "VOL_MA20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VOL_MA20"],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dot"),
            name="Vol MA20"
        ), row=2, col=1)

    # ===== ROW 3: MACD =====
    if show_macd and "MACD" in df.columns:
        # MACD line
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"],
            mode="lines",
            line=dict(color="#00bfff", width=1.5),
            name="MACD"
        ), row=3, col=1)

        # Signal line
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_SIGNAL"],
            mode="lines",
            line=dict(color="#ff8c42", width=1.5),
            name="Signal"
        ), row=3, col=1)

        # Histogram
        hist_colors = ["#00ff9f" if v >= 0 else "#ff4d6d" for v in df["MACD_HIST"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_HIST"],
            name="Histogram",
            marker_color=hist_colors,
            marker_line_width=0
        ), row=3, col=1)

    # ===== Backtest Trade Markers =====
    if trades_df is not None and not trades_df.empty:
        fig.add_trace(go.Scatter(
            x=trades_df["EntryDate"],
            y=trades_df["EntryPrice"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#00ff9f",
                        line=dict(color="white", width=1)),
            name="Entries"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=trades_df["ExitDate"],
            y=trades_df["ExitPrice"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#ff4d6d",
                        line=dict(color="white", width=1)),
            text=trades_df["Result"],
            textposition="top center",
            name="Exits"
        ), row=1, col=1)

    # ===== Layout =====
    fig.update_layout(
        height=800,
        template="plotly_dark",
        paper_bgcolor="#050816",
        plot_bgcolor="#0a0e1a",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Update axes
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="#1a1f2e")
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="#1a1f2e")
    fig.update_yaxes(title_text="MACD", row=3, col=1, gridcolor="#1a1f2e")
    fig.update_xaxes(gridcolor="#1a1f2e")

    return fig


def plot_volume_profile(df, symbol):
    """
    Create a volume profile histogram chart (Volume at Price).

    Returns a Plotly figure with horizontal volume bars.
    """
    vp = volume_profile(df, num_bins=30, lookback=min(60, len(df)))
    if vp is None:
        return None

    fig = go.Figure()

    # Horizontal volume bars
    fig.add_trace(go.Bar(
        x=vp["volumes"],
        y=vp["price_bins"],
        orientation="h",
        marker_color="#00bfff",
        marker_line_width=0,
        opacity=0.6,
        name="Volume",
        hovertemplate="Price: ₹%{y}<br>Volume: %{x:,.0f}<extra></extra>"
    ))

    # POC line
    fig.add_hline(y=vp["poc"], line_color="#f5c542", line_width=2,
                  annotation_text=f"POC ₹{vp['poc']}", annotation_position="top left")

    # Value Area
    fig.add_hrect(y0=vp["val"], y1=vp["vah"],
                  fillcolor="green", opacity=0.05, line_width=0,
                  annotation_text=f"VA ₹{vp['val']}–₹{vp['vah']}",
                  annotation_position="bottom right")

    fig.update_layout(
        title=f"{symbol} — Volume Profile",
        height=600,
        template="plotly_dark",
        paper_bgcolor="#050816",
        plot_bgcolor="#0a0e1a",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor="#1a1f2e", title="Volume"),
        yaxis=dict(gridcolor="#1a1f2e", title="Price"),
        hovermode="y unified",
        bargap=0.1
    )

    return fig


def plot_comparison_chart(
    dfs: dict,
    symbols: list,
    indicator: str = "Close",
    normalize: bool = True
):
    """
    Compare multiple stocks on a single chart for relative performance.

    Args:
        dfs: Dict of symbol -> DataFrame
        symbols: List of symbols to include
        indicator: Column to plot ('Close', 'RSI', 'RVOL', etc.)
        normalize: If True, normalize to 100 for relative comparison

    Returns Plotly figure.
    """
    fig = go.Figure()

    colors = ["#00ff9f", "#f5c542", "#00bfff", "#ff8c42", "#ff4d6d", "#aa66ff"]

    for i, sym in enumerate(symbols):
        df = dfs.get(sym)
        if df is None or df.empty:
            continue

        if indicator not in df.columns:
            continue

        values = df[indicator].values
        if normalize and len(values) > 0:
            base = values[0]
            if base > 0:
                values = (values / base) * 100

        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df.index,
            y=values,
            mode="lines",
            name=sym.replace(".NS", ""),
            line=dict(color=color, width=2)
        ))

    y_title = f"{indicator} (Normalized)" if normalize else indicator

    fig.update_layout(
        title=f"{'Relative Performance' if normalize else indicator} Comparison",
        height=500,
        template="plotly_dark",
        paper_bgcolor="#050816",
        plot_bgcolor="#0a0e1a",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor="#1a1f2e"),
        yaxis=dict(gridcolor="#1a1f2e", title=y_title),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_chart_with_drawings(df, symbol, show_bb=True, show_macd=True,
                              show_vwap=False, show_supertrend=True,
                              show_vp=False, shapes=None):
    """
    Enhanced chart with optional drawing annotations (shapes).

    Args:
        df: OHLCV DataFrame with indicators
        symbol: Ticker symbol
        show_bb: Show Bollinger Bands
        show_macd: Show MACD subplot
        show_vwap: Show VWAP overlay
        show_supertrend: Show Supertrend markers
        show_vp: Show Volume Profile as subplot (4 panels)
        shapes: List of Plotly shape dicts for drawings
        trades_df: Optional trade markers

    Returns Plotly figure.
    """
    if "EMA50" not in df.columns:
        from core.indicators import compute_indicators
        df = compute_indicators(df)

    rows = 4 if show_vp else 3
    row_heights = [0.4, 0.4, 0.12, 0.08] if show_vp else [0.55, 0.15, 0.15]
    subplot_titles = [f"{symbol} — Price"] + (["Volume Profile"] if show_vp else []) + ["Volume", "MACD"]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )

    # Row 1: Candlesticks + Overlays
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#00ff9f",
        decreasing_line_color="#ff4d6d"
    ), row=1, col=1)

    # EMA50
    if "EMA50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA50"],
            mode="lines", line=dict(color="#f5c542", width=1.5, dash="dash"),
            name="EMA50"
        ), row=1, col=1)

    # EMA200
    if "EMA200" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA200"],
            mode="lines", line=dict(color="#ff8c42", width=2),
            name="EMA200"
        ), row=1, col=1)

    # Bollinger Bands
    if show_bb and "BB_UPPER" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_UPPER"],
            mode="lines", line=dict(color="rgba(100,149,237,0.3)", width=1),
            name="BB Upper"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_LOWER"],
            mode="lines", line=dict(color="rgba(100,149,237,0.3)", width=1),
            fill="tonexty", fillcolor="rgba(100,149,237,0.05)",
            name="BB Lower"
        ), row=1, col=1)

    # VWAP
    if show_vwap:
        vwap_df = vwap_with_bands(df, lookback=len(df))
        if "VWAP" in vwap_df.columns:
            fig.add_trace(go.Scatter(
                x=vwap_df.index, y=vwap_df["VWAP"],
                mode="lines", line=dict(color="purple", width=1.5),
                name="VWAP"
            ), row=1, col=1)

    # Supertrend
    if show_supertrend and "SUPERTREND" in df.columns:
        st_colors = ["#00ff9f" if d == 1 else "#ff4d6d" for d in df["SUPERTREND_DIR"]]
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SUPERTREND"],
            mode="markers", marker=dict(size=3, color=st_colors),
            name="Supertrend"
        ), row=1, col=1)

    # Row 2 or 3: Volume Profile panel
    vp_row = 2
    if show_vp:
        vp_row = 2
        vp = volume_profile(df, num_bins=24, lookback=min(60, len(df)))
        if vp:
            fig.add_trace(go.Bar(
                x=vp["volumes"],
                y=vp["price_bins"],
                orientation="h",
                marker_color="#00bfff",
                opacity=0.4,
                name="Vol Profile",
                showlegend=True
            ), row=vp_row, col=1)
            fig.add_hline(y=vp["poc"], line_color="#f5c542", line_width=1,
                          row=vp_row, col=1)

    # Volume bars
    vol_row = vp_row + 1 if show_vp else 2
    vol_colors = ["#00ff9f" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ff4d6d"
                  for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=vol_colors, marker_line_width=0, opacity=0.8
    ), row=vol_row, col=1)

    if "VOL_MA20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VOL_MA20"],
            mode="lines", line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dot"),
            name="Vol MA20"
        ), row=vol_row, col=1)

    # MACD
    macd_row = vol_row + 1
    if show_macd and "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], mode="lines",
            line=dict(color="#00bfff", width=1.5), name="MACD"
        ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_SIGNAL"], mode="lines",
            line=dict(color="#ff8c42", width=1.5), name="Signal"
        ), row=macd_row, col=1)
        hist_colors = ["#00ff9f" if v >= 0 else "#ff4d6d" for v in df["MACD_HIST"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_HIST"],
            name="Histogram", marker_color=hist_colors, marker_line_width=0
        ), row=macd_row, col=1)

    # Apply drawing shapes if provided
    if shapes:
        for shape in shapes:
            fig.add_shape(**shape)

    # Layout
    fig.update_layout(
        height=900 if show_vp else 800,
        template="plotly_dark",
        paper_bgcolor="#050816",
        plot_bgcolor="#0a0e1a",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="#1a1f2e")
    if show_vp:
        fig.update_yaxes(title_text="Price", row=2, col=1, gridcolor="#1a1f2e")
    fig.update_yaxes(title_text="Volume", row=vol_row, col=1, gridcolor="#1a1f2e")
    fig.update_yaxes(title_text="MACD", row=macd_row, col=1, gridcolor="#1a1f2e")
    fig.update_xaxes(gridcolor="#1a1f2e")

    return fig


def create_drawing_shape(draw_type: str, x0, y0, x1=None, y1=None, color="#f5c542"):
    """
    Create a Plotly shape dict for drawing tools.

    Supported types: 'hline', 'vline', 'trendline', 'fib', 'rectangle'
    """
    if draw_type == "hline":
        return {
            "type": "line",
            "x0": 0, "x1": 1,
            "y0": y0, "y1": y0,
            "xref": "paper", "yref": "y",
            "line": {"color": color, "width": 2, "dash": "dash"},
            "name": f"HL {y0}"
        }
    elif draw_type == "trendline":
        return {
            "type": "line",
            "x0": x0, "x1": x1,
            "y0": y0, "y1": y1,
            "line": {"color": color, "width": 2},
            "name": "Trend"
        }
    return None
