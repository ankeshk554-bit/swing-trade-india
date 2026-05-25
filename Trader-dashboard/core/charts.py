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
        font=dict(color="#e6edf3"),
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
        font=dict(color="#e6edf3"),
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
    Enhanced chart with RSI (+auto divergence), ADX (+DI/-DI), MACD panels.

    Args:
        df: OHLCV DataFrame with indicators
        symbol: Ticker symbol
        show_bb: Show Bollinger Bands
        show_macd: Show MACD subplot
        show_vwap: Show VWAP overlay
        show_supertrend: Show Supertrend markers
        show_vp: Show Volume Profile as subplot
        shapes: List of Plotly shape dicts for drawings

    Returns Plotly figure.
    """
    if "EMA50" not in df.columns:
        from core.indicators import compute_indicators
        df = compute_indicators(df)

    # ── Divergence detection ──
    def _find_swings(series, order=5):
        """Find swing high/low indices. Returns (highs_idx, lows_idx)."""
        highs, lows = [], []
        for i in range(order, len(series) - order):
            if all(series.iloc[i] >= series.iloc[i - j] and series.iloc[i] >= series.iloc[i + j] for j in range(1, order + 1)):
                highs.append(i)
            if all(series.iloc[i] <= series.iloc[i - j] and series.iloc[i] <= series.iloc[i + j] for j in range(1, order + 1)):
                lows.append(i)
        return highs, lows

    def _detect_divergence(price_series, rsi_series, order=5):
        """Detect bullish and bearish divergences. Returns list of (idx, type, prev_idx, curr_idx)."""
        price_swing_h, price_swing_l = _find_swings(price_series, order)
        rsi_swing_h, rsi_swing_l = _find_swings(rsi_series, order)

        divs = []

        # Bearish divergence: price higher high, RSI lower high
        for i in range(len(price_swing_h)):
            for j in range(i + 1, min(i + 4, len(price_swing_h))):
                pi, pj = price_swing_h[i], price_swing_h[j]
                if pi >= len(price_series) or pj >= len(price_series):
                    continue
                if price_series.iloc[pj] > price_series.iloc[pi]:
                    # Find closest RSI swing after pj
                    for ri in rsi_swing_h:
                        if ri > pj and ri < pj + 30:
                            if rsi_series.iloc[ri] < rsi_series.iloc[pi]:
                                divs.append((ri, "bearish", pi, pj))
                                break
        # Bullish divergence: price lower low, RSI higher low
        for i in range(len(price_swing_l)):
            for j in range(i + 1, min(i + 4, len(price_swing_l))):
                pi, pj = price_swing_l[i], price_swing_l[j]
                if pi >= len(price_series) or pj >= len(price_series):
                    continue
                if price_series.iloc[pj] < price_series.iloc[pi]:
                    for ri in rsi_swing_l:
                        if ri > pj and ri < pj + 30:
                            if rsi_series.iloc[ri] > rsi_series.iloc[pi]:
                                divs.append((ri, "bullish", pi, pj))
                                break
        return divs

    # ── Layout: Price | Volume | RSI | ADX | MACD ──
    rows = 5
    row_heights = [0.40, 0.12, 0.12, 0.12, 0.12]
    subplot_titles = [f"{symbol} — Price", "Volume", "RSI", "ADX", "MACD"]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )

    # ═══════════════════════════════════════
    # ROW 1: CANDLESTICKS + OVERLAYS
    # ═══════════════════════════════════════
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

    # ═══════════════════════════════════════
    # ROW 2: VOLUME
    # ═══════════════════════════════════════
    vol_colors = ["#00ff9f" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ff4d6d"
                  for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=vol_colors, marker_line_width=0, opacity=0.8
    ), row=2, col=1)

    if "VOL_MA20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VOL_MA20"],
            mode="lines", line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dot"),
            name="Vol MA20"
        ), row=2, col=1)

    # ═══════════════════════════════════════
    # ROW 3: RSI + DIVERGENCE MARKERS
    # ═══════════════════════════════════════
    if "RSI" in df.columns:
        rsi = df["RSI"]
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi,
            mode="lines", line=dict(color="#c084fc", width=1.5),
            name="RSI (14)"
        ), row=3, col=1)

        # RSI levels
        fig.add_hline(y=70, line_color="rgba(255,77,109,0.4)", line_width=1, line_dash="dash", row=3, col=1)
        fig.add_hline(y=30, line_color="rgba(0,255,159,0.4)", line_width=1, line_dash="dash", row=3, col=1)
        fig.add_hline(y=50, line_color="rgba(255,255,255,0.15)", line_width=1, row=3, col=1)

        # Auto-detect divergences
        close_prices = df["Close"]
        divergences = _detect_divergence(close_prices, rsi, order=5)

        bearish_x, bearish_y = [], []
        bullish_x, bullish_y = [], []
        bearish_lines = []
        bullish_lines = []

        for div_idx, div_type, p_idx1, p_idx2 in divergences:
            idx = df.index[div_idx]
            if div_type == "bearish":
                bearish_x.append(idx)
                bearish_y.append(rsi.iloc[div_idx])
                # Draw line from price high to RSI high
                p1_idx, p2_idx = p_idx1, p_idx2
                if p1_idx < len(df) and p2_idx < len(df):
                    bearish_lines.append((df.index[p1_idx], float(df["High"].iloc[p1_idx]),
                                          df.index[p2_idx], float(df["High"].iloc[p2_idx])))
            else:  # bullish
                bullish_x.append(idx)
                bullish_y.append(rsi.iloc[div_idx])
                p1_idx, p2_idx = p_idx1, p_idx2
                if p1_idx < len(df) and p2_idx < len(df):
                    bullish_lines.append((df.index[p1_idx], float(df["Low"].iloc[p1_idx]),
                                          df.index[p2_idx], float(df["Low"].iloc[p2_idx])))

        # Plot RSI divergence markers
        if bearish_x:
            fig.add_trace(go.Scatter(
                x=bearish_x, y=bearish_y,
                mode="markers", marker=dict(symbol="triangle-down", size=10, color="#ff4d6d"),
                name="Bearish Div"
            ), row=3, col=1)

            # Price divergence lines
            for x1, y1, x2, y2 in bearish_lines:
                fig.add_shape(type="line", x0=x1, y0=y1, x1=x2, y1=y2,
                              line=dict(color="rgba(255,77,109,0.6)", width=1, dash="dot"),
                              row=1, col=1)

        if bullish_x:
            fig.add_trace(go.Scatter(
                x=bullish_x, y=bullish_y,
                mode="markers", marker=dict(symbol="triangle-up", size=10, color="#00ff9f"),
                name="Bullish Div"
            ), row=3, col=1)

            for x1, y1, x2, y2 in bullish_lines:
                fig.add_shape(type="line", x0=x1, y0=y1, x1=x2, y1=y2,
                              line=dict(color="rgba(0,255,159,0.6)", width=1, dash="dot"),
                              row=1, col=1)

    # ═══════════════════════════════════════
    # ROW 4: ADX + PLUS_DI / MINUS_DI
    # ═══════════════════════════════════════
    if "ADX" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ADX"],
            mode="lines", line=dict(color="#f5c542", width=1.5),
            name="ADX"
        ), row=4, col=1)

        # ADX level
        fig.add_hline(y=22, line_color="rgba(255,255,255,0.3)", line_width=1, line_dash="dash", row=4, col=1)
        fig.add_hline(y=25, line_color="rgba(0,255,159,0.3)", line_width=1, row=4, col=1)

    if "PLUS_DI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["PLUS_DI"],
            mode="lines", line=dict(color="#00ff9f", width=1.5),
            name="+DI"
        ), row=4, col=1)

    if "MINUS_DI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MINUS_DI"],
            mode="lines", line=dict(color="#ff4d6d", width=1.5),
            name="-DI"
        ), row=4, col=1)

    # ═══════════════════════════════════════
    # ROW 5: MACD
    # ═══════════════════════════════════════
    if show_macd and "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], mode="lines",
            line=dict(color="#00bfff", width=1.5), name="MACD"
        ), row=5, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_SIGNAL"], mode="lines",
            line=dict(color="#ff8c42", width=1.5), name="Signal"
        ), row=5, col=1)
        hist_colors = ["#00ff9f" if v >= 0 else "#ff4d6d" for v in df["MACD_HIST"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_HIST"],
            name="Histogram", marker_color=hist_colors, marker_line_width=0
        ), row=5, col=1)

    # Apply drawing shapes
    if shapes:
        for shape in shapes:
            fig.add_shape(**shape)

    # ── Layout ──
    fig.update_layout(
        height=1050,
        template="plotly_dark",
        paper_bgcolor="#050816",
        plot_bgcolor="#0a0e1a",
        font=dict(color="#e6edf3"),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="#1a1f2e")
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="#1a1f2e")
    fig.update_yaxes(title_text="RSI", row=3, col=1, gridcolor="#1a1f2e", range=[0, 100])
    fig.update_yaxes(title_text="ADX", row=4, col=1, gridcolor="#1a1f2e")
    fig.update_yaxes(title_text="MACD", row=5, col=1, gridcolor="#1a1f2e")
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
