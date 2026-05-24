import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from core.utils import load_data, clear_cache
from core.indicators import (
    compute_indicators, compute_institutional_score,
    generate_swing_signal, detect_vcp, calculate_avwap,
    compute_delivery_enhanced_score, generate_fo_enhanced_signal
)
from core.market_regime import get_market_regime
from core.backtest import run_backtest
from core.scanner import scan_universe, filter_by_strategy
from core.charts import plot_chart
from core.relative_strength import get_relative_strength
from core.data_providers import (
    get_market_overview, get_fo_chain, get_india_vix,
    get_fii_dii_data, get_market_breadth, get_block_deals,
    get_expiry_calendar, get_delivery_summary, get_vix_history,
    get_sector_performance, get_top_movers,
    clear_data_provider_cache
)
from data.sectors import get_sector, get_all_sectors
from core.risk_engine import (
    calculate_position_with_target, suggest_stop_loss
)
from core.watchlist import (
    get_watchlists, get_watchlist, create_watchlist, delete_watchlist,
    add_to_watchlist, remove_from_watchlist, get_watchlist_with_prices,
    export_watchlist
)
from core.trade_journal import (
    add_trade, close_trade, delete_trade, update_trade,
    get_all_trades, get_open_trades, get_closed_trades,
    get_portfolio_summary, get_monthly_returns, export_journal
)
from core.shoonya_bridge import (
    ShoonyaClient, connect as shoonya_connect,
    disconnect as shoonya_disconnect,
    load_credentials, save_credentials,
    clear_credentials, get_client
)
from core.support_resistance import (
    detect_support_resistance, fibonacci_levels, pivot_points
)
from core.strategy_optimizer import (
    run_strategy_comparison, walk_forward_optimize,
    get_rule_based_score, train_ai_scorer, score_signal_ai,
    _extract_features, SKLEARN_AVAILABLE
)
from core.profit_engine import (
    generate_v3_signal, run_v3_backtest, compare_v3_vs_old
)
from core.profit_engine_v4 import (
    generate_v4_signal, run_v4_backtest, compare_v4_vs_old,
    walk_forward_optimize, run_monte_carlo,
    kelly_fraction
)
from core.profit_engine_v5 import (
    run_v5_backtest, compare_v5_vs_old,
    compute_advanced_metrics, EntryFilters, RiskManager,
    anchored_walk_forward, parameter_stability_test,
    deflated_sharpe_ratio, detect_regime_hmm
)
from core.profit_engine_v6 import (
    Backtester, run_v6_backtest, compare_v6_vs_old,
    PerformanceAnalytics, IndiaMarketUtils,
    EntrySignalGenerator, ExitManager as ExitMgrV6,
    RiskManager as RiskMgrV6
)

# --- Stock Lists ---
from data.nifty50 import NIFTY50
from data.nifty200 import NIFTY200
from data.nifty500 import NIFTY500

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Sniper Terminal — India's #1 Swing Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# THEME SYSTEM
# =========================================================

# Load external stylesheet
import os
_css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(_css_path):
    with open(_css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Theme state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Inject theme CSS — overrides Streamlit's own CSS variables
_dark = st.session_state.theme == "dark"
_bg = "#050816" if _dark else "#f8fafc"
_bg2 = "#0a0e1a" if _dark else "#f1f5f9"
_bg_card = "#111827" if _dark else "#ffffff"
_text = "#f3f4f6" if _dark else "#0f172a"
_text2 = "#9ca3af" if _dark else "#475569"
_border = "#1f2937" if _dark else "#e2e8f0"
_gold = "#f5c542" if _dark else "#d97706"

theme_css = f"""
<style>
    .stApp, .stApp > header {{
        background-color: {_bg} !important;
    }}
    .main .block-container {{
        background-color: {_bg} !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {_bg2} !important;
        border-right: 1px solid {_border} !important;
    }}
    [data-testid="stSidebar"] *, .stMarkdown, p, span, label, .stCaption {{
        color: {_text} !important;
    }}
    [data-testid="stSidebar"] .stRadio > label {{
        color: {_text} !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {_gold} !important;
    }}
    div[data-testid="metric-container"] {{
        background-color: {_bg_card} !important;
        border-color: {_border} !important;
    }}
    div[data-testid="metric-container"] label {{
        color: {_text2} !important;
    }}
    div[data-testid="metric-container"] div[data-testid="metric-value"] {{
        color: {_text} !important;
    }}
    .stDataFrame {{
        border-color: {_border} !important;
    }}
    .stSelectbox > div > div, .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        background-color: {_bg_card} !important;
        border-color: {_border} !important;
        color: {_text} !important;
    }}
    .streamlit-expanderHeader {{
        background-color: {_bg_card} !important;
        border-color: {_border} !important;
    }}
    .stAlert {{
        background-color: {_bg_card} !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {_bg2} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {_text2} !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: {_bg_card} !important;
        color: {_gold} !important;
    }}
    [data-testid="StyledFullScreenButton"] {{
        color: {_text} !important;
    }}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚡ Sniper Terminal")
st.sidebar.caption("Institutional Swing Engine")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Screener", "Chart", "Backtest", "Market Data", "Analytics", "Portfolio", "🔍 Stock Research"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("v2.0 • NSE/BSE • Institutional Grade")

# Data source
data_source = st.sidebar.radio(
    "Data Source",
    ["Yahoo Finance (Free)", "Shoonya (Fast)"],
    index=0,
    help="Shoonya provides faster Indian market data. Requires Finvasia account."
)

st.sidebar.markdown("---")

# Cache management
with st.sidebar.expander("⚙️ Settings"):
    # Theme toggle
    theme_cols = st.columns(2)
    with theme_cols[0]:
        if st.button("🌙 Dark" if st.session_state.theme == "light" else "☀️ Light", 
                     key="theme_toggle", use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    with theme_cols[1]:
        st.caption(f"Current: {st.session_state.theme}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Cache"):
            n = clear_cache()
            st.success(f"Cleared {n} cached files")
            st.rerun()
    with col2:
        if st.button("🔄 Clear NSE Cache"):
            n = clear_data_provider_cache()
            st.success(f"Cleared {n} NSE cached files")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 🔑 Shoonya API (Finvasia)")

    # Show connection status
    shoonya_client = get_client()
    shoonya_connected = shoonya_client.is_connected
    status_emoji = "🟢 Connected" if shoonya_connected else "🔴 Disconnected"
    st.caption(f"Status: {status_emoji}")

    saved_creds = load_credentials()
    if not saved_creds:
        shoonya_uid = st.text_input("User ID", "", key="sh_uid", placeholder="e.g., FA123456")
        shoonya_pwd = st.text_input("Password", "", type="password", key="sh_pwd")
        shoonya_pan = st.text_input("PAN Card", "", type="password", key="sh_pan",
                                    placeholder="ABCDE1234F",
                                    help="Your PAN card number in UPPERCASE (not DOB/TOTP)")
        if st.button("🔗 Connect Shoonya", key="sh_connect"):
            if shoonya_uid and shoonya_pwd and shoonya_pan:
                with st.spinner("Connecting..."):
                    result = shoonya_connect(shoonya_uid, shoonya_pwd, shoonya_pan)
                    if isinstance(result, dict) and result.get("success"):
                        st.success("✅ Connected to Shoonya!")
                        st.rerun()
                    else:
                        err_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                        st.error(f"❌ {err_msg}")
                        if "IP" in err_msg or "whitelist" in err_msg.lower():
                            st.info(
                                "🔒 **IP Whitelisting Required** — You have an Oracle Cloud static IP. "
                                "Email Finvasia support (support@finvasia.com) with:\n"
                                "1. Your **User ID**\n"
                                "2. Your **Oracle Cloud static IP**\n"
                                "3. Request: *'Please whitelist my static IP for Shoonya API access'*\n\n"
                                "Once whitelisted, deploy this app on your Oracle VM and connect."
                            )
            else:
                st.warning("Fill all fields (User ID, Password, and PAN)")
    else:
        st.caption(f"Connected as: {saved_creds.get('uid', '?')}")
        if st.button("🔌 Disconnect", key="sh_disconnect"):
            shoonya_disconnect()
            clear_credentials()
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("v2.0 • NSE/BSE • Shoonya • Institutional Grade")

# =========================================================
# =========================================================
# PAGE: DASHBOARD
# =========================================================
if page == "Dashboard":
    st.subheader("📊 Command Center")

    # Market Regime Banner (dashboard only)
    try:
        regime_info = get_market_regime(detailed=True)
        if not isinstance(regime_info, dict):
            regime_info = {"regime": "UNKNOWN", "rsi": 50, "strength": ""}
    except Exception:
        regime_info = {"regime": "UNKNOWN", "rsi": 50, "strength": ""}

    regime = regime_info.get("regime", "UNKNOWN")
    regime_emoji = "🟢" if regime == "BULLISH" else ("🔴" if regime == "BEARISH" else "🟡")

    pulse_cols = st.columns(7)
    with pulse_cols[0]: st.metric(f"{regime_emoji} Market", f"{regime}", f"Nifty: {regime_info.get('nifty_close', '—')}", delta_color="off")
    with pulse_cols[1]: st.metric("RSI", regime_info.get("rsi", "—"), regime_info.get("rsi_zone", "—"))
    with pulse_cols[2]: st.metric("ADX", regime_info.get("adx", "—"), "Strength")
    with pulse_cols[3]: st.metric("vs EMA50", f"{regime_info.get('ema50_distance', 0):+.1f}%")
    with pulse_cols[4]: st.metric("vs EMA200", f"{regime_info.get('ema200_distance', 0):+.1f}%")
    with pulse_cols[5]: st.metric("VIX", regime_info.get("vix", "—"), regime_info.get("vix_regime", "—"))
    with pulse_cols[6]:
        fiidii = regime_info.get("net_fii_dii")
        st.metric("FII+DII", f"₹{fiidii:+,.0f}Cr" if fiidii else "N/A")
    st.markdown("---")

    # ===================================================================
    # ROW 1: EXECUTIVE SUMMARY — Market Pulse
    # ===================================================================
    with st.spinner("Loading market pulse..."):
        try:
            overview = get_market_overview()
        except Exception:
            overview = {}

    vix_info = overview.get("vix", {})
    fo_data = overview.get("fo", {})
    fii_data = overview.get("fii_dii", {})
    breadth_data = overview.get("breadth", {})
    expiry_cal = overview.get("expiry_calendar", [])
    market_status = overview.get("market_status", "UNKNOWN")

    # Quick signal scan from NIFTY200 sample
    total_buys = total_neutral = total_sells = total_scanned = 0
    for stock in NIFTY200[:30]:
        df_sig = load_data(stock, period="1y")
        if not df_sig.empty and len(df_sig) > 200:
            df_sig = compute_indicators(df_sig)
            sig = generate_swing_signal(df_sig)
            if sig["Signal"] == "BUY": total_buys += 1
            elif sig["Signal"] == "SELL": total_sells += 1
            else: total_neutral += 1
            total_scanned += 1

    pulse_cols = st.columns(7)
    with pulse_cols[0]:
        st.metric("📈 Universe", f"{total_scanned}", "NIFTY 200")
    with pulse_cols[1]:
        st.metric("🟢 Buys", total_buys, f"{total_buys/max(total_scanned,1)*100:.0f}%")
    with pulse_cols[2]:
        st.metric("🟡 Neutral", total_neutral)
    with pulse_cols[3]:
        st.metric("🔴 Sells", total_sells)

    # VIX
    vix_val = vix_info.get("vix", "—")
    vix_regime = vix_info.get("regime", "N/A")
    vix_emoji = "🟢" if vix_regime in ("LOW_VOL", "NORMAL") else ("🔴" if "FEAR" in vix_regime else "🟡")
    with pulse_cols[4]:
        st.metric(f"{vix_emoji} VIX", vix_val if vix_val else "N/A",
                  f"{vix_info.get('change', 0):+.2f}%" if vix_info.get('change') else None)

    # PCR
    pcr = fo_data.get("pcr")
    with pulse_cols[5]:
        if pcr is not None:
            st.metric("🔄 PCR", f"{pcr:.2f}", "Bullish" if pcr > 1.0 else "Bearish",
                      delta_color="normal" if pcr > 1.0 else "inverse")
        else:
            st.metric("🔄 PCR", "N/A")

    # FII/DII
    fii_net = fii_data.get("net_combined")
    with pulse_cols[6]:
        if fii_net is not None:
            st.metric("🏦 FII+DII", f"₹{fii_net:+,.0f}Cr",
                      f"FII: ₹{fii_data.get('fii_cash', 0):+,.0f}",
                      delta_color="normal" if fii_net > 0 else "inverse")
        else:
            st.metric("🏦 FII+DII", "N/A")

    st.markdown("---")

    # ===================================================================
    # ROW 2: VIX CHART + SECTOR HEATMAP (side by side)
    # ===================================================================
    dash_left, dash_right = st.columns([1.4, 1])

    # --- LEFT: VIX Historical Chart ---
    with dash_left:
        st.markdown("### 🌡️ India VIX — 6 Month History")
        vix_df = get_vix_history("6mo")
        if vix_df is not None and not vix_df.empty:
            vix_fig = go.Figure()
            vix_fig.add_trace(go.Scatter(
                x=vix_df.index, y=vix_df["Close"],
                mode="lines", name="VIX",
                line=dict(color="#f5c542", width=2),
                fill="tozeroy", fillcolor="rgba(245,197,66,0.1)"
            ))
            # Regime zones
            current_vix = float(vix_df["Close"].iloc[-1])
            max_vix = float(vix_df["Close"].max())
            y_max = max(max_vix * 1.3, 30)

            vix_fig.add_hrect(y0=0, y1=14, fillcolor="green", opacity=0.05, line_width=0, annotation_text="Low Vol")
            vix_fig.add_hrect(y0=14, y1=18, fillcolor="green", opacity=0.03, line_width=0, annotation_text="Normal")
            vix_fig.add_hrect(y0=18, y1=22, fillcolor="yellow", opacity=0.04, line_width=0, annotation_text="Elevated")
            vix_fig.add_hrect(y0=22, y1=28, fillcolor="red", opacity=0.04, line_width=0, annotation_text="High Vol")
            vix_fig.add_hrect(y0=28, y1=y_max, fillcolor="red", opacity=0.08, line_width=0, annotation_text="Extreme")

            vix_fig.update_layout(
                height=350, template="plotly_dark",
                paper_bgcolor="#050816", plot_bgcolor="#0a0e1a",
                margin=dict(l=10, r=10, t=10, b=10),
                hovermode="x unified",
                showlegend=False,
                yaxis=dict(gridcolor="#1a1f2e", range=[0, y_max]),
                xaxis=dict(gridcolor="#1a1f2e")
            )
            st.plotly_chart(vix_fig, width='stretch')
        else:
            st.info("VIX history unavailable")

    # --- RIGHT: Sector Rotation Heatmap ---
    with dash_right:
        st.markdown("### 🏭 Sector Rotation")
        sector_perf = get_sector_performance()
        if sector_perf:
            # Create a horizontal bar chart as heatmap
            sectors = [s["sector"][:15] for s in sector_perf]
            changes = [s["change_pct"] for s in sector_perf]
            strengths = [s["strength"] for s in sector_perf]

            colors = []
            for c in changes:
                if c > 0.5: colors.append("#00ff9f")
                elif c > 0: colors.append("#66d9a0")
                elif c > -0.5: colors.append("#ff8c42")
                else: colors.append("#ff4d6d")

            sec_fig = go.Figure(go.Bar(
                x=changes, y=sectors,
                orientation="h",
                marker_color=colors,
                text=[f"{c:+.2f}%" for c in changes],
                textposition="outside",
                textfont=dict(size=10)
            ))
            sec_fig.update_layout(
                height=350, template="plotly_dark",
                paper_bgcolor="#050816", plot_bgcolor="#0a0e1a",
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(gridcolor="#1a1f2e", title="Daily Change %"),
                yaxis=dict(gridcolor="#1a1f2e", autorange="reversed"),
                showlegend=False
            )
            st.plotly_chart(sec_fig, width='stretch')
        else:
            st.info("Sector data unavailable")

    st.markdown("---")

    # ===================================================================
    # ROW 3: TOP MOVERS (Gainers / Losers / Volume)
    # ===================================================================
    st.markdown("### 📊 Top Movers — NIFTY 200")

    with st.spinner("Scanning top movers..."):
        movers = get_top_movers(NIFTY200, top_n=8)

    if movers and movers.get("gainers"):
        m_cols = st.columns(3)

        # Gainers
        with m_cols[0]:
            st.markdown("#### 🟢 Top Gainers")
            for i, g in enumerate(movers["gainers"]):
                delta_color = "normal"
                pct = g["change"]
                bar_color = "#00ff9f"
                bg = "rgba(0,255,159,0.05)" if i % 2 == 0 else "transparent"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:6px 10px;"
                    f"background:{bg};border-radius:4px;margin:2px 0;'>"
                    f"<span>{i+1}. {g['symbol'].replace('.NS','')}</span>"
                    f"<span style='color:{bar_color};font-weight:bold;'>+{pct:.2f}%</span></div>",
                    unsafe_allow_html=True
                )

        # Losers
        with m_cols[1]:
            st.markdown("#### 🔴 Top Losers")
            for i, g in enumerate(movers["losers"]):
                pct = g["change"]
                bar_color = "#ff4d6d"
                bg = "rgba(255,77,109,0.05)" if i % 2 == 0 else "transparent"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:6px 10px;"
                    f"background:{bg};border-radius:4px;margin:2px 0;'>"
                    f"<span>{i+1}. {g['symbol'].replace('.NS','')}</span>"
                    f"<span style='color:{bar_color};font-weight:bold;'>{pct:.2f}%</span></div>",
                    unsafe_allow_html=True
                )

        # Most Volume
        with m_cols[2]:
            st.markdown("#### 📊 Most Active (Volume)")
            for i, g in enumerate(movers["most_volume"]):
                bg = "rgba(100,149,237,0.05)" if i % 2 == 0 else "transparent"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:6px 10px;"
                    f"background:{bg};border-radius:4px;margin:2px 0;'>"
                    f"<span>{i+1}. {g['symbol'].replace('.NS','')}</span>"
                    f"<span style='color:#64b5f6;'>₹{g['close']:,.2f}</span></div>",
                    unsafe_allow_html=True
                )
    else:
        st.info("Top movers data unavailable. Try during market hours.")

    st.markdown("---")

    # ===================================================================
    # ROW 4: INSTITUTIONAL DASHBOARD
    # ===================================================================
    inst_tabs = st.tabs(["🎯 F&O Chain", "🏦 FII/DII", "📊 Market Breadth", "📅 Expiry Calendar", "📋 Block Deals"])

    # Tab 1: F&O Chain
    with inst_tabs[0]:
        if fo_data and fo_data.get("pcr") is not None:
            f_cols = st.columns(6)
            with f_cols[0]:
                st.metric("📊 PCR", f"{fo_data['pcr']:.2f}",
                          "Bullish" if fo_data['pcr'] > 1.0 else "Bearish")
            with f_cols[1]:
                st.metric("📈 CE OI", f"{fo_data['total_ce_oi']/1e7:.1f}Cr")
            with f_cols[2]:
                st.metric("📉 PE OI", f"{fo_data['total_pe_oi']/1e7:.1f}Cr")
            with f_cols[3]:
                st.metric("🎯 Max Pain", fo_data.get('max_pain', '—'))
            with f_cols[4]:
                max_ce = fo_data.get('max_oi_call', {})
                st.metric("🔴 Max Call OI", max_ce.get('strike', '—'))
            with f_cols[5]:
                max_pe = fo_data.get('max_oi_put', {})
                st.metric("🟢 Max Put OI", max_pe.get('strike', '—'))
        else:
            st.caption("F&O data unavailable — NSE API may be blocked")

    # Tab 2: FII/DII
    with inst_tabs[1]:
        if fii_data and fii_data.get("fii_cash") is not None:
            fi_cols = st.columns(4)
            with fi_cols[0]:
                fc = fii_data.get("fii_cash", 0)
                st.metric("🇮🇳 FII Cash", f"₹{fc:+,.0f}Cr", delta_color="normal" if fc > 0 else "inverse")
            with fi_cols[1]:
                dc = fii_data.get("dii_cash", 0)
                st.metric("🇮🇳 DII Cash", f"₹{dc:+,.0f}Cr", delta_color="normal" if dc > 0 else "inverse")
            with fi_cols[2]:
                ff = fii_data.get("fii_fo", 0)
                st.metric("🎲 FII F&O", f"₹{ff:+,.0f}Cr", delta_color="normal" if ff > 0 else "inverse")
            with fi_cols[3]:
                total = fii_data.get("net_combined", 0)
                st.metric("💰 Combined", f"₹{total:+,.0f}Cr", delta_color="normal" if total > 0 else "inverse")
            if fii_data.get("date"):
                st.caption(f"Data as of: {fii_data['date']}")
        else:
            st.caption("FII/DII data unavailable")

    # Tab 3: Market Breadth
    with inst_tabs[2]:
        adv = breadth_data.get("advances", 0)
        dec = breadth_data.get("declines", 0)
        adr = breadth_data.get("advance_decline_ratio", 1.0)
        bstrength = breadth_data.get("breadth_strength", "N/A")

        if adv > 0 or dec > 0:
            b_cols = st.columns(4)
            with b_cols[0]:
                st.metric("🟢 Advances", adv, f"{adv/max(adv+dec,1)*100:.0f}%")
            with b_cols[1]:
                st.metric("🔴 Declines", dec, f"{dec/max(adv+dec,1)*100:.0f}%")
            with b_cols[2]:
                st.metric("📊 A/D Ratio", f"{adr:.2f}")
            with b_cols[3]:
                color = "normal" if adr > 1.0 else "inverse"
                st.metric("💪 Strength", bstrength, delta_color=color)

            # Visual A/D gauge
            total_b = adv + dec
            if total_b > 0:
                st.progress(adv / max(total_b, 1), text=f"🟢 Advances: {adv}/{total_b} ({adv/total_b*100:.0f}%)")
        else:
            st.caption("Breadth data unavailable — try during market hours")

    # Tab 4: Expiry Calendar
    with inst_tabs[3]:
        if expiry_cal:
            exp_df = pd.DataFrame(expiry_cal)
            st.dataframe(exp_df, width='stretch', hide_index=True,
                         column_config={
                             "days_to_expiry": st.column_config.NumberColumn("DTE"),
                             "expiry_date": "Expiry",
                             "type": "Type",
                             "symbol": "Symbol"
                         })
        else:
            st.caption("No expiry data available")

    # Tab 5: Block Deals
    with inst_tabs[4]:
        deals = overview.get("block_deals", [])
        if deals:
            deals_df = pd.DataFrame(deals)
            st.dataframe(deals_df, width='stretch', hide_index=True)
        else:
            st.caption("No block deals today")

    st.markdown("---")

    # ===================================================================
    # ROW 5: STRATEGY QUICK ACCESS
    # ===================================================================
    st.markdown("### 🎯 Quick Access Strategies")
    strat_cards = st.columns(4)
    quick_strats = [
        ("🏃 Momentum", "Buy + Conf≥3", "#00ff9f"),
        ("📐 VCP Breakout", "VCP + Buy signal", "#00bfff"),
        ("📦 Delivery Spurt", "Strong delivery", "#f5c542"),
        ("🚀 52W Breakout", "New highs + vol", "#ff8c42"),
    ]
    for i, (name, desc, color) in enumerate(quick_strats):
        with strat_cards[i]:
            st.markdown(
                f"<div style='background:#111827;border:1px solid {color}33;"
                f"border-radius:12px;padding:16px;text-align:center;'>"
                f"<div style='font-size:20px;font-weight:bold;color:{color};'>{name}</div>"
                f"<div style='font-size:12px;color:#6b7280;margin-top:4px;'>{desc}</div>"
                f"<div style='margin-top:8px;font-size:11px;color:#f5c542;'>"
                f"→ Use Screener</div></div>",
                unsafe_allow_html=True
            )

    st.info("💡 **Pro Tip**: Use the Screener for detailed scans with 56 conditions & 10 strategies. "
            "Check Market Data for F&O analysis.")

# =========================================================
# PAGE: SCREENER
# =========================================================
elif page == "Screener":
    st.subheader("🔍 Institutional Scanner")

    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

    with col1:
        universe = st.selectbox(
            "Select Universe",
            ["NIFTY50", "NIFTY200", "NIFTY500"],
            help="Choose the stock universe to scan"
        )

    with col2:
        strategy_filter = st.selectbox(
            "Strategy Filter",
            ["ALL", "MOMENTUM_RUNNER", "VCP_BREAKOUT", "BREAKOUT_52W", "DELIVERY_SPURT",
             "GOLDEN_CROSS", "MACD_MOMENTUM", "MEAN_REVERSION", "BULL_FLAG",
             "CONSOLIDATION_BREAKOUT", "STRONG_TREND", "WEAK"],
            format_func=lambda x: {
                "ALL": "All Stocks",
                "MOMENTUM_RUNNER": "🏃 Momentum Runner",
                "VCP_BREAKOUT": "📐 VCP Breakout",
                "BREAKOUT_52W": "🚀 52-Week Breakout",
                "DELIVERY_SPURT": "📦 Delivery Spurt",
                "GOLDEN_CROSS": "🥇 Golden Cross",
                "MACD_MOMENTUM": "📈 MACD Momentum",
                "MEAN_REVERSION": "🔄 Mean Reversion",
                "BULL_FLAG": "🚩 Bull Flag",
                "CONSOLIDATION_BREAKOUT": "📊 Consolidation Breakout",
                "STRONG_TREND": "💪 Strong Trend",
                "WEAK": "🔴 Weak / Sell"
            }.get(x, x)
        )

    with col3:
        min_score = st.slider("Min Score", 0, 100, 0, 10)

    with col4:
        include_fundamental = st.checkbox(
            "📦 Include Delivery & F&O",
            value=False,
            help="Also fetch delivery volume and F&O data (slower scan)"
        )

    if universe == "NIFTY50":
        stocks = NIFTY50
    elif universe == "NIFTY200":
        stocks = NIFTY200
    else:
        stocks = NIFTY500

    scan_button = st.button(f"🚀 Run Scan{' (Deep)' if include_fundamental else ''}", width='stretch')

    if scan_button:
        progress_bar = st.progress(0, text="Initializing scan...")
        status_text = st.empty()

        def update_progress(current, total):
            pct = current / total
            progress_bar.progress(pct, text=f"Scanning {current}/{total}...")
            status_text.caption(f"Processed {current} of {total} stocks | Found: scanning...")

        with st.spinner(f"Scanning {'with delivery/F&O data...' if include_fundamental else 'technical scan...'}"):
            results = scan_universe(stocks, progress_callback=update_progress, include_fundamental=include_fundamental)

        progress_bar.empty()
        status_text.empty()

        if results:
            # Apply filters
            results = filter_by_strategy(results, strategy_filter)
            results = [r for r in results if r.get("Score", 0) >= min_score]

            df_results = pd.DataFrame(results)

            def color_signal(signal):
                if signal == "BUY":
                    return "🟢 BUY"
                elif signal == "SELL":
                    return "🔴 SELL"
                return "🟡 NEUTRAL"

            df_results["Signal"] = df_results["Signal"].apply(color_signal)

            # Build display columns based on scan mode
            base_cols = [
                "Symbol", "Close", "Change%", "Signal", "Score", "Confidence",
                "RSI", "RVOL", "ATR%", "Supertrend", "MACD", "Pattern", "Reason"
            ]

            # If enhanced scan, show delivery/F&O columns
            enhanced_cols = ["Delivery%", "Delivery_Quality", "Delivery_Trend", "In_FO"]
            pattern_cols = ["Pattern", "Pattern_Conf", "Near_52W_High", "Uptrend_HH_HL"]
            all_extra = enhanced_cols + ["Near_52W_High", "Uptrend_HH_HL"]
            display_cols = base_cols + [c for c in all_extra if c in df_results.columns]
            display_cols = [c for c in display_cols if c in df_results.columns]

            st.success(f"✅ Found {len(results)} stocks matching criteria")

            col_config = {
                "Change%": st.column_config.NumberColumn(format="%.2f%%"),
                "Close": st.column_config.NumberColumn(format="₹%.2f"),
                "RSI": st.column_config.NumberColumn(format="%.1f"),
                "RVOL": st.column_config.NumberColumn(format="%.2f"),
                "Score": st.column_config.NumberColumn(format="%d"),
                "Confidence": st.column_config.NumberColumn(format="%d"),
                "ATR%": st.column_config.NumberColumn(format="%.2f%%"),
                "MACD": st.column_config.NumberColumn(format="%.4f"),
                "VCP": st.column_config.CheckboxColumn(),
                "Delivery%": st.column_config.NumberColumn(format="%.1f%%"),
                "In_FO": st.column_config.CheckboxColumn(),
            }

            st.dataframe(
                df_results[display_cols],
                width='stretch',
                column_config=col_config,
                hide_index=True
            )

            # Download button
            csv = df_results.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                data=csv,
                file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ No stocks found matching your criteria. Try expanding the universe or lowering the minimum score.")

    else:
        st.info(f"👆 Select your universe and filters, then click **Run Scan** to scan {len(stocks)} stocks")

# =========================================================
# PAGE: CHART
# =========================================================
elif page == "Chart":
    st.subheader("📈 Professional Charting Suite")

    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "📊 Main Chart", "📊 Stock Comparison", "📦 Volume Profile"
    ])

    # ══════════════════════════════════════════════
    # TAB 1: MAIN CHART
    # ══════════════════════════════════════════════
    with chart_tab1:
        c_col1, c_col2, c_col3, c_col4 = st.columns([2, 1, 1, 1])

        with c_col1:
            ticker = st.text_input("Symbol (e.g., RELIANCE.NS)", value="RELIANCE.NS", key="chart_ticker")

        with c_col2:
            period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                                  index=3, key="chart_period")

        with c_col3:
            interval = st.selectbox("Interval", ["1d", "1wk", "1mo"],
                                    index=0, key="chart_interval",
                                    help="1d=daily, 1wk=weekly, 1mo=monthly")

        with c_col4:
            st.markdown("<br>", unsafe_allow_html=True)
            show_vp = st.checkbox("📊 Vol Profile", value=False, key="chart_vp",
                                  help="Show Volume Profile sub-panel")

        if ticker:
            with st.spinner(f"Loading {ticker}..."):
                df = load_data(ticker, period=period)

            if not df.empty:
                df = compute_indicators(df)
                latest = df.iloc[-1]

                # Indicator toggles
                ind_cols = st.columns(7)
                with ind_cols[0]:
                    show_bb = st.checkbox("Bollinger", value=True, key="chart_bb")
                with ind_cols[1]:
                    show_macd = st.checkbox("MACD", value=True, key="chart_macd")
                with ind_cols[2]:
                    show_vwap = st.checkbox("VWAP", value=False, key="chart_vwap")
                with ind_cols[3]:
                    show_st = st.checkbox("SuperTrend", value=True, key="chart_st")
                with ind_cols[4]:
                    show_sr = st.checkbox("S/R Levels", value=False, key="chart_sr",
                                          help="Auto-detect support/resistance")
                with ind_cols[5]:
                    show_avwap_top = st.checkbox("AVWAP (Top)", value=False, key="chart_avwap_top",
                                                  help="Anchored VWAP from major swing high")
                with ind_cols[6]:
                    show_avwap_bot = st.checkbox("AVWAP (Bottom)", value=False, key="chart_avwap_bot",
                                                  help="Anchored VWAP from major swing low")
                with ind_cols[6]:
                    show_stats = st.checkbox("Stats Bar", value=True, key="chart_stats")

                # Key stats row
                if show_stats:
                    signal_info = generate_swing_signal(df)
                    k1, k2, k3, k4, k5, k6 = st.columns(6)
                    with k1:
                        st.metric("Close", f"₹{latest['Close']:.2f}",
                                  f"{float(latest['Close']/df.iloc[-2]['Close']-1)*100:.2f}%")
                    with k2:
                        st.metric("RSI (14)", f"{latest['RSI']:.1f}")
                    with k3:
                        st.metric("RVOL", f"{latest['RVOL']:.2f}")
                    with k4:
                        st.metric("MACD", f"{latest['MACD_HIST']:.4f}")
                    with k5:
                        st.metric("ATR", f"{latest['ATR_PCT']:.2f}%")
                    with k6:
                        direction = "🟢 Up" if latest["SUPERTREND_DIR"] == 1 else "🔴 Down"
                        st.metric("SuperTrend", direction)

                    # Signal
                    sig = signal_info["Signal"]
                    if sig == "BUY":
                        st.success(f"🟢 **BUY** — Conf: {signal_info['Confidence']}/6 — {signal_info['Reasons']}")
                    elif sig == "SELL":
                        st.error(f"🔴 **SELL** — Conf: {signal_info['Confidence']}/6 — {signal_info['Reasons']}")
                    else:
                        st.info(f"🟡 **NEUTRAL** — Conf: {signal_info['Confidence']}/6 — {signal_info['Reasons']}")

                    # VCP
                    vcp = detect_vcp(df)
                    if vcp["VCP_Flag"]:
                        st.markdown(
                            f'<span style="background:#1a3a1a;padding:4px 12px;border-radius:6px;'
                            f'color:#00ff9f;">📐 VCP Stage {vcp["Stage"]} | '
                            f'Pivot: ₹{vcp["Pivot"]} | Dry-Up: {vcp["VolumeDryUp"]}</span>',
                            unsafe_allow_html=True
                        )

                # S/R levels overlay
                shapes = []
                if show_sr:
                    from core.support_resistance import detect_support_resistance
                    sr = detect_support_resistance(df, lookback=80, order=5)
                    for s in sr["supports"][-5:]:
                        shapes.append({
                            "type": "line", "x0": 0, "x1": 1,
                            "y0": s, "y1": s,
                            "xref": "paper", "yref": "y",
                            "line": {"color": "rgba(0,255,159,0.4)", "width": 1, "dash": "dash"}
                        })
                    for r in sr["resistances"][-5:]:
                        shapes.append({
                            "type": "line", "x0": 0, "x1": 1,
                            "y0": r, "y1": r,
                            "xref": "paper", "yref": "y",
                            "line": {"color": "rgba(255,77,109,0.4)", "width": 1, "dash": "dash"}
                        })

                # The chart
                from core.charts import plot_chart_with_drawings
                fig = plot_chart_with_drawings(
                    df, ticker,
                    show_bb=show_bb, show_macd=show_macd,
                    show_vwap=show_vwap, show_supertrend=show_st,
                    show_vp=show_vp, shapes=shapes if shapes else None
                )

                # ── Anchored VWAP (from major top & bottom) ──
                if show_avwap_top or show_avwap_bot:
                    lookback = min(500, len(df))
                    seg = df.iloc[-lookback:]
                    offset = len(df) - lookback

                    # Find swing high (major top)
                    if show_avwap_top and len(seg) > 50:
                        # Find highest high in central 60% of lookback (avoid edges)
                        inner = seg.iloc[len(seg)//5:-len(seg)//5]
                        top_idx = inner["High"].idxmax()
                        top_iloc = df.index.get_loc(top_idx)
                        avwap_top = calculate_avwap(df, top_iloc)
                        if avwap_top is not None and len(avwap_top) > 0:
                            fig.add_trace(go.Scatter(
                                x=avwap_top.index, y=avwap_top,
                                mode="lines",
                                line=dict(color="#ff6b6b", width=1.5, dash="dot"),
                                name=f"AVWAP Top ({top_idx.strftime('%d-%b')})"
                            ), row=1, col=1)
                            # Mark the anchor
                            fig.add_annotation(
                                x=top_idx, y=float(df.loc[top_idx, "High"]),
                                text="🔺 Top", showarrow=True, arrowhead=1,
                                arrowcolor="#ff6b6b", font=dict(color="#ff6b6b", size=10),
                                row=1, col=1
                            )

                    # Find swing low (major bottom)
                    if show_avwap_bot and len(seg) > 50:
                        inner = seg.iloc[len(seg)//5:-len(seg)//5]
                        bot_idx = inner["Low"].idxmin()
                        bot_iloc = df.index.get_loc(bot_idx)
                        avwap_bot = calculate_avwap(df, bot_iloc)
                        if avwap_bot is not None and len(avwap_bot) > 0:
                            fig.add_trace(go.Scatter(
                                x=avwap_bot.index, y=avwap_bot,
                                mode="lines",
                                line=dict(color="#00ff9f", width=1.5, dash="dot"),
                                name=f"AVWAP Bottom ({bot_idx.strftime('%d-%b')})"
                            ), row=1, col=1)
                            fig.add_annotation(
                                x=bot_idx, y=float(df.loc[bot_idx, "Low"]),
                                text="🔻 Bottom", showarrow=True, arrowhead=1,
                                arrowcolor="#00ff9f", font=dict(color="#00ff9f", size=10),
                                row=1, col=1
                            )

                st.plotly_chart(fig, width='stretch')

                # Delivery volume section
                with st.expander("📦 Delivery Volume Analysis", expanded=False):
                    with st.spinner("Fetching..."):
                        delivery = get_delivery_summary(ticker)
                    if delivery:
                        dcols = st.columns(5)
                        with dcols[0]:
                            st.metric("Latest Delivery %", f"{delivery['latest_delivery_pct']:.1f}%")
                        with dcols[1]:
                            st.metric("10-Day Avg", f"{delivery['avg_delivery_10d']:.1f}%")
                        with dcols[2]:
                            st.metric("30-Day Avg", f"{delivery['avg_delivery_30d']:.1f}%")
                        with dcols[3]:
                            te = "📈" if delivery["delivery_trend"] == "RISING" else "📉"
                            st.metric(f"{te} Trend", delivery["delivery_trend"])
                        with dcols[4]:
                            qe = "🟢" if delivery["delivery_quality"] == "STRONG" else "🟡"
                            st.metric(f"{qe} Quality", delivery["delivery_quality"])
                    else:
                        st.caption("Delivery data unavailable")
            else:
                st.error(f"❌ Could not load {ticker}")

    # ══════════════════════════════════════════════
    # TAB 2: STOCK COMPARISON
    # ══════════════════════════════════════════════
    with chart_tab2:
        st.markdown("### 📊 Multi-Stock Comparison")
        st.caption("Compare up to 6 stocks on a single chart — normalized relative performance")

        comp_stocks = st.text_input(
            "Stock Symbols (comma separated, e.g., RELIANCE.NS, TCS.NS, INFY.NS)",
            "RELIANCE.NS, TCS.NS, INFY.NS",
            key="comp_stocks"
        )
        comp_period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"],
                                   index=2, key="comp_period")
        comp_indicator = st.selectbox(
            "Indicator",
            ["Close", "RSI", "RVOL", "ATR_PCT", "MACD_HIST", "ADX"],
            index=0, key="comp_ind"
        )
        comp_normalize = st.checkbox("Normalize (Base=100)", value=True, key="comp_norm")

        if st.button("🔄 Compare", key="comp_btn", width='stretch'):
            symbols = [s.strip().upper() for s in comp_stocks.split(",") if s.strip()]
            if symbols:
                dfs = {}
                progress = st.progress(0, text="Loading stocks...")
                for i, sym in enumerate(symbols):
                    df_sym = load_data(sym, period=comp_period)
                    if df_sym is not None and not df_sym.empty:
                        df_sym = compute_indicators(df_sym)
                        dfs[sym] = df_sym
                    progress.progress((i + 1) / len(symbols))

                if dfs:
                    from core.charts import plot_comparison_chart
                    fig = plot_comparison_chart(dfs, symbols, comp_indicator, comp_normalize)
                    st.plotly_chart(fig, width='stretch')

                    # Data table
                    st.markdown("#### 📋 Latest Values")
                    rows = []
                    for sym in symbols:
                        df_s = dfs.get(sym)
                        if df_s is not None and not df_s.empty:
                            latest_val = float(df_s[comp_indicator].iloc[-1]) if comp_indicator in df_s.columns else 0
                            rows.append({"Symbol": sym.replace(".NS", ""), comp_indicator: round(latest_val, 2)})
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
                else:
                    st.warning("Could not load data for any symbols")
            else:
                st.warning("Enter at least one symbol")

    # ══════════════════════════════════════════════
    # TAB 3: VOLUME PROFILE
    # ══════════════════════════════════════════════
    with chart_tab3:
        st.markdown("### 📦 Volume Profile — Volume at Price")
        st.caption("Shows which price levels have the highest trading volume")

        vp_ticker = st.text_input("Symbol", "RELIANCE.NS", key="vp_ticker")
        vp_period = st.selectbox("Data Period", ["3mo", "6mo", "1y", "2y"],
                                 index=2, key="vp_period")
        vp_bins = st.slider("Price Buckets", 10, 50, 30, 5, key="vp_bins",
                            help="More buckets = finer price resolution")

        if st.button("📊 Generate Volume Profile", key="vp_gen", width='stretch'):
            with st.spinner("Computing volume profile..."):
                df_vp = load_data(vp_ticker, period=vp_period)
                if df_vp is not None and not df_vp.empty:
                    df_vp = compute_indicators(df_vp)

                    from core.charts import plot_volume_profile
                    fig = plot_volume_profile(df_vp, vp_ticker)
                    if fig:
                        st.plotly_chart(fig, width='stretch')

                        # Volume metrics
                        from core.volume_profile import volume_profile, get_volume_metrics
                        vp = volume_profile(df_vp, num_bins=vp_bins, lookback=min(120, len(df_vp)))
                        if vp:
                            vm_cols = st.columns(5)
                            with vm_cols[0]:
                                st.metric("🎯 POC", f"₹{vp['poc']}")
                            with vm_cols[1]:
                                st.metric("📈 VAH", f"₹{vp['vah']}")
                            with vm_cols[2]:
                                st.metric("📉 VAL", f"₹{vp['val']}")
                            with vm_cols[3]:
                                st.metric("📊 Value Area", f"₹{vp['val']}–₹{vp['vah']}")
                            with vm_cols[4]:
                                st.metric("Total Vol", f"{vp['total_volume']/1e6:.1f}M")

                            # HVN / LVN
                            st.markdown("---")
                            hv_cols = st.columns(2)
                            with hv_cols[0]:
                                st.markdown("#### 🔴 High Volume Nodes (Support/Resistance)")
                                for node in vp["high_volume_nodes"][:5]:
                                    st.markdown(f"- **₹{node['price']:,.2f}** — {node['volume']:,.0f} shares")
                            with hv_cols[1]:
                                st.markdown("#### 🟢 Low Volume Nodes (Breakout Zones)")
                                for node in vp["low_volume_nodes"][:5]:
                                    st.markdown(f"- **₹{node['price']:,.2f}** — {node['volume']:,.0f} shares")

                            # VWAP info
                            st.markdown("---")
                            st.markdown("#### 📊 VWAP Analysis")
                            vwm = get_volume_metrics(df_vp)
                            if vwm:
                                vw_cols = st.columns(4)
                                with vw_cols[0]:
                                    st.metric("VWAP", f"₹{vwm.get('vwap', '—')}")
                                with vw_cols[1]:
                                    st.metric("VWAP Distance", f"{vwm.get('vwap_distance_pct', 0):+.2f}%",
                                              delta_color="normal" if vwm.get('above_vwap') else "inverse")
                                with vw_cols[2]:
                                    st.metric("Buy/Sell Ratio", vwm.get("buy_sell_ratio", "—"))
                                with vw_cols[3]:
                                    st.metric("Volume Trend", vwm.get("volume_trend", "—"))
                    else:
                        st.error("Could not compute volume profile")
                else:
                    st.error(f"Could not load {vp_ticker}")

# =========================================================
# PAGE: BACKTEST
# =========================================================
elif page == "Backtest":
    st.subheader("🧪 Profit Engine Backtest")

    bt_tab1, bt_tab2, bt_tab3, bt_tab4, bt_tab5 = st.tabs(["🔬 v6 Precision", "🏦 v5 Institutional", "🚀 v4 Precision", "📊 v3 (Legacy)", "⚖️ Comparison"])

    # ═══════════════════════════════════════
    # TAB 1: PROFIT ENGINE v6 (PRECISION)
    # ═══════════════════════════════════════
    with bt_tab1:
        st.markdown("### 🔬 v6 — Cost-Aware Precision Engine")
        st.caption("6 Entry Filters · 6 Exit Rules · Kelly Sizing · Chandelier Trail · Realistic SEBI Costs · Monte Carlo")

        v6_col1, v6_col2 = st.columns([2, 1])
        with v6_col1:
            v6_ticker = st.text_input("Symbol", value="RELIANCE.NS", key="v6_ticker")
        with v6_col2:
            v6_cap = st.number_input("Capital (₹)", 10000, 10000000, 100000, 50000, key="v6_cap")

        if st.button("🔬 Run v6 Backtest", key="v6_run", width='stretch', type="primary"):
            with st.spinner(f"Running v6 cost-aware backtest on {v6_ticker}..."):
                result = run_v6_backtest(v6_ticker, capital=v6_cap)

            if result and result.get("Total Trades", 0) > 0:
                m = result.get("metrics", {})
                # ── KPI Row ──
                kcols = st.columns(4)
                with kcols[0]:
                    pf = m.get("Profit Factor", 0)
                    st.metric("📊 Profit Factor", f"{pf:.2f}",
                              delta="✅ Institutional" if pf > 2.5 else ("✅ Good" if pf > 1.8 else "⚠️ Tune"),
                              delta_color="normal" if pf > 1.8 else "off")
                with kcols[1]:
                    sh = m.get("Sharpe Ratio", 0)
                    st.metric("🚀 Sharpe", f"{sh:.2f}",
                              delta="✅ Excellent" if sh > 2 else ("✅ Good" if sh > 1.5 else "⚠️"),
                              delta_color="normal" if sh > 1.5 else "off")
                with kcols[2]:
                    st.metric("💰 Total Return", f"{m.get('Total Return %', 0):+.2f}%")
                with kcols[3]:
                    st.metric("🏆 Win Rate", f"{m.get('Win Rate %', 0):.1f}%")

                kcols2 = st.columns(4)
                with kcols2[0]: st.metric("📉 Max DD", f"{m.get('Max Drawdown %', 0):.2f}%")
                with kcols2[1]: st.metric("🎯 Expectancy", f"{m.get('Expectancy %', 0):+.2f}%")
                with kcols2[2]: st.metric("Trades", m.get("Total Trades", 0))
                with kcols2[3]: st.metric("Final Capital", f"₹{result.get('Final Capital', 0):,.0f}")

                # ── Advanced Metrics ──
                st.markdown("---")
                st.markdown("#### 📊 Institutional Analytics")
                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    st.metric("Sortino", m.get("Sortino Ratio", "—"))
                    st.metric("Calmar", m.get("Calmar Ratio", "—"),
                              delta="✅ > 1.5" if m.get("Calmar Ratio", 0) > 1.5 else "⚠️")
                with a2:
                    st.metric("Payoff Ratio", m.get("Payoff Ratio", "—"),
                              delta="✅ > 2" if m.get("Payoff Ratio", 0) > 2 else "⚠️")
                    st.metric("Recovery Factor", m.get("Recovery Factor", "—"),
                              delta="✅ > 3" if m.get("Recovery Factor", 0) > 3 else "⚠️")
                with a3:
                    st.metric("Avg Win", f"{m.get('Avg Win %', 0):+.2f}%",
                              delta=f"{m.get('Avg Win Bars', 0)} bars")
                    st.metric("Avg Loss", f"{m.get('Avg Loss %', 0):.2f}%",
                              delta=f"{m.get('Avg Loss Bars', 0)} bars",
                              delta_color="inverse")
                with a4:
                    st.metric("Max Cons Losses", m.get("Max Consecutive Losses", "—"))
                    st.metric("Rolling Sharpe", f"{m.get('Rolling 20-Sharpe (Avg)', '—')}",
                              delta="⚠️ Degraded" if m.get("Rolling Sharpe < 0.5 Flag") else "✅ Stable")

                # ── Cost Breakdown ──
                st.markdown("---")
                st.markdown("#### 💰 SEBI Cost Model Applied")
                st.caption("STT 0.1% sell | Brokerage ₹20/0.03% | Exchange 0.00345% | SEBI 0.0001% | GST 18% | Stamp 0.015% buy")
                st.info("All costs deducted from every trade PnL before recording results.")

                # ── Exit Breakdown ──
                eb = m.get("Exit Breakdown", {})
                if eb:
                    st.markdown("---")
                    st.markdown("#### 🚪 Exit Reason Breakdown")
                    ecols = st.columns(min(len(eb), 6))
                    for i, (reason, count) in enumerate(sorted(eb.items(), key=lambda x: -x[1])[:6]):
                        with ecols[i % len(ecols)]:
                            pct = m.get("Exit % by Category", {}).get(reason, 0)
                            st.metric(reason, count, f"{pct:.1f}%")

                # ── Monte Carlo ──
                st.markdown("---")
                st.markdown("#### 🎲 Monte Carlo (2,000 runs)")
                trades_df = result.get("trades", pd.DataFrame())
                if not trades_df.empty:
                    bt = Backtester({"capital": v6_cap})
                    mc = bt.run_montecarlo(trades_df, n_runs=2000, confidence=0.90)
                    if mc.get("Simulations", 0) > 0:
                        mc_c = st.columns(4)
                        mc_c[0].metric("Median Return", f"{mc.get('Median Return %', 0):+.2f}%")
                        mc_c[1].metric("90% CI", f"{mc.get('CI Lower %', 0):+.2f}% to {mc.get('CI Upper %', 0):+.2f}%")
                        mc_c[2].metric("Prob Profit", f"{mc.get('Prob Profit %', 0):.1f}%",
                                       delta="✅ High" if mc.get('Prob Profit %', 0) > 80 else "⚠️")
                        mc_c[3].metric("Avg MaxDD (MC)", f"{mc.get('Avg MaxDD %', 0):.2f}%")

                # ── Trade Log ──
                if not trades_df.empty:
                    st.markdown("---")
                    st.subheader("📋 Trade Log")
                    disp = trades_df.copy()
                    if "Result" in disp.columns:
                        disp["Result"] = disp["Result"].apply(lambda v: "🟢 WIN" if v == "WIN" else "🔴 LOSS")
                    st.dataframe(disp, width='stretch', hide_index=True)

                    csv_data = trades_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Export CSV", csv_data, f"{v6_ticker}_v6_trades.csv", "text/csv", key="dl_v6")

            elif result and result.get("Total Trades", 0) == 0:
                if result.get("Error"):
                    st.error(f"❌ {result['Error']}")
                else:
                    st.warning("No trades generated. v6 has strict 6-filter entry — try a different symbol.")
            else:
                st.error(f"❌ {result.get('Error', 'Backtest failed')}")

    # ═══════════════════════════════════════
    # TAB 2: PROFIT ENGINE v5 (INSTITUTIONAL)
    # ═══════════════════════════════════════
    with bt_tab2:
        st.markdown("### 🏦 v5 — Institutional Precision")
        st.caption("8 Entry Filters · 5 Exit Rules · 7 Risk Upgrades · 16 Analytics · 5 Robustness Tests")

        col1, col2 = st.columns([2, 1])
        with col1:
            v5_ticker = st.text_input("Symbol", value="RELIANCE.NS", key="v5_ticker")
        with col2:
            v5_cap = st.number_input("Capital (₹)", 10000, 10000000, 100000, 50000, key="v5_cap")

        v5_cols = st.columns(4)
        with v5_cols[0]:
            v5_risk = st.slider("Base Risk %", 0.5, 5.0, 2.0, 0.25, key="v5_risk")
        with v5_cols[1]:
            v5_hold = st.slider("Max Hold Days", 5, 30, 15, 5, key="v5_hold")
        with v5_cols[2]:
            v5_mode = st.selectbox("Mode", ["All Filters", "Entry Filters Only", "Exit Rules Only", "Custom"], key="v5_mode")
        with v5_cols[3]:
            st.markdown("#### &nbsp;")
            v5_run = st.button("🏦 Run v5", key="v5_run", width='stretch', type="primary")

        # Advanced toggles
        with st.expander("⚙️ v5 Configuration — Toggle Each Filter", expanded=False):
            s1_col1, s1_col2, s1_col3, s1_col4 = st.columns(4)
            with s1_col1:
                st.markdown("**📥 Entry Filters**")
                f_vf = st.checkbox("Volume > 1.5x avg", False, key="v5_vf",
                                   help="Require RVOL > 1.5x 20d average")
                f_rs = st.checkbox("RS vs Nifty > 1.0", False, key="v5_rs",
                                   help="3-month relative strength > Nifty")
                f_52w = st.checkbox("Within 25% of 52w High", False, key="v5_52w")
            with s1_col2:
                f_stack = st.checkbox("EMA Stack (P>20>50>200)", False, key="v5_stack",
                                      help="Very strict: Price > EMA20 > EMA50 > EMA200")
                f_vwap = st.checkbox("Above VWAP", False, key="v5_vwap")
                f_breadth = st.checkbox("Market Breadth > 1.2", False, key="v5_breadth")
            with s1_col3:
                st.markdown("**📤 Exit Rules**")
                f_chandelier = st.checkbox("Chandelier Exit", False, key="v5_ch",
                                           help="HH(22) - 3x ATR trail (replaces basic trail)")
                f_vol_ex = st.checkbox("Volume Exhaustion", False, key="v5_ve")
                f_gap = st.checkbox("Gap-Up Exit (>3%)", False, key="v5_gap")
            with s1_col4:
                f_time = st.checkbox("Time Stop (dead money)", True, key="v5_ts",
                                     help="Exit if >15d without 1R profit")
                st.markdown("**⚡ Risk Upgrades**")
                f_heat = st.checkbox("Portfolio Heat Limit", False, key="v5_heat")
                f_corr = st.checkbox("Correlation Filter", False, key="v5_corr")
                f_cb = st.checkbox("Circuit Breaker", False, key="v5_cb")
                f_kelly = st.checkbox("Dynamic Kelly", False, key="v5_dk",
                                      help="Rolling 20-trade Kelly sizing")
                f_am = st.checkbox("Anti-Martingale", False, key="v5_am")

        if v5_run:
            with st.spinner(f"Running v5 institutional backtest on {v5_ticker}..."):
                result = run_v5_backtest(
                    v5_ticker, capital=v5_cap, risk_pct=v5_risk, max_hold=v5_hold,
                    use_volume_filter=f_vf, use_rs_filter=f_rs,
                    use_52w_filter=f_52w, use_ema_stack=f_stack,
                    use_vwap_filter=f_vwap, use_breadth_gate=f_breadth,
                    use_chandelier=f_chandelier, use_vol_exhaust=f_vol_ex,
                    use_gap_up_exit=f_gap, use_time_stop=f_time,
                    use_portfolio_heat=f_heat, use_correlation_filter=f_corr,
                    use_circuit_breaker=f_cb, use_dynamic_kelly=f_kelly,
                    use_anti_martingale=f_am,
                )

            if result and result.get("Total Trades", 0) > 0:
                # ── KPI Row ──
                mcols = st.columns(4)
                with mcols[0]:
                    pf = result['Profit Factor']
                    st.metric("📊 Profit Factor", f"{pf:.2f}",
                              delta="✅ Institutional" if pf > 2.5 else ("✅ Good" if pf > 1.8 else "⚠️ Tune"),
                              delta_color="normal" if pf > 1.8 else "off")
                with mcols[1]:
                    shp = result['Sharpe Ratio']
                    st.metric("🚀 Sharpe", f"{shp:.2f}",
                              delta="✅ Institutional" if shp > 2.5 else ("✅ Good" if shp > 1.8 else "⚠️"),
                              delta_color="normal" if shp > 1.8 else "off")
                with mcols[2]:
                    st.metric("💰 Total Return", f"{result['Total Return %']:+.2f}%")
                with mcols[3]:
                    st.metric("🏆 Win Rate", f"{result['Win Rate %']:.1f}%")

                mcols2 = st.columns(4)
                with mcols2[0]: st.metric("📉 Max DD", f"{result['Max Drawdown %']:.2f}%")
                with mcols2[1]: st.metric("🎯 Expectancy", f"{result['Expectancy %']:.2f}%")
                with mcols2[2]: st.metric("Trades", result["Total Trades"])
                with mcols2[3]: st.metric("Final Capital", f"₹{result['Final Capital']:,.0f}")

                # ── v5 Intelligence Metrics ──
                st.markdown("---")
                st.markdown("#### 🧠 v5 Intelligence")
                im = st.columns(6)
                with im[0]: st.metric("📅 Weekly", result.get("Weekly Trend", "N/A"))
                with im[1]: st.metric("🏛️ Regime", result.get("Market Regime", "N/A"))
                with im[2]: st.metric("🎭 HMM", result.get("HMM Regime", "N/A"))
                kf = result.get("Kelly Fraction", 0)
                with im[3]: st.metric("🎯 Kelly", f"{kf*100:.1f}%")
                dsr = result.get("Deflated Sharpe", 0)
                with im[4]: st.metric("🛡️ Defl Sharpe", f"{dsr:.2f}",
                                      delta="Robust" if dsr > 1.0 else "Fragile")
                with im[5]: st.metric("📐 Multiplier", f"{f_am and 'AM' or 'Fix'} / {'DK' if f_kelly else 'Fix'}")

                # ── Advanced Analytics ──
                adv = result.get("Advanced", {})
                if adv:
                    st.markdown("---")
                    st.markdown("#### 📊 Institutional Analytics (16 Metrics)")
                    a1, a2, a3, a4 = st.columns(4)
                    with a1:
                        st.metric("Sortino", adv.get("Sortino Ratio", "—"))
                        st.metric("Calmar", adv.get("Calmar Ratio", "—"))
                        st.metric("MAR", adv.get("MAR Ratio", "—"))
                    with a2:
                        st.metric("Omega", adv.get("Omega Ratio", "—"))
                        st.metric("Payoff Ratio", adv.get("Payoff Ratio", "—"),
                                  delta="✅ > 2" if adv.get("Payoff Ratio", 0) > 2 else "⚠️")
                        st.metric("Recovery Factor", adv.get("Recovery Factor", "—"),
                                  delta="✅ > 3" if adv.get("Recovery Factor", 0) > 3 else "⚠️")
                    with a3:
                        st.metric("Ulcer Index", adv.get("Ulcer Index", "—"),
                                  delta="Low" if adv.get("Ulcer Index", 999) < 10 else "High",
                                  delta_color="inverse")
                        st.metric("K-Ratio", adv.get("K-Ratio", "—"),
                                  delta="✅ > 1" if adv.get("K-Ratio", 0) > 1 else "⚠️")
                        st.metric("Tail Ratio", adv.get("Tail Ratio", "—"))
                    with a4:
                        st.metric("Max Loss Streak", adv.get("Max Loss Streak", "—"))
                        st.metric("Avg Loss Streak", adv.get("Avg Loss Streak", "—"))
                        st.metric("Rolling Sharpe", f"{adv.get('Rolling Sharpe (Avg)', '—')} ± {adv.get('Rolling Sharpe (Std)', '—')}")

                    # Duration analysis
                    st.markdown("#### ⏱️ Trade Duration")
                    d1, d2 = st.columns(2)
                    with d1:
                        st.metric("Avg Win Bars", adv.get("Avg Win Bars", "—"),
                                  delta="Holding winners longer ✅" if adv.get("Avg Win Bars", 0) > adv.get("Avg Loss Bars", 0) else "⚠️")
                    with d2:
                        st.metric("Avg Loss Bars", adv.get("Avg Loss Bars", "—"))

                # ── Exit Breakdown ──
                eb = result.get("Exit Breakdown", {})
                if eb:
                    st.markdown("---")
                    st.markdown("#### 🚪 Exit Reason Breakdown")
                    ecols = st.columns(min(len(eb), 6))
                    for i, (reason, count) in enumerate(sorted(eb.items(), key=lambda x: -x[1])[:6]):
                        with ecols[i % len(ecols)]:
                            st.metric(reason, count)

                # ── Trade Log ──
                trades_df = result.get("Trades")
                if trades_df is not None and not trades_df.empty:
                    st.markdown("---")
                    st.subheader("📋 Trade Log")
                    disp = trades_df.copy()
                    if "Result" in disp.columns:
                        disp["Result"] = disp["Result"].apply(lambda v: "🟢 WIN" if v == "WIN" else "🔴 LOSS")
                    st.dataframe(disp, width='stretch', hide_index=True)

                    # Export CSV
                    csv_raw = trades_df.copy()
                    if "Result" in csv_raw.columns:
                        csv_raw["Result"] = csv_raw["Result"].apply(lambda v: "WIN" if v == "WIN" else "LOSS")
                    csv_data = csv_raw.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Export CSV",
                        data=csv_data,
                        file_name=f"{v5_ticker}_v5_trades.csv",
                        mime="text/csv",
                        key="dl_v5"
                    )

            elif result and result.get("Total Trades", 0) == 0:
                if result.get("Error"):
                    st.error(f"❌ {result['Error']}")
                else:
                    st.warning("No trades generated. The filters may be too strict. Try 'Entry Filters Only' mode or disable some filters.")
            else:
                st.error(f"❌ {result.get('Error', 'Unknown error')}")

    # ═══════════════════════════════════════
    # TAB 3: PROFIT ENGINE v4 (PRECISION)
    # ═══════════════════════════════════════
    with bt_tab3:
        st.markdown("### 🏦 Institutional-Grade Backtest")
        st.caption("Weekly Trend Gate · ADX Filter · Partial Profit Booking · Kelly Sizing · Vol-Adaptive Stops · Monte Carlo")

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            v4_ticker = st.text_input("Symbol", value="RELIANCE.NS", key="v4_ticker")
        with col2:
            v4_cap = st.number_input("Capital (₹)", 10000, 10000000, 100000, 50000, key="v4_cap")
        with col3:
            v4_risk = st.slider("Risk/Trade %", 0.5, 5.0, 2.0, 0.5, key="v4_risk", format="%.1f%%")
        with col4:
            v4_hold = st.slider("Max Hold Days", 5, 40, 15, 5, key="v4_hold",
                                help="Exit after N bars to avoid dead money")

        with st.expander("⚙️ Advanced Toggles", expanded=False):
            adv_a, adv_b, adv_c, adv_d, adv_e = st.columns(5)
            with adv_a:
                use_weekly = st.checkbox("📅 Weekly Gate", True, key="v4_weekly",
                                         help="Skip stocks with bearish weekly trend")
            with adv_b:
                use_adx = st.checkbox("📈 ADX > 22", True, key="v4_adx",
                                      help="Only enter in trending markets")
            with adv_c:
                use_partial = st.checkbox("🔀 Partial Book", True, key="v4_partial",
                                          help="Book 40% at 1.5R, trail rest")
            with adv_d:
                use_kelly = st.checkbox("🎯 Kelly Sizing", True, key="v4_kelly",
                                        help="Kelly-optimal position sizing")
            with adv_e:
                use_vol_adapt = st.checkbox("🌊 Vol-Adapt Stops", True, key="v4_vol",
                                            help="Tighter stops in high volatility")

        if st.button("🚀 Run v4 Backtest", key="v4_run", width='stretch', type="primary"):
            with st.spinner(f"Running institutional backtest on {v4_ticker}..."):
                result = run_v4_backtest(
                    v4_ticker, capital=v4_cap, risk_pct=v4_risk,
                    max_hold=v4_hold, use_regime=True,
                    use_kelly=use_kelly, use_weekly_gate=use_weekly,
                    use_partial_booking=use_partial, use_adx_filter=use_adx,
                    use_vol_adapt=use_vol_adapt
                )

            if result and result.get("Total Trades", 0) > 0:
                # ── KPI Row ──
                mcols = st.columns(4)
                with mcols[0]:
                    pf = result['Profit Factor']
                    pf_delta = "✅ Institutional" if pf > 2 else ("✅ Good" if pf > 1.5 else "⚠️ Needs Tuning")
                    st.metric("📊 Profit Factor", f"{pf:.2f}", delta=pf_delta,
                              delta_color="normal" if pf > 1.5 else "off")
                with mcols[1]:
                    shp = result['Sharpe Ratio']
                    shp_delta = "✅ Institutional" if shp > 2 else ("✅ Good" if shp > 1.5 else "⚠️ Needs Work")
                    st.metric("🚀 Sharpe Ratio", f"{shp:.2f}", delta=shp_delta,
                              delta_color="normal" if shp > 1.5 else "off")
                with mcols[2]:
                    st.metric("💰 Total Return", f"{result['Total Return %']:+.2f}%")
                with mcols[3]:
                    st.metric("🏆 Win Rate", f"{result['Win Rate %']:.1f}%")

                mcols2 = st.columns(4)
                with mcols2[0]: st.metric("📉 Max DD", f"{result['Max Drawdown %']:.2f}%")
                with mcols2[1]: st.metric("🎯 Expectancy", f"{result['Expectancy %']:.2f}%")
                with mcols2[2]: st.metric("📈 Avg Win", f"{result['Avg Win %']:.2f}%")
                with mcols2[3]: st.metric("📉 Avg Loss", f"{result['Avg Loss %']:.2f}%")

                mcols3 = st.columns(4)
                with mcols3[0]: st.metric("Trades", result["Total Trades"])
                with mcols3[1]: st.metric("Final Capital", f"₹{result['Final Capital']:,.0f}")
                with mcols3[2]: st.metric("Best Trade", f"{result.get('Best Trade %', 0):+.2f}%")
                with mcols3[3]: st.metric("Worst Trade", f"{result.get('Worst Trade %', 0):+.2f}%")

                # ── v4 Specific Metrics ──
                st.markdown("---")
                st.markdown("#### 🧠 v4 Intelligence Metrics")
                im_cols = st.columns(4)
                with im_cols[0]:
                    st.metric("📅 Weekly Trend", result.get("Weekly Trend", "N/A"))
                with im_cols[1]:
                    st.metric("🏛️ Market Regime", result.get("Market Regime", "N/A"))
                with im_cols[2]:
                    st.metric("🎯 Kelly %", f"{result.get('Kelly Fraction', 0)*100:.1f}%",
                              help="Optimal fraction of capital per trade")
                with im_cols[3]:
                    wr = result.get("Win Rate %", 0)
                    aw = result.get("Avg Win %", 0)
                    al = result.get("Avg Loss %", 0)
                    kf = kelly_fraction(wr, aw, abs(al))
                    st.metric("📐 Suggested Kelly", f"{kf*100:.1f}%")

                # ── Monte Carlo ──
                mc = result.get("Monte Carlo", {})
                if mc and mc.get("Simulations", 0) > 0:
                    st.markdown("---")
                    st.markdown("#### 🎲 Monte Carlo Simulation (2,000 runs)")
                    mc_cols = st.columns(4)
                    with mc_cols[0]:
                        st.metric("Median Return", f"{mc.get('Median Return %', 0):+.2f}%")
                    with mc_cols[1]:
                        st.metric(f"90% CI Lower", f"{mc.get('CI Lower %', 0):+.2f}%",
                                  delta_color="inverse")
                    with mc_cols[2]:
                        st.metric(f"90% CI Upper", f"{mc.get('CI Upper %', 0):+.2f}%")
                    with mc_cols[3]:
                        st.metric("Prob of Profit", f"{mc.get('Prob Profit %', 0):.1f}%",
                                  delta="✅ High" if mc.get('Prob Profit %', 0) > 80 else "⚠️ Moderate")

                    mc_cols2 = st.columns(3)
                    with mc_cols2[0]:
                        st.metric("Avg MaxDD (MC)", f"{mc.get('Avg MaxDD %', 0):.2f}%")
                    with mc_cols2[1]:
                        st.metric("Worst MaxDD (MC)", f"{mc.get('Max MaxDD %', 0):.2f}%")
                    with mc_cols2[2]:
                        st.metric("Std Dev", f"{mc.get('Std Dev %', 0):.2f}%")

                # ── Exit Breakdown ──
                eb = result.get("Exit Breakdown", {})
                if eb:
                    st.markdown("---")
                    st.markdown("#### 🚪 Exit Reason Breakdown")
                    ecols = st.columns(len(eb))
                    for i, (reason, count) in enumerate(eb.items()):
                        with ecols[i]:
                            st.metric(reason, count)

                # ── Trade Log ──
                trades_df = result.get("Trades")
                if trades_df is not None and not trades_df.empty:
                    st.markdown("---")
                    st.subheader("📋 Trade Log")
                    disp = trades_df.copy()
                    if "Result" in disp.columns:
                        disp["Result"] = disp["Result"].apply(lambda v: "🟢 WIN" if v == "WIN" else "🔴 LOSS")
                    st.dataframe(disp, width='stretch', hide_index=True)

                    # Export CSV
                    csv_raw = trades_df.copy()
                    if "Result" in csv_raw.columns:
                        csv_raw["Result"] = csv_raw["Result"].apply(lambda v: "WIN" if v == "WIN" else "LOSS")
                    csv_data = csv_raw.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Export CSV",
                        data=csv_data,
                        file_name=f"{v4_ticker}_v4_trades.csv",
                        mime="text/csv",
                        key="dl_v4"
                    )

            elif result and result.get("Total Trades", 0) == 0:
                if result.get("Error"):
                    st.error(f"❌ {result['Error']}")
                else:
                    st.warning("No trades generated. Try a different symbol or relax the filters in Advanced Toggles.")
            else:
                st.error(f"❌ {result.get('Error', 'Unknown error')}")

    # ═══════════════════════════════════════
    # TAB 4: v3 (LEGACY)
    # ═══════════════════════════════════════
    with bt_tab4:
        st.markdown("### 🚀 Profit Engine v3 (Legacy)")
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            ticker = st.text_input("Symbol", value="RELIANCE.NS", key="bt_ticker")
        with col2:
            initial_capital = st.number_input("Capital (₹)", 10000, 10000000, 100000, 50000, key="bt_cap")
        with col3:
            risk_pct = st.slider("Risk/Trade %", 0.5, 5.0, 2.0, 0.5, key="bt_risk", format="%.1f%%")
        with col4:
            max_hold = st.slider("Max Hold Days", 5, 40, 15, 5, key="bt_hold",
                                 help="Exit after N bars to avoid dead money")

        st.caption("💡 **Profit Levers**: ATR Trailing Stop + Adaptive Market Regime + Volume Surge + Multi-Factor Entry")

        if st.button("🚀 Run v3 Backtest", key="bt_run", width='stretch'):
            with st.spinner(f"Running v3 backtest on {ticker}..."):
                result = run_v3_backtest(
                    ticker, capital=initial_capital, risk_pct=risk_pct,
                    max_hold=max_hold, use_regime=True
                )

            if result and "Total Trades" in result and result["Total Trades"] > 0:
                mcols = st.columns(4)
                with mcols[0]: st.metric("💰 Total Return", f"{result['Total Return %']:+.2f}%")
                with mcols[1]: st.metric("🏆 Win Rate", f"{result['Win Rate %']:.1f}%")
                with mcols[2]: st.metric("📊 Profit Factor", f"{result['Profit Factor']:.2f}",
                                          delta="Excellent" if result['Profit Factor'] > 2 else ("Good" if result['Profit Factor'] > 1.3 else "Needs work"))
                with mcols[3]: st.metric("🚀 Sharpe", f"{result['Sharpe Ratio']:.2f}")

                mcols2 = st.columns(4)
                with mcols2[0]: st.metric("📉 Max Drawdown", f"{result['Max Drawdown %']:.2f}%")
                with mcols2[1]: st.metric("🎯 Expectancy", f"{result['Expectancy %']:.2f}%")
                with mcols2[2]: st.metric("📈 Avg Win", f"{result['Avg Win %']:.2f}%")
                with mcols2[3]: st.metric("📉 Avg Loss", f"{result['Avg Loss %']:.2f}%")

                mcols3 = st.columns(4)
                with mcols3[0]: st.metric("Trades", result["Total Trades"])
                with mcols3[1]: st.metric("Final Capital", f"₹{result['Final Capital']:,.0f}")
                with mcols3[2]: st.metric("Best Trade", f"{result.get('Best Trade %', 0):+.2f}%")
                with mcols3[3]: st.metric("Worst Trade", f"{result.get('Worst Trade %', 0):+.2f}%")

                eb = result.get("Exit Breakdown", {})
                if eb:
                    st.markdown("---")
                    st.markdown("#### 🚪 Exit Reason Breakdown")
                    ecols = st.columns(len(eb))
                    for i, (reason, count) in enumerate(eb.items()):
                        with ecols[i]:
                            st.metric(reason, count)

                trades_df = result.get("Trades")
                if trades_df is not None and not trades_df.empty:
                    st.markdown("---")
                    st.subheader("📋 Trade Log")
                    disp = trades_df.copy()
                    if "Result" in disp.columns:
                        disp["Result"] = disp["Result"].apply(lambda v: "🟢 WIN" if v == "WIN" else "🔴 LOSS")
                    st.dataframe(disp, width='stretch', hide_index=True)

                    # Export CSV
                    csv_raw = trades_df.copy()
                    if "Result" in csv_raw.columns:
                        csv_raw["Result"] = csv_raw["Result"].apply(lambda v: "WIN" if v == "WIN" else "LOSS")
                    csv_data = csv_raw.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Export CSV",
                        data=csv_data,
                        file_name=f"{ticker}_v3_trades.csv",
                        mime="text/csv",
                        key="dl_v3"
                    )

            elif result and "Error" in result:
                st.error(f"❌ {result['Error']}")
            else:
                st.warning("No trades generated. Try a different symbol or adjust parameters.")

    # ═══════════════════════════════════════
    # TAB 3: OLD VS NEW COMPARISON
    # ═══════════════════════════════════════
    with bt_tab5:
        st.markdown("### ⚖️ Strategy Comparison")
        st.caption("Compare any two strategies side-by-side: Old Baseline, v3, v4, or v5")

        cmp_type = st.radio("Comparison Type",
                            ["v5 vs Old", "v5 vs v3", "v5 vs v4", "v4 vs Old", "v3 vs Old"],
                            horizontal=True, key="cmp_type")

        cmp_symbol = st.text_input("Symbol", value="RELIANCE.NS", key="cmp_sym")

        if st.button("⚖️ Compare", key="cmp_run", width='stretch'):
            from core.backtest import run_backtest as old_bt
            with st.spinner(f"Running {cmp_type} on {cmp_symbol}..."):
                old_data = old_bt(cmp_symbol) or {}
                if "v5" in cmp_type:
                    new_data = run_v5_backtest(cmp_symbol) or {}
                elif "v4" in cmp_type:
                    new_data = run_v4_backtest(cmp_symbol) or {}
                else:
                    new_data = run_v3_backtest(cmp_symbol) or {}

                if "Old" in cmp_type:
                    old_data_side = old_data
                elif "v4" in cmp_type and "v5" not in cmp_type:
                    old_data_side = run_v3_backtest(cmp_symbol) or {}
                elif "v5" in cmp_type and "v4" in cmp_type:
                    old_data_side = run_v4_backtest(cmp_symbol) or {}
                else:
                    old_data_side = old_data

                comp = {
                    "old": {
                        "Total Return %": old_data_side.get("Total Return %", 0),
                        "Win Rate %": old_data_side.get("Win Rate %", 0),
                        "Profit Factor": old_data_side.get("Profit Factor", 0),
                        "Max Drawdown %": old_data_side.get("Max Drawdown %", 0),
                        "Sharpe Ratio": old_data_side.get("Sharpe Ratio", 0),
                        "Total Trades": old_data_side.get("Total Trades", 0),
                    },
                    "new": {
                        "Total Return %": new_data.get("Total Return %", 0),
                        "Win Rate %": new_data.get("Win Rate %", 0),
                        "Profit Factor": new_data.get("Profit Factor", 0),
                        "Max Drawdown %": new_data.get("Max Drawdown %", 0),
                        "Sharpe Ratio": new_data.get("Sharpe Ratio", 0),
                        "Total Trades": new_data.get("Total Trades", 0),
                    },
                    "improvement": {
                        "PF Delta %": round((new_data.get("Profit Factor", 0) - old_data_side.get("Profit Factor", 0)) / max(old_data_side.get("Profit Factor", 0.1), 0.1) * 100, 1),
                        "Return Delta %": round(new_data.get("Total Return %", 0) - old_data_side.get("Total Return %", 0), 2),
                    },
                    "details": new_data
                }

            if comp and "old" in comp and "new" in comp:
                old = comp["old"]
                new = comp["new"]
                imp = comp["improvement"]

                st.markdown("#### 📊 Results Comparison")
                cmpc = st.columns([1, 1, 1])
                with cmpc[0]:
                    st.markdown("**Metric**")
                    st.markdown("Total Return")
                    st.markdown("Win Rate")
                    st.markdown("Profit Factor")
                    st.markdown("Max Drawdown")
                    st.markdown("Sharpe Ratio")
                    st.markdown("Total Trades")
                with cmpc[1]:
                    label = "v3" if "v3" in cmp_type else ("Old Baseline" if "v4" in cmp_type else "v3")
                    st.markdown(f"**{label}**")
                    st.markdown(f"{old['Total Return %']:+.2f}%")
                    st.markdown(f"{old['Win Rate %']:.1f}%")
                    st.markdown(f"{old['Profit Factor']:.2f}")
                    st.markdown(f"{old['Max Drawdown %']:.2f}%")
                    st.markdown(f"{old['Sharpe Ratio']:.2f}")
                    st.markdown(f"{old['Total Trades']}")
                with cmpc[2]:
                    label2 = "v4 🚀" if "v4" in cmp_type else "Profit Engine 🚀"
                    st.markdown(f"**{label2}**")
                    pf_color = ":green[" if new["Profit Factor"] > old["Profit Factor"] else ":red["
                    st.markdown(f"{pf_color}{new['Total Return %']:+.2f}%]")
                    st.markdown(f"{pf_color}{new['Win Rate %']:.1f}%]")
                    st.markdown(f"{pf_color}{new['Profit Factor']:.2f}]")
                    st.markdown(f"{pf_color}{new['Max Drawdown %']:.2f}%]")
                    st.markdown(f"{pf_color}{new['Sharpe Ratio']:.2f}]")
                    st.markdown(f"{pf_color}{new['Total Trades']}]")

                st.markdown("---")
                st.markdown("#### 📈 Improvement Summary")
                impcols = st.columns(3)
                impcols[0].metric("PF Δ", f"{imp['PF Delta %']:+.1f}%")
                impcols[1].metric("Return Δ", f"{imp['Return Delta %']:+.1f}%")
                pf_new = new["Profit Factor"]
                pf_improve = imp.get("PF Delta %", 0)
                impcols[2].metric("Result",
                                  "✅ Institutional" if pf_new > 2 else ("✅ Good" if pf_new > 1.3 else "⚠️ Tune"),
                                  delta=f"PF: {pf_new:.2f}")
                st.success(f"💡 PF changed by {pf_improve:+.1f}%")

                # v4-specific metrics
                if "v4" in cmp_type and "details" in comp:
                    det = comp["details"]
                    mc = det.get("Monte Carlo", {})
                    if mc and mc.get("Simulations", 0) > 0:
                        st.markdown("---")
                        st.markdown("#### 🎲 v4 Monte Carlo (2,000 runs)")
                        mc_c = st.columns(4)
                        mc_c[0].metric("Median Return", f"{mc.get('Median Return %', 0):+.2f}%")
                        mc_c[1].metric("90% CI", f"{mc.get('CI Lower %', 0):+.2f}% to {mc.get('CI Upper %', 0):+.2f}%")
                        mc_c[2].metric("Prob Profit", f"{mc.get('Prob Profit %', 0):.1f}%")
                        mc_c[3].metric("Avg MaxDD", f"{mc.get('Avg MaxDD %', 0):.2f}%")

                st.info(
                    "**v4 Features:**\n"
                    "- ✅ **Weekly Trend Gate** — Only trade bullish weekly structure\n"
                    "- ✅ **ADX > 22 Filter** — Trending markets only\n"
                    "- ✅ **Partial Profit Booking** — Book 40% at 1.5R, trail 60%\n"
                    "- ✅ **Volatility-Adaptive Stops** — Tighter in high vol, wider in low vol\n"
                    "- ✅ **Kelly Optimal Sizing** — f* = W - (1-W)/R, capped at 25%\n"
                    "- ✅ **Monte Carlo Simulation** — 90% CI on 2,000 randomized runs\n"
                    "- ✅ **Walk-Forward Ready** — 3yr train / 1yr test architecture"
                )
            else:
                st.warning("Comparison failed. Check the symbol or try a different stock.")

# =========================================================
# PAGE: MARKET DATA
# =========================================================
elif page == "Market Data":
    st.subheader("🏛️ Indian Market Data Hub")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 F&O Chain", "🌡️ India VIX", "🏦 FII/DII Flow",
        "📊 Market Breadth", "🏭 Sector Rotation"
    ])

    # ===== TAB 1: F&O Chain =====
    with tab1:
        st.markdown("### NIFTY Option Chain Overview")

        fo_col1, fo_col2 = st.columns([1, 1])
        with fo_col1:
            fo_symbol = st.selectbox("Index", ["NIFTY", "BANKNIFTY"], key="fo_symbol")
        with fo_col2:
            if st.button("🔄 Refresh F&O Data", key="fo_refresh"):
                clear_data_provider_cache()
                st.rerun()

        with st.spinner("Fetching F&O chain data..."):
            fo_data = get_fo_chain(fo_symbol)

        if fo_data and fo_data.get("pcr") is not None:
            metric_cols = st.columns(6)
            with metric_cols[0]:
                pcr = fo_data["pcr"]
                pcr_color = "normal" if pcr > 1.0 else "inverse"
                st.metric("📊 PCR (OI)", f"{pcr:.2f}", delta="Bullish" if pcr > 1.0 else "Bearish", delta_color=pcr_color)
            with metric_cols[1]:
                st.metric("📈 Total CE OI", f"{fo_data['total_ce_oi']/1e7:.2f} Cr")
            with metric_cols[2]:
                st.metric("📉 Total PE OI", f"{fo_data['total_pe_oi']/1e7:.2f} Cr")
            with metric_cols[3]:
                mp = fo_data.get("max_pain", "—")
                st.metric("🎯 Max Pain", mp)
            with metric_cols[4]:
                st.metric("💹 CE IV", f"{fo_data.get('iv_call', '—')}%")
            with metric_cols[5]:
                st.metric("💹 PE IV", f"{fo_data.get('iv_put', '—')}%")

            # Max OI strikes
            st.markdown("---")
            st.markdown("### 🔑 Key OI Levels")

            oi_cols = st.columns(2)
            with oi_cols[0]:
                max_ce = fo_data.get("max_oi_call", {})
                if max_ce:
                    st.markdown(
                        f"""<div class="card">
                        <div class="card-title">🔴 Max Call OI</div>
                        <div class="card-value">{max_ce.get('strike', '—')}</div>
                        <div class="card-sub">OI: {max_ce.get('oi', 0)/1e7:.2f} Cr — Resistance</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
            with oi_cols[1]:
                max_pe = fo_data.get("max_oi_put", {})
                if max_pe:
                    st.markdown(
                        f"""<div class="card">
                        <div class="card-title">🟢 Max Put OI</div>
                        <div class="card-value">{max_pe.get('strike', '—')}</div>
                        <div class="card-sub">OI: {max_pe.get('oi', 0)/1e7:.2f} Cr — Support</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

            # F&O Ban list
            st.markdown("---")
            st.markdown("### ⛔ Stocks in F&O Ban")
            ban_list = fo_data.get("fo_stocks_in_ban", [])
            if ban_list:
                ban_str = ", ".join(ban_list[:15])
                st.warning(f"🚫 {ban_str}{'...' if len(ban_list) > 15 else ''}")
            else:
                st.info("No stocks currently in F&O ban period.")

            # Expiry
            expiry = fo_data.get("expiry")
            if expiry:
                st.caption(f"📅 Next expiry: {expiry}")
        else:
            st.warning("F&O data unavailable. NSE may be blocking the request. Try clearing cache.")

    # ===== TAB 2: India VIX =====
    with tab2:
        st.markdown("### 🌡️ India VIX — Fear & Greed Index")

        with st.spinner("Fetching VIX data..."):
            vix_info = get_india_vix()

        if vix_info and vix_info.get("vix") is not None:
            vix_val = vix_info["vix"]

            # VIX gauge
            vix_cols = st.columns([1, 1, 1, 1, 1])
            regimes = [
                ("🟢 Low Vol", vix_val < 14, "< 14 — Complacency"),
                ("🟢 Normal", 14 <= vix_val < 18, "14-18 — Healthy"),
                ("🟡 Elevated", 18 <= vix_val < 22, "18-22 — Caution"),
                ("🔴 High Vol", 22 <= vix_val < 28, "22-28 — Fear"),
                ("🔴 Extreme", vix_val >= 28, "≥ 28 — Panic"),
            ]
            for i, (label, active, desc) in enumerate(regimes):
                with vix_cols[i]:
                    bg = "rgba(0,255,159,0.15)" if active else "rgba(255,255,255,0.05)"
                    border = "1px solid #00ff9f" if active else "1px solid #1f2937"
                    st.markdown(
                        f"<div style='background:{bg};border:{border};border-radius:8px;"
                        f"padding:12px;text-align:center;'>"
                        f"<div style='font-size:14px;'>{label}</div>"
                        f"<div style='font-size:11px;color:#6b7280;'>{desc}</div></div>",
                        unsafe_allow_html=True
                    )

            st.markdown("---")
            v_metrics = st.columns(4)
            with v_metrics[0]:
                st.metric("Current VIX", vix_val, delta=f"{vix_info.get('change', 0):+.2f}%")
            with v_metrics[1]:
                st.metric("Regime", vix_info.get("regime", "—").replace("_", " "))
            with v_metrics[2]:
                st.metric("52W High", vix_info.get("high_52w", "—"))
            with v_metrics[3]:
                st.metric("52W Low", vix_info.get("low_52w", "—"))

            # VIX interpretation
            regime = vix_info.get("regime", "UNKNOWN")
            if "LOW" in regime:
                st.success("✅ **Low Volatility** — Markets are complacent. Trend-following strategies work well.")
            elif regime == "NORMAL":
                st.info("✅ **Normal Volatility** — Healthy market conditions for swing trading.")
            elif "ELEVATED" in regime:
                st.warning("⚠️ **Elevated Volatility** — Increase position sizing caution. Consider wider stops.")
            elif "HIGH" in regime or "EXTREME" in regime:
                st.error("🔴 **High/Extreme Volatility** — High fear in markets. Reduce position size significantly.")

            # VIX Historical Chart
            st.markdown("---")
            st.markdown("### 📈 VIX Historical Trend")
            vix_df = get_vix_history("1y")
            if vix_df is not None and not vix_df.empty:
                vix_fig = go.Figure()
                vix_fig.add_trace(go.Scatter(
                    x=vix_df.index, y=vix_df["Close"],
                    mode="lines", name="VIX",
                    line=dict(color="#f5c542", width=2),
                    fill="tozeroy", fillcolor="rgba(245,197,66,0.1)"
                ))
                y_max = max(float(vix_df["Close"].max()) * 1.3, 30)
                vix_fig.add_hrect(y0=0, y1=14, fillcolor="green", opacity=0.05, line_width=0)
                vix_fig.add_hrect(y0=14, y1=18, fillcolor="green", opacity=0.03, line_width=0)
                vix_fig.add_hrect(y0=18, y1=22, fillcolor="yellow", opacity=0.04, line_width=0)
                vix_fig.add_hrect(y0=22, y1=28, fillcolor="red", opacity=0.04, line_width=0)
                vix_fig.add_hrect(y0=28, y1=y_max, fillcolor="red", opacity=0.08, line_width=0)
                vix_fig.update_layout(
                    height=400, template="plotly_dark",
                    paper_bgcolor="#050816", plot_bgcolor="#0a0e1a",
                    margin=dict(l=10, r=10, t=10, b=10),
                    hovermode="x unified", showlegend=False,
                    yaxis=dict(gridcolor="#1a1f2e", range=[0, y_max]),
                    xaxis=dict(gridcolor="#1a1f2e")
                )
                st.plotly_chart(vix_fig, width='stretch')
        else:
            st.warning("India VIX data not available. Check your internet connection.")

    # ===== TAB 3: FII/DII Flow =====
    with tab3:
        st.markdown("### 🏦 FII / DII Institutional Flow")

        with st.spinner("Fetching FII/DII data..."):
            fii_data = get_fii_dii_data()

        if fii_data and fii_data.get("fii_cash") is not None:
            fi_cols = st.columns(3)
            with fi_cols[0]:
                fii_cash = fii_data.get("fii_cash", 0)
                color = "normal" if fii_cash > 0 else "inverse"
                st.metric("🇮🇳 FII Cash Net", f"₹{fii_cash:+,.0f} Cr", delta_color=color)

            with fi_cols[1]:
                dii_cash = fii_data.get("dii_cash", 0)
                color = "normal" if dii_cash > 0 else "inverse"
                st.metric("🇮🇳 DII Cash Net", f"₹{dii_cash:+,.0f} Cr", delta_color=color)

            with fi_cols[2]:
                total = fii_data.get("net_combined", 0)
                color = "normal" if total > 0 else "inverse"
                st.metric("💰 Combined Net", f"₹{total:+,.0f} Cr", delta_color=color)

            st.markdown("---")
            st.markdown("### F&O Segment")
            fo_cols = st.columns(2)
            with fo_cols[0]:
                fii_fo = fii_data.get("fii_fo", 0)
                st.metric("FII F&O Net", f"₹{fii_fo:+,.0f} Cr")
            with fo_cols[1]:
                dii_fo = fii_data.get("dii_fo", 0)
                st.metric("DII F&O Net", f"₹{dii_fo:+,.0f} Cr")

            # Interpretation
            st.markdown("---")
            if total > 1000:
                st.success(f"✅ **Strong institutional buying** — Combined inflow of ₹{total:+,.0f} Cr suggests bullish sentiment.")
            elif total > 0:
                st.info(f"ℹ️ **Mild buying** — Combined inflow of ₹{total:+,.0f} Cr.")
            elif total > -1000:
                st.warning(f"⚠️ **Mild selling** — Combined outflow of ₹{total:+,.0f} Cr.")
            else:
                st.error(f"🔴 **Heavy institutional selling** — Combined outflow of ₹{total:+,.0f} Cr. Exercise caution.")

            if fii_data.get("date"):
                st.caption(f"Data as of: {fii_data['date']}")
        else:
            st.warning("FII/DII data unavailable. NSE may be blocking the request.")

    # ===== TAB 4: Market Breadth =====
    with tab4:
        st.markdown("### 📊 Market Breadth & Sentiment")

        with st.spinner("Fetching market breadth..."):
            breadth = get_market_breadth()

        if breadth and breadth.get("advances", 0) > 0:
            adv = breadth["advances"]
            dec = breadth["declines"]
            total = adv + dec + breadth.get("unchanged", 0)
            adr = breadth.get("advance_decline_ratio", 1.0)

            b_cols = st.columns(4)
            with b_cols[0]:
                st.metric("🟢 Advances", adv, delta=f"{adv/total*100:.1f}%" if total > 0 else None)
            with b_cols[1]:
                st.metric("🔴 Declines", dec, delta=f"{dec/total*100:.1f}%" if total > 0 else None)
            with b_cols[2]:
                st.metric("⚪ Unchanged", breadth.get("unchanged", 0))
            with b_cols[3]:
                st.metric("📊 A/D Ratio", f"{adr:.2f}")

            # Breadth gauge
            st.markdown("---")
            st.markdown("### Breadth Strength")
            strength = breadth.get("breadth_strength", "UNKNOWN")

            if strength == "STRONG":
                st.success(f"✅ **Strong Breadth** ({adr:.2f}) — More than 1.5 advancing stocks for every declining stock. Bullish.")
            elif strength == "MODERATE":
                st.info(f"ℹ️ **Moderate Breadth** ({adr:.2f}) — Slightly more advancers than decliners.")
            elif strength == "WEAK":
                st.warning(f"⚠️ **Weak Breadth** ({adr:.2f}) — Decliners outnumbering advancers.")
            else:
                st.error(f"🔴 **Bearish Breadth** ({adr:.2f}) — Strong decline pressure across the market.")

            st.progress(min(adr / 2.5, 1.0), text=f"A/D Ratio: {adr:.2f}")
        else:
            st.info("Market breadth data unavailable. Try during market hours.")

    # ===== TAB 5: Sector Rotation =====
    with tab5:
        st.markdown("### 🏭 Sector Rotation Heatmap")
        st.caption("Daily performance by sector — green = leading, red = lagging")

        with st.spinner("Computing sector performance..."):
            sector_perf = get_sector_performance()

        if sector_perf:
            # Gauge: how many sectors are positive
            pos_sectors = sum(1 for s in sector_perf if s["change_pct"] > 0)
            total_sectors = len(sector_perf)
            st.metric("Sectors Positive Today", f"{pos_sectors}/{total_sectors}",
                      f"{pos_sectors/total_sectors*100:.0f}%" if total_sectors > 0 else None)

            # Create the heatmap as a horizontal bar chart
            sectors_display = [s["sector"][:18] for s in sector_perf]
            changes = [s["change_pct"] for s in sector_perf]
            adv_pcts = [s.get("advance_pct", 50) for s in sector_perf]

            fig = go.Figure()
            for i, (sec, chg, adv) in enumerate(zip(sectors_display, changes, adv_pcts)):
                color = "#00ff9f" if chg > 0.5 else ("#66d9a0" if chg > 0 else ("#ff8c42" if chg > -0.5 else "#ff4d6d"))
                fig.add_trace(go.Bar(
                    x=[chg], y=[sec],
                    orientation="h",
                    marker_color=color,
                    name=sec,
                    text=[f"{chg:+.2f}%"],
                    textposition="outside",
                    hovertemplate=f"{sec}<br>Change: {chg:+.2f}%<br>Adv%: {adv:.0f}%<extra></extra>"
                ))

            fig.update_layout(
                height=max(400, len(sector_perf) * 28),
                template="plotly_dark",
                paper_bgcolor="#050816", plot_bgcolor="#0a0e1a",
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(gridcolor="#1a1f2e", title="Daily Change %"),
                yaxis=dict(gridcolor="#1a1f2e", autorange="reversed"),
                barmode="stack",
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')

            # Detailed sector table
            st.markdown("### 📋 Sector Detail")
            sec_df = pd.DataFrame(sector_perf)
            sec_df.columns = ["Sector", "Change %", "Strength", "Advance %", "Sample"]
            st.dataframe(sec_df, width='stretch', hide_index=True,
                         column_config={
                             "Change %": st.column_config.NumberColumn(format="%+.2f%%"),
                             "Advance %": st.column_config.NumberColumn(format="%.0f%%"),
                         })

            # Top/Bottom sectors
            top3 = sector_perf[:3]
            bot3 = sector_perf[-3:]
            sc_cols = st.columns(2)
            with sc_cols[0]:
                st.markdown("#### 🟢 Top 3 Sectors")
                for s in top3:
                    st.markdown(f"**{s['sector']}**: {s['change_pct']:+.2f}%")
            with sc_cols[1]:
                st.markdown("#### 🔴 Bottom 3 Sectors")
                for s in bot3:
                    st.markdown(f"**{s['sector']}**: {s['change_pct']:+.2f}%")
        else:
            st.info("Sector performance data unavailable. Try during market hours.")

# =========================================================
# PAGE: ANALYTICS
# =========================================================
elif page == "Analytics":
    st.subheader("🧠 Analytics & AI")

    atab1, atab2, atab3, atab4 = st.tabs([
        "📐 S/R Levels", "📊 Strategy Compare", "🤖 AI Scorer", "📈 Fib Levels"
    ])

    # ══════════════════════════════════════════════
    # TAB 1: SUPPORT & RESISTANCE
    # ══════════════════════════════════════════════
    with atab1:
        st.markdown("### 🔑 Support & Resistance Levels")
        sr_ticker = st.text_input("Stock Symbol", "RELIANCE.NS", key="sr_ticker")

        sr_col1, sr_col2 = st.columns([1, 2])
        with sr_col1:
            sr_lookback = st.slider("Lookback (days)", 30, 200, 100, 10, key="sr_look")
            sr_sensitivity = st.slider("Sensitivity", 2, 10, 5, 1, key="sr_sens",
                                       help="Lower = more swing points detected")

        if st.button("🔍 Detect Levels", key="sr_detect", width='stretch'):
            with st.spinner("Analyzing price structure..."):
                df = load_data(sr_ticker, period="1y")
                if df is not None and not df.empty:
                    df = compute_indicators(df)
                    sr = detect_support_resistance(df, lookback=sr_lookback, order=sr_sensitivity)
                    current = sr["current_price"]

                    st.success(f"Current Price: ₹{current:,.2f}")

                    # Key levels display
                    kcols = st.columns(3)
                    with kcols[0]:
                        st.metric("🟢 Nearest Support",
                                  f"₹{sr['nearest_support']:,.2f}" if sr['nearest_support'] else "—",
                                  f"{sr['dist_to_support_pct']:.2f}% away" if sr['dist_to_support_pct'] else None)
                    with kcols[1]:
                        st.metric("🔴 Nearest Resistance",
                                  f"₹{sr['nearest_resistance']:,.2f}" if sr['nearest_resistance'] else "—",
                                  f"{sr['dist_to_resistance_pct']:.2f}% away" if sr['dist_to_resistance_pct'] else None)
                    with kcols[2]:
                        range_pct = 0
                        if sr['supports'] and sr['resistances']:
                            range_pct = (max(sr['resistances']) / min(sr['supports']) - 1) * 100
                        st.metric("📊 Trading Range", f"{range_pct:.1f}%")

                    # Level visualization
                    st.markdown("#### Resistance Levels")
                    for i, r in enumerate(sr["resistances"][-5:]):
                        dist = (r / current - 1) * 100
                        bar_width = min(abs(dist) / 5, 1)
                        color = "#ff4d6d" if dist > 0 else "#00ff9f"
                        st.markdown(
                            f"<div style='display:flex;align-items:center;margin:4px 0;'>"
                            f"<span style='width:120px;'>₹{r:,.2f}</span>"
                            f"<div style='flex:1;height:8px;background:#1a1f2e;border-radius:4px;margin:0 10px;'>"
                            f"<div style='width:{bar_width*100:.0f}%;height:8px;background:{color};"
                            f"border-radius:4px;'></div></div>"
                            f"<span style='color:{color};'>{dist:+.2f}%</span></div>",
                            unsafe_allow_html=True
                        )

                    st.markdown("#### Support Levels")
                    for i, s in enumerate(sr["supports"][-5:]):
                        dist = (s / current - 1) * 100
                        bar_width = min(abs(dist) / 5, 1)
                        color = "#00ff9f" if dist < 0 else "#ff4d6d"
                        st.markdown(
                            f"<div style='display:flex;align-items:center;margin:4px 0;'>"
                            f"<span style='width:120px;'>₹{s:,.2f}</span>"
                            f"<div style='flex:1;height:8px;background:#1a1f2e;border-radius:4px;margin:0 10px;'>"
                            f"<div style='width:{bar_width*100:.0f}%;height:8px;background:{color};"
                            f"border-radius:4px;'></div></div>"
                            f"<span style='color:{color};'>{dist:+.2f}%</span></div>",
                            unsafe_allow_html=True
                        )

                    # Top important levels
                    st.markdown("#### ⭐ Most Tested Levels")
                    for level, touches in sr["important_levels"][:5]:
                        st.markdown(f"- **₹{level:,.2f}** — tested {touches} times")
                else:
                    st.error("Could not load data")

    # ══════════════════════════════════════════════
    # TAB 2: STRATEGY COMPARISON (ROBUST)
    # ══════════════════════════════════════════════
    with atab2:
        st.markdown("### 📊 Strategy Comparison — v3 vs v4 vs v5")
        st.caption("Side-by-side institutional comparison with Monte Carlo confidence intervals")

        comp_ticker = st.text_input("Symbol", "RELIANCE.NS", key="comp_ticker")
        comp_cap = st.number_input("Capital (₹)", 10000, 10000000, 100000, 50000, key="comp_cap")

        comp_cols = st.columns(3)
        with comp_cols[0]:
            run_v3 = st.checkbox("📊 v3 (Legacy)", True, key="cmp_v3")
        with comp_cols[1]:
            run_v4 = st.checkbox("🚀 v4 (Precision)", True, key="cmp_v4")
        with comp_cols[2]:
            run_v5 = st.checkbox("🏦 v5 (Institutional)", True, key="cmp_v5")

        if st.button("🔄 Deep Compare", key="comp_run", width='stretch', type="primary"):
            selected = []
            if run_v3: selected.append("v3")
            if run_v4: selected.append("v4")
            if run_v5: selected.append("v5")

            if not selected:
                st.warning("Select at least one strategy to compare.")
            else:
                all_results = {}
                with st.spinner("Running v3 backtest..."):
                    if "v3" in selected:
                        from core.profit_engine import run_v3_backtest
                        r3 = run_v3_backtest(comp_ticker, capital=comp_cap)
                        all_results["v3 (Legacy)"] = r3

                with st.spinner("Running v4 backtest..."):
                    if "v4" in selected:
                        from core.profit_engine_v4 import run_v4_backtest
                        r4 = run_v4_backtest(comp_ticker, capital=comp_cap)
                        all_results["v4 (Precision)"] = r4

                with st.spinner("Running v5 backtest..."):
                    if "v5" in selected:
                        from core.profit_engine_v5 import run_v5_backtest
                        r5 = run_v5_backtest(comp_ticker, capital=comp_cap)
                        all_results["v5 (Institutional)"] = r5

                if all_results:
                    # ── Master Comparison Table ──
                    st.markdown("---")
                    st.markdown("#### 🏆 Multi-Strategy Scorecard")
                    metrics_table = []
                    for name, r in all_results.items():
                        adv = r.get("Advanced", {})
                        mc = r.get("Monte Carlo", {}) or adv.get("Monte Carlo", {})
                        wf = r.get("WalkForward", {})

                        row = {
                            "Strategy": name,
                            "Trades": r.get("Total Trades", 0),
                            "Return%": r.get("Total Return %", 0),
                            "Win Rate%": r.get("Win Rate %", 0),
                            "PF": r.get("Profit Factor", 0),
                            "Sharpe": r.get("Sharpe Ratio", 0),
                            "MaxDD%": r.get("Max Drawdown %", 0),
                            "Sortino": adv.get("Sortino Ratio", "—"),
                            "Calmar": adv.get("Calmar Ratio", "—"),
                            "Payoff": adv.get("Payoff Ratio", "—"),
                            "Omega": adv.get("Omega Ratio", "—"),
                            "K-Ratio": adv.get("K-Ratio", "—"),
                            "Recovery": adv.get("Recovery Factor", "—"),
                            "Kelly%": r.get("Kelly Fraction", 0),
                            "MC_Med%": mc.get("Median Return %", "—"),
                            "MC_Profit%": mc.get("Prob Profit %", "—"),
                            "Regime": r.get("Market Regime", "—"),
                        }
                        metrics_table.append(row)

                    df_comp = pd.DataFrame(metrics_table).set_index("Strategy")
                    st.dataframe(df_comp, width='stretch')

                    # ── Best-in-Class Highlights ──
                    st.markdown("#### 🥇 Best-in-Class")
                    num = len(all_results)
                    bcols = st.columns(4)
                    metrics_to_check = [
                        ("Return%", "💰 Best Return", "+"),
                        ("Sharpe", "🚀 Best Sharpe", "+"),
                        ("PF", "📊 Best PF", "+"),
                        ("MaxDD%", "🛡️ Lowest DD", "-"),
                    ]
                    for i, (metric, label, direction) in enumerate(metrics_to_check):
                        with bcols[i]:
                            vals = {name: r.get(metric if metric != "MaxDD%" else "Max Drawdown %", 0) for name, r in all_results.items()}
                            if direction == "+":
                                best_name = max(vals, key=vals.get)
                                best_val = vals[best_name]
                            else:
                                best_name = min(vals, key=vals.get)
                                best_val = vals[best_name]
                            st.metric(label, f"{best_name}", f"{best_val:+.2f}" if isinstance(best_val, float) else str(best_val))

                    # ── Monte Carlo Comparison ──
                    st.markdown("---")
                    st.markdown("#### 🎲 Monte Carlo Comparison (Confidence Intervals)")
                    mc_cols = st.columns(len(all_results))
                    for i, (name, r) in enumerate(all_results.items()):
                        adv = r.get("Advanced", {})
                        mc = r.get("Monte Carlo", {}) or adv.get("Monte Carlo", {})
                        with mc_cols[i]:
                            st.markdown(f"**{name}**")
                            med = mc.get("Median Return %", "—")
                            ci_lo = mc.get("CI Lower %", "—")
                            ci_hi = mc.get("CI Upper %", "—")
                            prob = mc.get("Prob Profit %", "—")
                            st.metric("Median Return", f"{med:+.2f}%" if isinstance(med, (int, float)) else med)
                            st.metric("90% CI", f"{ci_lo:+.2f}% to {ci_hi:+.2f}%" if isinstance(ci_lo, (int, float)) else "—")
                            st.metric("Prob of Profit", f"{prob:.1f}%" if isinstance(prob, (int, float)) else "—",
                                      delta="✅ High" if isinstance(prob, (int, float)) and prob > 80 else "⚠️ Moderate")

                    # ── Walk-Forward Robustness ──
                    st.markdown("---")
                    st.markdown("#### 🔬 Walk-Forward Robustness Check")
                    for name, r in all_results.items():
                        wf = r.get("WalkForward", {})
                        if wf and "Folds" in wf:
                            folds = wf.get("Folds", [])
                            is_sharpe = wf.get("Avg IS Sharpe", "—")
                            oos_sharpe = wf.get("Avg OOS Sharpe", "—")
                            ratio = wf.get("OOS/IS Ratio %", "—")
                            overfit = wf.get("Overfit Flag", False)
                            st.markdown(
                                f"- **{name}**: IS Sharpe {is_sharpe} → OOS Sharpe {oos_sharpe} "
                                f"({ratio}% retention) {'⚠️ OVERFIT' if overfit else '✅ Robust'}"
                            )
                        else:
                            st.markdown(f"- **{name}**: Walk-forward data not available")

                    # ── Final Verdict ──
                    st.markdown("---")
                    st.markdown("#### 🏅 Final Verdict")
                    best_overall = max(all_results.items(),
                                       key=lambda x: x[1].get("Profit Factor", 0) * x[1].get("Sharpe Ratio", 0))
                    st.success(
                        f"**{best_overall[0]}** scores highest overall "
                        f"(PF {best_overall[1].get('Profit Factor', 0):.2f} × "
                        f"Sharpe {best_overall[1].get('Sharpe Ratio', 0):.2f} = "
                        f"{best_overall[1].get('Profit Factor', 0) * best_overall[1].get('Sharpe Ratio', 0):.2f}). "
                        f"Consider this your primary strategy."
                    )

                    # ── Trade Log Export ──
                    st.markdown("---")
                    with st.expander("📋 Export Individual Trade Logs", expanded=False):
                        for name, r in all_results.items():
                            tdf = r.get("Trades")
                            if tdf is not None and not tdf.empty:
                                csv_data = tdf.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label=f"📥 {name} Trades CSV",
                                    data=csv_data,
                                    file_name=f"{comp_ticker}_{name.replace(' ', '_')}_trades.csv",
                                    mime="text/csv",
                                    key=f"dl_cmp_{name[:3]}"
                                )
                else:
                    st.error("All backtests failed. Check the symbol or try again.")

    # ══════════════════════════════════════════════
    # TAB 3: AI SCORER
    # ══════════════════════════════════════════════
    with atab3:
        st.markdown("### 🤖 AI Signal Scorer")
        st.caption("Machine learning model that scores trading signals based on technical patterns")

        ai_ticker = st.text_input("Stock", "RELIANCE.NS", key="ai_ticker")

        if not SKLEARN_AVAILABLE:
            st.warning("⚠️ scikit-learn not installed. Run: pip install scikit-learn")
            st.info("Using rule-based scoring instead (no ML required)")

        ai_col1, ai_col2 = st.columns([1, 1])

        with ai_col1:
            ai_period = st.selectbox("Training Period", ["1y", "2y", "3y"], index=1, key="ai_period")

        with ai_col2:
            if st.button("🧠 Score Signal", key="ai_score", width='stretch'):
                with st.spinner("Loading data and computing score..."):
                    df = load_data(ai_ticker, period=ai_period)
                    if df is not None and not df.empty and len(df) > 60:
                        df = compute_indicators(df)
                        features = _extract_features(df)

                        if features:
                            # Get rule-based score (always works)
                            rule_score = get_rule_based_score(features)

                            # Try ML score
                            ml_score = None
                            if SKLEARN_AVAILABLE:
                                model_data = train_ai_scorer(df)
                                if model_data and model_data.get("available"):
                                    ml_score = score_signal_ai(features, model_data)

                            # Display results
                            sc_cols = st.columns(2)

                            with sc_cols[0]:
                                score = rule_score["score"]
                                signal = rule_score["signal"]
                                color = "#00ff9f" if score >= 55 else ("#ff4d6d" if score < 40 else "#f5c542")
                                st.markdown(
                                    f"<div style='background:#111827;border:2px solid {color};"
                                    f"border-radius:16px;padding:24px;text-align:center;'>"
                                    f"<div style='font-size:14px;color:#6b7280;'>Rule-Based Score</div>"
                                    f"<div style='font-size:48px;font-weight:bold;color:{color};'>{score}</div>"
                                    f"<div style='font-size:18px;color:{color};'>{signal}</div>"
                                    f"<div style='font-size:11px;color:#6b7280;margin-top:8px;'>"
                                    f"{rule_score['reasons']}</div></div>",
                                    unsafe_allow_html=True
                                )

                            with sc_cols[1]:
                                if ml_score:
                                    ml_color = "#00ff9f" if ml_score["score"] >= 55 else "#ff4d6d"
                                    st.markdown(
                                        f"<div style='background:#111827;border:2px solid {ml_color};"
                                        f"border-radius:16px;padding:24px;text-align:center;'>"
                                        f"<div style='font-size:14px;color:#6b7280;'>AI ML Score</div>"
                                        f"<div style='font-size:48px;font-weight:bold;color:{ml_color};'>"
                                        f"{ml_score['score']:.0f}</div>"
                                        f"<div style='font-size:18px;color:{ml_color};'>"
                                        f"{ml_score['confidence']}</div>"
                                        f"<div style='font-size:11px;color:#6b7280;margin-top:8px;'>"
                                        f"Buy Prob: {ml_score.get('probability_buy', 0):.1f}%</div></div>",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.info("ML model requires scikit-learn. Install it for AI-powered scoring.")

                            # Feature details
                            st.markdown("---")
                            st.markdown("#### 📋 Feature Analysis")
                            feat_cols = st.columns(3)
                            feat_display = [
                                ("RSI", features.get("rsi", 0), "Momentum", 30, 70),
                                ("RVOL", features.get("rvol", 0), "Volume", 0.7, 1.5),
                                ("ADX", features.get("adx", 0), "Trend", 20, 25),
                                ("MACD Hist", features.get("macd_hist", 0), "Momentum", None, 0),
                                ("ROC 5D", features.get("roc_5d", 0), "Momentum", -3, 3),
                                ("BB Pos", features.get("bb_pos", 0.5), "Structure", 0.2, 0.8),
                                ("Dist EMA50", features.get("dist_ema50", 0), "Trend", None, 0),
                                ("Dist EMA200", features.get("dist_ema200", 0), "Trend", None, 0),
                                ("Vol Ratio", features.get("vol_ratio", 1), "Volume", 0.8, 1.3),
                            ]
                            for i, (name, val, cat, low, high) in enumerate(feat_display):
                                with feat_cols[i % 3]:
                                    is_good = True
                                    if low is not None and high is not None:
                                        is_good = low <= val <= high
                                    elif val > 0:
                                        is_good = True
                                    else:
                                        is_good = False
                                    color = "#00ff9f" if is_good else "#ff4d6d"
                                    st.markdown(
                                        f"<div style='display:flex;justify-content:space-between;"
                                        f"padding:4px 0;border-bottom:1px solid #1f2937;'>"
                                        f"<span>{name}</span>"
                                        f"<span style='color:{color};'>{val:.2f}</span></div>",
                                        unsafe_allow_html=True
                                    )
                        else:
                            st.error("Could not extract features — insufficient data")
                    else:
                        st.error("Insufficient data (need 60+ bars)")

    # ══════════════════════════════════════════════
    # TAB 4: FIBONACCI LEVELS
    # ══════════════════════════════════════════════
    with atab4:
        st.markdown("### 📈 Fibonacci Levels")
        st.caption("Calculate Fibonacci retracement and extension levels")

        fib_ticker = st.text_input("Symbol", "RELIANCE.NS", key="fib_ticker")

        fib_method = st.radio("Method", ["Auto (Swing High/Low)", "Manual Entry"], horizontal=True, key="fib_method")

        if fib_method == "Manual Entry":
            fib_cols = st.columns(2)
            with fib_cols[0]:
                fib_high = st.number_input("Swing High (₹)", min_value=0.01, value=3500.0, step=10.0, key="fib_high")
            with fib_cols[1]:
                fib_low = st.number_input("Swing Low (₹)", min_value=0.01, value=2800.0, step=10.0, key="fib_low")

            if st.button("📐 Calculate Fib", key="fib_calc", width='stretch'):
                fib = fibonacci_levels(fib_high, fib_low)
                st.success(f"Range: ₹{fib['low']} → ₹{fib['high']} (₹{fib['range']})")

                fib_cols = st.columns(2)
                with fib_cols[0]:
                    st.markdown("#### 🔄 Retracement Levels")
                    for level, price in fib["retracement"].items():
                        st.markdown(f"**{level}**: ₹{price:,.2f}")
                with fib_cols[1]:
                    st.markdown("#### 🚀 Extension Levels")
                    for level, price in fib["extension"].items():
                        st.markdown(f"**{level}**: ₹{price:,.2f}")
        else:
            if st.button("🔍 Auto Detect Swing Points", key="fib_auto", width='stretch'):
                with st.spinner("Detecting swing points..."):
                    df = load_data(fib_ticker, period="1y")
                    if df is not None and not df.empty:
                        df = compute_indicators(df)
                        sr = detect_support_resistance(df, lookback=60, order=3)
                        current = float(df["Close"].iloc[-1])

                        if sr["swing_highs"] and sr["swing_lows"]:
                            # Use nearest swing high above and swing low below
                            recent_highs = [p for _, p in sr["swing_highs"] if p > current]
                            recent_lows = [p for _, p in sr["swing_lows"] if p < current]

                            if recent_highs and recent_lows:
                                high = min(recent_highs)
                                low = max(recent_lows)

                                fib = fibonacci_levels(high, low, current_price=current)
                                st.success(f"Swing Low: ₹{low:,.2f} → Swing High: ₹{high:,.2f}")

                                fcols = st.columns(3)
                                with fcols[0]:
                                    st.markdown("#### 🔄 Retracement")
                                    for level, price in fib["retracement"].items():
                                        color = "#00ff9f" if price <= current <= price * 1.01 else "#6b7280"
                                        arrow = "◀ Current" if fib.get("current_zone") and level in fib["current_zone"] else ""
                                        st.markdown(
                                            f"<span style='color:{color};'>**{level}**: ₹{price:,.2f} {arrow}</span>",
                                            unsafe_allow_html=True
                                        )
                                with fcols[1]:
                                    st.markdown("#### 🚀 Extension")
                                    for level, price in fib["extension"].items():
                                        st.markdown(f"**{level}**: ₹{price:,.2f}")
                                with fcols[2]:
                                    st.markdown("#### 📍 Current Zone")
                                    st.info(fib.get("current_zone", "Outside key levels"))
                                    st.metric("Current Price", f"₹{current:,.2f}")
                            else:
                                st.warning("Could not find suitable swing points")
                        else:
                            st.warning("Not enough swing points detected")
                    else:
                        st.error("Could not load data")
elif page == "Portfolio":
    st.subheader("⚡ Execution Hub")

    exec_tab1, exec_tab2, exec_tab3, exec_tab4, exec_tab5 = st.tabs([
        "📋 Watchlists", "📝 Trade Journal", "📊 Portfolio",
        "📐 Position Sizer", "🎯 Trading (Shoonya)"
    ])

    # ══════════════════════════════════════════════
    # TAB 1: WATCHLISTS
    # ══════════════════════════════════════════════
    with exec_tab1:
        st.markdown("### 📋 Watchlist Manager")

        wl_col1, wl_col2, wl_col3 = st.columns([2, 1, 1])

        with wl_col1:
            watchlists = get_watchlists()
            selected_wl = st.selectbox("Select Watchlist", watchlists, key="wl_select")

        with wl_col2:
            new_wl_name = st.text_input("New Watchlist Name", "", key="wl_new")
            if st.button("➕ Create", key="wl_create") and new_wl_name:
                if create_watchlist(new_wl_name):
                    st.success(f"Created '{new_wl_name}'")
                    st.rerun()
                else:
                    st.warning("Watchlist already exists")

        with wl_col3:
            if st.button("🗑️ Delete Current", key="wl_del"):
                if delete_watchlist(selected_wl):
                    st.success(f"Deleted '{selected_wl}'")
                    st.rerun()

        # Add/Remove stocks
        st.markdown("#### Manage Stocks")
        add_remove_cols = st.columns([2, 1, 1])
        with add_remove_cols[0]:
            stock_to_add = st.text_input("Add Stock (e.g., RELIANCE.NS)", "", key="wl_add")
        with add_remove_cols[1]:
            if st.button("➕ Add", key="wl_add_btn") and stock_to_add:
                symbol = stock_to_add.strip().upper()
                if not symbol.endswith(".NS") and "^" not in symbol and ".BO" not in symbol:
                    symbol = f"{symbol}.NS"
                if add_to_watchlist(symbol, selected_wl):
                    st.success(f"Added {symbol}")
                    st.rerun()
        with add_remove_cols[2]:
            wl_stocks = get_watchlist(selected_wl)
            if wl_stocks:
                stock_to_remove = st.selectbox("Remove Stock", [""] + [s.replace(".NS", "") for s in wl_stocks], key="wl_remove")
                if st.button("✖️ Remove", key="wl_rem_btn") and stock_to_remove:
                    remove_from_watchlist(f"{stock_to_remove}.NS", selected_wl)
                    st.rerun()

        # Display watchlist with prices
        st.markdown("#### Live Prices")
        wl_df = get_watchlist_with_prices(selected_wl)
        if wl_df is not None and not wl_df.empty:
            def color_signal(s):
                return f"🟢 {s}" if s == "BUY" else (f"🔴 {s}" if s == "SELL" else f"🟡 {s}")
            wl_df["Signal"] = wl_df["Signal"].apply(color_signal)

            st.dataframe(wl_df, width='stretch', hide_index=True,
                         column_config={
                             "Change%": st.column_config.NumberColumn(format="%+.2f%%"),
                             "Close": st.column_config.NumberColumn(format="₹%.2f"),
                             "RSI": st.column_config.NumberColumn(format="%.1f"),
                             "Score": st.column_config.NumberColumn(format="%d"),
                             "Confidence": st.column_config.NumberColumn(format="%d"),
                         })

            csv = wl_df.to_csv(index=False)
            st.download_button("📥 Export CSV", csv,
                               f"watchlist_{selected_wl}.csv", "text/csv")
        else:
            st.caption("Watchlist is empty or prices unavailable. Add stocks above.")

    # ══════════════════════════════════════════════
    # TAB 2: TRADE JOURNAL
    # ══════════════════════════════════════════════
    with exec_tab2:
        st.markdown("### 📝 Trade Journal")

        jtab1, jtab2 = st.tabs(["➕ New Trade", "📋 Trade History"])

        # --- New Trade Form ---
        with jtab1:
            with st.form("new_trade_form"):
                form_cols = st.columns(3)
                with form_cols[0]:
                    sym = st.text_input("Symbol", "RELIANCE.NS").strip().upper()
                    if not sym.endswith(".NS") and "^" not in sym:
                        sym = f"{sym}.NS"
                    direction = st.selectbox("Direction", ["LONG", "SHORT"])
                    qty = st.number_input("Quantity", min_value=1, value=100, step=10)
                with form_cols[1]:
                    entry_price = st.number_input("Entry Price (₹)", min_value=0.01, value=3000.0, step=1.0)
                    stop_loss = st.number_input("Stop Loss (₹)", min_value=0.0, value=2850.0, step=1.0)
                    target = st.number_input("Target (₹)", min_value=0.0, value=3400.0, step=1.0)
                with form_cols[2]:
                    entry_date = st.date_input("Entry Date", value=datetime.now())
                    tags = st.text_input("Tags (comma separated)", "swing, momentum")
                    notes = st.text_area("Notes", "Reason for entry...")

                submitted = st.form_submit_button("💾 Save Trade", width='stretch')
                if submitted:
                    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                    add_trade(sym, entry_price, qty, direction,
                              str(entry_date), stop_loss, target, tag_list, notes)
                    st.success(f"✅ Trade {sym} logged!")
                    st.rerun()

        # --- Trade History ---
        with jtab2:
            all_trades_df = get_all_trades()
            open_trades_df = get_open_trades()

            if not all_trades_df.empty:
                # Show open positions first
                if not open_trades_df.empty:
                    st.markdown("#### 🟢 Open Positions")
                    display_open = open_trades_df[["id", "symbol", "direction", "entry_price",
                                                    "live_price", "live_pnl", "live_pnl_pct",
                                                    "stop_loss", "target"]].copy()
                    display_open["symbol"] = display_open["symbol"].str.replace(".NS", "")
                    display_open.columns = ["ID", "Symbol", "Dir", "Entry", "LTP",
                                            "P&L", "P&L%", "SL", "Target"]

                    st.dataframe(display_open, width='stretch', hide_index=True,
                                 column_config={
                                     "P&L": st.column_config.NumberColumn(format="₹%.2f"),
                                     "P&L%": st.column_config.NumberColumn(format="%+.2f%%"),
                                     "Entry": st.column_config.NumberColumn(format="₹%.2f"),
                                     "LTP": st.column_config.NumberColumn(format="₹%.2f"),
                                     "SL": st.column_config.NumberColumn(format="₹%.2f"),
                                     "Target": st.column_config.NumberColumn(format="₹%.2f"),
                                 })

                    # Close trade
                    close_opts = [f"{r['symbol']} ({r['id']})" for _, r in open_trades_df.iterrows()]
                    close_cols = st.columns([2, 1, 1])
                    with close_cols[0]:
                        close_sel = st.selectbox("Close Trade", [""] + close_opts, key="close_sel")
                    with close_cols[1]:
                        exit_price = st.number_input("Exit Price", min_value=0.01, value=0.0, step=1.0, key="exit_pr")
                    with close_cols[2]:
                        if st.button("✖️ Close Trade", key="close_btn") and close_sel and exit_price > 0:
                            tid = close_sel.split("(")[-1].rstrip(")")
                            close_trade(tid, exit_price)
                            st.success("Trade closed!")
                            st.rerun()

                # All trades history
                st.markdown("#### 📋 All Trades")
                all_display = all_trades_df.copy()
                all_display["symbol"] = all_display["symbol"].str.replace(".NS", "")
                display_cols = ["symbol", "direction", "entry_price", "exit_price",
                                "quantity", "pnl", "pnl_pct", "result", "entry_date", "exit_date", "tags"]
                display_cols = [c for c in display_cols if c in all_display.columns]

                st.dataframe(all_display[display_cols], width='stretch', hide_index=True,
                             column_config={
                                 "pnl": st.column_config.NumberColumn("P&L", format="₹%.2f"),
                                 "pnl_pct": st.column_config.NumberColumn("P&L%", format="%+.2f%%"),
                                 "entry_price": st.column_config.NumberColumn("Entry", format="₹%.2f"),
                                 "exit_price": st.column_config.NumberColumn("Exit", format="₹%.2f"),
                                 "quantity": "Qty",
                             })

                # Export
                csv_data = export_journal()
                if csv_data:
                    st.download_button("📥 Export Journal (CSV)", csv_data,
                                       f"trade_journal_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

                # Delete trade
                with st.expander("🗑️ Delete Trade"):
                    del_opts = [""] + [f"{r['symbol']} ({r['id']})" for _, r in all_trades_df.iterrows()]
                    del_sel = st.selectbox("Select trade to delete", del_opts, key="del_trade")
                    if st.button("🗑️ Delete", key="del_btn") and del_sel:
                        tid = del_sel.split("(")[-1].rstrip(")")
                        delete_trade(tid)
                        st.success("Deleted!")
                        st.rerun()
            else:
                st.info("No trades yet. Use the 'New Trade' tab to add your first trade.")

    # ══════════════════════════════════════════════
    # TAB 3: PORTFOLIO DASHBOARD
    # ══════════════════════════════════════════════
    with exec_tab3:
        st.markdown("### 📊 Portfolio Dashboard")

        capital = st.number_input("Initial Capital (₹)", min_value=0, value=100000, step=50000, key="init_cap")

        summary = get_portfolio_summary(capital)

        if summary["total_trades"] > 0:
            # Summary metrics
            pnl_color = "normal" if summary["total_pnl"] > 0 else "inverse"
            m_cols = st.columns(5)
            with m_cols[0]:
                st.metric("📈 Total Trades", summary["total_trades"])
            with m_cols[1]:
                st.metric("🎯 Win Rate", f"{summary['win_rate']}%")
            with m_cols[2]:
                st.metric("💰 Total P&L", f"₹{summary['total_pnl']:+,.2f}", delta_color=pnl_color)
            with m_cols[3]:
                st.metric("📊 Profit Factor", summary["profit_factor"])
            with m_cols[4]:
                st.metric("🟢 Open Positions", summary["open_positions"])

            st.markdown("---")
            m2_cols = st.columns(4)
            with m2_cols[0]:
                st.metric("Avg Win", f"{summary['avg_win_pct']:.2f}%")
            with m2_cols[1]:
                st.metric("Avg Loss", f"{summary['avg_loss_pct']:.2f}%")
            with m2_cols[2]:
                st.metric("Best Trade", f"{summary['best_trade_symbol']} ({summary['best_trade_pct']:+.2f}%)")
            with m2_cols[3]:
                st.metric("Worst Trade", f"{summary['worst_trade_symbol']} ({summary['worst_trade_pct']:+.2f}%)")

            st.markdown("---")
            m3_cols = st.columns(3)
            with m3_cols[0]:
                st.metric("🔥 Max Consec Wins", summary["max_consec_wins"])
            with m3_cols[1]:
                st.metric("💀 Max Consec Losses", summary["max_consec_losses"])
            with m3_cols[2]:
                st.metric("📦 Total Exposure", f"₹{summary['total_exposure']:+,.0f}")

            # Monthly returns
            st.markdown("---")
            st.markdown("### 📅 Monthly Returns")
            monthly = get_monthly_returns()
            if not monthly.empty:
                st.dataframe(monthly, width='stretch', hide_index=True,
                             column_config={
                                 "month": "Month",
                                 "trades": "Trades",
                                 "wins": "Wins",
                                 "pnl": st.column_config.NumberColumn("P&L (₹)", format="₹%.2f"),
                                 "win_rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%")
                             })
        else:
            st.info("No trades recorded yet. Start by logging trades in the 'Trade Journal' tab.")

    # ══════════════════════════════════════════════
    # TAB 4: POSITION SIZER
    # ══════════════════════════════════════════════
    with exec_tab4:
        st.markdown("### 📐 Position Size Calculator")
        st.caption("Calculate optimal position size based on your risk tolerance")

        ps_col1, ps_col2 = st.columns(2)

        with ps_col1:
            ps_capital = st.number_input("Account Capital (₹)", min_value=1000, value=100000, step=10000, key="ps_cap")
            ps_risk = st.slider("Risk Per Trade (%)", 0.1, 5.0, 1.0, 0.1, key="ps_risk", format="%.1f%%")
            ps_entry = st.number_input("Entry Price (₹)", min_value=0.01, value=3000.0, step=1.0, key="ps_entry")

        with ps_col2:
            ps_sl = st.number_input("Stop Loss (₹)", min_value=0.01, value=2800.0, step=1.0, key="ps_sl")
            ps_target = st.number_input("Target Price (₹) (optional)", min_value=0.0, value=3500.0, step=1.0, key="ps_target")
            ps_max_risk = st.slider("Max Risk Cap (% of capital)", 0.5, 5.0, 2.0, 0.5, key="ps_max", format="%.1f%%")

        if st.button("🧮 Calculate", key="ps_calc", width='stretch'):
            result = calculate_position_with_target(
                ps_capital, ps_risk, ps_entry, ps_sl,
                target=ps_target if ps_target > 0 else None
            )

            if result and result.get("shares", 0) > 0:
                st.markdown("---")
                st.markdown("#### 📊 Position Sizing Results")

                res_cols = st.columns(4)
                with res_cols[0]:
                    st.metric("📈 Shares to Buy", result["shares"])
                with res_cols[1]:
                    st.metric("💰 Position Value", f"₹{result['position_value']:,.2f}")
                with res_cols[2]:
                    st.metric("🎯 Risk Amount", f"₹{result['risk_amount']:,.2f}",
                              delta=f"{result['risk_pct_portfolio']:.2f}% of capital")
                with res_cols[3]:
                    rr = result.get("risk_reward_ratio", 0)
                    st.metric("📊 Risk:Reward", f"1:{rr:.2f}" if rr > 0 else "N/A",
                              delta="Good" if rr >= 2 else ("Fair" if rr >= 1 else "Poor"))

                res2_cols = st.columns(3)
                with res2_cols[0]:
                    st.metric("Position Size %", f"{result['position_size_pct']:.1f}%")
                with res2_cols[1]:
                    target_pnl = result.get("target_pnl", 0)
                    st.metric("🎯 Target P&L", f"₹{target_pnl:+,.2f}",
                              delta=f"{result.get('target_pnl_pct', 0):+.2f}%")
                with res2_cols[2]:
                    st.metric("Risk Per Share", f"₹{result.get('risk_per_share', 0):.2f}")

                # Risk warning
                if result["position_size_pct"] > 20:
                    st.warning("⚠️ Position > 20% of portfolio — high concentration risk!")
                elif result["position_size_pct"] > 10:
                    st.info("ℹ️ Position is 10-20% of portfolio — moderate concentration")
            else:
                st.error("❌ Invalid inputs. Check that entry > stop loss.")

        # Stop Loss Suggestions
        st.markdown("---")
        st.markdown("#### 💡 Stop Loss Suggestions")
        sl_cols = st.columns(3)
        with sl_cols[0]:
            sl_entry = st.number_input("Entry", min_value=0.01, value=3000.0, step=1.0, key="sl_entry")
        with sl_cols[1]:
            sl_atr = st.number_input("ATR Value", min_value=0.0, value=60.0, step=1.0, key="sl_atr",
                                     help="Use ATR from chart page")
        with sl_cols[2]:
            sl_pct = st.slider("% Stop", 1, 10, 5, 1, key="sl_pct", format="%d%%")

        if st.button("💡 Suggest Stops", key="sl_suggest"):
            sl_suggestions = suggest_stop_loss(
                sl_entry, atr=sl_atr, pct_stop=float(sl_pct)
            )
            if sl_suggestions:
                st.markdown("**Recommended Stop:**")
                for method, price in sl_suggestions["suggestions"].items():
                    risk_pct = abs(sl_entry - price) / sl_entry * 100
                    st.markdown(f"- **{method}**: ₹{price} (risk: {risk_pct:.2f}%)")

                if sl_suggestions["recommended"]:
                    rec = sl_suggestions["recommended"]
                    st.success(f"✅ **Recommended**: ₹{rec} | Risk: {sl_suggestions['risk_if_recommended']:.2f}%")

    # ══════════════════════════════════════════════
    # TAB 5: TRADING (SHOONYA)
    # ══════════════════════════════════════════════
    with exec_tab5:
        st.markdown("### 🎯 Live Trading — Shoonya (Finvasia)")
        st.caption("Place, modify, and monitor orders directly from Sniper Terminal")

        shoonya_client = get_client()

        if not shoonya_client.is_connected:
            st.warning("🔴 Shoonya not connected. Go to Settings in the sidebar to connect.")
            st.info("💡 You need a Finvasia account. Get one at https://shoonya.finvasia.com")
        else:
            # Quote
            st.markdown("#### 🔍 Quick Quote")
            qq_cols = st.columns([2, 1])
            with qq_cols[0]:
                q_sym = st.text_input("Symbol", "RELIANCE", key="trd_qq")
            with qq_cols[1]:
                if st.button("📊 Get Quote", key="trd_gq"):
                    q = shoonya_client.get_quote(q_sym)
                    if q:
                        st.metric(f"{q_sym}", f"₹{q['ltp']:.2f}", f"{q['change_pct']:+.2f}%")

            st.markdown("---")

            # Order form
            st.markdown("#### 📝 Place Order")
            with st.form("shoonya_order"):
                o1 = st.columns(3)
                with o1[0]:
                    sym = st.text_input("Symbol", "RELIANCE", key="trd_sym").upper()
                with o1[1]:
                    bs = st.selectbox("Type", ["BUY", "SELL"], key="trd_bs")
                with o1[2]:
                    qty = st.number_input("Qty", 1, 10000, 10, key="trd_qty")

                o2 = st.columns(3)
                with o2[0]:
                    ot = st.selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"], key="trd_ot")
                with o2[1]:
                    pt = st.selectbox("Product", ["INTRADAY", "DELIVERY", "MARGIN"], key="trd_pt")
                with o2[2]:
                    st.markdown("<br>", unsafe_allow_html=True)

                o3 = st.columns(3)
                with o3[0]:
                    pr = st.number_input("Price (₹)", 0.0, 100000.0, 0.0, step=0.05, key="trd_pr",
                                         help="0 for MARKET orders")
                with o3[1]:
                    tr = st.number_input("Trigger (₹)", 0.0, 100000.0, 0.0, step=0.05, key="trd_tr",
                                         help="Required for SL orders")
                with o3[2]:
                    st.markdown("<br>", unsafe_allow_html=True)

                if st.form_submit_button("🚀 Place Order", width='stretch'):
                    p = 0 if ot == "MARKET" else pr
                    result = shoonya_client.place_order(
                        sym, bs, qty,
                        price=p,
                        trigger_price=tr if tr > 0 else 0.0,
                        order_type=ot,
                        product_type=pt
                    )
                    if result.get("success"):
                        oid = result.get("order_id", "?")
                        st.success(f"✅ Placed! ID: {oid}")
                        add_trade(f"{sym}.NS", p or 0, qty, bs,
                                  tags=["shoonya", "live"],
                                  notes=f"Order: {oid}")
                    else:
                        st.error(f"❌ {result.get('error', 'Failed')}")

            # Order Book
            st.markdown("---")
            st.markdown("#### 📋 Order Book & Positions")
            if st.button("🔄 Refresh", key="trd_ref"):
                ob = shoonya_client.get_order_book()
                pos = shoonya_client.get_positions()
                if ob is not None and not ob.empty:
                    st.dataframe(ob, width='stretch', hide_index=True)
                if pos is not None and not pos.empty:
                    st.dataframe(pos, width='stretch', hide_index=True)

# =========================================================
# PAGE: STOCK RESEARCH (SimplyWallSt-style)
# =========================================================
elif page == "🔍 Stock Research":
    st.subheader("🔍 Universal Stock Research")
    st.caption("Comprehensive company analysis — fundamentals, technicals, peers, and valuation")

    srch_col1, srch_col2 = st.columns([3, 1])
    with srch_col1:
        research_symbol = st.text_input("Search any Indian stock (e.g. RELIANCE, TCS, HDFCBANK)", value="RELIANCE",
                                        key="research_sym", help="No .NS suffix needed — auto-appended")
    with srch_col2:
        research_range = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3, key="research_range")

    sym = research_symbol.upper().strip()
    if not sym.endswith(".NS"):
        sym += ".NS"

    if st.button("🔍 Deep Research", key="research_go", width='stretch', type="primary"):
        with st.spinner(f"Gathering intelligence on {sym}..."):
            try:
                import yfinance as yf
                ticker = yf.Ticker(sym)
                info = ticker.info or {}

                df = load_data(sym, period=research_range)
                if df is not None and not df.empty:
                    df = compute_indicators(df)
                    latest = df.iloc[-1]
                    c = float(latest["Close"])

                    # ── Company Snapshot ──
                    st.markdown("### 📋 Company Snapshot")
                    snap = st.columns(5)
                    with snap[0]:
                        name = info.get("longName", sym.replace(".NS", ""))
                        st.markdown(f"**{name}**")
                        st.caption(f"{info.get('sector', 'N/A')} › {info.get('industry', 'N/A')}")
                    with snap[1]:
                        mcap = info.get("marketCap", 0)
                        st.metric("Market Cap", f"₹{mcap/1e7:.1f}Cr" if mcap > 1e7 else "N/A")
                    with snap[2]:
                        pe = info.get("trailingPE", info.get("forwardPE", "—"))
                        st.metric("P/E", f"{pe:.2f}" if isinstance(pe, (int, float)) else "—")
                    with snap[3]:
                        dy = info.get("dividendYield", 0)
                        if dy: dy *= 100
                        st.metric("Div Yield", f"{dy:.2f}%" if dy else "—")
                    with snap[4]:
                        beta = info.get("beta", "—")
                        st.metric("Beta", f"{beta:.2f}" if isinstance(beta, (int, float)) else "—")

                    # ── Price Performance ──
                    st.markdown("---")
                    st.markdown("### 📈 Price Performance")
                    perf = st.columns(4)
                    r1 = (c / float(df["Close"].iloc[-22]) - 1) * 100 if len(df) > 22 else 0
                    r3 = (c / float(df["Close"].iloc[-66]) - 1) * 100 if len(df) > 66 else 0
                    r6 = (c / float(df["Close"].iloc[-132]) - 1) * 100 if len(df) > 132 else 0
                    r1y = (c / float(df["Close"].iloc[-252]) - 1) * 100 if len(df) > 252 else 0
                    with perf[0]: st.metric("1 Month", f"{r1:+.2f}%")
                    with perf[1]: st.metric("3 Months", f"{r3:+.2f}%")
                    with perf[2]: st.metric("6 Months", f"{r6:+.2f}%")
                    with perf[3]: st.metric("1 Year", f"{r1y:+.2f}%")

                    # ── Technical Health ──
                    st.markdown("---")
                    st.markdown("### 🔧 Technical Health")
                    tech = st.columns(4)
                    rsi_v = float(latest.get("RSI", 50))
                    rsi_label = "Overbought" if rsi_v > 70 else ("Oversold" if rsi_v < 30 else "Neutral")
                    with tech[0]: st.metric("RSI (14)", f"{rsi_v:.1f}", rsi_label, delta_color="inverse" if rsi_v > 70 else "normal")
                    adx_v = float(latest.get("ADX", 0))
                    with tech[1]: st.metric("ADX (14)", f"{adx_v:.1f}", "Trending" if adx_v > 25 else "Ranging")
                    e50 = float(latest.get("EMA50", c))
                    e200 = float(latest.get("EMA200", c))
                    trend_s = "🟢 BULLISH" if c > e50 > e200 else ("🔴 BEARISH" if c < e50 < e200 else "🟡 SIDEWAYS")
                    with tech[2]: st.metric("Trend Structure", trend_s)
                    rvol_v = float(latest.get("RVOL", 1))
                    with tech[3]: st.metric("Volume (RVOL)", f"{rvol_v:.2f}x", "High" if rvol_v > 1.5 else "Normal")

                    # ── Fundamental Scorecard ──
                    st.markdown("---")
                    st.markdown("### 📊 Fundamental Scorecard")
                    fund = st.columns(4)
                    with fund[0]:
                        pb = info.get("priceToBook", "—")
                        st.metric("P/B", f"{pb:.2f}" if isinstance(pb, (int, float)) else "—")
                        h52 = info.get("fiftyTwoWeekHigh", 0)
                        l52 = info.get("fiftyTwoWeekLow", 0)
                        st.metric("52w Range", f"₹{l52:.0f}–₹{h52:.0f}" if h52 else "—")
                    with fund[1]:
                        roe = info.get("returnOnEquity", 0)
                        if roe: roe *= 100
                        st.metric("ROE", f"{roe:.1f}%" if roe else "—")
                        de = info.get("debtToEquity", "—")
                        st.metric("D/E", f"{de:.2f}" if isinstance(de, (int, float)) else "—")
                    with fund[2]:
                        eps = info.get("trailingEps", info.get("forwardEps", "—"))
                        st.metric("EPS", f"₹{eps:.2f}" if isinstance(eps, (int, float)) else "—")
                        rg = info.get("revenueGrowth", 0)
                        if rg: rg *= 100
                        st.metric("Rev Growth", f"{rg:+.1f}%" if rg else "—")
                    with fund[3]:
                        pm = info.get("profitMargins", 0)
                        if pm: pm *= 100
                        st.metric("Profit Margin", f"{pm:.1f}%" if pm else "—")
                        om = info.get("operatingMargins", 0)
                        if om: om *= 100
                        st.metric("Op Margin", f"{om:.1f}%" if om else "—")

                    # ── Valuation ──
                    st.markdown("---")
                    st.markdown("### 🏷️ Valuation vs Nifty 50")
                    try:
                        nifty_info = yf.Ticker("^NSEI").info or {}
                        n_pe = nifty_info.get("trailingPE", 0)
                        val_cols = st.columns(3)
                        with val_cols[0]:
                            sp = info.get("trailingPE", 0)
                            pct_vs = (sp / max(n_pe, 0.1) - 1) * 100 if sp and n_pe else 0
                            st.metric("P/E vs Nifty", f"{pct_vs:+.0f}%",
                                      delta="Expensive" if pct_vs > 20 else ("Cheap" if pct_vs < -20 else "Fair"),
                                      delta_color="inverse" if pct_vs > 20 else "normal")
                        with val_cols[1]:
                            target = info.get("targetMeanPrice", "—")
                            st.metric("Analyst Target", f"₹{target:.0f}" if isinstance(target, (int, float)) else "—",
                                      f"{((target / c) - 1) * 100:+.1f}%" if isinstance(target, (int, float)) and c else "")
                        with val_cols[2]:
                            rec = info.get("recommendationKey", "—")
                            rec_m = {"buy": "🟢 BUY", "strong_buy": "🟢 STRONG BUY", "hold": "🟡 HOLD", "sell": "🔴 SELL", "strong_sell": "🔴 STRONG SELL"}
                            st.metric("Analyst Rating", rec_m.get(rec, rec.upper() if rec != "—" else "—"))
                    except Exception:
                        st.caption("Valuation comparison unavailable")

                    # ── Peer Comparison ──
                    st.markdown("---")
                    st.markdown("### 👥 Peer Comparison")
                    base_name = sym.replace(".NS", "")
                    peer_map = {
                        "RELIANCE": ["TCS.NS", "WIPRO.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS"],
                        "TCS": ["RELIANCE.NS", "WIPRO.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS"],
                        "HDFCBANK": ["ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS", "INDUSINDBK.NS"],
                        "INFY": ["TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"],
                    }
                    peers = peer_map.get(base_name, ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"])
                    peer_rows = []
                    for p in peers[:5]:
                        try:
                            pi = yf.Ticker(p).info or {}
                            peer_rows.append({"Symbol": p.replace(".NS", ""),
                                "Mkt Cap": f"₹{pi.get('marketCap', 0)/1e7:.0f}Cr" if pi.get('marketCap') else "—",
                                "P/E": f"{pi.get('trailingPE', 0):.1f}" if pi.get('trailingPE') else "—",
                                "ROE": f"{pi.get('returnOnEquity', 0)*100:.1f}%" if pi.get('returnOnEquity') else "—",
                                "Div Yield": f"{pi.get('dividendYield', 0)*100:.2f}%" if pi.get('dividendYield') else "—"})
                        except Exception:
                            continue
                    if peer_rows:
                        st.dataframe(pd.DataFrame(peer_rows), width='stretch', hide_index=True)

                    # ── Signal Summary ──
                    st.markdown("---")
                    st.markdown("### 🧠 Trading Signal Summary")
                    sig_c = st.columns(3)
                    with sig_c[0]:
                        sv4 = generate_v4_signal(df)
                        st.metric("v4 Signal", sv4.get("Signal", "NEUTRAL"), f"Score: {sv4.get('Score', 0)}/10")
                    with sig_c[1]:
                        sv3 = generate_v3_signal(df)
                        st.metric("v3 Signal", sv3.get("Signal", "NEUTRAL"), f"Score: {sv3.get('Score', 0)}/8")
                    with sig_c[2]:
                        sw = generate_swing_signal(df)
                        st.metric("Classic Signal", sw.get("Signal", "NEUTRAL"), f"Conf: {sw.get('Confidence', 0)}/6")

                    # ── Key Levels ──
                    st.markdown("---")
                    st.markdown("### 🎯 Key Price Levels")
                    try:
                        sr = detect_support_resistance(df, lookback=100, order=5)
                        lc = st.columns(3)
                        with lc[0]: st.metric("🟢 Nearest Support", f"₹{sr.get('nearest_support', 0):,.2f}")
                        with lc[1]: st.metric("🔴 Nearest Resistance", f"₹{sr.get('nearest_resistance', 0):,.2f}")
                        with lc[2]:
                            ns = sr.get("nearest_support", c * 0.95)
                            nr_ = sr.get("nearest_resistance", c * 1.05)
                            fib = fibonacci_levels(c, ns, nr_)
                            st.metric("📐 50% Fib", f"₹{fib.get('level_50', 0):,.2f}" if fib else "—")
                    except Exception:
                        st.caption("Levels unavailable")

                    st.success(f"✅ Research complete for {sym.replace('.NS', '')} — {len(df)} bars")

                else:
                    st.error(f"No price data for {sym}. Check symbol.")
            except Exception as e:
                st.error(f"Research failed: {e}")