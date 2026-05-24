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
from core.profit_engine_v4 import (
    generate_v4_signal, run_v4_backtest,
    kelly_fraction
)
from core.intraday_scanner import get_daily_picks, scan_intraday_setups
from core.swing_scanner import get_weekly_picks, get_monthly_picks, scan_swing_setups
from core.ai_helper import call_ai, build_stock_analysis_prompt
from core.news_feed import fetch_rss_feeds, get_stock_news, scan_stock_news, categorize_news_batch_ai, categorize_news_keyword, CATEGORY_EMOJI
from core.heatmap import build_heatmap_data, plot_treemap_heatmap

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
_bg = "#0a0e1a" if _dark else "#f8fafc"
_bg2 = "#0d1117" if _dark else "#f1f5f9"
_bg_card = "#131722" if _dark else "#ffffff"
_text = "#e6edf3" if _dark else "#0f172a"
_text2 = "#8b949e" if _dark else "#64748b"
_border = "#21262d" if _dark else "#e2e8f0"
_gold = "#f0b90b" if _dark else "#d97706"
_green = "#00c9a7" if _dark else "#059669"
_red = "#f85149" if _dark else "#dc2626"

theme_css = f"""
<style>
    /* ── Core palette (minimal overrides — let external CSS handle layout) ── */
    .stApp, .stApp > header {{ background-color: {_bg} !important; }}
    .main .block-container {{ background-color: {_bg} !important; padding: 1rem 1.5rem !important; }}
    [data-testid="stSidebar"] {{ background-color: {_bg2} !important; border-right: 1px solid {_border} !important; }}
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{ color: {_text} !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: {_text} !important; font-weight: 600 !important; }}
    .stMarkdown, p, li {{ color: {_text} !important; }}
    .stCaption {{ color: {_text2} !important; }}
    div[data-testid="metric-container"] {{ background: {_bg_card} !important; border: 1px solid {_border} !important; }}
    div[data-testid="metric-container"] label {{ color: {_text2} !important; }}
    div[data-testid="metric-container"] div[data-testid="metric-value"] {{ color: {_text} !important; }}
    .stSelectbox > div > div, .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{ background-color: {_bg_card} !important; border: 1px solid {_border} !important; color: {_text} !important; }}
    .stButton > button {{ background: {_bg_card} !important; border: 1px solid {_border} !important; color: {_text} !important; }}
    .stButton > button:hover {{ border-color: {_gold} !important; color: {_gold} !important; background: rgba(240, 185, 11, 0.08) !important; }}
    .stButton > button[kind="primary"] {{ background: {_gold} !important; color: #0a0e1a !important; border: 1px solid {_gold} !important; }}
    .streamlit-expanderHeader {{ background-color: {_bg_card} !important; border: 1px solid {_border} !important; color: {_text} !important; }}
    .stAlert {{ background-color: {_bg_card} !important; border: 1px solid {_border} !important; }}
    .stTabs [data-baseweb="tab"] {{ color: {_text2} !important; }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {_text} !important; }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{ color: {_gold} !important; border-bottom-color: {_gold} !important; }}
    hr {{ border-color: {_border} !important; }}
    .stCheckbox label {{ color: {_text} !important; }}
    .stSpinner > div > div {{ border-color: {_gold} transparent transparent transparent !important; }}
    .stDataFrame {{ border-color: {_border} !important; }}
    .stDataFrame tbody tr:hover {{ background: rgba(240, 185, 11, 0.05) !important; }}
    .stSuccess {{ border-left: 3px solid {_green} !important; }}
    .stError {{ border-left: 3px solid {_red} !important; }}
    .stWarning {{ border-left: 3px solid {_gold} !important; }}
    .stInfo {{ border-left: 3px solid #58a6ff !important; }}
</style>"""
st.markdown(theme_css, unsafe_allow_html=True)

# =========================================================
# SESSION STATE: Navigation helpers
# =========================================================
if "navigate_to" not in st.session_state:
    st.session_state.navigate_to = None


def goto_chart(ticker):
    """Navigate to Chart page with the given ticker."""
    # Set chart_ticker BEFORE rerun so the widget reads it
    st.session_state.chart_ticker = ticker
    st.session_state.navigate_to = "Chart"
    st.rerun()


def stock_link(symbol, display_name=None, key_suffix=""):
    """
    Render a clickable stock name that navigates to the Chart page.
    key_suffix should include the list index for uniqueness, e.g. "gain_0", "loss_3".
    """
    label = display_name or symbol.replace(".NS", "")
    key = f"sl_{symbol.replace('.','_')}_{key_suffix}"
    if st.button(label, key=key, help=f"Open {symbol} on Chart", use_container_width=True):
        goto_chart(symbol)


def stock_link_inline(symbol, display_name=None):
    """
    Render a small inline button styled as a link. Use inside st.markdown or st.columns.
    """
    label = display_name or symbol.replace(".NS", "")
    key = f"sli_{symbol.replace('.','_')}"
    st.markdown(
        f"<a href='#' onclick='return false;' style='color:#FFD700;text-decoration:none;font-weight:bold;cursor:pointer;'>{label}</a>",
        unsafe_allow_html=True
    )
    # Also add hidden button for actual click handling
    if st.button(f"🔍 {label}", key=key, help=f"Analyze {symbol}"):
        goto_chart(symbol)


def _ensure_ns(symbol):
    """Auto-append .NS suffix for NSE stocks if not already present."""
    s = symbol.strip().upper()
    if s and not s.endswith(".NS") and "^" not in s and ".BO" not in s:
        s += ".NS"
    return s


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚡ SNIPER")
st.sidebar.caption("INSTITUTIONAL SWING TERMINAL · NSE/BSE")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Screener", "📈 Daily Trades", "📆 Swing Setups",
     "Chart", "Backtest", "Market Data", "Analytics", "📰 News",
     "Portfolio", "🔍 Stock Research"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("v3.0 · NSE/BSE · Bloomberg-Style Terminal")

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
st.sidebar.caption("v3.0 · NSE/BSE · Shoonya · Bloomberg-Style Terminal")

# ── AI Chat Toggle ──
if "ai_visible" not in st.session_state:
    st.session_state.ai_visible = True
if "ai_api_key" not in st.session_state:
    st.session_state.ai_api_key = ""
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []
if "ai_mode" not in st.session_state:
    st.session_state.ai_mode = "flash"

ai_tog = st.sidebar.checkbox("🤖 AI Chat", value=st.session_state.ai_visible,
                              key="ai_tog", help="Show/hide AI Analyst panel on the right")

# Sync toggle state
if ai_tog != st.session_state.ai_visible:
    st.session_state.ai_visible = ai_tog
    st.rerun()

# ── Navigation override (for stock links) ──
if st.session_state.navigate_to:
    page = st.session_state.navigate_to
    st.session_state.navigate_to = None  # Clear after use

# ═══════════════════════════════════════════════════════════════
# FLOATING AI CHAT PANEL (Right Side)
# ═══════════════════════════════════════════════════════════════
if st.session_state.get("ai_visible", True):
    # Page context dict
    pg_ctx = {
        "Dashboard": "Market Dashboard — Nifty pulse, FII/DII, VIX, breadth, heat map.",
        "Screener": "Stock Screener — scanning for breakouts, patterns, momentum.",
        "Chart": "Chart page — candlesticks with RSI, ADX, MACD panels.",
        "📈 Daily Trades": "Daily intraday high-probability trade setups.",
        "📆 Swing Setups": "Weekly/monthly swing trade setups.",
        "Backtest": "Backtest page — V4 Precision Engine with full analytics.",
        "Market Data": "Market Data — F&O chain, VIX, FII/DII, sector rotation.",
        "Analytics": "Analytics — S/R levels, strategy compare, AI scorer, Fibonacci.",
        "📰 News": "Live market news feed — RSS from Moneycontrol, ET, Livemint, Reuters.",
        "Portfolio": "Portfolio — watchlists, trade journal, position sizer, Shoonya.",
        "🔍 Stock Research": "Stock Research — fundamentals, technicals, peer comparison.",
    }

    # AI Settings in sidebar expander
    with st.sidebar.expander("⚙️ AI Settings", expanded=False):
        kc1, kc2 = st.columns(2)
        with kc1:
            st.session_state.ai_api_key = st.text_input(
                "API Key", type="password", value=st.session_state.ai_api_key,
                key="ai_key_float", label_visibility="collapsed",
                placeholder="DeepSeek/OpenAI key"
            )
        with kc2:
            ai_provider = st.selectbox("Provider", ["DS", "OA"], index=0,
                                       key="ai_prov_float", label_visibility="collapsed")
        ai_mode = st.selectbox("Mode", ["flash", "reasoning"],
                                index=0 if st.session_state.get("ai_mode") == "flash" else 1,
                                key="ai_mode_sel",
                                help="Flash=fast concise · Reasoning=detailed step-by-step")
        st.session_state.ai_mode = ai_mode

    # Fixed right-side panel using a placeholder container
    ai_panel = st.sidebar.container()
    with ai_panel:
        st.markdown("---")
        st.markdown(f"#### 🤖 AI Analyst <span style='color:#9ca3af;font-size:11px;'>({ai_mode})</span>", unsafe_allow_html=True)

        # Chat messages
        for msg in st.session_state.get("ai_messages", []):
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(f"<div style='background:#1f2937;padding:8px 12px;border-radius:10px;margin:6px 0;font-size:13px;'><strong>You:</strong> {content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#111827;border:1px solid #1f2937;padding:8px 12px;border-radius:10px;margin:6px 0;font-size:13px;'><strong>🤖 AI:</strong> {content}</div>", unsafe_allow_html=True)

        # Chat input
        ai_q = st.chat_input(f"Ask about {page}...", key="ai_chat_float")

        # Clear button
        if st.session_state.get("ai_messages", []):
            if st.button("🗑️ Clear", key="ai_clr_float", use_container_width=True):
                st.session_state.ai_messages = []
                st.rerun()

    # Process chat input outside the container
    if ai_q:
        if not st.session_state.get("ai_api_key"):
            st.warning("Enter API key in ⚙️ AI Settings")
        else:
            st.session_state.ai_messages.append({"role": "user", "content": ai_q})

            sp = f"""You are Sniper Terminal AI — Indian stock market analyst.

CURRENT PAGE: {page}
CONTEXT: {pg_ctx.get(page, 'Browsing platform')}

Answer about: technical analysis (RSI, MACD, ADX, BB, VWAP, Supertrend, ATR), fundamentals (P/E, P/B, ROE), NSE (Nifty, VIX, FII/DII, F&O), strategy (PF, Sharpe, Sortino, Calmar, Kelly, Monte Carlo).

RULES: Use ₹, use .NS symbols, be concise (under 200 words), educational, NO buy/sell advice."""

            msgs = [{"role": "system", "content": sp}]
            for m in st.session_state.ai_messages[-8:]:
                msgs.append({"role": m["role"], "content": m["content"]})

            with st.spinner("🤔"):
                reply = call_ai(
                    msgs,
                    api_key=st.session_state.ai_api_key,
                    provider=ai_provider,
                    mode=ai_mode,
                    max_tokens=2048,
                )

            if reply:
                st.session_state.ai_messages.append({"role": "assistant", "content": reply})
                st.rerun()
            else:
                st.error("❌ AI call failed. Check API key in Settings.")

# ═══════════════════════════════════════════════════════════════
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
                sym = g["symbol"]
                pct = g["change"]
                gcols = st.columns([3, 1])
                with gcols[0]:
                    stock_link(sym, display_name=f"{i+1}. {sym.replace('.NS','')}", key_suffix=f"gain_{i}")
                with gcols[1]:
                    st.markdown(f"<p style='color:#00ff9f;font-weight:bold;text-align:right;'>+{pct:.2f}%</p>", unsafe_allow_html=True)

        # Losers
        with m_cols[1]:
            st.markdown("#### 🔴 Top Losers")
            for i, g in enumerate(movers["losers"]):
                sym = g["symbol"]
                pct = g["change"]
                gcols = st.columns([3, 1])
                with gcols[0]:
                    stock_link(sym, display_name=f"{i+1}. {sym.replace('.NS','')}", key_suffix=f"loss_{i}")
                with gcols[1]:
                    st.markdown(f"<p style='color:#ff4d6d;font-weight:bold;text-align:right;'>{pct:.2f}%</p>", unsafe_allow_html=True)

        # Most Volume
        with m_cols[2]:
            st.markdown("#### 📊 Most Active (Volume)")
            for i, g in enumerate(movers["most_volume"]):
                sym = g["symbol"]
                gcols = st.columns([3, 1])
                with gcols[0]:
                    stock_link(sym, display_name=f"{i+1}. {sym.replace('.NS','')}", key_suffix=f"vol_{i}")
                with gcols[1]:
                    st.markdown(f"<p style='color:#64b5f6;font-weight:bold;text-align:right;'>₹{g['close']:,.2f}</p>", unsafe_allow_html=True)
    else:
        st.info("Top movers data unavailable. Try during market hours.")

    st.markdown("---")

    # ===================================================================
    # ROW 3.5: MARKET HEAT MAP
    # ===================================================================
    with st.expander("🔥 Market Heat Map — Nifty 50", expanded=True):
        hm_df = build_heatmap_data(NIFTY50)
        if not hm_df.empty:
            hm_tabs = st.tabs(["📊 Treemap", "📋 Grid View"])
            with hm_tabs[0]:
                hm_fig = plot_treemap_heatmap(hm_df)
                if hm_fig:
                    st.plotly_chart(hm_fig, use_container_width=True)
            with hm_tabs[1]:
                # Sortable table view
                hm_sorted = hm_df.sort_values("Change%", ascending=False)
                hm_sorted["Change%"] = hm_sorted["Change%"].apply(lambda v: f"{v:+.2f}%")
                hm_sorted["RSI"] = hm_sorted["RSI"].apply(lambda v: f"{v:.1f}")
                hm_sorted["RVOL"] = hm_sorted["RVOL"].apply(lambda v: f"{v:.2f}x")
                st.dataframe(
                    hm_sorted[["Symbol", "Close", "Change%", "RSI", "RVOL"]],
                    width='stretch', hide_index=True,
                    column_config={
                        "Close": st.column_config.NumberColumn(format="₹%.2f"),
                    }
                )
        else:
            st.info("Heat map data loading...")

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

            # ── Quick Chart Links for top results ──
            with st.expander("🔍 Quick Chart Access — Click any stock to open Chart tab", expanded=True):
                top_for_charts = min(len(results), 20)
                chart_links_rows = (top_for_charts + 4) // 5
                for row_i in range(chart_links_rows):
                    cols = st.columns(5)
                    for col_i in range(5):
                        idx = row_i * 5 + col_i
                        if idx < top_for_charts:
                            sym = results[idx].get("Symbol", "")
                            with cols[col_i]:
                                stock_link(sym, display_name=f"📊 {sym.replace('.NS','')}", key_suffix=f"scr_{idx}")

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
# PAGE: DAILY TRADES (Intraday High-Probability)
# =========================================================
elif page == "📈 Daily Trades":
    st.subheader("📈 Daily High-Probability Trades")
    st.caption("Intraday setups based on 15-min data · Updated every 5 minutes · Refresh to re-scan")

    # ── Universe selection ──
    dcol1, dcol2, dcol3 = st.columns([2, 1, 1])
    with dcol1:
        daily_universe = st.selectbox(
            "Universe",
            ["Nifty 50", "Nifty 200", "Nifty 500", "Nifty 50 + Nifty 200"],
            index=0, key="daily_uni"
        )
    with dcol2:
        daily_count = st.slider("Picks to show", 2, 8, 4, key="daily_count")
    with dcol3:
        daily_refresh = st.button("🔄 Refresh Scan", key="daily_refresh", type="primary", use_container_width=True)

    # Map universe selection
    uni_map = {
        "Nifty 50": NIFTY50,
        "Nifty 200": NIFTY200,
        "Nifty 500": NIFTY500,
        "Nifty 50 + Nifty 200": list(dict.fromkeys(NIFTY50 + NIFTY200)),
    }
    scan_symbols = uni_map.get(daily_universe, NIFTY50)

    # ── Clear stale cache for fresh data ──
    if daily_refresh:
        from core.utils import clear_cache as _cc
        _cc()

    # ── Run scan ──
    with st.spinner(f"Scanning {len(scan_symbols)} stocks for intraday setups..."):
        daily_picks = get_daily_picks(scan_symbols, top_n=daily_count)

    if daily_picks:
        st.success(f"✅ Found {len(daily_picks)} high-probability setups")

        # ── Summary metrics ──
        avg_score = np.mean([p["Score"] for p in daily_picks])
        avg_rr = np.mean([p["R:R"] for p in daily_picks if p["R:R"] > 0])
        buy_count = sum(1 for p in daily_picks if p["Direction"] == "BUY")
        st.markdown(f"**Avg Score:** {avg_score:.0f}/100 · **Avg R:R:** {avg_rr:.2f} · **Buy Signals:** {buy_count}/{len(daily_picks)}")

        # ── Display each pick as a card ──
        for i, pick in enumerate(daily_picks):
            with st.container(border=True):
                cols = st.columns([3, 2, 2, 2, 2])

                # Symbol + score (clickable)
                with cols[0]:
                    emoji = "🟢" if pick["Direction"] == "BUY" else ("🟡" if pick["Direction"] == "NEUTRAL" else "🔴")
                    st.markdown(f"### {emoji} {pick['Symbol'].replace('.NS','')}")
                    st.caption(f"Score **{pick['Score']}/100** · {pick['Direction']}")
                    stock_link(pick["Symbol"], display_name="🔍 Chart")

                # Entry / Stop / Targets
                with cols[1]:
                    st.metric("Entry", f"₹{pick['Entry']:,.2f}")
                    st.metric("Stop Loss", f"₹{pick['StopLoss']:,.2f}", delta_color="inverse")
                with cols[2]:
                    st.metric("Target 1 (1.5R)", f"₹{pick['Target1']:,.2f}")
                    st.metric("Target 2 (3R)", f"₹{pick['Target2']:,.2f}")
                with cols[3]:
                    st.metric("R:R", f"{pick['R:R']}")
                    st.metric("ATR%", f"{pick['ATR%']}%")
                with cols[4]:
                    st.metric("RSI", pick["RSI"])
                    st.metric("RVOL", f"{pick['RVOL']}x")

                # Reasons & Concerns
                rcols = st.columns(2)
                with rcols[0]:
                    for r in pick["Reasons"][:3]:
                        st.markdown(f"✅ {r}")
                with rcols[1]:
                    for c in pick["Concerns"][:2]:
                        st.markdown(f"⚠️ {c}")

                # ── AI Analysis button ──
                ai_btn_key = f"ai_daily_{i}_{pick['Symbol'].replace('.','_')}"
                ai_res_key = f"ai_daily_res_{i}_{pick['Symbol'].replace('.','_')}"
                if st.button("🤖 AI Analysis", key=ai_btn_key, use_container_width=True):
                    if not st.session_state.get("ai_api_key"):
                        st.warning("Enter API key in ⚙️ AI Settings first")
                    else:
                        with st.spinner("🤔 Analyzing setup..."):
                            from core.utils import load_data as _ld
                            from core.indicators import compute_indicators as _ci
                            _df = _ld(pick["Symbol"], interval="15m", period="5d")
                            if _df is not None and not _df.empty:
                                _df = _ci(_df)
                            sys_p, usr_p = build_stock_analysis_prompt(
                                pick["Symbol"], pick, timeframe="intraday", live_df=_df
                            )
                            msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
                            reply = call_ai(
                                msgs,
                                api_key=st.session_state.ai_api_key,
                                provider=st.session_state.get("ai_prov_float", "DS"),
                                mode=st.session_state.get("ai_mode", "flash"),
                                max_tokens=1024,
                            )
                            if reply:
                                st.session_state[ai_res_key] = reply
                            else:
                                st.error("❌ AI call failed. Check API key.")

                # Show AI analysis result if exists
                if ai_res_key in st.session_state:
                    with st.container(border=True):
                        st.markdown(f"**🤖 AI Analysis**")
                        st.markdown(st.session_state[ai_res_key])
                        if st.button("✕", key=f"close_{ai_res_key}", help="Dismiss"):
                            del st.session_state[ai_res_key]
                            st.rerun()

    else:
        st.warning("📭 No intraday setups found. Markets may be closed or data may not be available. Try:\n"
                   "1. Hit 🔄 Refresh Scan to clear cache\n"
                   "2. Use a larger universe (Nifty 200/500)\n"
                   "3. Scan during market hours (9:15 AM – 3:30 PM IST) for best results")

    # ── Stock-specific news alerts ──
    if daily_picks:
        with st.expander("📰 News Alerts for Picks", expanded=False):
            pick_symbols = [p["Symbol"] for p in daily_picks]
            with st.spinner("Checking stock-specific news..."):
                stock_news = scan_stock_news(pick_symbols, max_news_per_stock=3, min_impact=20)
            if stock_news:
                for item in stock_news:
                    trigger_tag = " 🚀 HIGH CERTAINTY" if item.get("is_trade_trigger") else ""
                    st.markdown(f"**{item['symbol'].replace('.NS','')}**{trigger_tag}: {item['title']}")
                    st.caption(f"{item.get('publisher', '')} · Impact: {item['impact']}/100")
            else:
                st.caption("No significant news for these stocks.")

    # ── Market condition note ──
    st.markdown("---")
    st.caption("💡 **Tips:** Best results during first 2 hours of market open. Always use proper risk management. "
               "These are directional ideas, not financial advice.")

# =========================================================
# PAGE: SWING SETUPS (Weekly/Monthly)
# =========================================================
elif page == "📆 Swing Setups":
    st.subheader("📆 Swing Trade Setups")
    st.caption("Weekly & Monthly timeframe setups for positional trading · Holding period: 2–8 weeks")

    scol1, scol2, scol3 = st.columns([2, 1, 1])
    with scol1:
        swing_universe = st.selectbox(
            "Universe",
            ["Nifty 50", "Nifty 200", "Nifty 500", "Nifty 50 + Nifty 200"],
            index=0, key="swing_uni"
        )
    with scol2:
        swing_tf = st.selectbox("Timeframe", ["Weekly", "Monthly", "Both"], index=0, key="swing_tf")
    with scol3:
        swing_refresh = st.button("🔄 Refresh", key="swing_refresh", type="primary", use_container_width=True)

    uni_map = {
        "Nifty 50": NIFTY50,
        "Nifty 200": NIFTY200,
        "Nifty 500": NIFTY500,
        "Nifty 50 + Nifty 200": list(dict.fromkeys(NIFTY50 + NIFTY200)),
    }
    swing_symbols = uni_map.get(swing_universe, NIFTY50)

    weekly_picks = []
    monthly_picks = []

    if swing_tf in ("Weekly", "Both"):
        with st.spinner(f"Scanning {len(swing_symbols)} stocks on weekly timeframe..."):
            weekly_picks = get_weekly_picks(swing_symbols, top_n=6)

    if swing_tf in ("Monthly", "Both"):
        with st.spinner(f"Scanning {len(swing_symbols)} stocks on monthly timeframe..."):
            monthly_picks = get_monthly_picks(swing_symbols, top_n=4)

    # ── Display Weekly Picks ──
    if swing_tf in ("Weekly", "Both") and weekly_picks:
        st.markdown("---")
        st.markdown("### 📅 Weekly Swing Setups")
        st.caption("Holding period: 2–6 weeks · Stop: 2.5× ATR")

        for i, pick in enumerate(weekly_picks):
            with st.container(border=True):
                cols = st.columns([3, 2, 2, 2, 2])
                with cols[0]:
                    st.markdown(f"### {pick['Symbol'].replace('.NS','')}")
                    st.caption(f"Score **{pick['Score']}/100** · {pick['EMA_Stack']} EMA Stack · SupTrend: {pick['SuperTrend']}")
                    stock_link(pick["Symbol"], display_name="🔍 Chart")
                with cols[1]:
                    st.metric("Entry", f"₹{pick['Entry']:,.2f}")
                    st.metric("Stop Loss", f"₹{pick['StopLoss']:,.2f}", delta_color="inverse")
                with cols[2]:
                    st.metric("Target 1 (2R)", f"₹{pick['Target1']:,.2f}")
                    st.metric("Target 2 (4R)", f"₹{pick['Target2']:,.2f}")
                with cols[3]:
                    st.metric("R:R", f"{pick['R:R']}")
                    st.metric("ATR%", f"{pick['ATR%']}%")
                with cols[4]:
                    st.metric("RSI", pick["RSI"])
                    st.metric("ADX", pick["ADX"])

                # Reasons
                reason_cols = st.columns(2)
                with reason_cols[0]:
                    for r in pick["Reasons"][:3]:
                        st.markdown(f"✅ {r}")
                with reason_cols[1]:
                    for c in pick["Concerns"][:2]:
                        st.markdown(f"⚠️ {c}")

                # ── AI Analysis button (Weekly) ──
                ai_btn_key = f"ai_swing_w_{i}_{pick['Symbol'].replace('.','_')}"
                ai_res_key = f"ai_swing_w_res_{i}_{pick['Symbol'].replace('.','_')}"
                if st.button("🤖 AI Analysis", key=ai_btn_key, use_container_width=True):
                    if not st.session_state.get("ai_api_key"):
                        st.warning("Enter API key in ⚙️ AI Settings first")
                    else:
                        with st.spinner("🤔 Analyzing setup..."):
                            from core.utils import load_data as _ldw
                            from core.indicators import compute_indicators as _ciw
                            _dfw = _ldw(pick["Symbol"], interval="1wk", period="1y")
                            if _dfw is not None and not _dfw.empty:
                                _dfw = _ciw(_dfw)
                            sys_p, usr_p = build_stock_analysis_prompt(
                                pick["Symbol"], pick, timeframe="swing", live_df=_dfw
                            )
                            msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
                            reply = call_ai(
                                msgs,
                                api_key=st.session_state.ai_api_key,
                                provider=st.session_state.get("ai_prov_float", "DS"),
                                mode=st.session_state.get("ai_mode", "flash"),
                                max_tokens=1024,
                            )
                            if reply:
                                st.session_state[ai_res_key] = reply
                            else:
                                st.error("❌ AI call failed. Check API key.")

                # Show AI analysis result if exists
                if ai_res_key in st.session_state:
                    with st.container(border=True):
                        st.markdown(f"**🤖 AI Analysis**")
                        st.markdown(st.session_state[ai_res_key])
                        if st.button("✕", key=f"close_{ai_res_key}", help="Dismiss"):
                            del st.session_state[ai_res_key]
                            st.rerun()

    # ── Display Monthly Picks ──
    if swing_tf in ("Monthly", "Both") and monthly_picks:
        st.markdown("---")
        st.markdown("### 📆 Monthly Swing Setups")
        st.caption("Holding period: 2–6 months · Wider stops · Larger targets")

        for i, pick in enumerate(monthly_picks):
            with st.container(border=True):
                cols = st.columns([3, 2, 2, 2, 2])
                with cols[0]:
                    st.markdown(f"### {pick['Symbol'].replace('.NS','')}")
                    st.caption(f"Score **{pick['Score']}/100** · {pick['EMA_Stack']} EMA Stack · SupTrend: {pick['SuperTrend']}")
                    stock_link(pick["Symbol"], display_name="🔍 Chart")
                with cols[1]:
                    st.metric("Entry", f"₹{pick['Entry']:,.2f}")
                    st.metric("Stop Loss", f"₹{pick['StopLoss']:,.2f}", delta_color="inverse")
                with cols[2]:
                    st.metric("Target 1 (2R)", f"₹{pick['Target1']:,.2f}")
                    st.metric("Target 2 (4R)", f"₹{pick['Target2']:,.2f}")
                with cols[3]:
                    st.metric("R:R", f"{pick['R:R']}")
                    st.metric("ATR%", f"{pick['ATR%']}%")
                with cols[4]:
                    st.metric("RSI", pick["RSI"])
                    st.metric("ADX", pick["ADX"])

                reason_cols = st.columns(2)
                with reason_cols[0]:
                    for r in pick["Reasons"][:3]:
                        st.markdown(f"✅ {r}")
                with reason_cols[1]:
                    for c in pick["Concerns"][:2]:
                        st.markdown(f"⚠️ {c}")

                # ── AI Analysis button (Monthly) ──
                ai_btn_key = f"ai_swing_m_{i}_{pick['Symbol'].replace('.','_')}"
                ai_res_key = f"ai_swing_m_res_{i}_{pick['Symbol'].replace('.','_')}"
                if st.button("🤖 AI Analysis", key=ai_btn_key, use_container_width=True):
                    if not st.session_state.get("ai_api_key"):
                        st.warning("Enter API key in ⚙️ AI Settings first")
                    else:
                        with st.spinner("🤔 Analyzing setup..."):
                            from core.utils import load_data as _ldm
                            from core.indicators import compute_indicators as _cim
                            _dfm = _ldm(pick["Symbol"], interval="1mo", period="2y")
                            if _dfm is not None and not _dfm.empty:
                                _dfm = _cim(_dfm)
                            sys_p, usr_p = build_stock_analysis_prompt(
                                pick["Symbol"], pick, timeframe="swing", live_df=_dfm
                            )
                            msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
                            reply = call_ai(
                                msgs,
                                api_key=st.session_state.ai_api_key,
                                provider=st.session_state.get("ai_prov_float", "DS"),
                                mode=st.session_state.get("ai_mode", "flash"),
                                max_tokens=1024,
                            )
                            if reply:
                                st.session_state[ai_res_key] = reply
                            else:
                                st.error("❌ AI call failed. Check API key.")

                # Show AI analysis result if exists
                if ai_res_key in st.session_state:
                    with st.container(border=True):
                        st.markdown(f"**🤖 AI Analysis**")
                        st.markdown(st.session_state[ai_res_key])
                        if st.button("✕", key=f"close_{ai_res_key}", help="Dismiss"):
                            del st.session_state[ai_res_key]
                            st.rerun()

    if not weekly_picks and not monthly_picks:
        st.warning("📭 No swing setups found. This can happen if: (1) Markets are closed — try during trading hours, (2) Filters are too strict — try Nifty 200 universe, (3) Data is cached — hit 🔄 Refresh to clear cache.")
        st.info("💡 Weekly data should always be available. Try switching to 'Nifty 200' universe and click Refresh.")

    st.markdown("---")
    st.caption("💡 **Tips:** Enter in tranches (50% now, 25% on confirmation, 25% on strength). "
               "Move stop to breakeven after 1R profit. These are directional ideas, not financial advice.")

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
            default_ticker = st.session_state.get("chart_ticker", "RELIANCE.NS")
            ticker = st.text_input("Symbol", value=default_ticker, key="chart_ticker")

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
            sym_clean = _ensure_ns(ticker)
            with st.spinner(f"Loading {sym_clean}..."):
                df = load_data(sym_clean, period=period)

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
            "Stock Symbols (comma separated, e.g., RELIANCE, TCS, INFY)",
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
    st.subheader("🧪 Profit Engine Backtest — V4 Precision Engine")

    bt_tab1 = st.tabs(["🚀 V4 Precision Engine"])[0]

    # ═══════════════════════════════════════
    # TAB 1: V4 PRECISION ENGINE (FULL)
    # ═══════════════════════════════════════
    with bt_tab1:
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

        if st.button("🚀 Run V4 Backtest", key="v4_run", width='stretch', type="primary"):
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

                # ── v4 Intelligence Metrics ──
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
                        st.metric("90% CI Lower", f"{mc.get('CI Lower %', 0):+.2f}%",
                                  delta_color="inverse")
                    with mc_cols[2]:
                        st.metric("90% CI Upper", f"{mc.get('CI Upper %', 0):+.2f}%")
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
                    st.warning("No trades generated. Try a different symbol or relax the filters.")
            else:
                st.error(f"❌ {result.get('Error', 'Failed')}")

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
        st.markdown("### 📊 Strategy Comparison — v3 vs v4")
        st.caption("Side-by-side comparison with Monte Carlo confidence intervals")

        comp_ticker = st.text_input("Symbol", "RELIANCE.NS", key="comp_ticker")
        comp_cap = st.number_input("Capital (₹)", 10000, 10000000, 100000, 50000, key="comp_cap")

        comp_cols = st.columns(2)
        with comp_cols[0]:
            run_v3 = st.checkbox("📊 v3 (Legacy)", True, key="cmp_v3")
        with comp_cols[1]:
            run_v4 = st.checkbox("🚀 v4 (Precision)", True, key="cmp_v4")

        if st.button("🔄 Deep Compare", key="comp_run", width='stretch', type="primary"):
            selected = []
            if run_v3: selected.append("v3")
            if run_v4: selected.append("v4")

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

# =========================================================
# PAGE: NEWS
# =========================================================
elif page == "📰 News":
    st.subheader("📰 Live Market Intelligence")
    st.caption("AI-categorized news from Moneycontrol, ET, Livemint, BS, Reuters · Auto-refresh 10min")

    # ── Controls row ──
    ncol1, ncol2, ncol3, ncol4 = st.columns([2, 1, 1, 1])
    with ncol1:
        cat_filter = st.selectbox("Category",
            ["All", "🔴 Market-Moving", "📈 Stock-Specific", "🏭 Sector", "🎯 IPO/FPO", "🌍 Macro", "📰 General"],
            index=0, key="news_cat", label_visibility="collapsed")
    with ncol2:
        news_refresh = st.button("🔄 Refresh", key="news_refresh", type="primary", use_container_width=True)
    with ncol3:
        use_ai = st.checkbox("🤖 AI Categorize", True, key="news_ai",
                              help="Use AI to categorize news (requires API key)")
    with ncol4:
        auto_refresh = st.checkbox("Auto", True, key="news_auto", help="Auto-refresh every 10min")

    # ── Init / Cache ──
    if "news_cache" not in st.session_state:
        st.session_state.news_cache = None
        st.session_state.news_cache_time = None

    should_refresh = news_refresh
    if auto_refresh and st.session_state.news_cache_time:
        if (datetime.now() - st.session_state.news_cache_time).total_seconds() > 600:
            should_refresh = True

    if should_refresh or st.session_state.news_cache is None:
        with st.spinner("📡 Fetching & categorizing news..."):
            raw_news = fetch_rss_feeds()
            if use_ai and st.session_state.get("ai_api_key"):
                raw_news = categorize_news_batch_ai(
                    raw_news, api_key=st.session_state.ai_api_key,
                    provider=st.session_state.get("ai_prov_float", "DS"),
                    mode=st.session_state.get("ai_mode", "flash"),
                )
            else:
                for n in raw_news:
                    categorize_news_keyword(n)
            st.session_state.news_cache = raw_news
            st.session_state.news_cache_time = datetime.now()
    else:
        raw_news = st.session_state.news_cache

    # ── Filter by category ──
    cat_map = {
        "All": lambda n: True,
        "🔴 Market-Moving": lambda n: n.get("category") == "market_moving",
        "📈 Stock-Specific": lambda n: n.get("category") == "stock_specific",
        "🏭 Sector": lambda n: n.get("category") == "sector",
        "🎯 IPO/FPO": lambda n: n.get("category") == "ipo_fpo",
        "🌍 Macro": lambda n: n.get("category") == "macro",
        "📰 General": lambda n: n.get("category") == "general",
    }
    filtered = [n for n in raw_news if cat_map.get(cat_filter, lambda n: True)(n)]

    if not filtered:
        st.info("No news items. Click Refresh to fetch.")
    else:
        time_str = st.session_state.news_cache_time.strftime("%H:%M:%S") if st.session_state.news_cache_time else "--"

        # ── BREAKING NEWS TICKER ──
        breaking = [n for n in raw_news if n.get("impact", 0) >= 60][:5]
        if breaking:
            ticker_text = "  🔸  ".join(
                f"{n['title'][:80]}... [{n['source']}]" for n in breaking
            )
            st.markdown(
                f'<div style="background:#1f2937;border-radius:8px;padding:8px 16px;'
                f'margin-bottom:12px;overflow:hidden;white-space:nowrap;">'
                f'<marquee behavior="scroll" direction="left" scrollamount="3" style="color:#f5c542;font-size:14px;">'
                f'🚨 {ticker_text}</marquee></div>',
                unsafe_allow_html=True
            )

        # ── COMPACT FEED TABLE ──
        st.caption(f"Showing {len(filtered)} items · Last refresh: {time_str}")

        # Build rows for compact display
        table_rows = []
        for item in filtered[:40]:
            cat = item.get("category", "general")
            emoji = CATEGORY_EMOJI.get(cat, "📰")
            sent = item.get("sentiment", "neutral")
            sent_icon = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(sent, "⚪")
            impact = item.get("impact", 0)
            impact_bar = "█" * min(10, max(1, impact // 10)) + "░" * max(0, 10 - min(10, impact // 10))
            source = item.get("source", "")
            stocks = item.get("stocks_mentioned", [])
            stocks_str = ", ".join(s.replace(".NS", "") for s in stocks[:3]) if stocks else ""
            published = item.get("published", "")
            time_ago = ""
            if published:
                diff = datetime.now() - published
                mins = int(diff.total_seconds() / 60)
                if mins < 60:
                    time_ago = f"{mins}m ago"
                elif mins < 1440:
                    time_ago = f"{mins // 60}h ago"
                else:
                    time_ago = f"{mins // 1440}d ago"

            table_rows.append({
                " ": emoji,
                "Impact": impact,
                "Title": item["title"],
                "Source": source,
                "Sentiment": sent_icon,
                "Stocks": stocks_str,
                "When": time_ago,
                "_link": item.get("link", "#"),
                "_summary": item.get("summary", ""),
            })

        # Render compact rows
        for i, row in enumerate(table_rows):
            bg = "#1a1f35" if i % 2 == 0 else "#111827"
            sent_color = {"positive": "#00ff9f", "negative": "#ff4d6d", "neutral": "#9ca3af"}.get(
                row["Sentiment"], "#9ca3af"
            )
            impact_pct = min(100, row["Impact"])
            bar_color = "#00ff9f" if row["Impact"] >= 50 else "#f5c542" if row["Impact"] >= 25 else "#4a5568"

            # Compact single-line card
            cols = st.columns([0.5, 3.5, 1.2, 0.8, 1, 0.7])
            with cols[0]:
                st.markdown(f"<span style='font-size:18px;'>{row[' ']}</span>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(
                    f"<div style='font-size:13px;line-height:1.3;'>"
                    f"<a href='{row['_link']}' target='_blank' style='color:#f3f4f6;text-decoration:none;'>{row['Title'][:100]}{'...' if len(row['Title'])>100 else ''}</a>"
                    f"<br><span style='color:#6b7280;font-size:11px;'>{row['Source']} · {row.get('_summary','')[:80]}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with cols[2]:
                st.markdown(
                    f"<div style='font-size:11px;color:#6b7280;'>{row['Stocks']}</div>",
                    unsafe_allow_html=True
                )
            with cols[3]:
                st.markdown(
                    f"<div style='background:{bar_color};border-radius:4px;height:6px;width:{impact_pct}%;min-width:4px;'></div>"
                    f"<span style='font-size:10px;color:#6b7280;'>{row['Impact']}/100</span>",
                    unsafe_allow_html=True
                )
            with cols[4]:
                st.markdown(f"<span style='color:{sent_color};font-size:14px;'>{row['Sentiment']}</span>", unsafe_allow_html=True)
            with cols[5]:
                st.markdown(f"<span style='font-size:11px;color:#6b7280;'>{row['When']}</span>", unsafe_allow_html=True)

        # ── Category summary bar ──
        st.markdown("---")
        cats = ["market_moving", "stock_specific", "sector", "ipo_fpo", "macro", "general"]
        cat_counts = {c: sum(1 for n in raw_news if n.get("category") == c) for c in cats}
        total = sum(cat_counts.values())
        if total > 0:
            cc = st.columns(len(cats))
            for i, c in enumerate(cats):
                pct = cat_counts[c] / total * 100
                emoji = CATEGORY_EMOJI.get(c, "📰")
                label = c.replace("_", " ").title()
                with cc[i]:
                    st.markdown(
                        f"<div style='text-align:center;font-size:12px;'>"
                        f"<span style='font-size:20px;'>{emoji}</span><br>"
                        f"<strong>{cat_counts[c]}</strong><br>"
                        f"<span style='color:#6b7280;'>{label}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

    # ── Stock news scanner ──
    with st.expander("🔍 Check News for a Specific Stock", expanded=False):
        nscol1, nscol2 = st.columns([3, 1])
        with nscol1:
            nsym = st.text_input("Symbol", value="RELIANCE.NS", key="nsym", label_visibility="collapsed",
                                  placeholder="Enter symbol (e.g., RELIANCE.NS)")
        with nscol2:
            if st.button("🔍 Check", key="nsym_btn", use_container_width=True):
                with st.spinner(f"Fetching news for {nsym}..."):
                    stock_news = get_stock_news(nsym, max_items=8)
                if stock_news:
                    for item in stock_news:
                        st.markdown(
                            f"• **[{item['title']}]({item['link']})** "
                            f"<span style='color:#6b7280;font-size:11px;'>{item.get('publisher','')}</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No recent news.")
            with st.spinner(f"Fetching news for {nsym}..."):
                stock_news = get_stock_news(nsym, max_items=10)
            if stock_news:
                for item in stock_news:
                    with st.container(border=True):
                        st.markdown(f"**{item['title']}**")
                        st.caption(f"{item.get('publisher', '')} · {item.get('published', '')}")
                        if item.get("link"):
                            st.markdown(f"[🔗 Read]({item['link']})")
            else:
                st.info("No recent news for this symbol.")

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
        with st.spinner(f"Gathering intelligence on {sym}... (may take 10-15s due to API rate limits)"):
            try:
                import yfinance as yf
                import time as _time

                # Rate limit: add delay before first call to be respectful
                _time.sleep(0.5)
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
                        _time.sleep(0.5)
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
                    for idx, p in enumerate(peers[:5]):
                        try:
                            _time.sleep(0.5)  # Rate limit: 500ms between peer calls
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
                    sig_c = st.columns(2)
                    with sig_c[0]:
                        sv4 = generate_v4_signal(df)
                        st.metric("v4 Signal", sv4.get("Signal", "NEUTRAL"), f"Score: {sv4.get('Score', 0)}/10")
                    with sig_c[1]:
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
                err_str = str(e)
                if "rate" in err_str.lower() or "429" in err_str or "many requests" in err_str.lower():
                    st.error("⏳ **Rate limited by Yahoo Finance.** This happens when too many requests are sent quickly. Wait 60 seconds and try again. The system auto-adds delays between calls, but Yahoo's free tier has tight limits.")
                    st.caption("💡 Tip: You can also check this stock on the **📈 Daily Trades** or **📆 Swing Setups** pages which use cached data.")
                else:
                    st.error(f"Research failed: {e}")

