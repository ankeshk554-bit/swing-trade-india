"""
Advanced Scan Engine — Sniper Terminal
=======================================
Provides 50+ scan conditions organized into 7 categories plus
pre-built trading strategy themes for the screener.

Categories:
  1. TREND       — ADX, EMA slope, MA alignment, SuperTrend, Golden/Death cross
  2. MOMENTUM    — RSI bands, Stochastic, MACD, Rate of Change, MFI, Williams %R
  3. VOLUME      — RVOL, volume spike, delivery %, accumulation, OBV, VPT
  4. PATTERN     — VCP, Bull Flag, Double Bottom, Cup&Handle, Breakout, Engulfing
  5. STRUCTURE   — Price vs MAs, BB position, ATR range, 52W proximity, HH/HL
  6. RS_FUNDA    — RS rating, sector rank, F&O status, delivery quality, score
  7. INDIAN      — F&O expiry week, OI strike proximity, delivery spurt, FII activity

Each condition returns a dict: {"passed": bool, "details": str}
"""

import numpy as np
import logging

from core.indicators import (
    compute_indicators, compute_institutional_score,
    generate_swing_signal, detect_vcp
)
from core.patterns import run_all_patterns, get_best_pattern
from core.data_providers import get_india_vix

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# CATEGORY 1: TREND CONDITIONS (10)
# ══════════════════════════════════════════════

def c_trend_adx_strong(df) -> dict:
    """ADX > 25 — Strong trend."""
    val = float(df["ADX"].iloc[-1]) if "ADX" in df.columns and not np.isnan(df["ADX"].iloc[-1]) else 0
    return {"passed": val > 25, "value": round(val, 1)}


def c_trend_adx_weak(df) -> dict:
    """ADX < 20 — Weak / ranging market."""
    val = float(df["ADX"].iloc[-1]) if "ADX" in df.columns and not np.isnan(df["ADX"].iloc[-1]) else 0
    return {"passed": val < 20, "value": round(val, 1)}


def c_trend_ema50_slope_positive(df) -> dict:
    """EMA50 slope positive over last 5 bars."""
    if "EMA50" not in df.columns or len(df) < 10:
        return {"passed": False, "value": 0}
    slope = (df["EMA50"].iloc[-1] / df["EMA50"].iloc[-5] - 1) * 100
    return {"passed": slope > 0.5, "value": round(slope, 2)}


def c_trend_golden_cross(df) -> dict:
    """EMA50 just crossed above EMA200 (golden cross)."""
    if "EMA50" not in df.columns or "EMA200" not in df.columns or len(df) < 3:
        return {"passed": False}
    prev = df["EMA50"].iloc[-2] > df["EMA200"].iloc[-2]
    curr = df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1]
    # Just crossed if not crossed before but is now
    just_crossed = not prev and curr
    return {"passed": just_crossed, "value": "Yes" if just_crossed else "No"}


def c_trend_death_cross(df) -> dict:
    """EMA50 just crossed below EMA200 (death cross)."""
    if "EMA50" not in df.columns or "EMA200" not in df.columns or len(df) < 3:
        return {"passed": False}
    prev = df["EMA50"].iloc[-2] < df["EMA200"].iloc[-2]
    curr = df["EMA50"].iloc[-1] < df["EMA200"].iloc[-1]
    just_crossed = not prev and curr
    return {"passed": just_crossed, "value": "Yes" if just_crossed else "No"}


def c_trend_price_above_ema50(df) -> dict:
    """Close > EMA50."""
    if "EMA50" not in df.columns:
        return {"passed": False}
    val = float(df["Close"].iloc[-1]) > float(df["EMA50"].iloc[-1])
    return {"passed": val}


def c_trend_price_above_ema200(df) -> dict:
    """Close > EMA200."""
    if "EMA200" not in df.columns:
        return {"passed": False}
    val = float(df["Close"].iloc[-1]) > float(df["EMA200"].iloc[-1])
    return {"passed": val}


def c_trend_alignment_bullish(df) -> dict:
    """Bullish alignment: Close > EMA50 > EMA200."""
    if "EMA50" not in df.columns or "EMA200" not in df.columns:
        return {"passed": False}
    c = float(df["Close"].iloc[-1])
    e50 = float(df["EMA50"].iloc[-1])
    e200 = float(df["EMA200"].iloc[-1])
    return {"passed": c > e50 > e200}


def c_trend_alignment_bearish(df) -> dict:
    """Bearish alignment: Close < EMA50 < EMA200."""
    if "EMA50" not in df.columns or "EMA200" not in df.columns:
        return {"passed": False}
    c = float(df["Close"].iloc[-1])
    e50 = float(df["EMA50"].iloc[-1])
    e200 = float(df["EMA200"].iloc[-1])
    return {"passed": c < e50 < e200}


def c_trend_supertrend_up(df) -> dict:
    """Supertrend direction is UP."""
    if "SUPERTREND_DIR" not in df.columns:
        return {"passed": False}
    return {"passed": df["SUPERTREND_DIR"].iloc[-1] == 1}


def c_trend_supertrend_down(df) -> dict:
    """Supertrend direction is DOWN."""
    if "SUPERTREND_DIR" not in df.columns:
        return {"passed": False}
    return {"passed": df["SUPERTREND_DIR"].iloc[-1] == -1}


# ══════════════════════════════════════════════
# CATEGORY 2: MOMENTUM CONDITIONS (10)
# ══════════════════════════════════════════════

def c_momentum_rsi_oversold(df) -> dict:
    """RSI < 35 — Oversold."""
    val = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 50
    return {"passed": val < 35, "value": round(val, 1)}


def c_momentum_rsi_overbought(df) -> dict:
    """RSI > 75 — Overbought."""
    val = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 50
    return {"passed": val > 75, "value": round(val, 1)}


def c_momentum_rsi_bullish(df) -> dict:
    """RSI between 55-75 — Bullish zone."""
    val = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else 50
    return {"passed": 55 <= val <= 75, "value": round(val, 1)}


def c_momentum_rsi_rising(df) -> dict:
    """RSI rising over last 3 bars."""
    if "RSI" not in df.columns or len(df) < 5:
        return {"passed": False}
    rsi_vals = df["RSI"].tail(5).values
    return {"passed": all(rsi_vals[i] >= rsi_vals[i - 1] for i in range(1, len(rsi_vals)))}


def c_momentum_macd_bullish(df) -> dict:
    """MACD histogram positive and MACD > Signal."""
    if "MACD_HIST" not in df.columns or "MACD" not in df.columns or "MACD_SIGNAL" not in df.columns:
        return {"passed": False}
    hist = float(df["MACD_HIST"].iloc[-1])
    macd = float(df["MACD"].iloc[-1])
    signal = float(df["MACD_SIGNAL"].iloc[-1])
    return {"passed": hist > 0 and macd > signal, "value": round(hist, 4)}


def c_momentum_macd_bearish(df) -> dict:
    """MACD histogram negative and MACD < Signal."""
    if "MACD_HIST" not in df.columns or "MACD" not in df.columns or "MACD_SIGNAL" not in df.columns:
        return {"passed": False}
    hist = float(df["MACD_HIST"].iloc[-1])
    macd = float(df["MACD"].iloc[-1])
    signal = float(df["MACD_SIGNAL"].iloc[-1])
    return {"passed": hist < 0 and macd < signal, "value": round(hist, 4)}


def c_momentum_macd_crossover(df) -> dict:
    """MACD line crossed above signal line."""
    if "MACD" not in df.columns or "MACD_SIGNAL" not in df.columns or len(df) < 3:
        return {"passed": False}
    prev = float(df["MACD"].iloc[-2]) <= float(df["MACD_SIGNAL"].iloc[-2])
    curr = float(df["MACD"].iloc[-1]) > float(df["MACD_SIGNAL"].iloc[-1])
    return {"passed": prev and curr}


def c_momentum_roc_positive(df) -> dict:
    """Rate of Change (10-bar) positive."""
    if len(df) < 12:
        return {"passed": False}
    roc = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-11]) - 1) * 100
    return {"passed": roc > 2, "value": round(roc, 2)}


def c_momentum_mfi_smart_money(df) -> dict:
    """MFI > 60 — Smart money flowing in."""
    try:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        money_flow = typical * df["Volume"]
        period = 14
        mf = money_flow.tail(period)
        pos = mf[mf.diff() > 0].sum()
        neg = abs(mf[mf.diff() < 0].sum())
        mfi = 100 - (100 / (1 + pos / max(neg, 1)))
        return {"passed": mfi > 60, "value": round(mfi, 1)}
    except Exception:
        return {"passed": False, "value": 0}


def c_momentum_bb_squeeze(df) -> dict:
    """Bollinger Band width contracting — squeeze setup."""
    if "BB_WIDTH" not in df.columns or len(df) < 20:
        return {"passed": False}
    current_width = float(df["BB_WIDTH"].iloc[-1])
    avg_width = df["BB_WIDTH"].tail(20).mean()
    return {"passed": current_width < avg_width * 0.8, "value": round(current_width / avg_width, 2)}


# ══════════════════════════════════════════════
# CATEGORY 3: VOLUME CONDITIONS (8)
# ══════════════════════════════════════════════

def c_volume_rvol_high(df) -> dict:
    """RVOL > 1.5 — Volume above average."""
    val = float(df["RVOL"].iloc[-1]) if "RVOL" in df.columns else 0
    return {"passed": val > 1.5, "value": round(val, 2)}


def c_volume_rvol_extreme(df) -> dict:
    """RVOL > 2.5 — Extreme volume spike."""
    val = float(df["RVOL"].iloc[-1]) if "RVOL" in df.columns else 0
    return {"passed": val > 2.5, "value": round(val, 2)}


def c_volume_rvol_low(df) -> dict:
    """RVOL < 0.7 — Low volume (drying up)."""
    val = float(df["RVOL"].iloc[-1]) if "RVOL" in df.columns else 1
    return {"passed": val < 0.7, "value": round(val, 2)}


def c_volume_spike(df) -> dict:
    """Volume > 2x 20-day average."""
    if "VOL_MA20" not in df.columns:
        return {"passed": False}
    vol = float(df["Volume"].iloc[-1])
    vol_ma = float(df["VOL_MA20"].iloc[-1])
    if vol_ma <= 0:
        return {"passed": False}
    ratio = vol / vol_ma
    return {"passed": ratio > 2, "value": round(ratio, 2)}


def c_volume_rising_trend(df) -> dict:
    """Volume trending up over last 10 days."""
    if len(df) < 15:
        return {"passed": False}
    vol_vals = df["Volume"].tail(10).values
    if np.mean(vol_vals[-5:]) > np.mean(vol_vals[:5]) * 1.3:
        return {"passed": True, "value": "Rising"}
    return {"passed": False, "value": "Flat/Falling"}


def c_volume_obv_rising(df) -> dict:
    """OBV making higher highs — accumulation."""
    if "OBV" not in df.columns or len(df) < 10:
        return {"passed": False}
    obv = df["OBV"].tail(10).values
    try:
        slope = np.polyfit(range(len(obv)), obv, 1)[0]
        return {"passed": slope > 0, "value": round(slope, 2)}
    except Exception:
        return {"passed": False}


def c_volume_delivery_high(df, delivery=None) -> dict:
    """Latest delivery % > 35."""
    if delivery:
        d = delivery.get("latest_delivery_pct", 0) or 0
        return {"passed": d > 35, "value": d}
    return {"passed": False}


def c_volume_delivery_spurt(df, delivery=None) -> dict:
    """Delivery spurt detected."""
    if delivery:
        spurt = delivery.get("delivery_spurt", False)
        return {"passed": spurt, "value": "Yes" if spurt else "No"}
    return {"passed": False}


# ══════════════════════════════════════════════
# CATEGORY 4: PATTERN CONDITIONS (8)
# ══════════════════════════════════════════════

def c_pattern_vcp(df, vcp=None) -> dict:
    """VCP pattern detected."""
    if vcp is None:
        vcp = detect_vcp(df)
    stage = vcp.get("Stage", 0)
    return {"passed": vcp["VCP_Flag"], "value": f"Stage {stage}"}


def c_pattern_vcp_volume_dryup(df, vcp=None) -> dict:
    """VCP with volume drying up (higher quality)."""
    if vcp is None:
        vcp = detect_vcp(df)
    return {"passed": vcp["VCP_Flag"] and vcp.get("VolumeDryUp", False), "value": "Yes" if vcp.get("VolumeDryUp") else "No"}


def c_pattern_bull_flag(df, patterns=None) -> dict:
    """Bull Flag pattern detected."""
    if patterns is None:
        patterns = run_all_patterns(df)
    bf = patterns.get("bull_flag", {})
    return {"passed": bf.get("flag", False), "value": f"Conf {bf.get('confidence', 0)}"}


def c_pattern_double_bottom(df, patterns=None) -> dict:
    """Double Bottom pattern."""
    if patterns is None:
        patterns = run_all_patterns(df)
    db = patterns.get("double_bottom", {})
    return {"passed": db.get("pattern", False), "value": f"Conf {db.get('confidence', 0)}"}


def c_pattern_cup_handle(df, patterns=None) -> dict:
    """Cup & Handle pattern."""
    if patterns is None:
        patterns = run_all_patterns(df)
    ch = patterns.get("cup_handle", {})
    return {"passed": ch.get("pattern", False), "value": f"Conf {ch.get('confidence', 0)}"}


def c_pattern_breakout(df, patterns=None) -> dict:
    """Consolidation Breakout pattern."""
    if patterns is None:
        patterns = run_all_patterns(df)
    br = patterns.get("consolidation_breakout", {})
    return {"passed": br.get("pattern", False), "value": f"{br.get('breakout_pct', 0):.1f}%"}


def c_pattern_52w_high(df, patterns=None) -> dict:
    """Near 52-week high (within 5%)."""
    if patterns is None:
        patterns = run_all_patterns(df)
    prox = patterns.get("52w", {})
    return {"passed": prox.get("near_high", False), "value": f"{prox.get('high_proximity', 0)*100:.0f}%"}


def c_pattern_engulfing_bullish(df, patterns=None) -> dict:
    """Bullish Engulfing candle pattern."""
    if patterns is None:
        patterns = run_all_patterns(df)
    eng = patterns.get("engulfing", {})
    return {"passed": eng.get("type") == "BULLISH_ENGULFING", "value": eng.get("type", "NONE")}


# ══════════════════════════════════════════════
# CATEGORY 5: STRUCTURE CONDITIONS (8)
# ══════════════════════════════════════════════

def c_structure_price_above_bb_mid(df) -> dict:
    """Close above BB middle band."""
    if "BB_MID" not in df.columns:
        return {"passed": False}
    c = float(df["Close"].iloc[-1])
    m = float(df["BB_MID"].iloc[-1])
    return {"passed": c > m}


def c_structure_price_near_bb_lower(df) -> dict:
    """Close near BB lower band (BB_Pos < 0.2)."""
    if "BB_POS" not in df.columns:
        return {"passed": False}
    pos = float(df["BB_POS"].iloc[-1])
    return {"passed": pos < 0.2, "value": round(pos, 2)}


def c_structure_price_near_bb_upper(df) -> dict:
    """Close near BB upper band (BB_Pos > 0.8)."""
    if "BB_POS" not in df.columns:
        return {"passed": False}
    pos = float(df["BB_POS"].iloc[-1])
    return {"passed": pos > 0.8, "value": round(pos, 2)}


def c_structure_atr_low(df) -> dict:
    """ATR% < 2% — Low volatility."""
    if "ATR_PCT" not in df.columns:
        return {"passed": False}
    atr = float(df["ATR_PCT"].iloc[-1])
    return {"passed": atr < 2, "value": round(atr, 2)}


def c_structure_atr_high(df) -> dict:
    """ATR% > 4% — High volatility."""
    if "ATR_PCT" not in df.columns:
        return {"passed": False}
    atr = float(df["ATR_PCT"].iloc[-1])
    return {"passed": atr > 4, "value": round(atr, 2)}


def c_structure_uptrend_structure(df, patterns=None) -> dict:
    """Higher highs + higher lows structure."""
    if patterns is None:
        patterns = run_all_patterns(df)
    s = patterns.get("structure", {})
    return {"passed": s.get("type") == "UPTREND_HH_HL", "value": s.get("type", "")}


def c_structure_consolidation_tight(df) -> dict:
    """Price action tight over last 20 days (range < 8%)."""
    if len(df) < 25:
        return {"passed": False}
    recent = df.tail(20)
    price_range = (recent["High"].max() / recent["Low"].min() - 1) * 100
    return {"passed": price_range < 8, "value": round(price_range, 1)}


def c_structure_inside_bar(df, patterns=None) -> dict:
    """Inside Bar pattern detected."""
    if patterns is None:
        patterns = run_all_patterns(df)
    ib = patterns.get("inside_bar", {})
    return {"passed": ib.get("pattern", False) and ib.get("bar_count", 0) >= 2, "value": f"{ib.get('bar_count', 0)} bars"}


# ══════════════════════════════════════════════
# CATEGORY 6: RS & FUNDAMENTAL CONDITIONS (6)
# ══════════════════════════════════════════════

def c_rs_score_high(df) -> dict:
    """Institutional score >= 75."""
    score = compute_institutional_score(df)
    return {"passed": score >= 75, "value": score}


def c_rs_score_medium(df) -> dict:
    """Institutional score 50-74."""
    score = compute_institutional_score(df)
    return {"passed": 50 <= score < 75, "value": score}


def c_rs_score_low(df) -> dict:
    """Institutional score < 35."""
    score = compute_institutional_score(df)
    return {"passed": score < 35, "value": score}


def c_rs_buy_signal(df) -> dict:
    """Swing signal = BUY."""
    sig = generate_swing_signal(df)
    return {"passed": sig["Signal"] == "BUY", "value": f"Conf {sig['Confidence']}"}


def c_rs_sell_signal(df) -> dict:
    """Swing signal = SELL."""
    sig = generate_swing_signal(df)
    return {"passed": sig["Signal"] == "SELL", "value": f"Conf {sig['Confidence']}"}


def c_rs_high_conviction(df) -> dict:
    """High conviction buy: BUY signal + Confidence >= 4."""
    sig = generate_swing_signal(df)
    return {"passed": sig["Signal"] == "BUY" and sig["Confidence"] >= 4, "value": f"Conf {sig['Confidence']}"}


# ══════════════════════════════════════════════
# CATEGORY 7: INDIAN MARKET-SPECIFIC (6)
# ══════════════════════════════════════════════

def c_indian_fo_available(stock_fo=None) -> dict:
    """Stock is F&O tradable."""
    if stock_fo:
        return {"passed": stock_fo.get("in_fo", False)}
    return {"passed": False}


def c_indian_delivery_quality(delivery=None) -> dict:
    """Delivery quality = STRONG."""
    if delivery:
        return {"passed": delivery.get("delivery_quality") == "STRONG", "value": delivery.get("delivery_quality", "")}
    return {"passed": False}


def c_indian_accumulation(delivery=None) -> dict:
    """Accumulation days >= 5 in last 10."""
    if delivery:
        acc = delivery.get("accumulation_days_10", 0)
        return {"passed": acc >= 5, "value": acc}
    return {"passed": False}


def c_indian_delivery_rising(delivery=None) -> dict:
    """Delivery trend is rising."""
    if delivery:
        return {"passed": delivery.get("delivery_trend") == "RISING", "value": delivery.get("delivery_trend", "")}
    return {"passed": False}


def c_indian_fo_ban(stock_fo=None) -> dict:
    """NOT in F&O ban period."""
    in_ban = stock_fo.get("in_ban", False) if stock_fo else False
    return {"passed": not in_ban, "value": "In ban" if in_ban else "Free"}


def c_indian_vix_favorable(df, vix=None) -> dict:
    """VIX regime favorable for swing trading."""
    if vix is None:
        vix = get_india_vix()
    regime = vix.get("regime", "UNKNOWN")
    favorable = regime in ("LOW_VOL", "NORMAL")
    return {"passed": favorable, "value": regime}


# ══════════════════════════════════════════════
# CONDITION REGISTRY
# ══════════════════════════════════════════════

# Conditions organized by category with metadata
CONDITION_REGISTRY = {
    # Category 1: Trend (10)
    "trend_adx_strong": {"name": "ADX > 25 (Strong Trend)", "category": "TREND", "fn": c_trend_adx_strong, "requires_df": True},
    "trend_adx_weak": {"name": "ADX < 20 (Weak Trend)", "category": "TREND", "fn": c_trend_adx_weak, "requires_df": True},
    "trend_ema50_slope": {"name": "EMA50 Slope Positive", "category": "TREND", "fn": c_trend_ema50_slope_positive, "requires_df": True},
    "trend_golden_cross": {"name": "Golden Cross (50>200)", "category": "TREND", "fn": c_trend_golden_cross, "requires_df": True},
    "trend_death_cross": {"name": "Death Cross (50<200)", "category": "TREND", "fn": c_trend_death_cross, "requires_df": True},
    "trend_price_above_ema50": {"name": "Close > EMA50", "category": "TREND", "fn": c_trend_price_above_ema50, "requires_df": True},
    "trend_price_above_ema200": {"name": "Close > EMA200", "category": "TREND", "fn": c_trend_price_above_ema200, "requires_df": True},
    "trend_bullish_alignment": {"name": "Bullish Align (C>50>200)", "category": "TREND", "fn": c_trend_alignment_bullish, "requires_df": True},
    "trend_bearish_alignment": {"name": "Bearish Align (C<50<200)", "category": "TREND", "fn": c_trend_alignment_bearish, "requires_df": True},
    "trend_supertrend_up": {"name": "Supertrend UP", "category": "TREND", "fn": c_trend_supertrend_up, "requires_df": True},
    "trend_supertrend_down": {"name": "Supertrend DOWN", "category": "TREND", "fn": c_trend_supertrend_down, "requires_df": True},

    # Category 2: Momentum (10)
    "momentum_rsi_oversold": {"name": "RSI < 35 (Oversold)", "category": "MOMENTUM", "fn": c_momentum_rsi_oversold, "requires_df": True},
    "momentum_rsi_overbought": {"name": "RSI > 75 (Overbought)", "category": "MOMENTUM", "fn": c_momentum_rsi_overbought, "requires_df": True},
    "momentum_rsi_bullish": {"name": "RSI 55-75 (Bullish)", "category": "MOMENTUM", "fn": c_momentum_rsi_bullish, "requires_df": True},
    "momentum_rsi_rising": {"name": "RSI Rising (3 bars)", "category": "MOMENTUM", "fn": c_momentum_rsi_rising, "requires_df": True},
    "momentum_macd_bullish": {"name": "MACD Bullish (Hist>0)", "category": "MOMENTUM", "fn": c_momentum_macd_bullish, "requires_df": True},
    "momentum_macd_bearish": {"name": "MACD Bearish (Hist<0)", "category": "MOMENTUM", "fn": c_momentum_macd_bearish, "requires_df": True},
    "momentum_macd_crossover": {"name": "MACD Crossover Up", "category": "MOMENTUM", "fn": c_momentum_macd_crossover, "requires_df": True},
    "momentum_roc_positive": {"name": "ROC 10d > 2%", "category": "MOMENTUM", "fn": c_momentum_roc_positive, "requires_df": True},
    "momentum_mfi_smart_money": {"name": "MFI > 60 (Smart $)", "category": "MOMENTUM", "fn": c_momentum_mfi_smart_money, "requires_df": True},
    "momentum_bb_squeeze": {"name": "BB Squeeze (Low Vol)", "category": "MOMENTUM", "fn": c_momentum_bb_squeeze, "requires_df": True},

    # Category 3: Volume (8)
    "volume_rvol_high": {"name": "RVOL > 1.5", "category": "VOLUME", "fn": c_volume_rvol_high, "requires_df": True},
    "volume_rvol_extreme": {"name": "RVOL > 2.5 (Extreme)", "category": "VOLUME", "fn": c_volume_rvol_extreme, "requires_df": True},
    "volume_rvol_low": {"name": "RVOL < 0.7 (Drying)", "category": "VOLUME", "fn": c_volume_rvol_low, "requires_df": True},
    "volume_spike": {"name": "Volume > 2x Avg", "category": "VOLUME", "fn": c_volume_spike, "requires_df": True},
    "volume_rising_trend": {"name": "Volume Trending Up", "category": "VOLUME", "fn": c_volume_rising_trend, "requires_df": True},
    "volume_obv_rising": {"name": "OBV Rising", "category": "VOLUME", "fn": c_volume_obv_rising, "requires_df": True},
    "volume_delivery_high": {"name": "Delivery % > 35%", "category": "VOLUME", "fn": c_volume_delivery_high, "requires_df": False, "needs_delivery": True},
    "volume_delivery_spurt": {"name": "Delivery Spurt", "category": "VOLUME", "fn": c_volume_delivery_spurt, "requires_df": False, "needs_delivery": True},

    # Category 4: Pattern (8)
    "pattern_vcp": {"name": "VCP Pattern", "category": "PATTERN", "fn": c_pattern_vcp, "requires_df": True},
    "pattern_vcp_dryup": {"name": "VCP + Vol Dry-Up", "category": "PATTERN", "fn": c_pattern_vcp_volume_dryup, "requires_df": True},
    "pattern_bull_flag": {"name": "Bull Flag", "category": "PATTERN", "fn": c_pattern_bull_flag, "requires_df": True},
    "pattern_double_bottom": {"name": "Double Bottom", "category": "PATTERN", "fn": c_pattern_double_bottom, "requires_df": True},
    "pattern_cup_handle": {"name": "Cup & Handle", "category": "PATTERN", "fn": c_pattern_cup_handle, "requires_df": True},
    "pattern_breakout": {"name": "Consolidation Breakout", "category": "PATTERN", "fn": c_pattern_breakout, "requires_df": True},
    "pattern_52w_high": {"name": "Near 52W High", "category": "PATTERN", "fn": c_pattern_52w_high, "requires_df": True},
    "pattern_engulfing_bullish": {"name": "Bullish Engulfing", "category": "PATTERN", "fn": c_pattern_engulfing_bullish, "requires_df": True},

    # Category 5: Structure (8)
    "structure_above_bb_mid": {"name": "Close > BB Mid", "category": "STRUCTURE", "fn": c_structure_price_above_bb_mid, "requires_df": True},
    "structure_near_bb_lower": {"name": "Near BB Lower", "category": "STRUCTURE", "fn": c_structure_price_near_bb_lower, "requires_df": True},
    "structure_near_bb_upper": {"name": "Near BB Upper", "category": "STRUCTURE", "fn": c_structure_price_near_bb_upper, "requires_df": True},
    "structure_atr_low": {"name": "ATR < 2% (Low Vol)", "category": "STRUCTURE", "fn": c_structure_atr_low, "requires_df": True},
    "structure_atr_high": {"name": "ATR > 4% (High Vol)", "category": "STRUCTURE", "fn": c_structure_atr_high, "requires_df": True},
    "structure_uptrend": {"name": "HH/HL Uptrend", "category": "STRUCTURE", "fn": c_structure_uptrend_structure, "requires_df": True},
    "structure_tight_range": {"name": "Tight Range (<8%)", "category": "STRUCTURE", "fn": c_structure_consolidation_tight, "requires_df": True},
    "structure_inside_bar": {"name": "Inside Bar Pattern", "category": "STRUCTURE", "fn": c_structure_inside_bar, "requires_df": True},

    # Category 6: RS & Score (6)
    "rs_score_high": {"name": "Score 75+", "category": "RS_SCORE", "fn": c_rs_score_high, "requires_df": True},
    "rs_score_medium": {"name": "Score 50-74", "category": "RS_SCORE", "fn": c_rs_score_medium, "requires_df": True},
    "rs_score_low": {"name": "Score < 35", "category": "RS_SCORE", "fn": c_rs_score_low, "requires_df": True},
    "rs_buy_signal": {"name": "BUY Signal", "category": "RS_SCORE", "fn": c_rs_buy_signal, "requires_df": True},
    "rs_sell_signal": {"name": "SELL Signal", "category": "RS_SCORE", "fn": c_rs_sell_signal, "requires_df": True},
    "rs_high_conviction": {"name": "High Conviction Buy", "category": "RS_SCORE", "fn": c_rs_high_conviction, "requires_df": True},

    # Category 7: Indian-specific (6)
    "indian_fo": {"name": "F&O Tradable", "category": "INDIAN", "fn": c_indian_fo_available, "requires_df": False, "needs_fo": True},
    "indian_delivery_quality": {"name": "Delivery Quality Strong", "category": "INDIAN", "fn": c_indian_delivery_quality, "requires_df": False, "needs_delivery": True},
    "indian_accumulation": {"name": "Accumulation 5+/10d", "category": "INDIAN", "fn": c_indian_accumulation, "requires_df": False, "needs_delivery": True},
    "indian_delivery_rising": {"name": "Delivery Trend Rising", "category": "INDIAN", "fn": c_indian_delivery_rising, "requires_df": False, "needs_delivery": True},
    "indian_not_ban": {"name": "NOT in F&O Ban", "category": "INDIAN", "fn": c_indian_fo_ban, "requires_df": False, "needs_fo": True},
    "indian_vix_favorable": {"name": "VIX Regime Favorable", "category": "INDIAN", "fn": c_indian_vix_favorable, "requires_df": False},
}

# ══════════════════════════════════════════════
# PRE-BUILT STRATEGIES
# ══════════════════════════════════════════════

STRATEGIES = {
    "MOMENTUM_RUNNER": {
        "name": "🏃 Momentum Runner",
        "description": "Stocks with strong momentum, high volume, and bullish structure",
        "icon": "🏃",
        "conditions": [
            "trend_bullish_alignment", "momentum_rsi_bullish",
            "momentum_macd_bullish", "volume_rvol_high",
            "rs_buy_signal", "trend_supertrend_up",
            "momentum_roc_positive"
        ],
        "min_conditions": 5,
        "timeframe": "Daily swing (3-10 days)"
    },
    "VCP_BREAKOUT": {
        "name": "📐 VCP Breakout",
        "description": "Volatility Contraction Pattern — Mark Minervini style",
        "icon": "📐",
        "conditions": [
            "pattern_vcp", "pattern_vcp_dryup",
            "volume_rvol_high", "trend_bullish_alignment",
            "rs_buy_signal", "structure_uptrend",
            "structure_tight_range"
        ],
        "min_conditions": 4,
        "timeframe": "Swing (5-20 days)"
    },
    "MEAN_REVERSION": {
        "name": "🔄 Mean Reversion",
        "description": "Oversold stocks bouncing from support levels",
        "icon": "🔄",
        "conditions": [
            "momentum_rsi_oversold", "structure_near_bb_lower",
            "trend_price_above_ema200", "volume_rvol_low",
            "structure_atr_low", "momentum_bb_squeeze"
        ],
        "min_conditions": 3,
        "timeframe": "Short-term (1-5 days)"
    },
    "BREAKOUT_52W": {
        "name": "🚀 52-Week Breakout",
        "description": "Stocks breaking to new 52-week highs with volume",
        "icon": "🚀",
        "conditions": [
            "pattern_52w_high", "volume_rvol_high",
            "trend_bullish_alignment", "rs_high_conviction",
            "momentum_macd_bullish", "trend_supertrend_up",
            "structure_uptrend"
        ],
        "min_conditions": 4,
        "timeframe": "Swing to Trend (5-30 days)"
    },
    "DELIVERY_SPURT": {
        "name": "📦 Delivery Spurt",
        "description": "High delivery volume indicating institutional accumulation",
        "icon": "📦",
        "conditions": [
            "volume_delivery_high", "volume_delivery_spurt",
            "indian_delivery_quality", "indian_accumulation",
            "indian_delivery_rising", "trend_bullish_alignment",
            "rs_buy_signal"
        ],
        "min_conditions": 4,
        "timeframe": "Swing (5-15 days)"
    },
    "GOLDEN_CROSS": {
        "name": "🥇 Golden Cross",
        "description": "Fresh EMA50 > EMA200 crossover with volume confirmation",
        "icon": "🥇",
        "conditions": [
            "trend_golden_cross", "volume_rvol_high",
            "momentum_rsi_bullish", "momentum_macd_bullish",
            "trend_price_above_ema200", "trend_supertrend_up"
        ],
        "min_conditions": 3,
        "timeframe": "Trend (10-60 days)"
    },
    "MACD_MOMENTUM": {
        "name": "📈 MACD Momentum",
        "description": "Fresh MACD crossover with rising momentum",
        "icon": "📈",
        "conditions": [
            "momentum_macd_crossover", "momentum_macd_bullish",
            "momentum_rsi_rising", "volume_rvol_high",
            "trend_price_above_ema50", "rs_high_conviction"
        ],
        "min_conditions": 3,
        "timeframe": "Swing (3-10 days)"
    },
    "CONSOLIDATION_BREAKOUT": {
        "name": "📊 Consolidation Breakout",
        "description": "Stocks breaking out of tight consolidation ranges",
        "icon": "📊",
        "conditions": [
            "pattern_breakout", "structure_tight_range",
            "volume_rvol_high", "trend_supertrend_up",
            "rs_buy_signal", "structure_inside_bar"
        ],
        "min_conditions": 3,
        "timeframe": "Swing (3-15 days)"
    },
    "BULL_FLAG": {
        "name": "🚩 Bull Flag",
        "description": "Bull flag pattern — sharp rally followed by consolidation",
        "icon": "🚩",
        "conditions": [
            "pattern_bull_flag", "trend_supertrend_up",
            "volume_rvol_high", "momentum_macd_bullish",
            "rs_buy_signal"
        ],
        "min_conditions": 3,
        "timeframe": "Swing (3-10 days)"
    },
    "WEAK_SELL": {
        "name": "🔴 Weak / Sell",
        "description": "Stocks showing weakness — potential short candidates",
        "icon": "🔴",
        "conditions": [
            "trend_bearish_alignment", "momentum_macd_bearish",
            "momentum_rsi_overbought", "trend_supertrend_down",
            "rs_sell_signal", "structure_near_bb_upper"
        ],
        "min_conditions": 3,
        "timeframe": "Short-term"
    }
}


# ══════════════════════════════════════════════
# CONDITION EVALUATOR
# ══════════════════════════════════════════════

def evaluate_conditions(
    condition_keys: list,
    df,
    delivery=None,
    stock_fo=None,
    vix=None,
    patterns=None
) -> dict:
    """
    Evaluate a list of conditions against stock data.

    Args:
        condition_keys: List of condition IDs (from CONDITION_REGISTRY)
        df: DataFrame with indicators
        delivery: Delivery summary dict (optional)
        stock_fo: F&O data dict (optional)
        vix: India VIX dict (optional)
        patterns: Pre-computed patterns dict (optional)

    Returns:
        dict with passed_conditions, total_conditions, results dict
    """
    results = {}
    passed = 0
    total = len(condition_keys)

    for key in condition_keys:
        info = CONDITION_REGISTRY.get(key)
        if not info:
            results[key] = {"passed": False, "value": "Unknown condition"}
            continue

        fn = info["fn"]
        try:
            if info.get("needs_delivery"):
                result = fn(df, delivery=delivery)
            elif info.get("needs_fo"):
                result = fn(stock_fo)
            elif info.get("requires_df"):
                if key.startswith("pattern_"):
                    result = fn(df, patterns=patterns)
                else:
                    result = fn(df)
            else:
                if key == "indian_vix_favorable":
                    result = fn(df, vix=vix)
                else:
                    result = fn()

            results[key] = result
            if result.get("passed", False):
                passed += 1

        except Exception as e:
            logger.debug(f"Condition {key} failed: {e}")
            results[key] = {"passed": False, "value": f"Error: {str(e)[:30]}"}

    return {
        "passed": passed,
        "total": total,
        "passed_pct": round(passed / max(total, 1) * 100, 1),
        "results": results
    }


def evaluate_strategy(
    strategy_id: str,
    df,
    delivery=None,
    stock_fo=None,
    vix=None,
    patterns=None
) -> dict:
    """
    Evaluate a pre-built strategy against stock data.

    Args:
        strategy_id: Key from STRATEGIES dict
        df: DataFrame with indicators
        delivery, stock_fo, vix, patterns: Optional context data

    Returns:
        dict with matched, total, min_required, conditions_passed, etc.
    """
    strategy = STRATEGIES.get(strategy_id)
    if not strategy:
        return {"match": False, "error": "Unknown strategy"}

    condition_keys = strategy["conditions"]
    min_required = strategy.get("min_conditions", len(condition_keys))

    eval_result = evaluate_conditions(condition_keys, df, delivery, stock_fo, vix, patterns)
    matched = eval_result["passed"] >= min_required

    return {
        "match": matched,
        "strategy_id": strategy_id,
        "strategy_name": strategy["name"],
        "passed": eval_result["passed"],
        "total": eval_result["total"],
        "min_required": min_required,
        "passed_pct": eval_result["passed_pct"],
        "conditions": eval_result["results"]
    }


def evaluate_all_strategies(df, delivery=None, stock_fo=None, vix=None, patterns=None) -> dict:
    """Evaluate all pre-built strategies against a stock."""
    results = {}
    for sid in STRATEGIES:
        results[sid] = evaluate_strategy(sid, df, delivery, stock_fo, vix, patterns)
    return results
