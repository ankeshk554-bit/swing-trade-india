import pandas as pd
import numpy as np


def compute_rsi(series, length=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def compute_indicators(df: pd.DataFrame):
    """Compute all technical indicators for a given dataframe."""
    df = df.copy()

    # --- Moving Averages ---
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    # --- RSI ---
    df["RSI"] = compute_rsi(df["Close"])

    # --- Volume ---
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["VOL_MA50"] = df["Volume"].rolling(50).mean()
    df["RVOL"] = df["Volume"] / df["VOL_MA20"].replace(0, np.nan)

    # --- Bollinger Bands ---
    bb_std = df["Close"].rolling(20).std()
    df["BB_MID"] = df["Close"].rolling(20).mean()
    df["BB_UPPER"] = df["BB_MID"] + 2 * bb_std
    df["BB_LOWER"] = df["BB_MID"] - 2 * bb_std
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"]
    df["BB_POS"] = (df["Close"] - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"]).replace(0, np.nan)

    # --- MACD ---
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # --- ATR ---
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_PCT"] = (df["ATR"] / df["Close"] * 100)

    # --- ADX ---
    df["ADX"] = _compute_adx(df, 14)

    # --- Supertrend ---
    supertrend_df = _compute_supertrend(df, period=10, multiplier=3)
    df["SUPERTREND"] = supertrend_df["SUPERTREND"]
    df["SUPERTREND_DIR"] = supertrend_df["SUPERTREND_DIR"]

    # --- OBV ---
    df["OBV"] = (df["Volume"] * (~(df["Close"].diff() < 0).astype(int) * 2 - 1)).cumsum()

    return df


def _compute_adx(df, period=14):
    """Compute Average Directional Index."""
    high, low, close = df["High"], df["Low"], df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.abs().rolling(period).mean() / atr.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(period).mean()

    return adx


def _compute_supertrend(df, period=10, multiplier=3):
    """Compute Supertrend indicator."""
    hl_avg = (df["High"] + df["Low"]) / 2
    atr = df["ATR"] if "ATR" in df else (df["High"] - df["Low"]).rolling(period).mean()

    upper_band = hl_avg + multiplier * atr
    lower_band = hl_avg - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        if df["Close"].iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif df["Close"].iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]

    return pd.DataFrame({"SUPERTREND": supertrend, "SUPERTREND_DIR": direction})


def calculate_avwap(df, anchor_idx=0):
    """Calculate Anchored VWAP from a given index anchor point."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    volume = df["Volume"]
    tpv = (typical_price * volume).iloc[anchor_idx:]
    cumulative_tpv = tpv.cumsum()
    cumulative_volume = volume.iloc[anchor_idx:].cumsum()
    avwap = cumulative_tpv / cumulative_volume.replace(0, np.nan)
    return avwap


def detect_vcp(df, lookback=60):
    """
    Detect Volatility Contraction Pattern (VCP) by Mark Minervini.

    Returns dict with:
      - VCP_Flag: bool
      - Stage: int (1-4 contraction count)
      - Pivot: float (highest close in contraction)
      - VolumeDryUp: bool
    """
    if len(df) < lookback:
        return {"VCP_Flag": False, "Stage": 0, "Pivot": None, "VolumeDryUp": False}

    recent = df.tail(lookback).copy()

    # Find swing highs
    highs = recent["High"].values
    closes = recent["Close"].values
    volumes = recent["Volume"].values

    # Detect contraction stages: each swing high should be lower than previous
    pivots = []
    for i in range(5, len(highs) - 5):
        if highs[i] == max(highs[i - 5:i + 6]):
            pivots.append((i, highs[i], volumes[i]))

    if len(pivots) < 2:
        return {"VCP_Flag": False, "Stage": 0, "Pivot": None, "VolumeDryUp": False}

    # Check contraction: each pivot lower than previous
    pivot_highs = [p[1] for p in pivots]
    pivot_volumes = [p[2] for p in pivots]

    contractions = sum(1 for i in range(1, len(pivot_highs)) if pivot_highs[i] < pivot_highs[i - 1])

    pivot_price = pivot_highs[-1]
    current_close = closes[-1]

    # Volume should dry up as contraction progresses
    recent_vol_avg = np.mean(volumes[-10:])
    prior_vol_avg = np.mean(volumes[:20])
    volume_dry_up = recent_vol_avg < prior_vol_avg * 0.7

    # VCP flag: at least 2 contractions, price near pivot (within 3%)
    vcp_flag = contractions >= 2 and current_close >= pivot_price * 0.97

    return {
        "VCP_Flag": vcp_flag,
        "Stage": min(contractions, 4),
        "Pivot": round(pivot_price, 2),
        "VolumeDryUp": volume_dry_up
    }


def compute_institutional_score(df):
    """
    Compute institutional accumulation score (0-100).
    Higher score = stronger institutional interest.
    """
    latest = df.iloc[-1]

    score = 0

    # Trend structure (max 30)
    if latest["Close"] > latest["EMA50"]:
        score += 15
    if latest["EMA50"] > latest["EMA200"]:
        score += 15

    # Momentum (max 25)
    if 55 < latest["RSI"] < 75:
        score += 15
    if latest["RSI"] > latest["RSI"] if "RSI" in df.columns else False:
        score += 10

    # Volume confirmation (max 25)
    if latest["RVOL"] > 1.5:
        score += 15
    if latest["RVOL"] > 2.0:
        score += 10

    # MACD confirmation (max 20)
    if latest["MACD"] > latest["MACD_SIGNAL"]:
        score += 10
    if latest["MACD_HIST"] > 0:
        score += 10

    return min(score, 100)


def generate_swing_signal(df):
    """
    Generate BUY/SELL/NEUTRAL signal with confidence and rationale.
    """
    latest = df.iloc[-1]

    reasons = []
    signal = "NEUTRAL"
    confidence = 0

    # --- Bullish checks ---
    if latest["Close"] > latest["EMA50"]:
        reasons.append("Price above 50-EMA")
        confidence += 1
    if latest["EMA50"] > latest["EMA200"]:
        reasons.append("Golden alignment (50>200)")
        confidence += 1
    if latest["RSI"] > 55 and latest["RSI"] < 75:
        reasons.append("RSI in bullish zone (55-75)")
        confidence += 1
    if latest["RVOL"] > 1.5:
        reasons.append("Volume >1.5x average")
        confidence += 1
    if latest["MACD_HIST"] > 0 and latest["MACD"] > latest["MACD_SIGNAL"]:
        reasons.append("MACD bullish crossover")
        confidence += 1
    if latest["SUPERTREND_DIR"] == 1:
        reasons.append("Supertrend bullish")
        confidence += 1

    # --- Bearish checks ---
    bearish_count = 0
    if latest["Close"] < latest["EMA50"]:
        bearish_count += 1
    if latest["RSI"] < 45:
        bearish_count += 1
    if latest["MACD_HIST"] < 0 and latest["MACD"] < latest["MACD_SIGNAL"]:
        bearish_count += 1
    if latest["SUPERTREND_DIR"] == -1:
        bearish_count += 1

    if confidence >= 4 and bearish_count <= 1:
        signal = "BUY"
    elif bearish_count >= 3 and confidence <= 2:
        signal = "SELL"

    return {
        "Signal": signal,
        "Confidence": confidence,
        "Reasons": "; ".join(reasons[:3])
    }


def compute_delivery_enhanced_score(df, delivery_summary=None):
    """
    Enhanced institutional score that incorporates delivery volume data.

    This extends the base compute_institutional_score with delivery quality.

    Args:
        df: OHLCV dataframe with computed indicators
        delivery_summary: Optional dict from data_providers.get_delivery_summary()

    Returns:
        dict with base_score, delivery_score, enhanced_score, signal
    """
    base = compute_institutional_score(df)
    delivery_score = 0
    delivery_notes = []

    if delivery_summary:
        # Delivery quality scoring (0-25 additional points)
        dq = delivery_summary.get("delivery_quality", "WEAK")
        if dq == "STRONG":
            delivery_score += 25
            delivery_notes.append("Strong delivery quality")
        elif dq == "MODERATE":
            delivery_score += 12
            delivery_notes.append("Moderate delivery quality")

        # Delivery trend
        trend = delivery_summary.get("delivery_trend", "STABLE")
        if trend == "RISING":
            delivery_score += 8
            delivery_notes.append("Delivery trend rising")

        # Accumulation days
        acc_days = delivery_summary.get("accumulation_days_10", 0)
        if acc_days >= 7:
            delivery_score += 10
            delivery_notes.append("Strong accumulation (7+ days)")
        elif acc_days >= 4:
            delivery_score += 5
            delivery_notes.append("Good accumulation")

        # Latest delivery spurt
        if delivery_summary.get("delivery_spurt", False):
            delivery_score += 7
            delivery_notes.append("Delivery spurt detected")

    enhanced = min(base + delivery_score, 100)

    return {
        "base_score": base,
        "delivery_score": delivery_score,
        "enhanced_score": enhanced,
        "delivery_notes": "; ".join(delivery_notes) if delivery_notes else "No delivery data"
    }


def generate_fo_enhanced_signal(
    df,
    delivery_summary=None,
    fo_data=None,
    vix_info=None
):
    """
    Generate trading signal enhanced with F&O and delivery data.

    Args:
        df: OHLCV dataframe with indicators
        delivery_summary: dict from get_delivery_summary()
        fo_data: dict from get_stock_fo_data()
        vix_info: dict from get_india_vix()

    Returns:
        dict with Signal, Confidence, Reasons, and sub-scores
    """
    base_signal = generate_swing_signal(df)
    signal = base_signal["Signal"]
    confidence = base_signal["Confidence"]
    reasons = [base_signal["Reasons"]] if base_signal["Reasons"] else []

    extra_confidence = 0

    # Delivery enhancement
    if delivery_summary:
        dq = delivery_summary.get("delivery_quality", "WEAK")
        if dq == "STRONG":
            extra_confidence += 2
            reasons.append("✅ Strong delivery")
        elif dq == "MODERATE":
            extra_confidence += 1
            reasons.append("📦 Decent delivery")

        if delivery_summary.get("delivery_spurt", False):
            extra_confidence += 1
            reasons.append("📊 Delivery spurt")

    # F&O enhancement
    if fo_data and fo_data.get("in_fo", False):
        reasons.append("🔷 F&O stock")
        oi_change = fo_data.get("oi_change_pct")
        if oi_change is not None:
            if oi_change > 10:
                extra_confidence += 1
                reasons.append(f"📈 OI +{oi_change}%")
            elif oi_change < -10:
                extra_confidence -= 1
                reasons.append(f"📉 OI {oi_change}%")

    # VIX regime context
    if vix_info and vix_info.get("vix") is not None:
        vix = vix_info["vix"]
        regime = vix_info.get("regime", "NORMAL")
        if regime in ("LOW_VOL", "NORMAL"):
            # Favorable for trend trading
            if signal == "BUY":
                extra_confidence += 1
                reasons.append(f"🌡️ VIX {vix} ({regime})")
        elif regime in ("HIGH_VOL", "EXTREME_FEAR"):
            # Caution
            if signal == "BUY":
                extra_confidence -= 1
                reasons.append(f"⚠️ VIX {vix} ({regime})")

    # Final confidence
    final_confidence = min(confidence + extra_confidence, 8)

    # Re-evaluate signal based on enhanced confidence
    if final_confidence >= 5:
        signal = "BUY"
    elif final_confidence <= 2:
        signal = "SELL"

    return {
        "Signal": signal,
        "Confidence": final_confidence,
        "BaseConfidence": confidence,
        "ExtraConfidence": extra_confidence,
        "Reasons": " | ".join(reasons[:5]) if reasons else "No strong signals"
    }
