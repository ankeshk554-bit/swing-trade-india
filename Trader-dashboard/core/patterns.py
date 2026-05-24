"""
Chart Pattern Recognition Engine
=================================
Detects common chart patterns for swing trading:
  - Bull Flag / Bear Flag
  - Double Bottom / Double Top
  - Cup & Handle
  - Head & Shoulders / Inverse H&S
  - Rising / Falling Wedge
  - Breakout from Consolidation
  - 52-Week High/Low Proximity
  - Inside Bar / Engulfing
  - Higher Highs / Higher Lows structure
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def detect_bull_flag(df, lookback=40):
    """
    Bull Flag pattern: sharp price rise (flagpole) followed by
    downward/ sideways consolidation (flag).
    """
    if len(df) < lookback:
        return {"flag": False, "confidence": 0}

    recent = df.tail(lookback)
    closes = recent["Close"].values
    highs = recent["High"].values
    lows = recent["Low"].values

    # Find the highest point in recent window (flagpole peak)
    peak_idx = np.argmax(highs)
    if peak_idx < 5:
        return {"flag": False, "confidence": 0}

    # Flagpole: sharp rise of at least 15%
    flagpole_start = closes[0]
    flagpole_end = highs[peak_idx]
    flagpole_rise = (flagpole_end / flagpole_start - 1) * 100

    if flagpole_rise < 15:
        return {"flag": False, "confidence": 0}

    # Flag: consolidation after the peak (lower highs, higher lows)
    if peak_idx >= len(highs) - 5:
        return {"flag": False, "confidence": 0}  # Too recent

    flag_highs = highs[peak_idx:]
    flag_lows = lows[peak_idx:]

    # Flag should be downward sloping or sideways
    flag_highs_slope = np.polyfit(range(len(flag_highs)), flag_highs, 1)[0]
    flag_lows_slope = np.polyfit(range(len(flag_lows)), flag_lows, 1)[0]

    # Bull flag: consolidating with slight downward drift
    is_consolidating = (
        len(flag_highs) >= 5 and
        flag_highs_slope < 0 and  # Lower highs
        abs(flag_highs_slope) < 2  # Not crashing
    )

    recent_vol = recent["Volume"].tail(10).mean()
    prior_vol = recent["Volume"].head(20).mean()
    vol_drying = recent_vol < prior_vol * 0.8 if prior_vol > 0 else False

    confidence = 0
    if is_consolidating:
        confidence += 1
    if vol_drying:
        confidence += 1
    if flagpole_rise > 25:
        confidence += 1
    if len(flag_highs) >= 8:
        confidence += 1

    return {
        "flag": confidence >= 2,
        "type": "BULL_FLAG",
        "confidence": min(confidence, 4),
        "flagpole_rise": round(flagpole_rise, 1),
        "consolidation_bars": len(flag_highs),
        "volume_drying": vol_drying
    }


def detect_bear_flag(df, lookback=40):
    """Bear Flag: sharp decline followed by upward consolidation."""
    if len(df) < lookback:
        return {"flag": False, "confidence": 0}

    recent = df.tail(lookback)
    closes = recent["Close"].values
    highs = recent["High"].values
    lows = recent["Low"].values

    # Find the lowest point
    trough_idx = np.argmin(lows)
    if trough_idx < 5:
        return {"flag": False, "confidence": 0}

    # Flagpole: sharp decline of at least 12%
    flagpole_start = closes[0]
    flagpole_end = lows[trough_idx]
    flagpole_drop = (flagpole_end / flagpole_start - 1) * 100

    if flagpole_drop > -12:
        return {"flag": False, "confidence": 0}

    if trough_idx >= len(closes) - 5:
        return {"flag": False, "confidence": 0}

    # Flag: upward consolidation
    flag_highs = highs[trough_idx:]
    flag_lows = lows[trough_idx:]

    flag_highs_slope = 0
    if len(flag_highs) > 1:
        flag_highs_slope = np.polyfit(range(len(flag_highs)), flag_highs, 1)[0]

    is_consolidating = len(flag_highs) >= 5 and flag_highs_slope > 0

    confidence = 0
    if is_consolidating:
        confidence += 2
    if flagpole_drop < -20:
        confidence += 1
    if len(flag_highs) >= 8:
        confidence += 1

    return {
        "flag": confidence >= 2,
        "type": "BEAR_FLAG",
        "confidence": min(confidence, 4),
        "flagpole_drop": round(abs(flagpole_drop), 1),
        "consolidation_bars": len(flag_highs)
    }


def detect_double_bottom(df, lookback=60):
    """
    Double Bottom pattern: price makes two similar lows with a peak in between.
    """
    if len(df) < lookback:
        return {"pattern": False, "confidence": 0}

    recent = df.tail(lookback)
    lows = recent["Low"].values
    highs = recent["High"].values

    # Find local minima
    local_min_indices = argrelextrema(lows, np.less, order=5)[0]

    if len(local_min_indices) < 2:
        return {"pattern": False, "confidence": 0}

    # Look for two bottoms at similar levels (within 5%)
    for i in range(len(local_min_indices) - 1):
        idx1 = local_min_indices[i]
        idx2 = local_min_indices[i + 1]

        # Need bars between them
        if idx2 - idx1 < 5:
            continue

        bottom1 = lows[idx1]
        bottom2 = lows[idx2]
        bottom_diff = abs(bottom2 / bottom1 - 1) * 100

        if bottom_diff > 5:
            continue

        # Peak between the bottoms should be at least 5% higher
        peak = np.max(highs[idx1:idx2 + 1])
        peak_rise = (peak / min(bottom1, bottom2) - 1) * 100

        if peak_rise < 5:
            continue

        # Check if price has broken above the middle peak
        current_close = recent["Close"].iloc[-1]
        middle_peak = peak  # The middle peak is the resistance
        breakout = (current_close / middle_peak - 1) * 100

        confidence = 2
        if bottom_diff < 2:
            confidence += 1
        if peak_rise > 10:
            confidence += 1
        if breakout > 1:
            confidence += 1

        return {
            "pattern": True,
            "type": "DOUBLE_BOTTOM",
            "confidence": min(confidence, 5),
            "bottom1": round(bottom1, 2),
            "bottom2": round(bottom2, 2),
            "neckline": round(middle_peak, 2),
            "breakout_pct": round(breakout, 1),
            "target": round(middle_peak + (middle_peak - min(bottom1, bottom2)), 2)
        }

    return {"pattern": False, "confidence": 0}


def detect_cup_handle(df, lookback=80):
    """
    Cup & Handle pattern (William O'Neil):
    - Cup: U-shaped consolidation (min 7 weeks)
    - Handle: slight downward drift on the right side
    """
    if len(df) < lookback:
        return {"pattern": False, "confidence": 0}

    recent = df.tail(lookback)
    closes = recent["Close"].values
    highs = recent["High"].values

    # Find the cup (highest point on left, lowest in middle)
    left_peak_idx = np.argmax(highs[:lookback // 3])
    if left_peak_idx == 0:
        return {"pattern": False, "confidence": 0}

    cup_bottom_idx = np.argmin(highs[left_peak_idx:]) + left_peak_idx
    if cup_bottom_idx >= len(highs) - 10:
        return {"pattern": False, "confidence": 0}

    left_peak = highs[left_peak_idx]
    cup_bottom = lows = recent["Low"].values[cup_bottom_idx]  # Use Low for bottom
    cup_depth = (1 - cup_bottom / left_peak) * 100

    # Cup should be 15-40% deep
    if cup_depth < 10 or cup_depth > 50:
        return {"pattern": False, "confidence": 0}

    # Right side should recover at least halfway
    right_side = highs[cup_bottom_idx:]
    right_peak = np.max(right_side)
    right_recovery = (right_peak / cup_bottom - 1) * 100

    if right_recovery < cup_depth / 2:
        return {"pattern": False, "confidence": 0}

    # Handle: last 10-15 bars should show slight downward drift
    handle_bars = min(15, len(right_side))
    if handle_bars >= 5:
        handle_slice = right_side[-handle_bars:]
        handle_slope = np.polyfit(range(len(handle_slice)), handle_slice, 1)[0]
        has_handle = handle_slope < 0 and abs(handle_slope) < 1.5
    else:
        has_handle = False

    confidence = 2
    if 20 <= cup_depth <= 35:
        confidence += 1
    if has_handle:
        confidence += 1
    if right_recovery > cup_depth * 0.7:
        confidence += 1

    return {
        "pattern": confidence >= 2,
        "type": "CUP_HANDLE",
        "confidence": min(confidence, 5),
        "cup_depth": round(cup_depth, 1),
        "cup_bottom": round(cup_bottom, 2),
        "has_handle": has_handle,
        "target": round(left_peak + (left_peak - cup_bottom), 2)
    }


def detect_head_shoulders(df, lookback=60):
    """
    Head & Shoulders (bearish) pattern detection.
    """
    if len(df) < lookback:
        return {"pattern": False, "confidence": 0}

    recent = df.tail(lookback)
    highs = recent["High"].values

    # Find local maxima
    local_max_indices = argrelextrema(highs, np.greater, order=5)[0]

    if len(local_max_indices) < 3:
        return {"pattern": False, "confidence": 0}

    # Look for three peaks: left shoulder < head > right shoulder
    for i in range(len(local_max_indices) - 2):
        ls_idx = local_max_indices[i]
        h_idx = local_max_indices[i + 1]
        rs_idx = local_max_indices[i + 2]

        left_shoulder = highs[ls_idx]
        head = highs[h_idx]
        right_shoulder = highs[rs_idx]

        # Head should be highest
        if not (head > left_shoulder and head > right_shoulder):
            continue

        # Shoulders should be roughly equal
        shoulder_diff = abs(right_shoulder / left_shoulder - 1) * 100
        if shoulder_diff > 8:
            continue

        # Head should be at least 3% higher than shoulders
        head_rise = (head / max(left_shoulder, right_shoulder) - 1) * 100
        if head_rise < 3:
            continue

        # Neckline: lowest point between shoulders
        neckline = np.min(recent["Low"].values[ls_idx:rs_idx + 1])
        current_close = recent["Close"].iloc[-1]

        # Pattern is confirmed if price breaks below neckline
        broken = current_close < neckline

        confidence = 2
        if head_rise > 7:
            confidence += 1
        if shoulder_diff < 3:
            confidence += 1
        if broken:
            confidence += 1

        return {
            "pattern": True,
            "type": "HEAD_SHOULDERS",
            "confidence": min(confidence, 5),
            "left_shoulder": round(left_shoulder, 2),
            "head": round(head, 2),
            "right_shoulder": round(right_shoulder, 2),
            "neckline": round(neckline, 2),
            "broken": broken,
            "target": round(neckline - (head - neckline), 2)
        }

    return {"pattern": False, "confidence": 0}


def detect_consolidation_breakout(df, lookback=30):
    """
    Breakout from consolidation / range-bound trading.
    """
    if len(df) < lookback:
        return {"pattern": False, "confidence": 0}

    recent = df.tail(lookback)
    highs = recent["High"].values
    lows = recent["Low"].values
    closes = recent["Close"].values

    # Range in first 70% of window
    range_start = 0
    range_end = int(lookback * 0.7)

    range_high = np.max(highs[range_start:range_end])
    range_low = np.min(lows[range_start:range_end])
    range_pct = (range_high / range_low - 1) * 100

    # Tight range: less than 12% over the period
    if range_pct > 12:
        return {"pattern": False, "confidence": 0}

    # Recent bar should break above the range
    recent_max = np.max(highs[range_end:]) if len(highs) > range_end else 0
    breakout_pct = (recent_max / range_high - 1) * 100

    if breakout_pct < 1:
        return {"pattern": False, "confidence": 0}

    # Volume confirmation
    recent_vol = df["Volume"].tail(min(5, len(df))).mean()
    range_vol = df["Volume"].tail(lookback).head(int(lookback * 0.7)).mean()
    vol_surge = recent_vol > range_vol * 1.3 if range_vol > 0 else False

    confidence = 2
    if range_pct < 8:
        confidence += 1
    if breakout_pct > 3:
        confidence += 1
    if vol_surge:
        confidence += 1

    return {
        "pattern": True,
        "type": "CONSOLIDATION_BREAKOUT",
        "confidence": min(confidence, 5),
        "range_pct": round(range_pct, 1),
        "resistance": round(range_high, 2),
        "breakout_pct": round(breakout_pct, 1),
        "volume_surge": vol_surge,
        "target": round(range_high + (range_high - range_low), 2)
    }


def detect_inside_bar(df, lookback=10):
    """
    Inside Bar pattern: current bar's high and low are within prev bar's range.
    Signals potential breakout.
    """
    if len(df) < lookback:
        return {"pattern": False, "confidence": 0}

    recent = df.tail(lookback)
    inside_bars = []
    for i in range(1, len(recent)):
        curr_high = recent["High"].iloc[i]
        curr_low = recent["Low"].iloc[i]
        prev_high = recent["High"].iloc[i - 1]
        prev_low = recent["Low"].iloc[i - 1]

        if curr_high <= prev_high and curr_low >= prev_low:
            inside_bars.append(i)

    if not inside_bars:
        return {"pattern": False, "confidence": 0}

    latest_breakout = False
    # Check if last bar breaks out of the mother bar
    if len(inside_bars) >= 2:
        mother_high = recent["High"].iloc[inside_bars[0] - 1]
        mother_low = recent["Low"].iloc[inside_bars[0] - 1]
        latest_close = recent["Close"].iloc[-1]
        if latest_close > mother_high:
            latest_breakout = True

    return {
        "pattern": len(inside_bars) >= 2,
        "type": "INSIDE_BAR",
        "confidence": min(len(inside_bars), 4),
        "bar_count": len(inside_bars),
        "breakout_up": latest_breakout
    }


def detect_higher_highs(df, lookback=40):
    """
    Detect Higher Highs / Higher Lows (uptrend) or
    Lower Highs / Lower Lows (downtrend) structure.
    """
    if len(df) < lookback:
        return {"pattern": False, "type": "NEUTRAL", "confidence": 0}

    recent = df.tail(lookback)
    highs = recent["High"].values
    lows = recent["Low"].values

    # Find swing highs and lows
    swing_high_indices = argrelextrema(highs, np.greater, order=5)[0]
    swing_low_indices = argrelextrema(lows, np.less, order=5)[0]

    if len(swing_high_indices) < 2 or len(swing_low_indices) < 2:
        return {"pattern": False, "type": "NEUTRAL", "confidence": 0}

    swing_highs = highs[swing_high_indices]
    swing_lows = lows[swing_low_indices]

    # Check if each successive swing high is higher
    rising_highs = all(swing_highs[i] > swing_highs[i - 1] for i in range(1, len(swing_highs)))
    rising_lows = all(swing_lows[i] > swing_lows[i - 1] for i in range(1, len(swing_lows)))
    falling_highs = all(swing_highs[i] < swing_highs[i - 1] for i in range(1, len(swing_highs)))
    falling_lows = all(swing_lows[i] < swing_lows[i - 1] for i in range(1, len(swing_lows)))

    if rising_highs and rising_lows:
        return {
            "pattern": True, "type": "UPTREND_HH_HL",
            "confidence": 4, "swing_points": len(swing_high_indices)
        }
    elif falling_highs and falling_lows:
        return {
            "pattern": True, "type": "DOWNTREND_LH_LL",
            "confidence": 4, "swing_points": len(swing_high_indices)
        }
    elif rising_lows and not rising_highs:
        return {
            "pattern": True, "type": "BULLISH_DIVERGENCE",
            "confidence": 3, "swing_points": len(swing_low_indices)
        }

    return {"pattern": False, "type": "NEUTRAL", "confidence": 1, "swing_points": 0}


def detect_engulfing(df, lookback=10):
    """
    Bullish / Bearish Engulfing candle pattern.
    """
    if len(df) < 3:
        return {"pattern": False, "type": "NONE", "confidence": 0}

    recent = df.tail(lookback)

    for i in range(1, len(recent)):
        prev = recent.iloc[i - 1]
        curr = recent.iloc[i]

        prev_open = float(prev["Open"])
        prev_close = float(prev["Close"])
        curr_open = float(curr["Open"])
        curr_close = float(curr["Close"])

        # Bullish Engulfing
        if (prev_close < prev_open and  # Previous red candle
                curr_close > curr_open and  # Current green candle
                curr_open < prev_close and  # Open below prev close
                curr_close > prev_open):    # Close above prev open
            return {
                "pattern": True,
                "type": "BULLISH_ENGULFING",
                "confidence": 3,
                "index": i
            }

        # Bearish Engulfing
        if (prev_close > prev_open and  # Previous green candle
                curr_close < curr_open and  # Current red candle
                curr_open > prev_close and  # Open above prev close
                curr_close < prev_open):    # Close below prev open
            return {
                "pattern": True,
                "type": "BEARISH_ENGULFING",
                "confidence": 3,
                "index": i
            }

    return {"pattern": False, "type": "NONE", "confidence": 0}


def get_52w_proximity(df):
    """
    Check how close price is to 52-week high/low.
    Returns proximity ratio (0-1) for both high and low.
    """
    if len(df) < 252:
        # Use available data as proxy
        lookback = min(len(df), 200)

    high_52w = df["High"].rolling(252).max().iloc[-1] if len(df) >= 252 else df["High"].max()
    low_52w = df["Low"].rolling(252).min().iloc[-1] if len(df) >= 252 else df["Low"].min()

    # Use 200-day if 52-week data not available
    try:
        high_52w = df["High"].tail(252).max() if len(df) >= 252 else df["High"].tail(len(df)).max()
        low_52w = df["Low"].tail(252).min() if len(df) >= 252 else df["Low"].tail(len(df)).min()
    except Exception:
        return {"near_high": False, "near_low": False, "high_proximity": 0, "low_proximity": 0}

    current_close = float(df["Close"].iloc[-1])

    if high_52w == low_52w:
        return {"near_high": False, "near_low": False, "high_proximity": 0, "low_proximity": 0}

    high_proximity = (current_close / high_52w) if high_52w > 0 else 0
    low_proximity = (current_close / low_52w) if low_52w > 0 else 0

    return {
        "near_high": high_proximity >= 0.95,
        "near_low": low_proximity <= 1.05,
        "high_proximity": round(high_proximity, 3),
        "low_proximity": round(low_proximity, 3),
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2)
    }


def run_all_patterns(df) -> dict:
    """
    Run all pattern detection algorithms and return combined results.
    """
    patterns = {}

    try:
        patterns["bull_flag"] = detect_bull_flag(df)
    except Exception:
        patterns["bull_flag"] = {"flag": False, "confidence": 0}

    try:
        patterns["bear_flag"] = detect_bear_flag(df)
    except Exception:
        patterns["bear_flag"] = {"flag": False, "confidence": 0}

    try:
        patterns["double_bottom"] = detect_double_bottom(df)
    except Exception:
        patterns["double_bottom"] = {"pattern": False, "confidence": 0}

    try:
        patterns["cup_handle"] = detect_cup_handle(df)
    except Exception:
        patterns["cup_handle"] = {"pattern": False, "confidence": 0}

    try:
        patterns["head_shoulders"] = detect_head_shoulders(df)
    except Exception:
        patterns["head_shoulders"] = {"pattern": False, "confidence": 0}

    try:
        patterns["consolidation_breakout"] = detect_consolidation_breakout(df)
    except Exception:
        patterns["consolidation_breakout"] = {"pattern": False, "confidence": 0}

    try:
        patterns["inside_bar"] = detect_inside_bar(df)
    except Exception:
        patterns["inside_bar"] = {"pattern": False, "confidence": 0}

    try:
        patterns["structure"] = detect_higher_highs(df)
    except Exception:
        patterns["structure"] = {"pattern": False, "type": "NEUTRAL", "confidence": 0}

    try:
        patterns["engulfing"] = detect_engulfing(df)
    except Exception:
        patterns["engulfing"] = {"pattern": False, "type": "NONE", "confidence": 0}

    try:
        patterns["52w"] = get_52w_proximity(df)
    except Exception:
        patterns["52w"] = {"near_high": False, "near_low": False}

    return patterns


def get_best_pattern(patterns: dict) -> dict:
    """
    From all detected patterns, return the one with highest confidence.
    """
    best = {"name": "NONE", "confidence": 0, "details": None}

    pattern_map = {
        "bull_flag": ("Bull Flag", "flag"),
        "bear_flag": ("Bear Flag", "flag"),
        "double_bottom": ("Double Bottom", "pattern"),
        "cup_handle": ("Cup & Handle", "pattern"),
        "head_shoulders": ("H&S", "pattern"),
        "consolidation_breakout": ("Breakout", "pattern"),
        "inside_bar": ("Inside Bar", "pattern"),
        "structure": ("Trend Structure", "pattern"),
        "engulfing": ("Engulfing", "pattern")
    }

    for key, (name, flag_key) in pattern_map.items():
        pat = patterns.get(key, {})
        if pat.get(flag_key, False):
            conf = pat.get("confidence", 0)
            if conf > best["confidence"]:
                best = {
                    "name": name,
                    "confidence": conf,
                    "details": pat
                }

    return best
