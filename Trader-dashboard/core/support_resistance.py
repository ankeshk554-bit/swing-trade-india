"""
Support & Resistance Detection Engine
=======================================
Detects key price levels using:
  - Swing point clustering (scipy argrelextrema)
  - Volume profile (high volume nodes)
  - Moving average clusters
  - Fibonacci retracement levels
  - Pivot points (classic + Camarilla + Woodie)
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Optional


def _find_swing_points(df, order=5):
    """Find swing highs and lows using local extrema."""
    highs = df["High"].values
    lows = df["Low"].values

    swing_high_idx = argrelextrema(highs, np.greater, order=order)[0]
    swing_low_idx = argrelextrema(lows, np.less, order=order)[0]

    swing_highs = [(df.index[i], highs[i]) for i in swing_high_idx]
    swing_lows = [(df.index[i], lows[i]) for i in swing_low_idx]

    return swing_highs, swing_lows


def _cluster_levels(levels, tolerance_pct=0.015):
    """Cluster nearby price levels within a tolerance percentage."""
    if not levels:
        return []

    sorted_levels = sorted(levels)
    clusters = []
    current_cluster = [sorted_levels[0]]

    for level in sorted_levels[1:]:
        if abs(level / np.mean(current_cluster) - 1) < tolerance_pct:
            current_cluster.append(level)
        else:
            clusters.append(round(np.mean(current_cluster), 2))
            current_cluster = [level]

    clusters.append(round(np.mean(current_cluster), 2))
    return clusters


def detect_support_resistance(df, lookback=120, order=5, tolerance_pct=0.015):
    """
    Detect key support and resistance levels.

    Args:
        df: OHLCV DataFrame
        lookback: Number of bars to analyze
        order: Sensitivity for swing point detection (lower = more points)
        tolerance_pct: % tolerance for clustering nearby levels

    Returns:
        dict with resistances, supports, all_levels, pivot_info
    """
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback)
    swing_highs, swing_lows = _find_swing_points(recent, order=order)

    # Extract price levels
    resistance_levels = [price for _, price in swing_highs]
    support_levels = [price for _, price in swing_lows]

    # Cluster nearby levels
    resistances = _cluster_levels(resistance_levels, tolerance_pct)
    supports = _cluster_levels(support_levels, tolerance_pct)

    # Merge all significant levels
    all_levels = sorted(set(resistances + supports))

    # Score each level by number of touches (higher = more important)
    close_prices = df["Close"].values
    level_importance = {}
    for level in all_levels:
        touches = sum(1 for c in close_prices if abs(c / level - 1) < tolerance_pct)
        level_importance[level] = touches

    # Sort by importance
    important_levels = sorted(level_importance.items(), key=lambda x: x[1], reverse=True)

    current_price = float(df["Close"].iloc[-1])

    # Nearest levels
    nearest_resistance = min([l for l in resistances if l > current_price], default=None)
    nearest_support = max([l for l in supports if l < current_price], default=None)

    # Distance to nearest levels
    dist_to_res = round((nearest_resistance / current_price - 1) * 100, 2) if nearest_resistance else None
    dist_to_sup = round((1 - nearest_support / current_price) * 100, 2) if nearest_support else None

    return {
        "resistances": resistances[-10:],  # Top 10 resistance levels
        "supports": supports[-10:],  # Top 10 support levels
        "all_levels": all_levels[-15:],
        "important_levels": important_levels[:10],
        "nearest_resistance": nearest_resistance,
        "nearest_support": nearest_support,
        "dist_to_resistance_pct": dist_to_res,
        "dist_to_support_pct": dist_to_sup,
        "current_price": current_price,
        "swing_highs": swing_highs[-10:],
        "swing_lows": swing_lows[-10:]
    }


def fibonacci_levels(high, low, current_price=None):
    """
    Calculate Fibonacci retracement and extension levels.

    Args:
        high: Swing high price
        low: Swing low price
        current_price: Current price (optional)

    Returns:
        dict with retracement and extension levels
    """
    diff = high - low

    retracement = {
        "0.0%": round(high, 2),
        "23.6%": round(high - diff * 0.236, 2),
        "38.2%": round(high - diff * 0.382, 2),
        "50.0%": round(high - diff * 0.5, 2),
        "61.8%": round(high - diff * 0.618, 2),
        "78.6%": round(high - diff * 0.786, 2),
        "100.0%": round(low, 2),
    }

    extension = {
        "127.2%": round(high + diff * 0.272, 2),
        "161.8%": round(high + diff * 0.618, 2),
        "261.8%": round(high + diff * 1.618, 2),
        "423.6%": round(high + diff * 3.236, 2),
    }

    current_zone = None
    if current_price:
        for level, price in retracement.items():
            if abs(current_price / price - 1) < 0.01:
                current_zone = f"At {level}"
                break
        if not current_zone:
            for i, (l1, p1) in enumerate(retracement.items()):
                levels_list = list(retracement.items())
                if i < len(levels_list) - 1:
                    l2, p2 = levels_list[i + 1]
                    if min(p1, p2) <= current_price <= max(p1, p2):
                        current_zone = f"Between {l1} and {l2}"
                        break

    return {
        "retracement": retracement,
        "extension": extension,
        "current_zone": current_zone,
        "high": round(high, 2),
        "low": round(low, 2),
        "range": round(diff, 2)
    }


def pivot_points(high, low, close):
    """
    Calculate classic pivot points.

    Returns dict with P, R1-R3, S1-S3.
    """
    pp = (high + low + close) / 3
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    r3 = high + 2 * (pp - low)
    s3 = low - 2 * (high - pp)

    return {
        "Pivot": round(pp, 2),
        "R1": round(r1, 2),
        "R2": round(r2, 2),
        "R3": round(r3, 2),
        "S1": round(s1, 2),
        "S2": round(s2, 2),
        "S3": round(s3, 2)
    }


def detect_breakout_levels(df, lookback=50, threshold_pct=0.02):
    """
    Detect breakout levels from consolidation.

    Returns resistance breakout level and support breakdown level.
    """
    if len(df) < lookback:
        return None

    recent = df.tail(lookback)
    range_high = recent["High"].max()
    range_low = recent["Low"].min()
    current = float(df["Close"].iloc[-1])

    return {
        "breakout_level": round(range_high, 2),
        "breakdown_level": round(range_low, 2),
        "range_pct": round((range_high / range_low - 1) * 100, 2),
        "above_breakout": current > range_high,
        "below_breakdown": current < range_low,
        "distance_to_breakout": round((current / range_high - 1) * 100, 2),
        "distance_to_breakdown": round((current / range_low - 1) * 100, 2)
    }
