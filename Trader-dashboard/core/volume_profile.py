"""
Volume Profile & Market Profile — Sniper Terminal
===================================================
Calculates:
  - Volume Profile (Volume at Price)
  - Volume Weighted Average Price (VWAP) with bands
  - Market Profile / TPO (Time Price Opportunity)
  - High Volume Nodes (HVN) / Low Volume Nodes (LVN)
  - Point of Control (POC) — price with highest volume
"""

import numpy as np
import pandas as pd
from typing import Optional


def volume_profile(df, num_bins: int = 24, lookback: int = None):
    """
    Calculate Volume Profile — Volume distributed across price levels.

    Args:
        df: OHLCV DataFrame
        num_bins: Number of price buckets
        lookback: Number of bars to analyze (default: all)

    Returns:
        dict with price_bins, volumes, poc, vah, val, high_volume_nodes
    """
    if lookback:
        df = df.tail(lookback)

    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values
    closes = df["Close"].values

    price_min = lows.min()
    price_max = highs.max()
    price_range = price_max - price_min

    if price_range <= 0:
        return None

    bin_size = price_range / num_bins
    price_bins = np.linspace(price_min, price_max, num_bins + 1)

    # Distribute volume across price bins
    bin_volumes = np.zeros(num_bins)

    for i in range(len(df)):
        high = highs[i]
        low = lows[i]
        vol = volumes[i]

        # Which bins does this bar span?
        low_bin = max(0, int((low - price_min) / bin_size))
        high_bin = min(num_bins - 1, int((high - price_min) / bin_size))

        if high_bin == low_bin:
            bin_volumes[low_bin] += vol
        else:
            # Distribute volume proportionally
            bar_range = high - low
            for b in range(low_bin, high_bin + 1):
                bin_low = max(low, price_min + b * bin_size)
                bin_high = min(high, price_min + (b + 1) * bin_size)
                ratio = (bin_high - bin_low) / bar_range if bar_range > 0 else 0
                bin_volumes[b] += vol * ratio

    # Midpoint of each bin
    bin_prices = price_bins[:-1] + bin_size / 2

    # Point of Control (price with highest volume)
    poc_idx = np.argmax(bin_volumes)
    poc = round(bin_prices[poc_idx], 2)

    # Value Area (70% of volume around POC)
    total_vol = bin_volumes.sum()
    target_vol = total_vol * 0.70

    # Expand outward from POC until we capture 70% of volume
    cum_vol = 0
    vah_idx = poc_idx
    val_idx = poc_idx
    lower_idx = poc_idx - 1
    upper_idx = poc_idx + 1

    while cum_vol < target_vol:
        vol_added = False
        if lower_idx >= 0:
            cum_vol += bin_volumes[lower_idx]
            val_idx = lower_idx
            lower_idx -= 1
            vol_added = True
        if cum_vol < target_vol and upper_idx < num_bins:
            cum_vol += bin_volumes[upper_idx]
            vah_idx = upper_idx
            upper_idx += 1
            vol_added = True
        if not vol_added:
            break

    vah = round(bin_prices[vah_idx] + bin_size / 2, 2)
    val = round(bin_prices[val_idx] - bin_size / 2, 2)

    # High Volume Nodes (HVN) — bins with volume > 2x average
    avg_bin_vol = total_vol / num_bins
    hvn = []
    lvn = []
    for i in range(num_bins):
        price = round(bin_prices[i], 2)
        if bin_volumes[i] > avg_bin_vol * 1.5:
            hvn.append({"price": price, "volume": round(bin_volumes[i], 0)})
        elif bin_volumes[i] < avg_bin_vol * 0.3 and bin_volumes[i] > 0:
            lvn.append({"price": price, "volume": round(bin_volumes[i], 0)})

    return {
        "price_bins": [round(p, 2) for p in bin_prices],
        "volumes": [round(v, 0) for v in bin_volumes],
        "poc": poc,
        "vah": vah,
        "val": val,
        "bin_size": round(bin_size, 2),
        "high_volume_nodes": hvn[:10],
        "low_volume_nodes": lvn[:10],
        "total_volume": round(total_vol, 0),
        "price_min": round(price_min, 2),
        "price_max": round(price_max, 2)
    }


def vwap_with_bands(df, lookback: int = None):
    """
    Calculate VWAP with standard deviation bands.

    Returns DataFrame with VWAP, VWAP_Upper, VWAP_Lower columns.
    """
    result_df = df.copy()

    typical_price = (result_df["High"] + result_df["Low"] + result_df["Close"]) / 3

    if lookback:
        result_df = result_df.tail(lookback).copy()
        typical_price = typical_price.tail(lookback)

    result_df["VWAP"] = (typical_price * result_df["Volume"]).cumsum() / result_df["Volume"].cumsum().replace(0, np.nan)

    # Standard deviation of VWAP
    vwap_diff = (typical_price - result_df["VWAP"]) ** 2
    vwap_std = np.sqrt((vwap_diff * result_df["Volume"]).cumsum() / result_df["Volume"].cumsum().replace(0, np.nan))

    result_df["VWAP_Upper1"] = result_df["VWAP"] + vwap_std
    result_df["VWAP_Lower1"] = result_df["VWAP"] - vwap_std
    result_df["VWAP_Upper2"] = result_df["VWAP"] + 2 * vwap_std
    result_df["VWAP_Lower2"] = result_df["VWAP"] - 2 * vwap_std

    return result_df


def market_profile(df, lookback: int = 50, tpo_bins: int = 12):
    """
    Simplified Market Profile (TPO) chart data.

    TPO = Time Price Opportunity — shows which prices were traded
    during which time periods.

    Returns dict with TPO letters by price level.
    """
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback)
    highs = recent["High"].values
    lows = recent["Low"].values

    price_min = lows.min()
    price_max = highs.max()
    price_range = price_max - price_min

    if price_range <= 0:
        return None

    bin_size = price_range / tpo_bins

    # Assign each time period a letter (A, B, C, ...)
    tpo_data = {}
    for i in range(len(recent)):
        period_letter = chr(65 + (i % 26))  # A-Z, repeats
        low = lows[i]
        high = highs[i]

        low_bin = max(0, int((low - price_min) / bin_size))
        high_bin = min(tpo_bins - 1, int((high - price_min) / bin_size))

        for b in range(low_bin, high_bin + 1):
            price_level = round(price_min + (b + 0.5) * bin_size, 2)
            if price_level not in tpo_data:
                tpo_data[price_level] = ""
            if period_letter not in tpo_data[price_level]:
                tpo_data[price_level] += period_letter

    # Find initial balance (first hour = first 2 periods)
    ib_high = max(highs[:2]) if len(highs) >= 2 else highs[0]
    ib_low = min(lows[:2]) if len(lows) >= 2 else lows[0]

    # Sort by price (descending)
    sorted_prices = sorted(tpo_data.keys(), reverse=True)

    return {
        "tpo_data": {str(p): tpo_data[p] for p in sorted_prices},
        "initial_balance_high": round(ib_high, 2),
        "initial_balance_low": round(ib_low, 2),
        "value_area_high": None,  # Computed from volume profile
        "value_area_low": None,
        "bin_size": round(bin_size, 2),
        "price_min": round(price_min, 2),
        "price_max": round(price_max, 2)
    }


def get_volume_metrics(df) -> dict:
    """
    Get key volume-based metrics for a stock.

    Returns dict with buy/sell volume ratio, accumulation/distribution,
    and volume-weighted indicators.
    """
    if df.empty or len(df) < 20:
        return {}

    latest = df.iloc[-1]
    close = float(latest["Close"])

    # Compute buy/sell volume approximation
    # Using the Close相对于Open的位置来估计
    buy_vol = 0
    sell_vol = 0
    for i in range(min(20, len(df))):
        bar = df.iloc[-i - 1]
        if float(bar["Close"]) >= float(bar["Open"]):
            buy_vol += float(bar["Volume"])
        else:
            sell_vol += float(bar["Volume"])

    buy_sell_ratio = round(buy_vol / max(sell_vol, 1), 2)

    # VWAP distance
    vwap_df = vwap_with_bands(df, lookback=20)
    if "VWAP" in vwap_df.columns:
        vwap = float(vwap_df["VWAP"].iloc[-1])
        vwap_dist = round((close / vwap - 1) * 100, 2)
    else:
        vwap = None
        vwap_dist = None

    return {
        "buy_volume_20d": round(buy_vol, 0),
        "sell_volume_20d": round(sell_vol, 0),
        "buy_sell_ratio": buy_sell_ratio,
        "vwap": round(vwap, 2) if vwap else None,
        "vwap_distance_pct": vwap_dist,
        "above_vwap": vwap is not None and close > vwap,
        "volume_trend": "RISING" if buy_sell_ratio > 1.2 else ("FALLING" if buy_sell_ratio < 0.8 else "NEUTRAL")
    }
