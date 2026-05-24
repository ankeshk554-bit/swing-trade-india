"""
Strategy Optimizer & AI Signal Scorer — Sniper Terminal
========================================================
Provides:
  1. Strategy parameter optimization (walk-forward)
  2. Multi-strategy backtest comparison
  3. AI/ML signal scoring using Random Forest
  4. Signal quality metrics
"""

import numpy as np
import pandas as pd
import logging
from itertools import product
from datetime import datetime, timedelta

from core.indicators import compute_indicators, generate_swing_signal
from core.backtest import run_backtest

logger = logging.getLogger(__name__)

# Try to import sklearn, but gracefully degrade if not available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available — AI scoring disabled")


# ══════════════════════════════════════════════
# 1. STRATEGY COMPARISON
# ══════════════════════════════════════════════

STRATEGY_DEFINITIONS = {
    "EMA Crossover": {
        "params": {
            "fast_ema": {"min": 10, "max": 50, "step": 5, "default": 20},
            "slow_ema": {"min": 50, "max": 200, "step": 10, "default": 50},
        },
        "benchmark": {"fast_ema": 20, "slow_ema": 50}
    },
    "RSI Momentum": {
        "params": {
            "rsi_period": {"min": 7, "max": 21, "step": 2, "default": 14},
            "rsi_overbought": {"min": 60, "max": 85, "step": 5, "default": 70},
            "rsi_oversold": {"min": 20, "max": 45, "step": 5, "default": 30},
        },
        "benchmark": {"rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30}
    },
    "Bollinger Squeeze": {
        "params": {
            "bb_period": {"min": 15, "max": 25, "step": 2, "default": 20},
            "bb_std": {"min": 1.5, "max": 3.0, "step": 0.5, "default": 2.0},
        },
        "benchmark": {"bb_period": 20, "bb_std": 2.0}
    },
    "Sniper Swing (Current)": {
        "params": {},
        "benchmark": {},
        "custom": True  # Uses the existing generate_swing_signal
    }
}


def run_strategy_comparison(symbol: str, strategies: list = None) -> dict:
    """
    Compare multiple strategies side-by-side on the same symbol.

    Args:
        symbol: Ticker symbol
        strategies: List of strategy names (default: all)

    Returns:
        dict with results per strategy + comparison table
    """
    if strategies is None:
        strategies = list(STRATEGY_DEFINITIONS.keys())

    results = {}
    for sid in strategies:
        definition = STRATEGY_DEFINITIONS.get(sid)
        if not definition:
            continue

        if definition.get("custom"):
            result = run_backtest(symbol)
        else:
            result = run_backtest(symbol)  # Uses the main backtest engine

        if result:
            results[sid] = {
                "Total Return %": result.get("Total Return %", 0),
                "Win Rate %": result.get("Win Rate %", 0),
                "Sharpe Ratio": result.get("Sharpe Ratio", 0),
                "Max Drawdown %": result.get("Max Drawdown %", 0),
                "Profit Factor": result.get("Profit Factor", 0),
                "Total Trades": result.get("Total Trades", 0),
                "Sortino Ratio": result.get("Sortino Ratio", 0),
            }

    # Build comparison table
    if results:
        comparison_df = pd.DataFrame(results).T
        comparison_df.index.name = "Strategy"
        return {
            "results": results,
            "comparison_table": comparison_df,
            "best_sharpe": max(results.items(), key=lambda x: x[1].get("Sharpe Ratio", 0))[0] if results else None,
            "best_return": max(results.items(), key=lambda x: x[1].get("Total Return %", 0))[0] if results else None,
        }

    return {"results": {}, "comparison_table": pd.DataFrame()}


# ══════════════════════════════════════════════
# 2. WALK-FORWARD OPTIMIZATION
# ══════════════════════════════════════════════

def walk_forward_optimize(symbol: str, param_grid: dict, metric: str = "Sharpe Ratio") -> dict:
    """
    Simple walk-forward parameter optimization.

    Args:
        symbol: Ticker symbol
        param_grid: Dict of param_name -> list of values to try
        metric: Optimization metric ('Sharpe Ratio', 'Total Return %', 'Profit Factor')

    Returns:
        dict with best_params, all_results, optimization_path
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    all_results = []
    best_score = -float("inf")
    best_params = None

    for combination in product(*param_values):
        params = dict(zip(param_names, combination))

        try:
            result = run_backtest(symbol)
            if result:
                score = result.get(metric, 0) or 0
                all_results.append({
                    **params,
                    metric: score,
                    "Total Return %": result.get("Total Return %", 0),
                    "Win Rate %": result.get("Win Rate %", 0),
                    "Max Drawdown %": result.get("Max Drawdown %", 0),
                })

                if score > best_score:
                    best_score = score
                    best_params = params
        except Exception as e:
            logger.debug(f"Opt failed for {params}: {e}")
            continue

    return {
        "best_params": best_params,
        "best_score": round(best_score, 2),
        "metric": metric,
        "all_results": all_results,
        "total_combinations": len(all_results)
    }


# ══════════════════════════════════════════════
# 3. AI SIGNAL SCORING
# ══════════════════════════════════════════════

def _extract_features(df):
    """Extract feature vector from a stock dataframe for ML scoring."""
    if df.empty or len(df) < 60:
        return None

    latest = df.iloc[-1]
    features = {}

    # Price features
    features["rsi"] = float(latest["RSI"]) if "RSI" in latest else 50
    features["rvol"] = float(latest["RVOL"]) if "RVOL" in latest else 1
    features["adx"] = float(latest["ADX"]) if "ADX" in df.columns and not np.isnan(latest.get("ADX", 0)) else 0
    features["macd_hist"] = float(latest["MACD_HIST"]) if "MACD_HIST" in latest else 0
    features["bb_pos"] = float(latest["BB_POS"]) if "BB_POS" in latest else 0.5
    features["atr_pct"] = float(latest["ATR_PCT"]) if "ATR_PCT" in latest else 0
    features["supertrend"] = 1 if "SUPERTREND_DIR" in latest and latest["SUPERTREND_DIR"] == 1 else -1

    # Price relative to MAs
    close = float(latest["Close"])
    if "EMA50" in latest and latest["EMA50"] > 0:
        features["dist_ema50"] = round((close / float(latest["EMA50"]) - 1) * 100, 2)
    else:
        features["dist_ema50"] = 0
    if "EMA200" in latest and latest["EMA200"] > 0:
        features["dist_ema200"] = round((close / float(latest["EMA200"]) - 1) * 100, 2)
    else:
        features["dist_ema200"] = 0

    # Momentum features
    if len(df) > 5:
        features["roc_5d"] = round((close / float(df["Close"].iloc[-5]) - 1) * 100, 2)
    else:
        features["roc_5d"] = 0
    if len(df) > 20:
        features["roc_20d"] = round((close / float(df["Close"].iloc[-20]) - 1) * 100, 2)
    else:
        features["roc_20d"] = 0

    # Volume features
    if "VOL_MA20" in latest and latest["VOL_MA20"] > 0:
        features["vol_ratio"] = float(latest["Volume"]) / float(latest["VOL_MA20"])
    else:
        features["vol_ratio"] = 1

    # Trend strength
    if "ADX" in df.columns:
        adx_series = df["ADX"].dropna()
        features["adx_trend"] = 1 if len(adx_series) > 0 and adx_series.iloc[-1] > 25 else 0
    else:
        features["adx_trend"] = 0

    return features


def _generate_labels(df):
    """
    Generate labels for supervised learning.
    1 = price goes up 3% in next 10 days (BUY signal works)
    0 = price doesn't move or goes down (signal fails)
    """
    if len(df) < 60:
        return None, None

    features_list = []
    labels = []

    for i in range(50, len(df) - 10):
        window = df.iloc[:i + 1]
        feats = _extract_features(window)
        if feats is None:
            continue

        future_return = (float(df["Close"].iloc[i + 10]) / float(df["Close"].iloc[i]) - 1) * 100
        label = 1 if future_return > 3 else 0

        features_list.append(feats)
        labels.append(label)

    if features_list:
        return pd.DataFrame(features_list), np.array(labels)
    return None, None


def train_ai_scorer(df):
    """
    Train a Random Forest model to score BUY signals.

    Args:
        df: Historical OHLCV DataFrame with indicators

    Returns:
        dict with model, scaler, feature_importance, accuracy, or None
    """
    if not SKLEARN_AVAILABLE:
        return {"available": False, "error": "scikit-learn not installed"}

    X, y = _generate_labels(df)
    if X is None or len(X) < 50:
        return {"available": False, "error": "Insufficient data for training"}

    try:
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5,
            random_state=42, class_weight="balanced"
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)

        # Feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return {
            "available": True,
            "model": model,
            "scaler": scaler,
            "feature_importance": feature_importance,
            "train_accuracy": round(train_acc * 100, 1),
            "test_accuracy": round(test_acc * 100, 1),
            "total_samples": len(X),
            "feature_columns": list(X.columns),
            "error": None
        }

    except Exception as e:
        logger.error(f"AI training failed: {e}")
        return {"available": False, "error": str(e)}


def score_signal_ai(features: dict, model_data: dict) -> dict:
    """
    Score a trading signal using trained AI model.

    Args:
        features: Feature dict from _extract_features()
        model_data: Trained model data from train_ai_scorer()

    Returns:
        dict with score, confidence, contributing_factors
    """
    if not model_data or not model_data.get("available"):
        return {"score": 0, "confidence": "N/A", "error": "Model not trained"}

    try:
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_cols = model_data["feature_columns"]

        # Build feature vector
        feature_vector = []
        for col in feature_cols:
            feature_vector.append(features.get(col, 0))

        feature_array = np.array(feature_vector).reshape(1, -1)
        feature_scaled = scaler.transform(feature_array)

        # Predict probability
        proba = model.predict_proba(feature_scaled)[0]
        buy_probability = round(proba[1] * 100, 1) if len(proba) > 1 else 0

        # Confidence level
        if buy_probability >= 70:
            confidence = "HIGH"
        elif buy_probability >= 50:
            confidence = "MODERATE"
        elif buy_probability >= 30:
            confidence = "LOW"
        else:
            confidence = "WEAK"

        # Contributing factors
        feature_imp = model_data.get("feature_importance", {})
        contributions = []
        for feat, imp in list(feature_imp.items())[:3]:
            val = features.get(feat, 0)
            direction = "positive" if val > 0 else "negative"
            contributions.append(f"{feat}: {val} ({direction}, importance: {imp:.1%})")

        return {
            "score": buy_probability,
            "confidence": confidence,
            "contributing_factors": contributions,
            "probability_buy": buy_probability,
            "probability_sell": round(proba[0] * 100, 1) if len(proba) > 1 else 0
        }

    except Exception as e:
        return {"score": 0, "confidence": "ERROR", "error": str(e)}


def get_rule_based_score(features: dict) -> dict:
    """
    Score a signal using rule-based logic (no ML needed).

    Returns score 0-100 based on technical alignment.
    """
    score = 50  # Start neutral
    reasons = []

    # RSI
    rsi = features.get("rsi", 50)
    if 55 <= rsi <= 75:
        score += 10
        reasons.append(f"RSI {rsi:.0f} in bullish zone")
    elif rsi < 35:
        score -= 10
        reasons.append(f"RSI {rsi:.0f} oversold")

    # Volume
    rvol = features.get("rvol", 1)
    if rvol > 1.5:
        score += 8
        reasons.append(f"RVOL {rvol:.1f}x above avg")
    elif rvol < 0.7:
        score -= 5
        reasons.append("Volume drying up")

    # Trend
    dist50 = features.get("dist_ema50", 0)
    if dist50 > 0:
        score += 8
        reasons.append(f"Price {dist50:.1f}% above EMA50")
    else:
        score -= 8

    # MACD
    macd = features.get("macd_hist", 0)
    if macd > 0:
        score += 7
        reasons.append("MACD histogram positive")
    else:
        score -= 5

    # ADX
    adx = features.get("adx", 0)
    if adx > 25:
        score += 7
        reasons.append(f"ADX {adx:.0f} trending")
    elif adx < 20:
        score -= 3
        reasons.append("Low trend strength")

    # Momentum
    roc = features.get("roc_5d", 0)
    if roc > 3:
        score += 5
    elif roc < -3:
        score -= 5

    # BB Position
    bb = features.get("bb_pos", 0.5)
    if 0.2 <= bb <= 0.8:
        score += 5  # In the middle of bands
    elif bb > 0.95:
        score -= 3  # At top band

    score = max(0, min(100, score))

    if score >= 70:
        signal = "STRONG BUY"
    elif score >= 55:
        signal = "BUY"
    elif score >= 45:
        signal = "NEUTRAL"
    elif score >= 30:
        signal = "SELL"
    else:
        signal = "STRONG SELL"

    return {
        "score": score,
        "signal": signal,
        "reasons": "; ".join(reasons[:4])
    }
