import numpy as np
import logging
from core.utils import load_data
from core.indicators import compute_indicators

logger = logging.getLogger(__name__)


def get_market_regime(detailed=False):
    """
    Determine current NIFTY 50 market regime with multi-factor analysis.

    Incorporates:
      - NIFTY 50 price action (EMA structure, RSI, ADX)
      - India VIX (volatility regime)
      - FII/DII flows (institutional sentiment)
      - Market breadth (advance/decline)

    Args:
        detailed: If True, returns full regime dict with all metrics.
                 If False, returns short string: BULLISH / BEARISH / SIDEWAYS / UNKNOWN
    """
    try:
        df = load_data("^NSEI", period="1y")
        if df.empty:
            return "UNKNOWN" if not detailed else {"regime": "UNKNOWN"}

        df = compute_indicators(df)
        latest = df.iloc[-1]
        close = float(latest["Close"])
        ema50 = float(latest["EMA50"])
        ema200 = float(latest["EMA200"])
        rsi = float(latest["RSI"])
        adx = float(latest["ADX"]) if "ADX" in df.columns and not np.isnan(latest["ADX"]) else 0

        # === Price-based regime ===
        if close > ema50 > ema200:
            trend = "BULLISH"
        elif close < ema50 < ema200:
            trend = "BEARISH"
        else:
            trend = "SIDEWAYS"

        # Trend strength
        if adx > 25:
            strength = "STRONG"
        elif adx > 20:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        # RSI zone
        if rsi > 70:
            rsi_zone = "Overbought"
        elif rsi > 55:
            rsi_zone = "Bullish"
        elif rsi > 45:
            rsi_zone = "Neutral"
        elif rsi > 30:
            rsi_zone = "Bearish"
        else:
            rsi_zone = "Oversold"

        if not detailed:
            return trend

        # === Multi-factor regime (detailed only) ===
        # Lazy imports to avoid circular dependency on core package
        from core.data_providers import get_india_vix, get_fii_dii_data, get_market_breadth
        vix_info = get_india_vix()
        fii_info = get_fii_dii_data()
        breadth = get_market_breadth()

        # Combined regime score (-10 to +10)
        regime_score = 0

        # Price contribution (max ±4)
        if trend == "BULLISH":
            regime_score += 2
            if strength == "STRONG":
                regime_score += 2
        elif trend == "BEARISH":
            regime_score -= 2
            if strength == "STRONG":
                regime_score -= 2

        # VIX contribution (max ±2)
        vix_regime = vix_info.get("regime", "NORMAL") if vix_info else "NORMAL"
        if vix_regime in ("LOW_VOL", "NORMAL"):
            regime_score += 1
        elif vix_regime in ("HIGH_VOL", "EXTREME_FEAR"):
            regime_score -= 2
        elif vix_regime == "ELEVATED":
            regime_score -= 1

        # FII contribution (max ±2)
        if fii_info and fii_info.get("net_combined") is not None:
            if fii_info["net_combined"] > 500:  # Cr
                regime_score += 2
                fii_sentiment = "🟢 Strong buying"
            elif fii_info["net_combined"] > 0:
                regime_score += 1
                fii_sentiment = "🟢 Mild buying"
            elif fii_info["net_combined"] > -500:
                regime_score -= 1
                fii_sentiment = "🔴 Mild selling"
            else:
                regime_score -= 2
                fii_sentiment = "🔴 Heavy selling"
        else:
            fii_sentiment = "N/A"

        # Breadth contribution (max ±2)
        adr = breadth.get("advance_decline_ratio", 1.0) if breadth else 1.0
        if adr > 1.5:
            regime_score += 2
        elif adr > 1.0:
            regime_score += 1
        elif adr > 0.7:
            regime_score -= 1
        else:
            regime_score -= 2

        # Final composite regime
        if regime_score >= 4:
            composite = "STRONG_BULLISH 🟢🟢"
        elif regime_score >= 1:
            composite = "BULLISH 🟢"
        elif regime_score >= -2:
            composite = "SIDEWAYS 🟡"
        elif regime_score >= -5:
            composite = "BEARISH 🔴"
        else:
            composite = "STRONG_BEARISH 🔴🔴"

        # RSI bias
        rsi_bias = rsi - 50

        return {
            "regime": trend,
            "composite": composite,
            "regime_score": regime_score,
            "strength": strength,
            "nifty_close": round(close, 2),
            "ema50": round(ema50, 2),
            "ema200": round(ema200, 2),
            "rsi": round(rsi, 2),
            "rsi_zone": rsi_zone,
            "rsi_bias": round(rsi_bias, 1),
            "adx": round(adx, 2),
            "ema50_distance": round((close / ema50 - 1) * 100, 2),
            "ema200_distance": round((close / ema200 - 1) * 100, 2),
            # VIX
            "vix": vix_info.get("vix") if vix_info else None,
            "vix_regime": vix_regime,
            "vix_change": vix_info.get("change") if vix_info else None,
            # FII/DII
            "fii_cash": fii_info.get("fii_cash") if fii_info else None,
            "dii_cash": fii_info.get("dii_cash") if fii_info else None,
            "fii_sentiment": fii_sentiment,
            "net_fii_dii": fii_info.get("net_combined") if fii_info else None,
            # Breadth
            "advances": breadth.get("advances", 0) if breadth else 0,
            "declines": breadth.get("declines", 0) if breadth else 0,
            "advance_decline_ratio": breadth.get("advance_decline_ratio", 1.0) if breadth else 1.0,
            "breadth_strength": breadth.get("breadth_strength", "UNKNOWN") if breadth else "UNKNOWN"
        }

    except Exception as e:
        logger.error(f"Market Regime Error: {e}")
        return "UNKNOWN" if not detailed else {"regime": "UNKNOWN"}