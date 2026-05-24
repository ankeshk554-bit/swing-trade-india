import math


def calculate_position_size(
    capital: float,
    risk_pct: float,
    entry: float,
    stoploss: float,
    max_risk_per_trade: float = None,
) -> dict:
    """
    Calculate optimal position size with full risk analysis.

    Args:
        capital: Total available capital
        risk_pct: % of capital to risk on this trade (e.g., 1 = 1%)
        entry: Entry price per share
        stoploss: Stop loss price per share
        max_risk_per_trade: Max % of capital to risk (cap)

    Returns:
        dict with shares, risk_amount, position_value, risk_pct_portfolio,
        risk_reward_ratio, share_size_details
    """
    risk_amount = capital * risk_pct / 100

    # Apply max risk cap if provided
    if max_risk_per_trade is not None:
        max_risk = capital * max_risk_per_trade / 100
        risk_amount = min(risk_amount, max_risk)

    sl_distance = abs(entry - stoploss)
    if sl_distance <= 0:
        return {
            "shares": 0,
            "risk_amount": 0,
            "position_value": 0,
            "risk_pct_portfolio": 0,
            "risk_reward_ratio": 0,
            "error": "Stop loss cannot equal entry price"
        }

    shares = int(risk_amount / sl_distance)
    shares = max(shares, 0)

    position_value = shares * entry
    actual_risk_pct = (risk_amount / capital) * 100 if capital > 0 else 0

    return {
        "shares": shares,
        "risk_amount": round(risk_amount, 2),
        "position_value": round(position_value, 2),
        "risk_pct_portfolio": round(actual_risk_pct, 2),
        "risk_per_share": round(sl_distance, 2),
        "position_size_pct": round((position_value / capital) * 100, 2) if capital > 0 else 0,
        "error": None
    }


def calculate_position_with_target(
    capital: float,
    risk_pct: float,
    entry: float,
    stoploss: float,
    target: float = None,
) -> dict:
    """
    Calculate position size including risk-reward ratio and target analysis.

    Returns dict with all position sizing info + risk:reward.
    """
    result = calculate_position_size(capital, risk_pct, entry, stoploss)

    if target is not None and stoploss is not None and result["shares"] > 0:
        risk_per_share = abs(entry - stoploss)
        reward_per_share = abs(target - entry)
        result["risk_reward_ratio"] = round(reward_per_share / max(risk_per_share, 0.01), 2)
        result["target_price"] = target
        result["target_pnl"] = round(reward_per_share * result["shares"], 2)
        result["target_pnl_pct"] = round((target / entry - 1) * 100, 2)
    else:
        result["risk_reward_ratio"] = 0
        result["target_price"] = target
        result["target_pnl"] = 0
        result["target_pnl_pct"] = 0

    return result


def portfolio_heat(open_risks: list) -> dict:
    """
    Calculate total portfolio risk heat.

    Args:
        open_risks: List of individual trade risk amounts

    Returns dict with total_risk, risk_by_trade, warnings
    """
    total = sum(open_risks) if open_risks else 0

    warnings = []
    if total > 10:
        warnings.append("⚠️ Total risk > 10% — reduce position sizes")
    elif total > 6:
        warnings.append("⚠️ Total risk > 6% — consider reducing exposure")

    return {
        "total_risk_pct": round(total, 2),
        "trade_count": len(open_risks),
        "avg_risk_per_trade": round(total / max(len(open_risks), 1), 2),
        "warnings": warnings
    }


def suggest_stop_loss(
    entry: float,
    atr: float = None,
    atr_multiplier: float = 2.0,
    swing_low: float = None,
    ema: float = None,
    pct_stop: float = None,
) -> dict:
    """
    Suggest a stop loss level using multiple methods.

    Args:
        entry: Entry price
        atr: Average True Range value
        atr_multiplier: Multiplier for ATR-based stop
        swing_low: Recent swing low price
        ema: Key EMA value (e.g., EMA50 or EMA200)
        pct_stop: Percentage-based stop (e.g., 5 = 5% below entry)

    Returns dict with suggested stops and method used.
    """
    suggestions = {}

    if atr is not None and atr > 0:
        suggestions["ATR Based"] = round(entry - (atr * atr_multiplier), 2)

    if swing_low is not None:
        suggestions["Swing Low"] = round(swing_low * 0.995, 2)  # Slightly below swing low

    if ema is not None:
        suggestions["EMA Based"] = round(ema, 2)

    if pct_stop is not None:
        suggestions["Percentage"] = round(entry * (1 - pct_stop / 100), 2)

    # Best recommendation: use the tightest logical stop
    valid_stops = [v for v in suggestions.values() if v > 0 and v < entry]
    best = min(valid_stops) if valid_stops else None

    return {
        "suggestions": suggestions,
        "recommended": best,
        "entry": entry,
        "risk_if_recommended": round(abs(entry - best) * 100 / entry, 2) if best else None
    }


def calculate_pyramid_size(
    capital: float,
    entry_price: float,
    total_positions: int = 3,
    risk_per_trade: float = 0.5,
) -> list:
    """
    Calculate pyramiding position sizes (adding to winners).

    Returns list of position sizes for each pyramid level.
    """
    sizes = []
    for i in range(total_positions):
        level_risk = risk_per_trade * (1 + i * 0.5)
        pos_capital = capital / total_positions
        risk_amt = pos_capital * level_risk / 100
        size = risk_amt  # This would be shares = risk_amt / sl_distance
        sizes.append({
            "level": i + 1,
            "risk_pct": round(level_risk, 2),
            "allocation_ratio": f"{1}/{total_positions}"
        })
    return sizes
