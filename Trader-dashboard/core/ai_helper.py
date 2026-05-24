"""
AI helper functions for the Trader Dashboard.
Provides a call_ai() function to query DeepSeek/OpenAI APIs,
and stock analysis prompt builders.
"""
import requests as req


def call_ai(
    messages,
    api_key,
    provider="DS",
    mode="flash",
    max_tokens=2048,
    temperature=0.3,
    timeout=30,
):
    """
    Send messages to an AI API and return the response text.

    Parameters
    ----------
    messages : list[dict]
        Messages in {"role": "...", "content": "..."} format
    api_key : str
        API key for the provider
    provider : str
        "DS" for DeepSeek, "OA" for OpenAI
    mode : str
        "flash" for fast model, "reasoning" for detailed step-by-step
    max_tokens : int
        Maximum response tokens
    temperature : float
        Response creativity (0.0-1.0)
    timeout : int
        HTTP request timeout in seconds

    Returns
    -------
    str
        Response content, or None on failure
    """
    base_url = "https://api.deepseek.com" if provider == "DS" else "https://api.openai.com/v1"

    if provider == "DS":
        model = "deepseek-reasoner" if mode == "reasoning" else "deepseek-chat"
    else:
        model = "o3-mini" if mode == "reasoning" else "gpt-4o-mini"

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = req.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def build_stock_analysis_prompt(symbol, setup_data, timeframe="intraday", live_df=None):
    """
    Build a system + user prompt pair for analyzing a stock setup.

    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "RELIANCE.NS")
    setup_data : dict
        Scanner output with keys: Score, Entry, StopLoss, Target1, Target2,
        R:R, RSI, ADX, RVOL, Reasons, Concerns, etc.
    timeframe : str
        "intraday" or "swing"
    live_df : pd.DataFrame or None
        Live indicator DataFrame for chart-level analysis

    Returns
    -------
    tuple (system_prompt, user_prompt)
    """
    system = (
        "You are Sniper Terminal AI — a professional Indian stock market analyst "
        "with 20 years of experience. Analyze the given trade setup with SPECIFIC, "
        "ACTIONABLE guidance. Use ₹ symbols. Be concise (max 250 words). "
        "Focus on: entry timing, stop placement logic, target probability, "
        "key levels to watch, and risk management. "
        "NO buy/sell advice — educational only.\n\n"
        "Structure your response as:\n"
        "1. **Setup Quality**: Brief verdict on this setup\n"
        "2. **Entry Strategy**: Best way to enter (now, pullback, breakout confirm)\n"
        "3. **Key Levels**: Support/resistance around entry\n"
        "4. **Risk Notes**: What could go wrong, how to manage\n"
        "5. **Actionable Plan**: If-then scenarios"
    )

    sym_clean = symbol.replace(".NS", "")
    reasons = "; ".join(setup_data.get("Reasons", [])[:3])
    concerns = "; ".join(setup_data.get("Concerns", [])[:2])

    # Build live chart context if available
    chart_context = ""
    if live_df is not None and not live_df.empty:
        try:
            latest = live_df.iloc[-1]
            prev5 = live_df.iloc[-6] if len(live_df) > 5 else live_df.iloc[-2]
            prev20 = live_df.iloc[-21] if len(live_df) > 20 else live_df.iloc[0]

            # Recent price action
            change_5 = ((latest.get("Close", 0) - prev5.get("Close", 0)) / prev5.get("Close", 1) * 100)
            change_20 = ((latest.get("Close", 0) - prev20.get("Close", 0)) / prev20.get("Close", 1) * 100)

            # Volume trend
            vol_5_avg = live_df["Volume"].iloc[-6:].mean()
            vol_20_avg = live_df["Volume"].iloc[-21:].mean()

            # Candle pattern detection (last 3 candles)
            candles = []
            for i in range(-3, 0):
                row = live_df.iloc[i]
                body = abs(row.get("Close", 0) - row.get("Open", 0))
                upper = row.get("High", 0) - max(row.get("Close", 0), row.get("Open", 0))
                lower = min(row.get("Close", 0), row.get("Open", 0)) - row.get("Low", 0)
                dir_c = "GREEN" if row.get("Close", 0) > row.get("Open", 0) else "RED"
                candles.append(f"{dir_c}(O:{row.get('Open',0):.1f} H:{row.get('High',0):.1f} L:{row.get('Low',0):.1f} C:{row.get('Close',0):.1f})")

            # Support/Resistance from recent price action
            recent_high = live_df["High"].iloc[-10:].max()
            recent_low = live_df["Low"].iloc[-10:].min()

            chart_context = (
                f"\n\n📊 **Live Chart Context ({timeframe}):**\n"
                f"- Last 5-bar change: {change_5:+.2f}% | Last 20-bar: {change_20:+.2f}%\n"
                f"- Recent range: ₹{recent_low:,.2f} – ₹{recent_high:,.2f}\n"
                f"- Position in range: {(latest.get('Close',0)-recent_low)/(recent_high-recent_low)*100:.1f}%\n"
                f"- Volume trend: last 5 avg {vol_5_avg:,.0f} vs 20 avg {vol_20_avg:,.0f}\n"
                f"- Last 3 candles: {', '.join(candles)}\n"
            )

            # Add indicator context
            if all(k in latest for k in ["BB_Upper", "BB_Lower", "BB_Mid"]):
                bb_pos = (latest["Close"] - latest["BB_Lower"]) / (latest["BB_Upper"] - latest["BB_Lower"]) * 100
                chart_context += f"- Bollinger Band position: {bb_pos:.0f}% (lower=0, middle=50, upper=100)\n"

            if all(k in latest for k in ["VWAP", "Close"]):
                chart_context += f"- VWAP distance: {(latest['Close']/latest['VWAP']-1)*100:+.2f}%\n"

        except Exception:
            pass

    user = (
        f"Analyze this {timeframe} setup for **{sym_clean}**:\n"
        f"- Score: {setup_data.get('Score', 0)}/100 | Direction: {setup_data.get('Direction', 'BUY')}\n"
        f"- Entry: ₹{setup_data.get('Entry', 0):,.2f} | Stop: ₹{setup_data.get('StopLoss', 0):,.2f}\n"
        f"- Target 1: ₹{setup_data.get('Target1', 0):,.2f} | Target 2: ₹{setup_data.get('Target2', 0):,.2f}\n"
        f"- R:R: {setup_data.get('R:R', 0)} | ATR%: {setup_data.get('ATR%', 0)}%\n"
        f"- RSI: {setup_data.get('RSI', 50)} | ADX: {setup_data.get('ADX', 20)} | RVOL: {setup_data.get('RVOL', 1.0)}x\n"
        f"- Reasons: {reasons}\n"
        f"- Concerns: {concerns}"
        f"{chart_context}\n\n"
        "Provide:\n"
        "1. **Setup Verdict** — Is this a tradeable setup right now?\n"
        "2. **Entry Plan** — Market order, limit order at discount, or wait for confirmation?\n"
        "3. **Key Levels** — Where is the nearest support/resistance?\n"
        "4. **Risk Scenarios** — What invalidates the trade?\n"
        "5. **Actionable Plan** — If price does X, then do Y."
    )

    return system, user
