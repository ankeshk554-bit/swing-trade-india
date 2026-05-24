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


def build_stock_analysis_prompt(symbol, setup_data, timeframe="intraday"):
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

    Returns
    -------
    tuple (system_prompt, user_prompt)
    """
    system = (
        "You are Sniper Terminal AI — a professional Indian stock market analyst. "
        "Analyze the given trade setup objectively. Use ₹ symbols. "
        "Be concise (max 200 words). Focus on: risk assessment, "
        "technical alignment, market context. NO buy/sell advice — educational only."
    )

    sym_clean = symbol.replace(".NS", "")
    reasons = "; ".join(setup_data.get("Reasons", [])[:3])
    concerns = "; ".join(setup_data.get("Concerns", [])[:2])

    user = (
        f"Analyze this {timeframe} setup for {sym_clean}:\n"
        f"- Score: {setup_data.get('Score', 0)}/100\n"
        f"- Entry: ₹{setup_data.get('Entry', 0):,.2f}\n"
        f"- Stop Loss: ₹{setup_data.get('StopLoss', 0):,.2f}\n"
        f"- Target 1: ₹{setup_data.get('Target1', 0):,.2f}\n"
        f"- Target 2: ₹{setup_data.get('Target2', 0):,.2f}\n"
        f"- R:R: {setup_data.get('R:R', 0)}\n"
        f"- RSI: {setup_data.get('RSI', 50)}\n"
        f"- ADX: {setup_data.get('ADX', 20)}\n"
        f"- RVOL: {setup_data.get('RVOL', 1.0)}x\n"
        f"- Reasons: {reasons}\n"
        f"- Concerns: {concerns}\n\n"
        "Provide: 1) Risk assessment of this setup 2) Key levels to watch "
        "3) Market context check 4) Overall verdict."
    )

    return system, user
