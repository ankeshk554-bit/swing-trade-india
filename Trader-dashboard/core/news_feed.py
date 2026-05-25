"""
News Feed Aggregator + Stock-Specific News Scanner
Fetches live RSS feeds from Indian/global sources and stock-specific news via Yahoo Finance.
"""
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import logging
import re

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# ── RSS Feed Sources ──
RSS_FEEDS = {
    "Moneycontrol": "https://www.moneycontrol.com/rss/latestnews.xml",
    "Economic Times Mkts": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Livemint Markets": "https://www.livemint.com/rss/money",
    "BS Markets": "https://www.business-standard.com/rss/markets-101.rss",
    "Reuters India": "https://www.reuters.com/tools/rss/topic/indiamarkets/",
}

# Keywords that signal market-moving news for Indian stocks
HIGH_IMPACT_KEYWORDS = [
    "buyback", "bonus", "split", "dividend", "results", "earnings",
    "order win", "contract win", "joint venture", "partnership",
    "approval", "clearance", "launch", "expansion", "acquisition",
    "stake buy", "stake sale", "promoter", "FII", "DII", "QIP",
    "FPO", "IPO", "allotment", "rights issue", "secures",
    "board meeting", "record date", "ex-date", "ex-date",
    "upgrade", "downgrade", "target price", "overweight",
    "crude", "inflation", "GDP", "GDP growth", "rate cut",
    "repo rate", "RBI", "budget", "finance minister",
]

MARKET_MOVING_KEYWORDS = [
    "sensex", "nifty", "market crash", "rally", "bull run",
    "bear market", "correction", "volatility", "FII outflow",
    "FII inflow", "foreign fund", "domestic fund",
    "global cues", "wall street", "fed rate", "US Fed",
    "trade war", "tariff", "geopolitical", "crude oil",
    "rupee", "dollar", "bond yield", "US treasury",
]


def fetch_rss_feeds(timeout=10, max_items=60):
    """
    Fetch all RSS feeds and return merged, sorted news items.

    Returns
    -------
    list[dict]
        News items sorted by published time (newest first), each with:
        title, link, summary, source, published, keywords, impact
    """
    all_items = []

    for source_name, url in RSS_FEEDS.items():
        try:
            resp = requests.get(url, timeout=timeout, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            resp.raise_for_status()

            root = ET.fromstring(resp.content)
            # RSS 2.0: channel → item
            items = []
            channel = root.find("channel")
            if channel is not None:
                items = channel.findall("item")
            else:
                # Maybe it's an Atom feed or direct items
                items = root.findall(".//item")

            for item in items:
                title = _get_text(item, "title")
                link = _get_text(item, "link")
                desc = _get_text(item, "description") or ""
                pub_str = _get_text(item, "pubDate") or _get_text(item, "dc:date") or ""

                # Parse publication date
                pub_date = _parse_date(pub_str)

                # Skip if older than 72 hours
                if pub_date and (datetime.now() - pub_date) > timedelta(hours=72):
                    continue

                # Detect market-moving impact
                title_lower = title.lower() if title else ""
                desc_lower = desc.lower() if desc else ""
                combined = title_lower + " " + desc_lower

                impact = _score_impact(combined)
                keywords_found = [kw for kw in HIGH_IMPACT_KEYWORDS if kw in combined]

                # Extract potential stock symbols mentioned
                stocks = _extract_stocks(combined)

                all_items.append({
                    "title": title or "Untitled",
                    "link": link or "#",
                    "summary": _clean_html(desc)[:300],
                    "source": source_name,
                    "published": pub_date or datetime.now(),
                    "impact": impact,
                    "keywords": keywords_found[:5],
                    "stocks_mentioned": stocks,
                })

        except Exception as e:
            logger.warning(f"RSS feed failed for {source_name}: {e}")
            continue

    # Sort by published time (newest first), then by impact
    all_items.sort(key=lambda x: (x["published"] or datetime.now()), reverse=True)
    return all_items[:max_items]


def _get_text(element, tag):
    """Get text from an XML element, trying several namespaces."""
    try:
        ns_map = {
            "dc": "http://purl.org/dc/elements/1.1/",
            "content": "http://purl.org/rss/1.0/modules/content/",
        }
        # Try standard tag
        el = element.find(tag)
        if el is not None and el.text:
            return el.text.strip()
        # Try with namespaces
        for prefix, ns in ns_map.items():
            el = element.find(f"{{{ns}}}{tag.split(':')[-1]}")
            if el is not None and el.text:
                return el.text.strip()
    except Exception:
        pass
    return ""


def _parse_date(date_str):
    """Parse various RSS date formats to datetime."""
    if not date_str:
        return None
    # Common RSS formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            # Convert to timezone-naive
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except (ValueError, TypeError):
            continue
    return None


def _score_impact(text):
    """Score market-moving impact of a news item (0-100)."""
    score = 0

    # High-impact keywords
    high_impact = [
        "rbi", "repo rate", "budget", "finance minister", "fed rate",
        "crude oil", "inflation", "gdp", "sensex", "nifty",
    ]
    for kw in high_impact:
        if kw in text:
            score += 20

    # Medium impact
    medium_impact = [
        "fii", "dii", "foreign", "result", "earnings", "acquisition",
        "buyback", "bonus", "split", "dividend", "order",
        "upgrade", "downgrade", "target",
    ]
    for kw in medium_impact:
        if kw in text:
            score += 10

    # Multiple matches
    count = sum(1 for kw in HIGH_IMPACT_KEYWORDS if kw in text)
    score += count * 3

    return min(100, score)


def _extract_stocks(text):
    """Try to extract NSE stock symbols mentioned in news text."""
    # Common NSE stock names
    known_stocks = [
        "reliance", "tcs", "hdfc", "infosys", "icici", "sbi", "bharti",
        "bajaj", "kotiak", "l&t", "lt", "wipro", "axis", "maruti",
        "sun pharma", "tatamotors", "tata motors", "tatasteel", "tata steel",
        "jsw", "hindalco", "vedanta", "coal india", "ongc", "nTPC",
        "powergrid", "nestle", "hul", "itc", "asian paints", "hdfcbank",
        "hdfc life", "sbin", "bajaj finance", "bajaj finserv", "maruti",
        "mahindra", "m&m", "tech mahindra", "titan", "adani",
    ]
    found = []
    text_lower = text.lower()
    for stock in known_stocks:
        if stock in text_lower:
            # Map to NSE symbol
            symbol = stock.upper().replace(" ", "")
            if symbol == "L&T" or symbol == "LT":
                symbol = "LT.NS"
            elif symbol == "HUL":
                symbol = "HINDUNILVR.NS"
            elif symbol == "SBI":
                symbol = "SBIN.NS"
            elif symbol == "M&M":
                symbol = "M&M.NS"
            elif symbol == "TATAMOTORS":
                symbol = "TATAMOTORS.NS"
            elif symbol == "TATASTEEL":
                symbol = "TATASTEEL.NS"
            elif symbol == "BAJAJFINSERV":
                symbol = "BAJAJFINSV.NS"
            else:
                symbol = f"{symbol}.NS"
            found.append(symbol)
    return list(set(found))


def _clean_html(text):
    """Remove HTML tags from text."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def get_stock_news(symbol, max_items=5):
    """
    Fetch news for a specific stock using Yahoo Finance.

    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "RELIANCE.NS")
    max_items : int
        Max news items to return

    Returns
    -------
    list[dict]
        News items with title, link, published, publisher, type
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return []

        results = []
        for item in news[:max_items]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)) if item.get("providerPublishTime") else datetime.now(),
                "publisher": item.get("publisher", ""),
                "type": item.get("type", "STORY"),
            })
        return results
    except Exception:
        return []


def scan_stock_news(symbols, max_news_per_stock=3, min_impact=30):
    """
    Scan a universe of stocks for significant recent news.

    Parameters
    ----------
    symbols : list
        Stock symbols to scan
    max_news_per_stock : int
        Max news items per stock
    min_impact : int
        Minimum impact score to include

    Returns
    -------
    list[dict]
        News items sorted by impact, with stock symbol
    """
    all_news = []
    seen_titles = set()

    for symbol in symbols:
        try:
            news = get_stock_news(symbol, max_items=max_news_per_stock)
            for item in news:
                title = item.get("title", "")
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                # Score impact
                impact = _score_impact(title.lower())
                if impact < min_impact:
                    continue

                # Check if this is a high-certainty trade trigger
                is_trade_trigger = any(
                    kw in title.lower()
                    for kw in ["buyback", "bonus", "split", "order win",
                               "contract", "results beat", "upgrade",
                               "stake buy", "approval"]
                )

                all_news.append({
                    "symbol": symbol,
                    "title": title,
                    "link": item.get("link", ""),
                    "published": item.get("published", datetime.now()),
                    "publisher": item.get("publisher", ""),
                    "impact": impact,
                    "is_trade_trigger": is_trade_trigger,
                })
        except Exception:
            continue

    all_news.sort(key=lambda x: (x["impact"], x["published"]), reverse=True)
    return all_news


# ── AI-Powered Categorization ──

NEWS_CATEGORIES = [
    "market_moving",   # RBI, budget, Fed, crude, inflation, GDP
    "stock_specific",  # Earnings, orders, buybacks, analyst actions
    "sector",          # Banking, IT, pharma, auto, etc.
    "ipo_fpo",         # IPO, FPO, QIP, rights
    "macro",           # Global cues, rupee, bond yields
    "general",         # Everything else
]

CATEGORY_EMOJI = {
    "market_moving": "🔴",
    "stock_specific": "📈",
    "sector": "🏭",
    "ipo_fpo": "🎯",
    "macro": "🌍",
    "general": "📰",
}


def categorize_news_keyword(item):
    """
    Categorize a news item using keyword rules (fast, no API cost).
    Adds 'category' and 'sentiment' fields to the item dict.
    """
    title = (item.get("title", "") + " " + item.get("summary", "")).lower()

    # Market-moving (macro policy, central bank)
    if any(kw in title for kw in ["rbi", "repo rate", "mpc", "monetary policy",
                                    "budget", "finance minister", "finance ministry",
                                    "fed rate", "federal reserve", "interest rate"]):
        item["category"] = "market_moving"
        item["sentiment"] = "neutral"
        return item

    # IPO / FPO
    if any(kw in title for kw in ["ipo", "fpo", "qip", "rights issue",
                                    "public issue", "listing"]):
        item["category"] = "ipo_fpo"
        item["sentiment"] = "positive" if any(kw in title for kw in ["oversubscribed", "listing gain", "strong demand"]) else "neutral"
        return item

    # Stock-specific
    if item.get("stocks_mentioned") or any(kw in title for kw in [
        "buyback", "bonus", "split", "dividend", "results", "earnings",
        "order win", "contract", "acquisition", "stake", "upgrade",
        "downgrade", "target price", "board meeting", "record date",
        "secures", "approval", "launch",
    ]):
        item["category"] = "stock_specific"
        item["sentiment"] = "positive" if any(kw in title for kw in [
            "upgrade", "buyback", "bonus", "order win", "profit",
            "growth", "strong", "beat", "oversubscribed", "secures",
        ]) else ("negative" if any(kw in title for kw in [
            "downgrade", "loss", "decline", "fall", "probe", "investigation",
            "fraud", "penalty", "default",
        ]) else "neutral")
        return item

    # Macro (global cues, currency, commodities)
    if any(kw in title for kw in ["crude", "rupee", "dollar", "bond yield",
                                    "wall street", "global cues", "trade war",
                                    "geopolitical", "inflation", "gdp",
                                    "sensex", "nifty", "market crash",
                                    "fii", "dii", "foreign fund"]):
        item["category"] = "macro"
        item["sentiment"] = "positive" if any(kw in title for kw in [
            "rally", "gain", "surplus", "inflow", "cut", "boost"
        ]) else ("negative" if any(kw in title for kw in [
            "crash", "fall", "decline", "outflow", "sell-off", "fear"
        ]) else "neutral")
        return item

    # Sector
    sectors = {
        "banking": ["bank", "bnk", "hdfc bank", "icici", "sbi", "axis", "kotak"],
        "it": ["it", "tech", "tcs", "infosys", "wipro", "hcl", "software"],
        "pharma": ["pharma", "drug", "healthcare", "cipla", "sun pharma", "dr reddy"],
        "auto": ["auto", "car", "vehicle", "maruti", "tata motors", "mahindra"],
        "energy": ["oil", "gas", "energy", "power", "renewable", "solar"],
        "metal": ["metal", "steel", "aluminium", "mining", "coal"],
    }
    for sector, kws in sectors.items():
        if any(kw in title for kw in kws):
            item["category"] = "sector"
            item["sentiment"] = "neutral"
            item["sector_name"] = sector.capitalize()
            return item

    # Default
    item["category"] = "general"
    item["sentiment"] = "neutral"
    return item


def categorize_news_batch_ai(news_items, api_key, provider="DS", mode="flash"):
    """
    Use AI to categorize and add sentiment to a batch of news items.
    Falls back to keyword categorization if API fails.

    Parameters
    ----------
    news_items : list[dict]
        News items from fetch_rss_feeds()
    api_key : str
        API key
    provider : str
        "DS" or "OA"
    mode : str
        "flash" or "reasoning"

    Returns
    -------
    list[dict]
        News items with 'category', 'sentiment', 'ai_summary' fields added
    """
    # Start with keyword categorization as base
    for item in news_items:
        categorize_news_keyword(item)

    if not api_key:
        return news_items

    # Batch: send headlines to AI for smart categorization
    headlines = []
    for i, item in enumerate(news_items[:30]):  # Cap at 30 items
        headlines.append(f"{i}: {item['title'][:120]}")

    prompt = (
        "Categorize each Indian market news headline into one of:\n"
        "- market_moving: RBI, policy, budget, Fed, inflation, GDP\n"
        "- stock_specific: earnings, orders, buybacks, analyst actions, M&A\n"
        "- sector: sector-wide news (banking, IT, pharma, auto, etc)\n"
        "- ipo_fpo: IPOs, FPOs, QIPs, rights issues\n"
        "- macro: global cues, rupee, crude, bond yields, FII flows\n"
        "- general: everything else\n\n"
        "Also assign sentiment: positive, negative, or neutral.\n\n"
        "Return ONLY a JSON array like:\n"
        '[{"id":0,"category":"macro","sentiment":"negative"},'
        '{"id":1,"category":"stock_specific","sentiment":"positive"},...]\n\n'
        "Headlines:\n" + "\n".join(headlines)
    )

    try:
        from core.ai_helper import call_ai
        msgs = [{"role": "user", "content": prompt}]
        reply = call_ai(msgs, api_key=api_key, provider=provider,
                        mode=mode, max_tokens=2000, temperature=0.1)
        if not reply:
            return news_items

        # Parse JSON from response
        import json as _json
        # Find JSON array in response
        match = re.search(r"\[.*?\]", reply, re.DOTALL)
        if match:
            parsed = _json.loads(match.group())
            for entry in parsed:
                idx = entry.get("id")
                if 0 <= idx < len(news_items):
                    news_items[idx]["category"] = entry.get("category", news_items[idx].get("category", "general"))
                    news_items[idx]["sentiment"] = entry.get("sentiment", news_items[idx].get("sentiment", "neutral"))
    except Exception:
        pass  # Fall back to keyword categorization

    return news_items
