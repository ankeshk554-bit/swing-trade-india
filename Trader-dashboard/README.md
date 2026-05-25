# ⚡ Sniper Terminal — India's Institutional-Grade Swing Trading Platform

A professional-grade swing trading platform for Indian markets (NSE/BSE) with institutional-level screening, charting, backtesting, and market regime analysis.

## Features

- **🔍 Institutional Screener** — Scan NIFTY 50/200/500 with 50+ technical criteria
- **📈 Advanced Charts** — Multi-panel charts with EMA, Bollinger Bands, MACD, Supertrend, VCP detection
- **📊 Full Backtest Engine** — Sharpe ratio, max drawdown, win rate, profit factor, Monte Carlo-ready
- **🎯 Strategy Themes** — Momentum, VCP Breakout, Mean Reversion, Strong Trend
- **📉 Market Regime** — Real-time NIFTY regime analysis with ADX strength, RSI zone
- **💾 Smart Caching** — Disk + Streamlit dual caching to minimize API calls
- **📥 CSV Export** — Download screener results

## Tech Stack

- **Streamlit** — UI framework
- **Python** — Core engine
- **Plotly** — Interactive charts
- **Yahoo Finance (yfinance)** — Market data
- **Pandas / NumPy** — Data processing

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── app.py                 # Main UI (Streamlit)
├── core/
│   ├── indicators.py      # All TA indicators + signal generation + VCP
│   ├── scanner.py         # Stock screening engine
│   ├── backtest.py        # Full backtest with metrics
│   ├── market_regime.py   # NIFTY regime detection
│   ├── relative_strength.py  # RS rating vs NIFTY
│   ├── risk_engine.py     # Position sizing
│   ├── charts.py          # Multi-panel Plotly charts
│   └── utils.py           # Data loading, caching, helpers
├── data/
│   ├── nifty50.py         # Full NIFTY 50 constituents
│   ├── nifty200.py        # Full NIFTY 200 constituents
│   └── nifty500.py        # Full NIFTY 500 constituents
└── assets/
    └── style.css          # Custom styles
```

## Roadmap

- [x] Critical bugs fixed (incomplete lists, silent errors, duplicated code)
- [x] Full NIFTY 50/200/500 stock lists with .NS suffix
- [x] Advanced indicator engine (Bollinger, MACD, ATR, ADX, Supertrend, OBV)
- [x] Multi-panel professional charts
- [x] Full backtest metrics suite
- [x] Strategy filtering themes
- [x] Disk caching to avoid rate limits
- [x] **F&O data integration** — PCR, OI change, IV, Max Pain, Max OI strikes
- [x] **Delivery volume analysis** — Delivery %, trend, quality score, accumulation days
- [x] **FII/DII flow data** — Cash & F&O net flows, combined sentiment
- [x] **India VIX integration** — Regime classification, fear/greed gauge
- [x] **Market breadth** — Advance/decline ratio, breadth strength
- [x] **Block/bulk deals** — Recent large transactions
- [x] **Expiry calendar** — Weekly & monthly expiry countdowns
- [x] **F&O ban list** — Stocks currently in F&O ban period
- [x] **50+ scan conditions** across 7 categories (Trend, Momentum, Volume, Pattern, Structure, RS, Indian)
- [x] **10 pre-built trading strategies** (Momentum Runner, VCP Breakout, 52W Breakout, Delivery Spurt, Golden Cross, etc.)
- [x] **Chart pattern recognition** — Bull Flag, Double Bottom, Cup & Handle, H&S, Breakout, Engulfing, Inside Bar
- [x] **Sector/industry mapping** — All NIFTY 500 stocks mapped to 13 sectors
- [x] **52-Week High/Low proximity** detection
- [x] **Higher Highs/Lower Lows** trend structure analysis
- [x] **Professional Dashboard** — Command Center with market pulse, VIX chart, sector rotation
- [x] **VIX Historical Chart** — 6-month trend with regime zones (Low/ Normal/ Elevated/ High/ Extreme)
- [x] **Sector Rotation Heatmap** — 13-sector performance tracking with visual ranking
- [x] **Top Movers Panel** — Gainers, Losers, & Most Active by volume
- [x] **Institutional Tabs** — F&O Chain, FII/DII Flows, Market Breadth, Expiry Calendar, Block Deals
- [x] **Market Pulse Bar** — One-row executive summary of all key indicators
- [x] **Strategy Quick Access** — One-click strategy cards leading to the Screener
- [x] **Watchlist Manager** — Create/delete multiple watchlists with live prices & signals
- [x] **Trade Journal** — Log entries/exits with tags, notes, SL, targets, P&L tracking
- [x] **Portfolio Dashboard** — Win rate, profit factor, monthly returns, best/worst trades
- [x] **Position Size Calculator** — Risk-based sizing with R:R analysis, stop suggestions
- [x] **Open Positions** — Live P&L tracking with close functionality
- [x] **Monthly Returns** — Performance breakdown by month
- [x] **Support & Resistance Detection** — Swing point clustering, most-tested levels, visual bars
- [x] **Fibonacci Levels** — Auto-detect swing points + manual retracement/extension calculator
- [x] **Strategy Comparison** — Side-by-side backtest comparison (Sniper Swing, EMA, RSI, Bollinger)
- [x] **AI Signal Scoring** — ML-based (Random Forest) + rule-based signal scoring with feature analysis
- [x] **Pivot Points** — Classic pivot calculation (P, R1-R3, S1-S3)
- [x] **Volume Profile** — Volume at Price, POC, VAH/VAL, HVN/LVN, VWAP bands
- [x] **Multi-Stock Comparison** — Side-by-side relative performance (up to 6 stocks)
- [x] **Volume Profile Chart** — Dedicated volume at price histogram with POC & Value Area
- [x] **Indicator Toggles** — On/off for BB, MACD, VWAP, Supertrend, Volume Profile, S/R levels
- [x] **S/R Overlay** — Auto-detect support/resistance levels directly on chart
- [x] **Interval Switching** — Daily, Weekly, Monthly charts
- [x] **VWAP Analysis** — VWAP distance, buy/sell ratio, volume trend
- [ ] Real-time alerts (Telegram/Email)
- [ ] BSE data support