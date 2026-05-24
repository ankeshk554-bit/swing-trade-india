
# =========================================================
# PAGE: STOCK RESEARCH (SimplyWallSt-style)
# =========================================================
elif page == "🔍 Stock Research":
    st.subheader("🔍 Universal Stock Research")
    st.caption("Comprehensive company analysis — fundamentals, technicals, peers, and valuation")

    # ── Search ──
    srch_col1, srch_col2 = st.columns([3, 1])
    with srch_col1:
        research_symbol = st.text_input("Search any Indian stock (e.g. RELIANCE, TCS, HDFCBANK)", value="RELIANCE",
                                        key="research_sym", help="No .NS suffix needed — auto-appended")
    with srch_col2:
        research_range = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3, key="research_range")

    sym = research_symbol.upper().strip()
    if not sym.endswith(".NS"):
        sym += ".NS"

    if st.button("🔍 Research", key="research_go", width='stretch', type="primary"):
        with st.spinner(f"Gathering intelligence on {sym}..."):
            try:
                import yfinance as yf
                ticker = yf.Ticker(sym)
                info = ticker.info or {}

                # Load price data
                df = load_data(sym, period=research_range)
                if df is not None and not df.empty:
                    df = compute_indicators(df)
                    latest = df.iloc[-1]
                    c = float(latest["Close"])

                    # ── Company Snapshot ──
                    st.markdown("### 📋 Company Snapshot")
                    snap = st.columns(5)
                    with snap[0]:
                        name = info.get("longName", sym.replace(".NS", ""))
                        sector = info.get("sector", "N/A")
                        industry = info.get("industry", "N/A")
                        st.markdown(f"**{name}**")
                        st.caption(f"{sector} › {industry}")
                    with snap[1]:
                        mcap = info.get("marketCap", 0)
                        st.metric("Market Cap", f"₹{mcap/1e7:.1f}Cr" if mcap > 1e7 else "N/A")
                    with snap[2]:
                        pe = info.get("trailingPE", info.get("forwardPE", "—"))
                        st.metric("P/E", f"{pe:.2f}" if isinstance(pe, (int, float)) else "—")
                    with snap[3]:
                        div_yield = info.get("dividendYield", 0)
                        if div_yield:
                            div_yield *= 100
                        st.metric("Div Yield", f"{div_yield:.2f}%" if div_yield else "—")
                    with snap[4]:
                        beta = info.get("beta", "—")
                        st.metric("Beta", f"{beta:.2f}" if isinstance(beta, (int, float)) else "—")

                    # ── Price & Performance ──
                    st.markdown("---")
                    st.markdown("### 📈 Price Performance")
                    perf = st.columns(4)
                    ret_1m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-22]) - 1) * 100 if len(df) > 22 else 0
                    ret_3m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-66]) - 1) * 100 if len(df) > 66 else 0
                    ret_6m = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-132]) - 1) * 100 if len(df) > 132 else 0
                    ret_1y = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-252]) - 1) * 100 if len(df) > 252 else 0
                    with perf[0]: st.metric("1 Month", f"{ret_1m:+.2f}%")
                    with perf[1]: st.metric("3 Months", f"{ret_3m:+.2f}%")
                    with perf[2]: st.metric("6 Months", f"{ret_6m:+.2f}%")
                    with perf[3]: st.metric("1 Year", f"{ret_1y:+.2f}%")

                    # ── Technical Health ──
                    st.markdown("---")
                    st.markdown("### 🔧 Technical Health")
                    tech = st.columns(4)
                    with tech[0]:
                        rsi = float(latest.get("RSI", 50))
                        rsi_signal = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
                        st.metric("RSI (14)", f"{rsi:.1f}", rsi_signal,
                                  delta_color="inverse" if rsi > 70 else "normal")
                    with tech[1]:
                        adx = float(latest.get("ADX", 0))
                        st.metric("ADX (14)", f"{adx:.1f}", "Trending" if adx > 25 else "Ranging")
                    with tech[2]:
                        ema50 = float(latest.get("EMA50", c))
                        ema200 = float(latest.get("EMA200", c))
                        trend = "🟢 BULLISH" if c > ema50 > ema200 else ("🔴 BEARISH" if c < ema50 < ema200 else "🟡 SIDEWAYS")
                        st.metric("Trend Structure", trend)
                    with tech[3]:
                        rvol = float(latest.get("RVOL", 1))
                        st.metric("Volume (RVOL)", f"{rvol:.2f}x", "High" if rvol > 1.5 else "Normal")

                    # ── Fundamental Scorecard ──
                    st.markdown("---")
                    st.markdown("### 📊 Fundamental Scorecard")
                    fund = st.columns(4)
                    with fund[0]:
                        pb = info.get("priceToBook", "—")
                        st.metric("P/B", f"{pb:.2f}" if isinstance(pb, (int, float)) else "—")
                        high52 = info.get("fiftyTwoWeekHigh", 0)
                        low52 = info.get("fiftyTwoWeekLow", 0)
                        st.metric("52w Range", f"₹{low52:.0f}–₹{high52:.0f}" if high52 else "—")
                    with fund[1]:
                        roe = info.get("returnOnEquity", 0)
                        if roe: roe *= 100
                        st.metric("ROE", f"{roe:.1f}%" if roe else "—")
                        debt_eq = info.get("debtToEquity", "—")
                        st.metric("D/E", f"{debt_eq:.2f}" if isinstance(debt_eq, (int, float)) else "—")
                    with fund[2]:
                        eps = info.get("trailingEps", info.get("forwardEps", "—"))
                        st.metric("EPS", f"₹{eps:.2f}" if isinstance(eps, (int, float)) else "—")
                        rev_growth = info.get("revenueGrowth", 0)
                        if rev_growth: rev_growth *= 100
                        st.metric("Rev Growth", f"{rev_growth:+.1f}%" if rev_growth else "—")
                    with fund[3]:
                        profit_margin = info.get("profitMargins", 0)
                        if profit_margin: profit_margin *= 100
                        st.metric("Profit Margin", f"{profit_margin:.1f}%" if profit_margin else "—")
                        op_margin = info.get("operatingMargins", 0)
                        if op_margin: op_margin *= 100
                        st.metric("Op Margin", f"{op_margin:.1f}%" if op_margin else "—")

                    # ── Valuation vs Peers ──
                    st.markdown("---")
                    st.markdown("### 🏷️ Valuation vs Nifty 50")
                    try:
                        nifty = yf.Ticker("^NSEI")
                        nifty_info = nifty.info or {}
                        n_pe = nifty_info.get("trailingPE", 0)
                        val_cols = st.columns(3)
                        with val_cols[0]:
                            stock_pe = info.get("trailingPE", 0)
                            pct_vs_nifty = (stock_pe / max(n_pe, 0.1) - 1) * 100 if stock_pe and n_pe else 0
                            st.metric("P/E vs Nifty",
                                      f"{pct_vs_nifty:+.0f}%",
                                      delta="Expensive" if pct_vs_nifty > 20 else ("Cheap" if pct_vs_nifty < -20 else "Fair"),
                                      delta_color="inverse" if pct_vs_nifty > 20 else "normal")
                        with val_cols[1]:
                            target = info.get("targetMeanPrice", "—")
                            st.metric("Analyst Target",
                                      f"₹{target:.0f}" if isinstance(target, (int, float)) else "—",
                                      f"{((target / c) - 1) * 100:+.1f}%" if isinstance(target, (int, float)) and c else "")
                        with val_cols[2]:
                            rec = info.get("recommendationKey", "—")
                            rec_map = {"buy": "🟢 BUY", "strong_buy": "🟢 STRONG BUY",
                                       "hold": "🟡 HOLD", "sell": "🔴 SELL", "strong_sell": "🔴 STRONG SELL"}
                            st.metric("Analyst Rating", rec_map.get(rec, rec.upper() if rec != "—" else "—"))
                    except Exception:
                        st.caption("Valuation comparison unavailable")

                    # ── Peer Comparison Table ──
                    st.markdown("---")
                    st.markdown("### 👥 Peer Comparison")
                    sector_peers = {
                        "RELIANCE": ["TCS.NS", "WIPRO.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS"],
                        "TCS": ["RELIANCE.NS", "WIPRO.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS"],
                        "HDFCBANK": ["ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS", "INDUSINDBK.NS"],
                        "INFY": ["TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"],
                    }
                    base = sym.replace(".NS", "")
                    peer_list = sector_peers.get(base, ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"])

                    peer_data = []
                    for p in peer_list[:5]:
                        try:
                            pt = yf.Ticker(p)
                            pi = pt.info or {}
                            peer_data.append({
                                "Symbol": p.replace(".NS", ""),
                                "Mkt Cap (Cr)": f"₹{pi.get('marketCap', 0)/1e7:.0f}" if pi.get('marketCap') else "—",
                                "P/E": f"{pi.get('trailingPE', 0):.1f}" if pi.get('trailingPE') else "—",
                                "ROE": f"{pi.get('returnOnEquity', 0)*100:.1f}%" if pi.get('returnOnEquity') else "—",
                                "Div Yield": f"{pi.get('dividendYield', 0)*100:.2f}%" if pi.get('dividendYield') else "—",
                            })
                        except Exception:
                            continue
                    if peer_data:
                        st.dataframe(pd.DataFrame(peer_data), width='stretch', hide_index=True)
                    else:
                        st.caption("Peer data unavailable — yfinance rate limits")

                    # ── Signal Summary ──
                    st.markdown("---")
                    st.markdown("### 🧠 Trading Signal Summary")
                    sig_cols = st.columns(3)
                    with sig_cols[0]:
                        from core.profit_engine_v5 import generate_v4_signal
                        sig = generate_v4_signal(df)
                        st.metric("v4 Signal", sig.get("Signal", "NEUTRAL"),
                                  f"Score: {sig.get('Score', 0)}/10")
                    with sig_cols[1]:
                        from core.profit_engine import generate_v3_signal
                        sig3 = generate_v3_signal(df)
                        st.metric("v3 Signal", sig3.get("Signal", "NEUTRAL"),
                                  f"Score: {sig3.get('Score', 0)}/8")
                    with sig_cols[2]:
                        from core.indicators import generate_swing_signal
                        swing = generate_swing_signal(df)
                        st.metric("Classic Signal", swing.get("Signal", "NEUTRAL"),
                                  f"Conf: {swing.get('Confidence', 0)}/6")

                    # ── Key Levels ──
                    st.markdown("---")
                    st.markdown("### 🎯 Key Price Levels")
                    try:
                        from core.support_resistance import detect_support_resistance
                        sr = detect_support_resistance(df, lookback=100, order=5)
                        lvl_cols = st.columns(3)
                        with lvl_cols[0]:
                            ns = sr.get("nearest_support")
                            st.metric("🟢 Nearest Support", f"₹{ns:,.2f}" if ns else "—")
                        with lvl_cols[1]:
                            nr = sr.get("nearest_resistance")
                            st.metric("🔴 Nearest Resistance", f"₹{nr:,.2f}" if nr else "—")
                        with lvl_cols[2]:
                            fib_levels = fibonacci_levels(c, ns if ns else c * 0.9, nr if nr else c * 1.1) if ns and nr else {}
                            st.metric("📐 50% Fib", f"₹{fib_levels.get('level_50', 0):,.2f}" if fib_levels else "—")
                    except Exception:
                        st.caption("Levels unavailable")

                    st.success(f"✅ Research complete for {sym.replace('.NS', '')} — {len(df)} bars analyzed")

                else:
                    st.error(f"Could not load price data for {sym}. Check if the symbol exists.")

            except Exception as e:
                st.error(f"Research failed: {e}")
                import traceback
                traceback.print_exc()
