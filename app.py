"""
Alpha-Graham Market Scanner — Streamlit Dashboard
====================================================
Run with: streamlit run app.py
"""

import time
import math
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

from scanner import (
    get_sp1500_tickers,
    scan_universe,
    scan_ticker,
    save_results,
    get_scan_history,
    add_to_portfolio,
    get_portfolio,
    remove_from_portfolio,
    init_db,
)

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Alpha-Graham Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0a0e14; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { color: #e0e6ed; }
    .metric-card {
        background: linear-gradient(135deg, #0d1520, #111d33);
        border: 1px solid #1e3a5f30;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .signal-strong-buy { color: #00e676; font-weight: 800; }
    .signal-buy { color: #66bb6a; font-weight: 700; }
    .signal-hold { color: #ffd740; font-weight: 600; }
    .signal-sell { color: #ff5252; font-weight: 600; }
    .signal-strong-sell { color: #d50000; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

init_db()

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Alpha-Graham Scanner")
    st.markdown("*Graham Valuation × Quant Factors*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🔍 Market Scanner", "📋 Leaderboard", "🔎 Stock Lookup", "💼 Portfolio", "📜 Scan History"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Scoring Model")
    st.markdown("""
    | Factor | Weight |
    |--------|--------|
    | 📉 Graham Value | 30% |
    | 📈 Momentum | 30% |
    | ⭐ Quality (F-Score) | 25% |
    | 🎯 Analyst Sentiment | 15% |
    """)

    st.markdown("---")
    st.markdown("### Signal Guide")
    st.markdown("""
    - **≥80** → 🟢 Strong Buy
    - **65-79** → 🟩 Buy
    - **45-64** → 🟡 Hold
    - **30-44** → 🟠 Sell
    - **<30** → 🔴 Strong Sell
    """)


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def signal_color(signal):
    colors = {
        "Strong Buy": "#00e676",
        "Buy": "#66bb6a",
        "Hold": "#ffd740",
        "Sell": "#ff5252",
        "Strong Sell": "#d50000",
    }
    return colors.get(signal, "#888")


def signal_emoji(signal):
    emojis = {
        "Strong Buy": "🟢",
        "Buy": "🟩",
        "Hold": "🟡",
        "Sell": "🟠",
        "Strong Sell": "🔴",
    }
    return emojis.get(signal, "⚪")


def score_bar(score, max_score, label=""):
    pct = min(100, score / max_score * 100) if max_score > 0 else 0
    color = "#00e676" if pct >= 70 else "#66bb6a" if pct >= 50 else "#ffd740" if pct >= 35 else "#ff5252"
    st.markdown(f"""
    <div style="margin-bottom:6px;">
        <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#8a9aba;">
            <span>{label}</span><span>{score:.0f}/{max_score}</span>
        </div>
        <div style="height:6px;background:#1e3a5f20;border-radius:3px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{color};border-radius:3px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_stock_card(result, expanded=True):
    """Display a detailed stock analysis card."""
    sig = result["signal"]
    sig_col = signal_color(sig)
    emoji = signal_emoji(sig)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1520,#111d33);border:1px solid #1e3a5f30;
                border-left:4px solid {sig_col};border-radius:10px;padding:20px;margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.6rem;font-weight:800;color:#e0e6ed;">{result['ticker']}</span>
                <span style="font-size:0.9rem;color:{sig_col};margin-left:12px;font-weight:700;">
                    {emoji} {sig}
                </span>
                <div style="color:#6a8aba;font-size:0.85rem;margin-top:2px;">{result['name']}</div>
                <div style="color:#4a6a8a;font-size:0.75rem;">{result['sector']} · ${result['price']:.2f}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:2.5rem;font-weight:900;color:{sig_col};">{result['total_score']}</div>
                <div style="font-size:0.65rem;color:#4a6a8a;letter-spacing:0.1em;">SCORE</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if expanded:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Score Breakdown**")
            score_bar(result["value_score"], 30, "📉 Graham Value")
            score_bar(result["momentum_score"], 30, "📈 Momentum")
            score_bar(result["quality_score"], 25, "⭐ Quality")
            score_bar(result["sentiment_score"], 15, "🎯 Sentiment")

        with col2:
            st.markdown("**Key Metrics**")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("P/E Ratio", f"{result.get('pe', 0):.1f}" if result.get('pe') else "N/A")
                st.metric("P/B Ratio", f"{result.get('pb', 0):.1f}" if result.get('pb') else "N/A")
                st.metric("Graham #", f"${result['graham_number']:.2f}" if result['graham_number'] > 0 else "N/A")
            with m2:
                st.metric("ROE", f"{result.get('roe', 0):.1f}%")
                st.metric("Margin", f"{result.get('profit_margin', 0):.1f}%")
                st.metric("Div Yield", f"{result.get('dividend_yield', 0):.2f}%")

        # Why column
        st.markdown(f"**Why:** {result['why']}")

        # Detailed reasons
        with st.expander("Full Analysis Details"):
            for reason in result.get("reasons", []):
                st.markdown(f"- {reason}")


# ═══════════════════════════════════════════════════════════════
# PAGE: MARKET SCANNER
# ═══════════════════════════════════════════════════════════════

if page == "🔍 Market Scanner":
    st.title("🔍 Market Scanner")
    st.markdown("Scan the S&P 1500 for Alpha-Graham opportunities")

    col1, col2, col3 = st.columns(3)
    with col1:
        scan_size = st.selectbox("Scan Universe", [
            "Quick Scan (50 stocks)",
            "S&P 500",
            "S&P 500 + MidCap",
            "Full S&P 1500",
        ])
    with col2:
        max_workers = st.slider("Parallel Threads", 2, 16, 8)
    with col3:
        min_score = st.slider("Min Score Filter", 0, 80, 40)

    if st.button("🚀 Start Scan", type="primary", use_container_width=True):
        with st.spinner("Loading ticker universe..."):
            all_tickers = get_sp1500_tickers()

            if scan_size == "Quick Scan (50 stocks)":
                # A curated set of interesting tickers across sectors
                tickers = all_tickers[:50] if len(all_tickers) >= 50 else all_tickers
            elif scan_size == "S&P 500":
                tickers = all_tickers[:500]
            elif scan_size == "S&P 500 + MidCap":
                tickers = all_tickers[:900]
            else:
                tickers = all_tickers

            st.info(f"Scanning {len(tickers)} tickers with {max_workers} threads...")

        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()

        def update_progress(completed, total, ticker):
            progress_bar.progress(completed / total)
            status_text.text(f"Scanning {ticker}... ({completed}/{total})")

        results = scan_universe(tickers, max_workers=max_workers, progress_callback=update_progress)

        progress_bar.progress(1.0)
        status_text.text(f"✅ Scan complete! {len(results)} stocks scored.")

        # Filter by min score
        results = [r for r in results if r["total_score"] >= min_score]

        # Save to DB
        save_results(results)

        # Store in session
        st.session_state["last_results"] = results

        # Display top results
        if results:
            st.markdown(f"### Top {min(20, len(results))} Results")
            for r in results[:20]:
                display_stock_card(r, expanded=False)
        else:
            st.warning("No stocks met the minimum score threshold.")


# ═══════════════════════════════════════════════════════════════
# PAGE: LEADERBOARD
# ═══════════════════════════════════════════════════════════════

elif page == "📋 Leaderboard":
    st.title("📋 Alpha-Graham Leaderboard")

    results = st.session_state.get("last_results", [])

    if not results:
        # Try loading from DB
        hist = get_scan_history()
        if not hist.empty:
            # Get the latest scan date
            latest_date = hist["scan_date"].max()
            latest = hist[hist["scan_date"] == latest_date]
            results = latest.to_dict("records")
            st.info(f"Showing results from last scan: {latest_date[:19]}")
        else:
            st.info("No scan results yet. Run a Market Scan first!")
            st.stop()

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    strong_buys = [r for r in results if r.get("signal") == "Strong Buy"]
    buys = [r for r in results if r.get("signal") == "Buy"]
    holds = [r for r in results if r.get("signal") == "Hold"]
    sells = [r for r in results if r.get("signal") in ("Sell", "Strong Sell")]

    c1.metric("🟢 Strong Buy", len(strong_buys))
    c2.metric("🟩 Buy", len(buys))
    c3.metric("🟡 Hold", len(holds))
    c4.metric("🟠 Sell+", len(sells))
    c5.metric("📊 Total", len(results))

    # Leaderboard table
    st.markdown("### Top 10 Picks")

    top10 = results[:10]
    if top10:
        df = pd.DataFrame(top10)
        display_cols = ["ticker", "name", "sector", "price", "total_score", "signal",
                        "value_score", "momentum_score", "quality_score", "sentiment_score",
                        "graham_number", "pe", "roe", "why"]
        available_cols = [c for c in display_cols if c in df.columns]
        df_display = df[available_cols].copy()

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "name": st.column_config.TextColumn("Company"),
                "total_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                "signal": st.column_config.TextColumn("Signal"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "graham_number": st.column_config.NumberColumn("Graham #", format="$%.2f"),
                "pe": st.column_config.NumberColumn("P/E", format="%.1f"),
                "roe": st.column_config.NumberColumn("ROE %", format="%.1f"),
                "why": st.column_config.TextColumn("Why", width="large"),
            },
        )

    # Detailed cards for top 3
    st.markdown("### Detailed Analysis — Top 3")
    for r in results[:3]:
        display_stock_card(r, expanded=True)
        st.markdown("---")

    # Score distribution chart
    if len(results) > 5:
        st.markdown("### Score Distribution")
        df_all = pd.DataFrame(results)
        fig = px.histogram(
            df_all, x="total_score", nbins=20,
            color_discrete_sequence=["#00d4ff"],
            labels={"total_score": "Alpha-Graham Score"},
        )
        fig.update_layout(
            plot_bgcolor="#0a0e14",
            paper_bgcolor="#0a0e14",
            font_color="#8a9aba",
            xaxis=dict(gridcolor="#1e3a5f30"),
            yaxis=dict(gridcolor="#1e3a5f30"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: STOCK LOOKUP
# ═══════════════════════════════════════════════════════════════

elif page == "🔎 Stock Lookup":
    st.title("🔎 Stock Lookup")
    st.markdown("Analyze any ticker — not limited to the S&P 1500")

    ticker_input = st.text_input("Enter Ticker Symbol", placeholder="e.g. AAPL, TSLA, CSTM").upper().strip()

    if ticker_input:
        with st.spinner(f"Analyzing {ticker_input}..."):
            result = scan_ticker(ticker_input)

        if result:
            display_stock_card(result, expanded=True)

            # Price chart
            st.markdown("### Price History")
            try:
                hist = yf.Ticker(ticker_input).history(period="1y")
                if not hist.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=hist.index, open=hist["Open"], high=hist["High"],
                        low=hist["Low"], close=hist["Close"], name="Price",
                    ))

                    # Add 200-SMA
                    if len(hist) >= 200:
                        sma200 = hist["Close"].rolling(200).mean()
                        fig.add_trace(go.Scatter(
                            x=hist.index, y=sma200, name="200-SMA",
                            line=dict(color="#ffd740", width=1.5),
                        ))

                    # Add 50-SMA
                    if len(hist) >= 50:
                        sma50 = hist["Close"].rolling(50).mean()
                        fig.add_trace(go.Scatter(
                            x=hist.index, y=sma50, name="50-SMA",
                            line=dict(color="#00d4ff", width=1),
                        ))

                    fig.update_layout(
                        plot_bgcolor="#0a0e14",
                        paper_bgcolor="#0a0e14",
                        font_color="#8a9aba",
                        xaxis_rangeslider_visible=False,
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Could not load price chart")

            # Add to portfolio button
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            with col1:
                shares = st.number_input("Shares", min_value=0.01, value=10.0, step=1.0)
            with col2:
                if st.button(f"📥 Add {ticker_input} to Portfolio", type="primary"):
                    add_to_portfolio(
                        ticker_input, result["price"], shares,
                        result["total_score"], result["signal"]
                    )
                    st.success(f"Added {shares} shares of {ticker_input} at ${result['price']:.2f}")

        else:
            st.error(f"Could not analyze {ticker_input}. Check the ticker symbol.")


# ═══════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO
# ═══════════════════════════════════════════════════════════════

elif page == "💼 Portfolio":
    st.title("💼 Portfolio Tracker")
    st.markdown("Track performance of your Alpha-Graham picks vs S&P 500")

    portfolio = get_portfolio()

    if portfolio.empty:
        st.info("No positions yet. Use Stock Lookup to add picks to your portfolio.")
        st.stop()

    # Fetch current prices and compute P&L
    portfolio_data = []
    total_invested = 0
    total_current = 0

    for _, row in portfolio.iterrows():
        try:
            tk = yf.Ticker(row["ticker"])
            current_price = tk.info.get("currentPrice") or tk.info.get("regularMarketPrice") or row["buy_price"]
        except Exception:
            current_price = row["buy_price"]

        invested = row["buy_price"] * row["shares"]
        current_val = current_price * row["shares"]
        pnl = current_val - invested
        pnl_pct = (pnl / invested * 100) if invested > 0 else 0

        total_invested += invested
        total_current += current_val

        portfolio_data.append({
            "id": row["id"],
            "Ticker": row["ticker"],
            "Shares": row["shares"],
            "Buy Price": row["buy_price"],
            "Current": current_price,
            "Invested": invested,
            "Value": current_val,
            "P&L": pnl,
            "P&L %": pnl_pct,
            "Score": row["score_at_buy"],
            "Signal": row["signal_at_buy"],
            "Buy Date": row["buy_date"][:10],
        })

    # Summary
    total_pnl = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Invested", f"${total_invested:,.2f}")
    c2.metric("Current Value", f"${total_current:,.2f}")
    c3.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl_pct:+.1f}%")
    c4.metric("Positions", len(portfolio_data))

    # Portfolio table
    df_pf = pd.DataFrame(portfolio_data)
    st.dataframe(
        df_pf[["Ticker", "Shares", "Buy Price", "Current", "P&L", "P&L %", "Score", "Signal", "Buy Date"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Buy Price": st.column_config.NumberColumn(format="$%.2f"),
            "Current": st.column_config.NumberColumn(format="$%.2f"),
            "P&L": st.column_config.NumberColumn(format="$%.2f"),
            "P&L %": st.column_config.NumberColumn(format="%.1f%%"),
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=100),
        },
    )

    # P&L chart
    if len(portfolio_data) > 0:
        fig = px.bar(
            df_pf, x="Ticker", y="P&L %",
            color="P&L %",
            color_continuous_scale=["#d50000", "#ffd740", "#00e676"],
            color_continuous_midpoint=0,
        )
        fig.update_layout(
            plot_bgcolor="#0a0e14", paper_bgcolor="#0a0e14",
            font_color="#8a9aba",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Remove positions
    st.markdown("### Manage Positions")
    for item in portfolio_data:
        col1, col2 = st.columns([4, 1])
        with col1:
            pnl_str = f"{'🟢' if item['P&L'] >= 0 else '🔴'} {item['Ticker']} — {item['Shares']} shares — P&L: ${item['P&L']:.2f} ({item['P&L %']:+.1f}%)"
            st.text(pnl_str)
        with col2:
            if st.button("🗑️", key=f"remove_{item['id']}"):
                remove_from_portfolio(item["id"])
                st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE: SCAN HISTORY
# ═══════════════════════════════════════════════════════════════

elif page == "📜 Scan History":
    st.title("📜 Scan History")

    hist = get_scan_history()

    if hist.empty:
        st.info("No scan history yet. Run a Market Scan first!")
        st.stop()

    # Show unique scan dates
    scan_dates = hist["scan_date"].unique()
    st.markdown(f"**{len(scan_dates)} scans recorded**")

    selected_date = st.selectbox("Select Scan Date", scan_dates)
    scan_data = hist[hist["scan_date"] == selected_date]

    st.dataframe(
        scan_data[["ticker", "name", "sector", "price", "total_score", "signal",
                    "value_score", "momentum_score", "quality_score", "sentiment_score",
                    "graham_number", "why"]],
        use_container_width=True,
        hide_index=True,
    )

    # Export option
    if st.button("📥 Export to CSV"):
        csv = scan_data.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"alpha_graham_scan_{selected_date[:10]}.csv",
            "text/csv",
        )
