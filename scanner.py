"""
Alpha-Graham Market Scanner — Core Scoring Engine
===================================================
Merges Benjamin Graham's Intelligent Investor valuation with modern
quant factors (Momentum, Quality, Revisions) to identify high-alpha stocks.

Scoring Model (0-100):
  Value (30%)      — Graham Number, PE, PB, PE×PB, Earnings Yield
  Momentum (30%)   — 200-SMA, RSI, 3/6/12-month returns
  Quality (25%)    — Piotroski F-Score, ROE, margins
  Sentiment (15%)  — EPS revision trend, analyst recommendations
"""

import time
import math
import sqlite3
import logging
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("alpha_graham")

DB_PATH = Path(__file__).parent / "alpha_graham.db"

# ═══════════════════════════════════════════════════════════════
# S&P 1500 UNIVERSE
# ═══════════════════════════════════════════════════════════════

def get_sp1500_tickers():
    """Get S&P 500 + S&P 400 MidCap + S&P 600 SmallCap tickers."""
    try:
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].tolist()
    except Exception:
        sp500 = []

    try:
        sp400 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")[0]["Symbol"].tolist()
    except Exception:
        sp400 = []

    try:
        sp600 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")[0]["Symbol"].tolist()
    except Exception:
        sp600 = []

    tickers = list(set(sp500 + sp400 + sp600))
    # Clean tickers: replace dots with hyphens for yfinance
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(tickers)


# ═══════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════

def fetch_stock_data(ticker: str, retries: int = 2) -> dict | None:
    """Fetch all required data for a single ticker. Returns dict or None on failure."""
    for attempt in range(retries):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}

            # Skip if missing critical data
            eps = info.get("trailingEps")
            bvps = info.get("bookValue")
            price = info.get("currentPrice") or info.get("regularMarketPrice")

            if not price or price <= 0:
                return None

            # Financial statements for Piotroski
            try:
                bs = tk.balance_sheet
                inc = tk.income_stmt
                cf = tk.cashflow
            except Exception:
                bs, inc, cf = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            # Price history for momentum
            try:
                hist = tk.history(period="2y")
            except Exception:
                hist = pd.DataFrame()

            # EPS estimates for sentiment
            try:
                earnings_est = tk.earnings_estimate
            except Exception:
                earnings_est = None

            # Analyst recommendations
            try:
                recs = tk.recommendations
            except Exception:
                recs = None

            return {
                "ticker": ticker,
                "info": info,
                "price": price,
                "eps": eps,
                "bvps": bvps,
                "balance_sheet": bs,
                "income_stmt": inc,
                "cashflow": cf,
                "history": hist,
                "earnings_est": earnings_est,
                "recommendations": recs,
            }

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5)
            else:
                log.debug(f"Failed to fetch {ticker}: {e}")
                return None

    return None


# ═══════════════════════════════════════════════════════════════
# GRAHAM VALUATION (30 points)
# From "The Intelligent Investor" by Benjamin Graham
# ═══════════════════════════════════════════════════════════════

def score_value(data: dict) -> tuple[float, list[str]]:
    """
    Graham-based value scoring (0-30 points).

    Components:
    - Graham Number: √(22.5 × EPS × BVPS) vs current price (0-12)
    - PE ≤ 15 test (0-6)
    - PE × PB ≤ 22.5 blended multiplier (0-6)
    - Earnings Yield margin of safety (0-6)
    """
    pts = 0.0
    reasons = []
    info = data["info"]
    price = data["price"]
    eps = data.get("eps") or info.get("trailingEps", 0)
    bvps = data.get("bvps") or info.get("bookValue", 0)

    if not eps or eps <= 0 or not bvps or bvps <= 0:
        return 0, ["No positive EPS or Book Value"]

    pe = info.get("trailingPE", 0) or (price / eps if eps > 0 else 0)
    pb = info.get("priceToBook", 0) or (price / bvps if bvps > 0 else 0)

    # 1. Graham Number (0-12 points)
    graham_number = math.sqrt(22.5 * eps * bvps)
    discount = (graham_number - price) / graham_number * 100 if graham_number > 0 else -100

    if discount >= 30:
        pts += 12
        reasons.append(f"Deep Value: {discount:.0f}% below Graham# ${graham_number:.1f}")
    elif discount >= 15:
        pts += 9
        reasons.append(f"Value: {discount:.0f}% below Graham# ${graham_number:.1f}")
    elif discount >= 0:
        pts += 6
        reasons.append(f"Fair Value: at Graham# ${graham_number:.1f}")
    elif discount >= -20:
        pts += 3
        reasons.append(f"Slightly above Graham# ${graham_number:.1f}")
    else:
        reasons.append(f"Overvalued vs Graham# ${graham_number:.1f}")

    # 2. PE ≤ 15 test (0-6)
    if 0 < pe <= 12:
        pts += 6
        reasons.append(f"PE {pe:.1f} — deep value")
    elif pe <= 15:
        pts += 5
        reasons.append(f"PE {pe:.1f} ≤ 15 ✓")
    elif pe <= 20:
        pts += 3
        reasons.append(f"PE {pe:.1f} — moderate")
    elif pe <= 30:
        pts += 1
    # else 0

    # 3. PE × PB ≤ 22.5 (0-6)
    pepb = pe * pb if pe > 0 and pb > 0 else 999
    if 0 < pepb <= 22.5:
        pts += 6
        reasons.append(f"PE×PB {pepb:.1f} ≤ 22.5 ✓")
    elif pepb <= 35:
        pts += 3
        reasons.append(f"PE×PB {pepb:.1f} — moderate")
    elif pepb <= 50:
        pts += 1

    # 4. Earnings Yield margin of safety (0-6)
    ey = (1 / pe * 100) if pe > 0 else 0
    if ey >= 10:
        pts += 6
        reasons.append(f"Earnings Yield {ey:.1f}% — strong margin")
    elif ey >= 7:
        pts += 4
        reasons.append(f"Earnings Yield {ey:.1f}% — good margin")
    elif ey >= 5:
        pts += 2
        reasons.append(f"Earnings Yield {ey:.1f}% — adequate")

    return min(30, pts), reasons


# ═══════════════════════════════════════════════════════════════
# MOMENTUM (30 points)
# ═══════════════════════════════════════════════════════════════

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Compute RSI from price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50


def score_momentum(data: dict) -> tuple[float, list[str]]:
    """
    Momentum scoring (0-30 points).

    Components:
    - Above 200-day SMA (0-8)
    - RSI between 50-70 sweet spot (0-8)
    - 3-month return positive (0-5)
    - 6-month return positive (0-5)
    - 12-month return positive (0-4)
    """
    pts = 0.0
    reasons = []
    hist = data.get("history")

    if hist is None or hist.empty or len(hist) < 50:
        return 0, ["Insufficient price history"]

    closes = hist["Close"]
    current_price = closes.iloc[-1]

    # 1. Above 200-SMA (0-8)
    if len(closes) >= 200:
        sma200 = closes.rolling(200).mean().iloc[-1]
        pct_above = (current_price - sma200) / sma200 * 100
        if pct_above > 10:
            pts += 8
            reasons.append(f"{pct_above:.0f}% above 200-SMA")
        elif pct_above > 0:
            pts += 5
            reasons.append(f"Above 200-SMA (+{pct_above:.0f}%)")
        else:
            reasons.append(f"Below 200-SMA ({pct_above:.0f}%)")
    else:
        # Use 50-SMA as fallback
        sma50 = closes.rolling(50).mean().iloc[-1]
        if current_price > sma50:
            pts += 4
            reasons.append("Above 50-SMA")

    # 2. RSI sweet spot 50-70 (0-8)
    rsi = compute_rsi(closes)
    if 50 <= rsi <= 70:
        pts += 8
        reasons.append(f"RSI {rsi:.0f} — trending, not overbought")
    elif 40 <= rsi < 50:
        pts += 4
        reasons.append(f"RSI {rsi:.0f} — neutral")
    elif 70 < rsi <= 80:
        pts += 3
        reasons.append(f"RSI {rsi:.0f} — strong but overbought risk")
    elif rsi > 80:
        reasons.append(f"RSI {rsi:.0f} — overbought ✗")
    else:
        reasons.append(f"RSI {rsi:.0f} — weak")

    # 3. 3-month return (0-5)
    if len(closes) >= 63:
        ret_3m = (current_price / closes.iloc[-63] - 1) * 100
        if ret_3m > 10:
            pts += 5
        elif ret_3m > 0:
            pts += 3
        if ret_3m > 0:
            reasons.append(f"3M: +{ret_3m:.1f}%")

    # 4. 6-month return (0-5)
    if len(closes) >= 126:
        ret_6m = (current_price / closes.iloc[-126] - 1) * 100
        if ret_6m > 15:
            pts += 5
        elif ret_6m > 0:
            pts += 3
        if ret_6m > 0:
            reasons.append(f"6M: +{ret_6m:.1f}%")

    # 5. 12-month return (0-4)
    if len(closes) >= 252:
        ret_12m = (current_price / closes.iloc[-252] - 1) * 100
        if ret_12m > 20:
            pts += 4
        elif ret_12m > 0:
            pts += 2
        if ret_12m > 0:
            reasons.append(f"12M: +{ret_12m:.1f}%")

    return min(30, pts), reasons


# ═══════════════════════════════════════════════════════════════
# QUALITY & PROFITABILITY (25 points)
# Piotroski F-Score + ROE + Margins
# ═══════════════════════════════════════════════════════════════

def compute_piotroski(data: dict) -> int:
    """
    Piotroski F-Score (0-9).
    9 binary signals covering profitability, leverage, and operating efficiency.
    """
    fscore = 0
    info = data["info"]
    bs = data.get("balance_sheet", pd.DataFrame())
    inc = data.get("income_stmt", pd.DataFrame())
    cf = data.get("cashflow", pd.DataFrame())

    # Need at least 2 years of data
    if bs.empty or inc.empty or cf.empty:
        # Fallback: estimate from info
        roe = info.get("returnOnEquity", 0) or 0
        roa = info.get("returnOnAssets", 0) or 0
        if roa > 0:
            fscore += 1  # positive ROA
        if roe > 0:
            fscore += 1  # proxy for positive operating CF
        margin = info.get("profitMargins", 0) or 0
        if margin > 0:
            fscore += 1
        return min(9, fscore * 2)  # rough estimate

    try:
        # Get most recent and prior year columns
        cols = sorted(bs.columns, reverse=True)
        if len(cols) < 2:
            return fscore

        curr_year = cols[0]
        prev_year = cols[1]

        # Helper to safely get values
        def get_val(df, row, col, default=0):
            try:
                v = df.loc[row, col]
                return float(v) if pd.notna(v) else default
            except (KeyError, TypeError):
                return default

        # --- PROFITABILITY ---
        # 1. Positive Net Income
        ni_curr = get_val(inc, "Net Income", curr_year)
        if ni_curr > 0:
            fscore += 1

        # 2. Positive ROA (Net Income / Total Assets)
        ta_curr = get_val(bs, "Total Assets", curr_year, 1)
        roa_curr = ni_curr / ta_curr if ta_curr > 0 else 0
        if roa_curr > 0:
            fscore += 1

        # 3. Positive Operating Cash Flow
        ocf_curr = get_val(cf, "Operating Cash Flow", curr_year)
        if ocf_curr > 0:
            fscore += 1

        # 4. Cash flow > Net Income (accrual quality)
        if ocf_curr > ni_curr:
            fscore += 1

        # --- LEVERAGE ---
        # 5. Decrease in long-term debt ratio
        ltd_curr = get_val(bs, "Long Term Debt", curr_year)
        ltd_prev = get_val(bs, "Long Term Debt", prev_year)
        ta_prev = get_val(bs, "Total Assets", prev_year, 1)
        if ta_curr > 0 and ta_prev > 0:
            if (ltd_curr / ta_curr) < (ltd_prev / ta_prev):
                fscore += 1

        # 6. Increase in current ratio
        ca_curr = get_val(bs, "Current Assets", curr_year)
        cl_curr = get_val(bs, "Current Liabilities", curr_year, 1)
        ca_prev = get_val(bs, "Current Assets", prev_year)
        cl_prev = get_val(bs, "Current Liabilities", prev_year, 1)
        cr_curr = ca_curr / cl_curr if cl_curr > 0 else 0
        cr_prev = ca_prev / cl_prev if cl_prev > 0 else 0
        if cr_curr > cr_prev:
            fscore += 1

        # 7. No new shares issued
        shares_curr = get_val(bs, "Ordinary Shares Number", curr_year)
        shares_prev = get_val(bs, "Ordinary Shares Number", prev_year)
        if shares_curr > 0 and shares_prev > 0 and shares_curr <= shares_prev:
            fscore += 1

        # --- OPERATING EFFICIENCY ---
        # 8. Increasing gross margin
        rev_curr = get_val(inc, "Total Revenue", curr_year, 1)
        cogs_curr = get_val(inc, "Cost Of Revenue", curr_year)
        rev_prev = get_val(inc, "Total Revenue", prev_year, 1)
        cogs_prev = get_val(inc, "Cost Of Revenue", prev_year)
        gm_curr = (rev_curr - cogs_curr) / rev_curr if rev_curr > 0 else 0
        gm_prev = (rev_prev - cogs_prev) / rev_prev if rev_prev > 0 else 0
        if gm_curr > gm_prev:
            fscore += 1

        # 9. Increasing asset turnover
        at_curr = rev_curr / ta_curr if ta_curr > 0 else 0
        at_prev = rev_prev / ta_prev if ta_prev > 0 else 0
        if at_curr > at_prev:
            fscore += 1

    except Exception as e:
        log.debug(f"Piotroski error: {e}")

    return fscore


def score_quality(data: dict) -> tuple[float, list[str]]:
    """
    Quality & Profitability scoring (0-25 points).

    Components:
    - Piotroski F-Score (0-12): ~1.33 pts per F-Score point
    - ROE > 15% (0-7)
    - Profit margins (0-6)
    """
    pts = 0.0
    reasons = []
    info = data["info"]

    # 1. Piotroski F-Score (0-12)
    fscore = compute_piotroski(data)
    f_pts = min(12, fscore * 1.33)
    pts += f_pts
    if fscore >= 7:
        reasons.append(f"Piotroski {fscore}/9 — strong")
    elif fscore >= 5:
        reasons.append(f"Piotroski {fscore}/9 — decent")
    else:
        reasons.append(f"Piotroski {fscore}/9 — weak")

    # 2. ROE (0-7)
    roe = (info.get("returnOnEquity") or 0) * 100
    if roe >= 25:
        pts += 7
        reasons.append(f"ROE {roe:.0f}% — excellent")
    elif roe >= 15:
        pts += 5
        reasons.append(f"ROE {roe:.0f}% > 15% ✓")
    elif roe >= 10:
        pts += 3
        reasons.append(f"ROE {roe:.0f}%")
    elif roe > 0:
        pts += 1

    # 3. Profit Margin (0-6)
    margin = (info.get("profitMargins") or 0) * 100
    if margin >= 20:
        pts += 6
        reasons.append(f"Margin {margin:.0f}% — high quality")
    elif margin >= 10:
        pts += 4
        reasons.append(f"Margin {margin:.0f}%")
    elif margin >= 5:
        pts += 2

    return min(25, pts), reasons


# ═══════════════════════════════════════════════════════════════
# ANALYST SENTIMENT (15 points)
# ═══════════════════════════════════════════════════════════════

def score_sentiment(data: dict) -> tuple[float, list[str]]:
    """
    Analyst Sentiment scoring (0-15 points).

    Components:
    - EPS estimate revisions — current vs 90-day-ago (0-8)
    - Analyst recommendation distribution (0-7)
    """
    pts = 0.0
    reasons = []
    info = data["info"]

    # 1. EPS Revisions (0-8)
    # Use yfinance's earnings estimate data
    est = data.get("earnings_est")
    revision_positive = False

    if est is not None and not est.empty:
        try:
            # Check for upward revision in current year estimate
            if "avg" in est.columns:
                # Compare growth vs current
                avg_est = est["avg"].iloc[0] if len(est) > 0 else None
                if avg_est and avg_est > 0:
                    pts += 4
                    revision_positive = True
                    reasons.append("Positive EPS estimates")
        except Exception:
            pass

    # Fallback: use forward PE vs trailing PE as proxy for estimate direction
    if not revision_positive:
        fwd_pe = info.get("forwardPE", 0)
        trail_pe = info.get("trailingPE", 0)
        if fwd_pe and trail_pe and 0 < fwd_pe < trail_pe:
            pts += 5
            reasons.append("FwdPE < TrailPE — earnings growth expected")
        elif fwd_pe and trail_pe and fwd_pe > 0:
            pts += 2
            reasons.append("Stable EPS outlook")

    # Also check earnings growth estimate
    eg = info.get("earningsGrowth")
    if eg and eg > 0.10:
        pts += 3
        reasons.append(f"Earnings growth {eg*100:.0f}%")
    elif eg and eg > 0:
        pts += 1

    # 2. Analyst Recommendations (0-7)
    rec_mean = info.get("recommendationMean")  # 1=Strong Buy, 5=Sell
    if rec_mean:
        if rec_mean <= 1.5:
            pts += 7
            reasons.append(f"Analysts: Strong Buy ({rec_mean:.1f})")
        elif rec_mean <= 2.0:
            pts += 5
            reasons.append(f"Analysts: Buy ({rec_mean:.1f})")
        elif rec_mean <= 2.5:
            pts += 3
            reasons.append(f"Analysts: Overweight ({rec_mean:.1f})")
        elif rec_mean <= 3.0:
            pts += 1
            reasons.append(f"Analysts: Hold ({rec_mean:.1f})")

    return min(15, pts), reasons


# ═══════════════════════════════════════════════════════════════
# COMPOSITE SCORE
# ═══════════════════════════════════════════════════════════════

def get_alpha_score(data: dict) -> dict:
    """
    Master scoring function. Returns a dict with:
    - total_score (0-100)
    - signal (Strong Buy / Buy / Hold / Sell / Strong Sell)
    - component scores
    - reasons
    """
    if data is None:
        return None

    val_score, val_reasons = score_value(data)
    mom_score, mom_reasons = score_momentum(data)
    qual_score, qual_reasons = score_quality(data)
    sent_score, sent_reasons = score_sentiment(data)

    total = val_score + mom_score + qual_score + sent_score
    total = min(100, max(0, round(total)))

    # Signal classification
    if total >= 80:
        signal = "Strong Buy"
    elif total >= 65:
        signal = "Buy"
    elif total >= 45:
        signal = "Hold"
    elif total >= 30:
        signal = "Sell"
    else:
        signal = "Strong Sell"

    # Build "why" summary
    why_parts = []
    if val_score >= 20:
        why_parts.append("Deep Value")
    elif val_score >= 12:
        why_parts.append("Value")
    if mom_score >= 20:
        why_parts.append("High Momentum")
    elif mom_score >= 12:
        why_parts.append("Momentum")
    if qual_score >= 18:
        why_parts.append("High Quality")
    elif qual_score >= 12:
        why_parts.append("Quality")
    if sent_score >= 10:
        why_parts.append("Rising Estimates")
    elif sent_score >= 6:
        why_parts.append("Positive Sentiment")

    why = " + ".join(why_parts) if why_parts else "Mixed signals"

    info = data["info"]
    eps = data.get("eps") or info.get("trailingEps", 0) or 0
    bvps = data.get("bvps") or info.get("bookValue", 0) or 0
    graham_num = math.sqrt(22.5 * eps * bvps) if eps > 0 and bvps > 0 else 0

    return {
        "ticker": data["ticker"],
        "name": info.get("shortName", data["ticker"]),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "price": data["price"],
        "market_cap": info.get("marketCap", 0),
        "pe": info.get("trailingPE", 0),
        "pb": info.get("priceToBook", 0),
        "eps": eps,
        "bvps": bvps,
        "graham_number": round(graham_num, 2),
        "roe": round((info.get("returnOnEquity") or 0) * 100, 1),
        "profit_margin": round((info.get("profitMargins") or 0) * 100, 1),
        "dividend_yield": round((info.get("dividendYield") or 0) * 100, 2),
        "total_score": total,
        "signal": signal,
        "value_score": round(val_score, 1),
        "momentum_score": round(mom_score, 1),
        "quality_score": round(qual_score, 1),
        "sentiment_score": round(sent_score, 1),
        "why": why,
        "reasons": val_reasons + mom_reasons + qual_reasons + sent_reasons,
        "scan_date": datetime.datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════
# SCANNER
# ═══════════════════════════════════════════════════════════════

def scan_ticker(ticker: str) -> dict | None:
    """Fetch data and score a single ticker."""
    time.sleep(0.2)  # Rate limiting
    data = fetch_stock_data(ticker)
    if data is None:
        return None
    try:
        result = get_alpha_score(data)
        return result
    except Exception as e:
        log.debug(f"Scoring error for {ticker}: {e}")
        return None


def scan_universe(tickers: list[str], max_workers: int = 8, progress_callback=None) -> list[dict]:
    """Scan a list of tickers using thread pool. Returns sorted results."""
    results = []
    total = len(tickers)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(scan_ticker, t): t for t in tickers}

        for future in as_completed(future_map):
            completed += 1
            ticker = future_map[future]
            try:
                result = future.result()
                if result and result.get("total_score", 0) > 0:
                    results.append(result)
            except Exception:
                pass

            if progress_callback:
                progress_callback(completed, total, ticker)

    results.sort(key=lambda x: x["total_score"], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════
# DATABASE (SQLite)
# ═══════════════════════════════════════════════════════════════

def init_db():
    """Create the database tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT,
            ticker TEXT,
            name TEXT,
            sector TEXT,
            price REAL,
            total_score INTEGER,
            signal TEXT,
            value_score REAL,
            momentum_score REAL,
            quality_score REAL,
            sentiment_score REAL,
            graham_number REAL,
            pe REAL,
            pb REAL,
            roe REAL,
            why TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            buy_date TEXT,
            buy_price REAL,
            shares REAL,
            score_at_buy INTEGER,
            signal_at_buy TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_results(results: list[dict]):
    """Save scan results to the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.datetime.now().isoformat()

    for r in results:
        c.execute("""
            INSERT INTO scan_results
            (scan_date, ticker, name, sector, price, total_score, signal,
             value_score, momentum_score, quality_score, sentiment_score,
             graham_number, pe, pb, roe, why)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now, r["ticker"], r["name"], r["sector"], r["price"],
            r["total_score"], r["signal"], r["value_score"],
            r["momentum_score"], r["quality_score"], r["sentiment_score"],
            r["graham_number"], r.get("pe", 0), r.get("pb", 0),
            r.get("roe", 0), r["why"]
        ))

    conn.commit()
    conn.close()


def get_scan_history() -> pd.DataFrame:
    """Retrieve past scan results."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM scan_results ORDER BY scan_date DESC, total_score DESC", conn)
    conn.close()
    return df


def add_to_portfolio(ticker: str, price: float, shares: float, score: int, signal: str):
    """Add a stock to the tracked portfolio."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO portfolio (ticker, buy_date, buy_price, shares, score_at_buy, signal_at_buy)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ticker, datetime.datetime.now().isoformat(), price, shares, score, signal))
    conn.commit()
    conn.close()


def get_portfolio() -> pd.DataFrame:
    """Get the current tracked portfolio."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM portfolio ORDER BY buy_date DESC", conn)
    conn.close()
    return df


def remove_from_portfolio(row_id: int):
    """Remove a position from the portfolio."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()
