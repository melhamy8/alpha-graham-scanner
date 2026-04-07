"""
Alpha-Graham Market Scanner — Core Scoring Engine
"""

import time
import math
import sqlite3
import logging
import datetime
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("alpha_graham")

DB_PATH = Path(__file__).parent / "alpha_graham.db"


def get_sp1500_tickers() -> List[str]:
    """Get S&P 500 + S&P 400 MidCap + S&P 600 SmallCap tickers."""
    tickers = []
    for url in [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    ]:
        try:
            df = pd.read_html(url)[0]
            col = "Symbol" if "Symbol" in df.columns else "Ticker symbol" if "Ticker symbol" in df.columns else df.columns[0]
            tickers.extend(df[col].tolist())
        except Exception as e:
            log.warning(f"Could not fetch from {url}: {e}")
    
    tickers = list(set([str(t).replace(".", "-").strip() for t in tickers if isinstance(t, str)]))
    return sorted(tickers)


def fetch_stock_data(ticker: str, retries: int = 2) -> Optional[Dict]:
    """Fetch all required data for a single ticker."""
    for attempt in range(retries):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}

            eps = info.get("trailingEps")
            bvps = info.get("bookValue")
            price = info.get("currentPrice") or info.get("regularMarketPrice")

            if not price or price <= 0:
                return None

            try:
                bs = tk.balance_sheet
                inc = tk.income_stmt
                cf = tk.cashflow
            except Exception:
                bs, inc, cf = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            try:
                hist = tk.history(period="2y")
            except Exception:
                hist = pd.DataFrame()

            try:
                earnings_est = tk.earnings_estimate
            except Exception:
                earnings_est = None

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
# ═══════════════════════════════════════════════════════════════

def score_value(data: Dict) -> Tuple[float, List[str]]:
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

    # Graham Number
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
        reasons.append(f"Above Graham# ${graham_number:.1f}")
    else:
        reasons.append(f"Overvalued vs Graham# ${graham_number:.1f}")

    # PE <= 15
    if 0 < pe <= 12:
        pts += 6
        reasons.append(f"PE {pe:.1f} — deep value")
    elif pe <= 15:
        pts += 5
        reasons.append(f"PE {pe:.1f} <= 15")
    elif pe <= 20:
        pts += 3
    elif pe <= 30:
        pts += 1

    # PE x PB <= 22.5
    pepb = pe * pb if pe > 0 and pb > 0 else 999
    if 0 < pepb <= 22.5:
        pts += 6
        reasons.append(f"PE*PB {pepb:.1f} <= 22.5")
    elif pepb <= 35:
        pts += 3
    elif pepb <= 50:
        pts += 1

    # Earnings Yield
    ey = (1 / pe * 100) if pe > 0 else 0
    if ey >= 10:
        pts += 6
        reasons.append(f"Earnings Yield {ey:.1f}%")
    elif ey >= 7:
        pts += 4
    elif ey >= 5:
        pts += 2

    return min(30, pts), reasons


# ═══════════════════════════════════════════════════════════════
# MOMENTUM (30 points)
# ═══════════════════════════════════════════════════════════════

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1] if len(rsi) > 0 else 50
    return float(val) if not pd.isna(val) else 50


def score_momentum(data: Dict) -> Tuple[float, List[str]]:
    pts = 0.0
    reasons = []
    hist = data.get("history")

    if hist is None or hist.empty or len(hist) < 50:
        return 0, ["Insufficient price history"]

    closes = hist["Close"]
    current_price = float(closes.iloc[-1])

    # Above 200-SMA
    if len(closes) >= 200:
        sma200 = float(closes.rolling(200).mean().iloc[-1])
        pct_above = (current_price - sma200) / sma200 * 100
        if pct_above > 10:
            pts += 8
            reasons.append(f"{pct_above:.0f}% above 200-SMA")
        elif pct_above > 0:
            pts += 5
            reasons.append(f"Above 200-SMA")
    elif len(closes) >= 50:
        sma50 = float(closes.rolling(50).mean().iloc[-1])
        if current_price > sma50:
            pts += 4
            reasons.append("Above 50-SMA")

    # RSI
    rsi = compute_rsi(closes)
    if 50 <= rsi <= 70:
        pts += 8
        reasons.append(f"RSI {rsi:.0f} — sweet spot")
    elif 40 <= rsi < 50:
        pts += 4
    elif 70 < rsi <= 80:
        pts += 3
        reasons.append(f"RSI {rsi:.0f} — overbought risk")

    # Returns
    if len(closes) >= 63:
        ret_3m = (current_price / float(closes.iloc[-63]) - 1) * 100
        if ret_3m > 10:
            pts += 5
        elif ret_3m > 0:
            pts += 3
        if ret_3m > 0:
            reasons.append(f"3M: +{ret_3m:.1f}%")

    if len(closes) >= 126:
        ret_6m = (current_price / float(closes.iloc[-126]) - 1) * 100
        if ret_6m > 15:
            pts += 5
        elif ret_6m > 0:
            pts += 3
        if ret_6m > 0:
            reasons.append(f"6M: +{ret_6m:.1f}%")

    if len(closes) >= 252:
        ret_12m = (current_price / float(closes.iloc[-252]) - 1) * 100
        if ret_12m > 20:
            pts += 4
        elif ret_12m > 0:
            pts += 2
        if ret_12m > 0:
            reasons.append(f"12M: +{ret_12m:.1f}%")

    return min(30, pts), reasons


# ═══════════════════════════════════════════════════════════════
# QUALITY (25 points)
# ═══════════════════════════════════════════════════════════════

def compute_piotroski(data: Dict) -> int:
    fscore = 0
    info = data["info"]
    bs = data.get("balance_sheet", pd.DataFrame())
    inc = data.get("income_stmt", pd.DataFrame())
    cf = data.get("cashflow", pd.DataFrame())

    if bs.empty or inc.empty or cf.empty:
        roe = info.get("returnOnEquity", 0) or 0
        roa = info.get("returnOnAssets", 0) or 0
        margin = info.get("profitMargins", 0) or 0
        if roa > 0: fscore += 1
        if roe > 0: fscore += 1
        if margin > 0: fscore += 1
        return min(9, fscore * 2)

    try:
        cols = sorted(bs.columns, reverse=True)
        if len(cols) < 2:
            return fscore

        curr, prev = cols[0], cols[1]

        def gv(df, row, col, default=0):
            try:
                v = df.loc[row, col]
                return float(v) if pd.notna(v) else default
            except (KeyError, TypeError):
                return default

        ni = gv(inc, "Net Income", curr)
        if ni > 0: fscore += 1

        ta = gv(bs, "Total Assets", curr, 1)
        if ni / ta > 0: fscore += 1

        ocf = gv(cf, "Operating Cash Flow", curr)
        if ocf > 0: fscore += 1
        if ocf > ni: fscore += 1

        ltd_c = gv(bs, "Long Term Debt", curr)
        ltd_p = gv(bs, "Long Term Debt", prev)
        ta_p = gv(bs, "Total Assets", prev, 1)
        if ta > 0 and ta_p > 0 and (ltd_c / ta) < (ltd_p / ta_p): fscore += 1

        ca_c = gv(bs, "Current Assets", curr)
        cl_c = gv(bs, "Current Liabilities", curr, 1)
        ca_p = gv(bs, "Current Assets", prev)
        cl_p = gv(bs, "Current Liabilities", prev, 1)
        if cl_c > 0 and cl_p > 0 and (ca_c / cl_c) > (ca_p / cl_p): fscore += 1

        sh_c = gv(bs, "Ordinary Shares Number", curr)
        sh_p = gv(bs, "Ordinary Shares Number", prev)
        if sh_c > 0 and sh_p > 0 and sh_c <= sh_p: fscore += 1

        rev_c = gv(inc, "Total Revenue", curr, 1)
        cogs_c = gv(inc, "Cost Of Revenue", curr)
        rev_p = gv(inc, "Total Revenue", prev, 1)
        cogs_p = gv(inc, "Cost Of Revenue", prev)
        if rev_c > 0 and rev_p > 0:
            if (rev_c - cogs_c) / rev_c > (rev_p - cogs_p) / rev_p: fscore += 1

        if ta > 0 and ta_p > 0 and (rev_c / ta) > (rev_p / ta_p): fscore += 1

    except Exception:
        pass

    return fscore


def score_quality(data: Dict) -> Tuple[float, List[str]]:
    pts = 0.0
    reasons = []
    info = data["info"]

    fscore = compute_piotroski(data)
    f_pts = min(12, fscore * 1.33)
    pts += f_pts
    if fscore >= 7:
        reasons.append(f"Piotroski {fscore}/9 — strong")
    elif fscore >= 5:
        reasons.append(f"Piotroski {fscore}/9 — decent")
    else:
        reasons.append(f"Piotroski {fscore}/9")

    roe = (info.get("returnOnEquity") or 0) * 100
    if roe >= 25:
        pts += 7
        reasons.append(f"ROE {roe:.0f}%")
    elif roe >= 15:
        pts += 5
        reasons.append(f"ROE {roe:.0f}%")
    elif roe >= 10:
        pts += 3
    elif roe > 0:
        pts += 1

    margin = (info.get("profitMargins") or 0) * 100
    if margin >= 20:
        pts += 6
        reasons.append(f"Margin {margin:.0f}%")
    elif margin >= 10:
        pts += 4
    elif margin >= 5:
        pts += 2

    return min(25, pts), reasons


# ═══════════════════════════════════════════════════════════════
# SENTIMENT (15 points)
# ═══════════════════════════════════════════════════════════════

def score_sentiment(data: Dict) -> Tuple[float, List[str]]:
    pts = 0.0
    reasons = []
    info = data["info"]

    fwd_pe = info.get("forwardPE", 0)
    trail_pe = info.get("trailingPE", 0)
    if fwd_pe and trail_pe and 0 < fwd_pe < trail_pe * 0.85:
        pts += 6
        reasons.append("Rising estimates")
    elif fwd_pe and trail_pe and 0 < fwd_pe < trail_pe:
        pts += 4
        reasons.append("Positive outlook")
    elif fwd_pe and fwd_pe > 0:
        pts += 1

    eg = info.get("earningsGrowth")
    if eg and eg > 0.10:
        pts += 3
        reasons.append(f"EPS growth {eg*100:.0f}%")
    elif eg and eg > 0:
        pts += 1

    rec_mean = info.get("recommendationMean")
    if rec_mean:
        if rec_mean <= 1.5:
            pts += 6
            reasons.append(f"Strong Buy ({rec_mean:.1f})")
        elif rec_mean <= 2.0:
            pts += 4
            reasons.append(f"Buy ({rec_mean:.1f})")
        elif rec_mean <= 2.5:
            pts += 2
        elif rec_mean <= 3.0:
            pts += 1

    return min(15, pts), reasons


# ═══════════════════════════════════════════════════════════════
# COMPOSITE
# ═══════════════════════════════════════════════════════════════

def get_alpha_score(data: Dict) -> Optional[Dict]:
    if data is None:
        return None

    val_score, val_reasons = score_value(data)
    mom_score, mom_reasons = score_momentum(data)
    qual_score, qual_reasons = score_quality(data)
    sent_score, sent_reasons = score_sentiment(data)

    total = val_score + mom_score + qual_score + sent_score
    total = min(100, max(0, round(total)))

    if total >= 80: signal = "Strong Buy"
    elif total >= 65: signal = "Buy"
    elif total >= 45: signal = "Hold"
    elif total >= 30: signal = "Sell"
    else: signal = "Strong Sell"

    why_parts = []
    if val_score >= 20: why_parts.append("Deep Value")
    elif val_score >= 12: why_parts.append("Value")
    if mom_score >= 20: why_parts.append("High Momentum")
    elif mom_score >= 12: why_parts.append("Momentum")
    if qual_score >= 18: why_parts.append("High Quality")
    elif qual_score >= 12: why_parts.append("Quality")
    if sent_score >= 10: why_parts.append("Rising Estimates")
    elif sent_score >= 6: why_parts.append("Positive Sentiment")

    info = data["info"]
    eps = data.get("eps") or info.get("trailingEps", 0) or 0
    bvps = data.get("bvps") or info.get("bookValue", 0) or 0
    gn = math.sqrt(22.5 * eps * bvps) if eps > 0 and bvps > 0 else 0

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
        "graham_number": round(gn, 2),
        "roe": round((info.get("returnOnEquity") or 0) * 100, 1),
        "profit_margin": round((info.get("profitMargins") or 0) * 100, 1),
        "dividend_yield": round((info.get("dividendYield") or 0) * 100, 2),
        "total_score": total,
        "signal": signal,
        "value_score": round(val_score, 1),
        "momentum_score": round(mom_score, 1),
        "quality_score": round(qual_score, 1),
        "sentiment_score": round(sent_score, 1),
        "why": " + ".join(why_parts) if why_parts else "Mixed signals",
        "reasons": val_reasons + mom_reasons + qual_reasons + sent_reasons,
        "scan_date": datetime.datetime.now().isoformat(),
    }


def scan_ticker(ticker: str) -> Optional[Dict]:
    time.sleep(0.2)
    data = fetch_stock_data(ticker)
    if data is None:
        return None
    try:
        return get_alpha_score(data)
    except Exception:
        return None


def scan_universe(tickers: List[str], max_workers: int = 8, progress_callback=None) -> List[Dict]:
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
# DATABASE
# ═══════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT, ticker TEXT, name TEXT, sector TEXT,
            price REAL, total_score INTEGER, signal TEXT,
            value_score REAL, momentum_score REAL, quality_score REAL, sentiment_score REAL,
            graham_number REAL, pe REAL, pb REAL, roe REAL, why TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, buy_date TEXT, buy_price REAL, shares REAL,
            score_at_buy INTEGER, signal_at_buy TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_results(results: List[Dict]):
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    now = datetime.datetime.now().isoformat()
    for r in results:
        c.execute("""INSERT INTO scan_results (scan_date,ticker,name,sector,price,total_score,signal,
            value_score,momentum_score,quality_score,sentiment_score,graham_number,pe,pb,roe,why)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now, r["ticker"], r["name"], r["sector"], r["price"],
             r["total_score"], r["signal"], r["value_score"],
             r["momentum_score"], r["quality_score"], r["sentiment_score"],
             r["graham_number"], r.get("pe", 0), r.get("pb", 0),
             r.get("roe", 0), r["why"]))
    conn.commit()
    conn.close()


def get_scan_history() -> pd.DataFrame:
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM scan_results ORDER BY scan_date DESC, total_score DESC", conn)
    conn.close()
    return df


def add_to_portfolio(ticker, price, shares, score, signal):
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("INSERT INTO portfolio (ticker,buy_date,buy_price,shares,score_at_buy,signal_at_buy) VALUES (?,?,?,?,?,?)",
              (ticker, datetime.datetime.now().isoformat(), price, shares, score, signal))
    conn.commit()
    conn.close()


def get_portfolio() -> pd.DataFrame:
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM portfolio ORDER BY buy_date DESC", conn)
    conn.close()
    return df


def remove_from_portfolio(row_id):
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.cursor().execute("DELETE FROM portfolio WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()
