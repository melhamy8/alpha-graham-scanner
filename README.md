# Alpha-Graham Market Scanner

**Live stock scanner covering 1,500+ NYSE & NASDAQ stocks.**
Graham Intelligent Investor valuation × Modern quant factors.

---

## 🚀 Deploy in 3 Steps (No Code Required)

### Step 1: Create a GitHub Account (skip if you have one)
1. Go to **github.com** → Click **Sign Up**
2. Pick a username, enter your email, set a password
3. Verify your email

### Step 2: Upload This Project to GitHub
1. Log into **github.com**
2. Click the **+** button (top right) → **New repository**
3. Name it: `alpha-graham-scanner`
4. Keep it **Public**
5. Click **Create repository**
6. On the next page, click **"uploading an existing file"** link
7. **Drag and drop ALL the files** from this folder:
   - `app.py`
   - `scanner.py`
   - `requirements.txt`
   - `packages.txt`
   - `.streamlit/config.toml` (you may need to create this folder on GitHub — see note below)
8. Click **Commit changes**

**For the `.streamlit/config.toml` file:**
- On your repo page, click **Add file** → **Create new file**
- In the name field type: `.streamlit/config.toml`
- Paste the contents from the config.toml file
- Click **Commit changes**

### Step 3: Deploy on Streamlit Cloud (Free)
1. Go to **share.streamlit.io**
2. Click **Sign in with GitHub**
3. Authorize Streamlit to access your GitHub
4. Click **New app**
5. Select your repository: `alpha-graham-scanner`
6. Main file path: `app.py`
7. Click **Deploy!**
8. Wait 2-3 minutes for it to build
9. **Done!** You get a live URL like `https://your-name-alpha-graham-scanner.streamlit.app`

### Access From Your Phone
- Open that URL in Safari or Chrome
- Tap **Share** → **Add to Home Screen**
- Now it works like a native app on your phone

---

## 📱 What The App Does

### 🔍 Market Scanner
- Scans the **entire S&P 1500** (500 large + 400 mid + 600 small cap stocks)
- Pulls **live market data** from Yahoo Finance
- Scores every stock on a 0-100 scale
- Choose Quick Scan (50), S&P 500, S&P 500 + MidCap, or Full 1500

### 📋 Leaderboard
- **Top 10 picks** ranked by composite score
- "Why" column explains each pick (e.g., "Deep Value + High Momentum + Rising Estimates")
- Detailed analysis cards for top 3
- Score distribution chart

### 🔎 Stock Lookup
- **Search ANY ticker** — not limited to the S&P 1500
- Full candlestick price chart with 50/200-day moving averages
- Complete score breakdown with all Graham metrics
- One-click add to portfolio tracker

### 💼 Portfolio Tracker
- Track performance of your saved picks
- **Live P&L** calculated from current market prices
- Shows profit/loss per position and total portfolio
- Performance bar chart

### 📜 Scan History
- Every scan saved automatically to a database
- Browse past scans by date
- Export any scan to CSV
- Track how scores change over time

---

## 📊 Scoring Model (0-100)

### Graham Value — 30%
From *The Intelligent Investor* by Benjamin Graham:
- **Graham Number**: √(22.5 × EPS × Book Value) — full points if price is 30%+ below
- **PE ≤ 15**: Graham's defensive investor rule
- **PE × PB ≤ 22.5**: The blended multiplier
- **Earnings Yield**: Must exceed bond rate (margin of safety)

### Momentum — 30%
- Above 200-day simple moving average
- RSI between 50-70 (trending, not overbought)
- Positive returns over 3, 6, and 12 months

### Quality — 25%
- **Piotroski F-Score** (0-9): Profitability, leverage, efficiency
- **ROE > 15%**: Return on equity threshold
- **Profit margins**: Higher = better quality business

### Analyst Sentiment — 15%
- **EPS revisions**: Forward PE lower than trailing (growth expected)
- **Analyst recommendations**: Strong Buy to Sell scale
- **Earnings growth rate**

### Signal Classification
| Score | Signal | Action |
|-------|--------|--------|
| ≥ 80 | 🟢 **Strong Buy** | High conviction — all factors aligned |
| 65-79 | 🟩 **Buy** | Most factors positive |
| 45-64 | 🟡 **Hold** | Mixed signals |
| 30-44 | 🟠 **Sell** | Weak on most factors |
| < 30 | 🔴 **Strong Sell** | Fails most screens |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is not financial advice. Always conduct your own due diligence before making investment decisions. Past performance does not guarantee future results.

---

## 🛠 Technical Details (for reference)

- **Data source**: Yahoo Finance (yfinance)
- **Universe**: S&P 500 + S&P 400 MidCap + S&P 600 SmallCap
- **Database**: SQLite (auto-created, stores scan history + portfolio)
- **Parallel scanning**: ThreadPoolExecutor with configurable threads
- **Rate limiting**: 0.2s between API calls to avoid throttling
- **Framework**: Streamlit + Plotly for interactive charts
