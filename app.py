import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Nifty 50 Screener", layout="wide")
st.title("🇮🇳 Nifty 50 Fundamental + Technical Screener")
st.caption("Combines SHORT-TERM + LONG-TERM + ULTRA LONG-TERM analysis")

# =========================================================================
# NIFTY 50 STOCKS
# =========================================================================
NIFTY50 = {
    "RELIANCE.NS": ("Reliance Industries", "Oil & Gas"),
    "TCS.NS": ("Tata Consultancy Services", "IT"),
    "HDFCBANK.NS": ("HDFC Bank", "Banking"),
    "INFY.NS": ("Infosys", "IT"),
    "ITC.NS": ("ITC Limited", "FMCG"),
    "LT.NS": ("Larsen & Toubro", "Engineering"),
    "SBIN.NS": ("State Bank of India", "Banking"),
    "MARUTI.NS": ("Maruti Suzuki", "Auto"),
    "BAJAJFINSV.NS": ("Bajaj Finserv", "Finance"),
    "WIPRO.NS": ("Wipro", "IT"),
    "ASIANPAINT.NS": ("Asian Paints", "Paints"),
    "AXISBANK.NS": ("Axis Bank", "Banking"),
    "BAJAJ-AUTO.NS": ("Bajaj Auto", "Auto"),
    "BHARTIARTL.NS": ("Bharti Airtel", "Telecom"),
    "HINDUNILVR.NS": ("Hindustan Unilever", "FMCG"),
    "ICICIBANK.NS": ("ICICI Bank", "Banking"),
    "JSWSTEEL.NS": ("JSW Steel", "Steel"),
    "KOTAKBANK.NS": ("Kotak Mahindra Bank", "Banking"),
    "M&M.NS": ("Mahindra & Mahindra", "Auto"),
    "NTPC.NS": ("NTPC Limited", "Power"),
    "ONGC.NS": ("Oil & Natural Gas", "Oil & Gas"),
    "POWERGRID.NS": ("Power Grid Corporation", "Power"),
    "SUNPHARMA.NS": ("Sun Pharmaceutical", "Pharma"),
    "TATAMOTORS.NS": ("Tata Motors", "Auto"),
    "TATASTEEL.NS": ("Tata Steel", "Steel"),
    "TECHM.NS": ("Tech Mahindra", "IT"),
    "TITAN.NS": ("Titan Company", "Retail"),
    "ULTRACEMCO.NS": ("UltraTech Cement", "Cement"),
    "UPL.NS": ("UPL Limited", "Chemicals"),
    "EICHERMOT.NS": ("Eicher Motors", "Auto"),
    "HEROMOTOCO.NS": ("Hero MotoCorp", "Auto"),
    "GRASIM.NS": ("Grasim Industries", "Materials"),
    "HINDALCO.NS": ("Hindalco Industries", "Metals"),
    "HCLTECH.NS": ("HCL Technologies", "IT"),
    "DRREDDY.NS": ("Dr. Reddy's Laboratories", "Pharma"),
    "CIPLA.NS": ("Cipla", "Pharma"),
    "COALINDIA.NS": ("Coal India", "Mining"),
    "BEL.NS": ("Bharat Electronics", "Defense"),
    "INDIGO.NS": ("IndiGo", "Aviation"),
}

# =========================================================================
# TECHNICAL INDICATOR FUNCTIONS
# =========================================================================

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return (macd - macd_signal).fillna(0)

def calculate_stochastic(high, low, close, window=14, smooth=3):
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    return k.rolling(window=smooth).mean().fillna(50)

def calculate_roc(data, window=10):
    return ((data - data.shift(window)) / data.shift(window) * 100).fillna(0)

def calculate_williams_r(high, low, close, window=14):
    highest = high.rolling(window=window).max()
    lowest = low.rolling(window=window).min()
    return -100 * (highest - close) / (highest - lowest)

def calculate_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad)

# =========================================================================
# MAIN ANALYSIS FUNCTION
# =========================================================================

def analyze_stock(ticker, name, sector):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="3y")
        info = getattr(stock, 'info', {})

        if df.empty or len(df) < 100:
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']

        # ===== SHORT-TERM SCORE =====
        st_rsi = calculate_rsi(close)
        st_macd = calculate_macd(close)
        st_stoch = calculate_stochastic(high, low, close)
        st_roc = calculate_roc(close)

        st_score = 50.0
        if st_rsi.iloc[-1] < 30: st_score += 15
        elif st_rsi.iloc[-1] > 70: st_score -= 15
        if st_macd.iloc[-1] > 0: st_score += 10
        else: st_score -= 5
        if st_stoch.iloc[-1] < 20: st_score += 10
        elif st_stoch.iloc[-1] > 80: st_score -= 10
        if st_roc.iloc[-1] > 0: st_score += 5

        st_score = max(0, min(100, st_score))
        st_rec = "BUY" if st_score >= 65 else ("SELL" if st_score <= 35 else "HOLD")

        # ===== LONG-TERM SCORE =====
        lt_sma200 = calculate_sma(close, 200)
        if len(close) >= 252:
            annual_return = ((close.iloc[-1] - close.iloc[-252]) / close.iloc[-252]) * 100
        else:
            annual_return = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100

        lt_score = 50.0
        if annual_return > 15: lt_score += 20
        elif annual_return > 5: lt_score += 10
        elif annual_return < -10: lt_score -= 20
        if close.iloc[-1] > lt_sma200.iloc[-1]: lt_score += 15

        lt_score = max(0, min(100, lt_score))
        lt_rec = "BUY" if lt_score >= 65 else ("SELL" if lt_score <= 35 else "HOLD")

        # ===== ULTRA LONG-TERM SCORE =====
        three_year_return = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100

        ult_score = 50.0
        if three_year_return > 30: ult_score += 25
        elif three_year_return > 10: ult_score += 15
        elif three_year_return < -20: ult_score -= 25

        ult_score = max(0, min(100, ult_score))
        ult_rec = "BUY" if ult_score >= 65 else ("SELL" if ult_score <= 35 else "HOLD")

        # ===== FINAL ACTION =====
        if st_rec == "BUY" and lt_rec == "BUY" and ult_rec == "BUY":
            action = "🚀 STRONG BUY"
        elif st_rec == "BUY" and lt_rec == "BUY":
            action = "📈 SOLID BUY"
        elif lt_rec == "BUY" and ult_rec == "BUY":
            action = "🔄 BUY on Dips"
        elif ult_rec == "BUY":
            action = "💎 ACCUMULATE"
        elif ult_rec == "SELL":
            action = "🔴 AVOID"
        elif st_rec == "SELL":
            action = "📉 SELL"
        else:
            action = "→ HOLD"

        # ===== ADDITIONAL METRICS =====
        high_52w = high.rolling(window=252, min_periods=1).max().iloc[-1]
        low_52w = low.rolling(window=252, min_periods=1).min().iloc[-1]
        distance_to_support = ((close.iloc[-1] - low_52w) / low_52w) * 100

        # P/E Ratio
        pe_ratio = info.get('trailingPE')
        try:
            pe_ratio = f"{float(pe_ratio):.2f}" if pe_ratio else "N/A"
        except:
            pe_ratio = "N/A"

        # Market Cap
        mcap = info.get('marketCap')
        try:
            mcap = f"₹{int(mcap / 10000000)} Cr" if mcap else "N/A"
        except:
            mcap = "N/A"

        # Volatility
        daily_returns = close.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0

        return {
            "Stock": f"{name} ({ticker})",
            "Sector": sector,
            "Price": f"₹{close.iloc[-1]:.2f}",
            "P/E Ratio": pe_ratio,
            "Market Cap": mcap,
            "ST Score": round(st_score, 1),
            "ST Rec": st_rec,
            "LT Score": round(lt_score, 1),
            "LT Rec": lt_rec,
            "ULT Score": round(ult_score, 1),
            "ULT Rec": ult_rec,
            "Action": action,
            "RSI(14)": f"{st_rsi.iloc[-1]:.1f}",
            "MACD": f"{st_macd.iloc[-1]:.2f}",
            "52W High": f"₹{high_52w:.2f}",
            "52W Low": f"₹{low_52w:.2f}",
            "Distance to Support": f"{distance_to_support:.1f}%",
            "Volatility (%)": f"{volatility:.1f}%",
        }

    except Exception:
        return None

# =========================================================================
# STREAMLIT UI
# =========================================================================

st.sidebar.header("⚙️ Scanning Framework")
num_stocks = st.sidebar.slider("Number of stocks to scan", 5, len(NIFTY50), 20)
scan_button = st.sidebar.button("🔍 RUN FULL ANALYSIS", use_container_width=True, type="primary")

if scan_button:
    st.info("⏳ Analyzing stocks... This may take 2-3 minutes")

    selected_stocks = dict(list(NIFTY50.items())[:num_stocks])
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (ticker, (name, sector)) in enumerate(selected_stocks.items()):
        status_text.text(f"Analyzing {idx+1}/{len(selected_stocks)}: {name}...")
        analysis = analyze_stock(ticker, name, sector)
        if analysis:
            results.append(analysis)
        progress_bar.progress((idx + 1) / len(selected_stocks))

    progress_bar.empty()
    status_text.empty()

    if results:
        df_results = pd.DataFrame(results)

        st.divider()
        st.subheader("📊 Screening Results")
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        # Summary
        st.divider()
        st.subheader("📈 Summary")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🚀 Strong Buy", len(df_results[df_results['Action'].str.contains('STRONG BUY', na=False)]))
        with col2:
            st.metric("📈 Solid Buy", len(df_results[df_results['Action'].str.contains('SOLID BUY', na=False)]))
        with col3:
            st.metric("💎 Accumulate", len(df_results[df_results['Action'].str.contains('ACCUMULATE', na=False)]))
        with col4:
            st.metric("🔄 Buy on Dips", len(df_results[df_results['Action'].str.contains('BUY on Dips', na=False)]))
        with col5:
            st.metric("🔴 Avoid", len(df_results[df_results['Action'].str.contains('AVOID', na=False)]))

    else:
        st.error("❌ No data found for selected stocks")
else:
    st.info("👈 Click 'RUN FULL ANALYSIS' button to start screening")
