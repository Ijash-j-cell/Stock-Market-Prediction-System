import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Import prediction and training components
from predict import predict_stock
from train import train_model

# Page Configuration
st.set_page_config(
    page_title="STOCK MARKET PREDICTION SYSTEM 📈",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Indian Markets UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title and Subtitle */
    .main-title {
        background: linear-gradient(135deg, #FF9933 0%, #ffffff 50%, #128807 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Custom Cards for Metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 153, 51, 0.2);
    }
    .metric-title {
        color: #a0aec0;
        font-size: 0.9rem;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Recommendations styling */
    .recommendation-buy {
        background: rgba(72, 187, 120, 0.15);
        border: 1px solid #48bb78;
        color: #48bb78;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    .recommendation-sell {
        background: rgba(245, 101, 101, 0.15);
        border: 1px solid #f56565;
        color: #f56565;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-title'>STOCK MARKET PREDICTION SYSTEM 📈</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-time tracking, technical indicator grids, and AI predictions (Random Forest) tailored for Indian Stock Exchanges (NSE & BSE).</p>", unsafe_allow_html=True)

# Sidebar - Indian Markets Controls
st.sidebar.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=80)
st.sidebar.header("🇮🇳 Indian Market Control")

# 1. Selection for popular Indian Equities / Indices
indian_stocks = {
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
    "State Bank of India (SBIN.NS)": "SBIN.NS",
    "Nifty 50 Index (^NSEI)": "^NSEI",
    "Sensex Index (^BSESN)": "^BSESN"
}

selected_stock = st.sidebar.selectbox(
    "Select Indian Stock / Index",
    options=list(indian_stocks.keys()),
    index=0
)

custom_ticker_input = st.sidebar.text_input(
    "Or Enter Custom Symbol (e.g. ITC, WIPRO)",
    value=indian_stocks[selected_stock],
    help="Enter an Indian stock code. If suffix is omitted, .NS will be appended automatically."
).upper().strip()

# Format Ticker
ticker = custom_ticker_input
if ticker and not ticker.startswith('^') and not (ticker.endswith('.NS') or ticker.endswith('.BO')):
    ticker = f"{ticker}.NS"

# Checkbox for selecting the timeperiod / interval type (Minutes / Hours)
if 'interval_type' not in st.session_state:
    st.session_state.interval_type = 'Minutes'

def set_minutes():
    st.session_state.interval_type = 'Minutes'
    st.session_state.hours_cb = False

def set_hours():
    st.session_state.interval_type = 'Hours'
    st.session_state.minutes_cb = False

st.sidebar.markdown("**Select Time Period (Interval Type):**")
col_min, col_hr = st.sidebar.columns(2)
with col_min:
    minutes_selected = st.checkbox(
        "Minutes", 
        value=(st.session_state.interval_type == 'Minutes'), 
        key="minutes_cb", 
        on_change=set_minutes
    )
with col_hr:
    hours_selected = st.checkbox(
        "Hours", 
        value=(st.session_state.interval_type == 'Hours'), 
        key="hours_cb", 
        on_change=set_hours
    )

interval_type = st.session_state.interval_type

if interval_type == 'Minutes':
    minutes_interval = st.sidebar.selectbox(
        "Select Minutes Interval",
        options=["1 Minute", "5 Minutes", "15 Minutes", "30 Minutes", "90 Minutes"],
        index=2  # Default to 15 Minutes
    )
    
    if minutes_interval == "1 Minute":
        period_options = ["1 Day", "5 Days", "7 Days"]
    else:
        period_options = ["1 Day", "5 Days", "7 Days", "1 Month"]
        
    minutes_period = st.sidebar.selectbox(
        "Select Period",
        options=period_options,
        index=1 if len(period_options) > 1 else 0  # Default to 5 Days
    )
    
    interval_map = {
        "1 Minute": "1m",
        "5 Minutes": "5m",
        "15 Minutes": "15m",
        "30 Minutes": "30m",
        "90 Minutes": "90m"
    }
    period_map = {
        "1 Day": "1d",
        "5 Days": "5d",
        "7 Days": "7d",
        "1 Month": "1mo"
    }
    interval = interval_map[minutes_interval]
    period = period_map[minutes_period]
else:
    hours_interval = st.sidebar.selectbox(
        "Select Hours Interval",
        options=["1 Hour"],
        index=0
    )
    hours_period = st.sidebar.selectbox(
        "Select Period",
        options=["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"],
        index=0  # Default to 1 Month
    )
    
    interval_map = {
        "1 Hour": "1h"
    }
    period_map = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y"
    }
    interval = interval_map[hours_interval]
    period = period_map[hours_period]

# Live Market Quote Panel in Sidebar (INR)
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Live Market Quote (IST)")

try:
    ticker_obj = yf.Ticker(ticker)
    fast_info = ticker_obj.fast_info
    
    live_price = float(fast_info.last_price)
    prev_close = float(fast_info.previous_close)
    day_high = float(fast_info.day_high)
    day_low = float(fast_info.day_low)
    open_price = float(fast_info.open)
    volume = int(fast_info.last_volume)
    mcap = float(fast_info.market_cap)
    currency = "INR" if fast_info.currency == "INR" or ticker.endswith('.NS') or ticker.endswith('.BO') or ticker.startswith('^NSE') or ticker.startswith('^BSE') else fast_info.currency
    
    # Calculate daily change
    daily_change = live_price - prev_close
    daily_change_pct = (daily_change / prev_close) * 100
    
    change_color = "#48bb78" if daily_change >= 0 else "#f56565"
    change_arrow = "▲" if daily_change >= 0 else "▼"
    curr_symbol = "₹" if currency == "INR" else "$"
    
    st.sidebar.markdown(f"""
    <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;">
        <div style="font-size: 0.8rem; color: #a0aec0; text-transform: uppercase; font-weight: 600;">Last Traded Price</div>
        <div style="font-size: 1.8rem; font-weight: 700; margin-top: 0.2rem; color: #ffffff;">{curr_symbol}{live_price:.2f} <span style="font-size: 0.9rem; font-weight: normal; color: #a0aec0;">{currency}</span></div>
        <div style="font-size: 0.9rem; font-weight: 600; color: {change_color};">{change_arrow} {curr_symbol}{abs(daily_change):.2f} ({daily_change_pct:+.2f}%)</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.8rem; font-size: 0.75rem; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 0.6rem;">
            <div><span style="color: #a0aec0;">Open:</span> <br><b>{curr_symbol}{open_price:.2f}</b></div>
            <div><span style="color: #a0aec0;">Prev Close:</span> <br><b>{curr_symbol}{prev_close:.2f}</b></div>
            <div><span style="color: #a0aec0;">High:</span> <br><b>{curr_symbol}{day_high:.2f}</b></div>
            <div><span style="color: #a0aec0;">Low:</span> <br><b>{curr_symbol}{day_low:.2f}</b></div>
            <div style="grid-column: span 2;"><span style="color: #a0aec0;">Market Cap:</span> <br><b>{curr_symbol}{mcap/1e9:.2f}B</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.sidebar.warning(f"Could not load live stock statistics: {e}")
    curr_symbol = "₹"  # Fallback

# Display Model status in Sidebar
model_path = f"models/{ticker}_model.joblib"
model_exists = os.path.exists(model_path)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Model Training Status")

if model_exists:
    try:
        checkpoint = joblib.load(model_path)
        metrics = checkpoint.get('metrics', {})
        st.sidebar.success(f"✅ Model trained for {ticker}")
        st.sidebar.markdown(f"**R² Score:** `{metrics.get('r2', 0.0):.4f}`")
        st.sidebar.markdown(f"**MAE:** `{curr_symbol}{metrics.get('mae', 0.0):.2f}`")
    except Exception:
        st.sidebar.warning(f"⚠️ Model found but failed to load.")
else:
    st.sidebar.warning(f"⚠️ No model trained for {ticker}")

# Force Retrain Button in Sidebar
if st.sidebar.button("⚙️ Train / Re-train Model", use_container_width=True):
    with st.spinner(f"Downloading NSE history and training RandomForest model for {ticker}..."):
        try:
            checkpoint = train_model(ticker, period="5y")
            st.sidebar.success(f"🎉 Model trained and saved!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to train model: {e}")

# Run analysis trigger
analyze_clicked = st.sidebar.button("🔍 Run Stock Analysis", type="primary", use_container_width=True)

# Load data and run prediction on click or on load or if parameters change
if (analyze_clicked or 
    "last_ticker" not in st.session_state or 
    st.session_state.last_ticker != ticker or
    "last_period" not in st.session_state or
    st.session_state.last_period != period or
    "last_interval" not in st.session_state or
    st.session_state.last_interval != interval):
    
    st.session_state.last_ticker = ticker
    st.session_state.last_period = period
    st.session_state.last_interval = interval
    
    with st.spinner(f"Analyzing {ticker} from Yahoo Finance..."):
        # 1. Download Stock Data
        data = yf.download(ticker, period=period, interval=interval)
        
        if data.empty:
            st.error(f"❌ Could not download data for ticker '{ticker}'. Make sure the stock is listed on NSE/BSE.")
        else:
            # Flatten columns if MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            # Convert timezone to Indian Standard Time (IST) if timezone-aware
            if data.index.tz is not None:
                data.index = data.index.tz_convert('Asia/Kolkata')
                
            st.session_state.data = data
            
            # 2. Get Model Prediction
            predicted_price = predict_stock(ticker)
            st.session_state.predicted_price = predicted_price

# Display dashboard results
if "data" in st.session_state and "predicted_price" in st.session_state:
    data = st.session_state.data.copy()
    predicted_price = st.session_state.predicted_price
    
    # Calculate indicators locally for plotting and verification
    data['Close'] = data['Close'].astype(float)
    data['High'] = data['High'].astype(float)
    data['Low'] = data['Low'].astype(float)
    data['Open'] = data['Open'].astype(float)
    
    # Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['SMA_20'] + (2 * bb_std)
    data['BB_Lower'] = data['SMA_20'] - (2 * bb_std)
    
    # RSI (14)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator (%K and %D)
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['Stochastic_K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14 + 1e-9))
    data['Stochastic_D'] = data['Stochastic_K'].rolling(window=3).mean()
    
    # ATR (Average True Range)
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    true_tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = true_tr.rolling(window=14).mean()
    
    current_price = float(data['Close'].iloc[-1])
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Banners showing live price status
    st.info(f"⚡ Live Integration Active: Predicted price is calculated using the latest live price of **{curr_symbol}{current_price:.2f}**.")
    
    # Top Row Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Current Indian Stock Price</div>
            <div class="metric-value">{curr_symbol}{current_price:.2f}</div>
            <div style="color: #a0aec0; font-size: 0.85rem;">Last recorded close (IST)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">AI Projected Price (Next Close)</div>
            <div class="metric-value" style="color: #6366f1;">{curr_symbol}{predicted_price:.2f}</div>
            <div style="color: #a0aec0; font-size: 0.85rem;">Random Forest Regressor</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        color = "#48bb78" if price_change >= 0 else "#f56565"
        arrow = "▲" if price_change >= 0 else "▼"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Expected Movement</div>
            <div class="metric-value" style="color: {color};">{arrow} {curr_symbol}{abs(price_change):.2f}</div>
            <div style="color: {color}; font-size: 0.85rem; font-weight: 600;">{price_change_pct:+.2f}% change</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        if predicted_price > current_price:
            rec_html = '<span class="recommendation-buy">🟢 BUY SIGNAL</span>'
            explanation = "Model forecasts upward movement"
        else:
            rec_html = '<span class="recommendation-sell">🔴 SELL SIGNAL</span>'
            explanation = "Model forecasts correction"
            
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Signal Recommendation</div>
            <div style="margin-top: 0.5rem; margin-bottom: 0.5rem;">{rec_html}</div>
            <div style="color: #a0aec0; font-size: 0.85rem;">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab_live, tab2, tab3, tab4 = st.tabs([
        "🕯️ Intraday Candlestick Chart", 
        "⏱️ Live Intraday Chart (1m)",
        "📊 Advanced Technical Indicators", 
        "🧠 AI Model Insights",
        "📋 Historical Data Table"
    ])
    
    with tab1:
        st.subheader(f"Intraday Candlestick Pattern & Bollinger Bands Overlay ({interval} interval - {period} period)")
        
        # Interactive plot with plotly
        fig = go.Figure()
        
        # 1. Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick',
                increasing=dict(line=dict(color='#26a69a'), fillcolor='#26a69a'),
                decreasing=dict(line=dict(color='#ef5350'), fillcolor='#ef5350')
            )
        )
        
        # 2. Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Upper'],
                name='Bollinger Upper',
                line=dict(color='rgba(255, 165, 0, 0.4)', width=1, dash='dash')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                name='SMA 20 (BB Middle)',
                line=dict(color='rgba(255, 255, 255, 0.3)', width=1)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Lower'],
                name='Bollinger Lower',
                line=dict(color='rgba(255, 165, 0, 0.4)', width=1, dash='dash')
            )
        )
        
        # 3. Add prediction marker for next day
        next_day = data.index[-1] + pd.Timedelta(days=1)
        if next_day.weekday() == 5: # Saturday
            next_day += pd.Timedelta(days=2)
        elif next_day.weekday() == 6: # Sunday
            next_day += pd.Timedelta(days=1)
            
        # Draw dotted line from last close to prediction
        fig.add_trace(
            go.Scatter(
                x=[data.index[-1], next_day],
                y=[current_price, predicted_price],
                name='AI Prediction Link',
                line=dict(color='#8B5CF6', width=2, dash='dot'),
                mode='lines'
            )
        )
        
        # Prediction node point
        fig.add_trace(
            go.Scatter(
                x=[next_day],
                y=[predicted_price],
                name='Predicted Close',
                mode='markers',
                marker=dict(color='#EC4899', size=10, symbol='diamond')
            )
        )
        
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=20, b=20),
            height=550,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title=f"Price ({curr_symbol})")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tab_live:
        st.subheader("Real-Time Intraday Stock Activity (1-Minute Intervals)")
        with st.spinner(f"Downloading 1-minute intraday candles for {ticker}..."):
            try:
                intraday_data = yf.download(ticker, period="1d", interval="1m")
                if intraday_data.empty:
                    st.warning("No intraday data found (the Indian stock market is open Monday to Friday, 9:15 AM - 3:30 PM IST).")
                else:
                    if isinstance(intraday_data.columns, pd.MultiIndex):
                        intraday_data.columns = intraday_data.columns.get_level_values(0)
                        
                    # Convert index timezone to IST (Asia/Kolkata)
                    if intraday_data.index.tz is not None:
                        intraday_data.index = intraday_data.index.tz_convert('Asia/Kolkata')
                        
                    fig_intra = go.Figure()
                    fig_intra.add_trace(
                        go.Scatter(
                            x=intraday_data.index,
                            y=intraday_data['Close'],
                            mode='lines',
                            name='Intraday price',
                            line=dict(color='#10B981', width=1.8)
                        )
                    )
                    fig_intra.update_layout(
                        template="plotly_dark",
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=450,
                        hovermode="x unified",
                        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title="Indian Standard Time (IST)"),
                        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title=f"Price ({curr_symbol})")
                    )
                    st.plotly_chart(fig_intra, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load intraday data: {e}")
                
    with tab2:
        st.subheader("📊 Multiple Technical Indicators Grid")
        
        # 1. MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='#F59E0B', width=1.5)))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['Signal'], name='Signal Line', line=dict(color='#10B981', width=1.5)))
        macd_hist = data['MACD'] - data['Signal']
        colors = ['#10B981' if val >= 0 else '#EF4444' for val in macd_hist]
        fig_macd.add_trace(go.Bar(x=data.index, y=macd_hist, name='Histogram', marker_color=colors, opacity=0.4))
        
        fig_macd.update_layout(
            title="1. MACD (Moving Average Convergence Divergence)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            height=250,
            hovermode="x unified",
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)")
        )
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # 2. RSI (14)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI (14)', line=dict(color='#EC4899', width=1.5)))
        # Threshold lines
        fig_rsi.add_trace(go.Scatter(x=data.index, y=[70]*len(data), name='Overbought (70)', line=dict(color='#EF4444', width=1, dash='dash'), mode='lines'))
        fig_rsi.add_trace(go.Scatter(x=data.index, y=[30]*len(data), name='Oversold (30)', line=dict(color='#10B981', width=1, dash='dash'), mode='lines'))
        
        fig_rsi.update_layout(
            title="2. RSI (Relative Strength Index - 14 Days)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            height=250,
            hovermode="x unified",
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", range=[10, 90], title="RSI Score")
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # 3. Stochastic Oscillator
        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(x=data.index, y=data['Stochastic_K'], name='%K Line', line=dict(color='#3B82F6', width=1.5)))
        fig_stoch.add_trace(go.Scatter(x=data.index, y=data['Stochastic_D'], name='%D SMA', line=dict(color='#93C5FD', width=1.5)))
        # Thresholds
        fig_stoch.add_trace(go.Scatter(x=data.index, y=[80]*len(data), name='Upper (80)', line=dict(color='#EF4444', width=1, dash='dash'), mode='lines'))
        fig_stoch.add_trace(go.Scatter(x=data.index, y=[20]*len(data), name='Lower (20)', line=dict(color='#10B981', width=1, dash='dash'), mode='lines'))
        
        fig_stoch.update_layout(
            title="3. Stochastic Oscillator (14-day %K & 3-day %D)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            height=250,
            hovermode="x unified",
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", range=[-5, 105], title="Stoch Score")
        )
        st.plotly_chart(fig_stoch, use_container_width=True)
        
        # ATR Display
        st.markdown(f"""
        **💡 Additional Volatility Metrics**:
        * **Average True Range (ATR - 14 Days)**: `{curr_symbol}{data['ATR'].iloc[-1]:.2f}` (indicates the average daily trading range).
        * **Simple Moving Average (SMA - 50 Days)**: `{curr_symbol}{data['SMA_50'].iloc[-1]:.2f}` (medium-term support baseline).
        * **Bollinger Band Width**: `{curr_symbol}{(data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1]):.2f}`
        """)
        
    with tab3:
        st.subheader("🤖 Artificial Intelligence Explainability (INR)")
        
        if os.path.exists(model_path):
            try:
                chk = joblib.load(model_path)
                metrics = chk.get('metrics', {})
                features = chk.get('features', [])
                model_obj = chk.get('model', None)
                
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric("R² Score (Goodness of Fit)", f"{metrics.get('r2', 0.0):.4f}")
                with col_met2:
                    st.metric("Mean Absolute Error (MAE)", f"{curr_symbol}{metrics.get('mae', 0.0):.4f}")
                with col_met3:
                    st.metric("Root Mean Squared Error (RMSE)", f"{curr_symbol}{metrics.get('rmse', 0.0):.4f}")
                
                # Feature Importance Chart
                if hasattr(model_obj, 'feature_importances_'):
                    st.markdown("#### Feature Importance (What the AI model prioritizes)")
                    importances = model_obj.feature_importances_
                    
                    df_imp = pd.DataFrame({
                        'Feature Name': features,
                        'Importance Score': importances
                    }).sort_values(by='Importance Score', ascending=True)
                    
                    fig_imp = go.Figure(go.Bar(
                        x=df_imp['Importance Score'],
                        y=df_imp['Feature Name'],
                        orientation='h',
                        marker=dict(
                            color=df_imp['Importance Score'],
                            colorscale='Viridis'
                        )
                    ))
                    
                    fig_imp.update_layout(
                        template="plotly_dark",
                        margin=dict(l=20, r=20, t=10, b=10),
                        height=500,
                        xaxis=dict(title="Importance Weight", gridcolor="rgba(255, 255, 255, 0.05)"),
                        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)")
                    )
                    
                    st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.info(f"Could not load feature importances: {e}")
        else:
            st.info("Train the model using the sidebar option to view explainability reports and feature importances.")
            
    with tab4:
        st.subheader("Historical Stock Data Table")
        st.dataframe(
            data.tail(50).sort_index(ascending=False),
            use_container_width=True
        )
else:
    st.info("👈 Select an Indian stock ticker symbol and click 'Run Stock Analysis' in the sidebar to start.")