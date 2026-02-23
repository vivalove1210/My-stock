import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- 網頁設定 ---
st.set_page_config(page_title="高手指標回測系統", layout="wide")
st.title("🛡️ 終極全能版 - 股價 + 成交量 + 指標同步顯示 (修正版)")

# --- 側邊欄：全域設定 ---
st.sidebar.header("1. 設定標的與資金")
ticker = st.sidebar.text_input("股票代碼 (台股請加 .TW)", value="3167.TW")
start_date = st.sidebar.date_input("開始日期", value=datetime.now() - timedelta(days=365 * 2))
end_date = st.sidebar.date_input("結束日期", value=datetime.now())
initial_capital = st.sidebar.number_input("初始資金 (元)", value=1000000)

# --- 側邊欄：策略選擇器 ---
st.sidebar.header("2. 選擇交易策略")
strategy_type = st.sidebar.selectbox(
    "請選擇一種操盤指標",
    ("超級趨勢 + ADX濾網 (推薦🔥)", "超級趨勢 (原始版)", "RSI 逆勢抄底", "MACD 動能指標")
)

# 參數設定區 (預設值)
st_period, st_multiplier = 10, 3.0
adx_threshold = 25
fast_period, slow_period, signal_period = 12, 26, 9
rsi_period, rsi_lower, rsi_upper = 14, 30, 70

if "超級趨勢" in strategy_type:
    st.sidebar.subheader("SuperTrend 參數")
    st_period = st.sidebar.slider("ATR 天數", 5, 20, 10)
    st_multiplier = st.sidebar.slider("波動倍數 (建議 3.0)", 1.0, 5.0, 3.0)

    if "ADX" in strategy_type:
        st.sidebar.subheader("🛡️ ADX 濾網參數")
        adx_threshold = st.sidebar.slider("ADX 門檻", 15, 40, 25)

elif "MACD" in strategy_type:
    st.sidebar.subheader("MACD 參數")
    fast_period = st.sidebar.slider("快線天數", 5, 20, 12)
    slow_period = st.sidebar.slider("慢線天數", 20, 40, 26)
    signal_period = st.sidebar.slider("訊號線天數", 5, 15, 9)

elif "RSI" in strategy_type:
    st.sidebar.subheader("RSI 參數")
    rsi_period = st.sidebar.slider("計算天數", 5, 30, 14)
    rsi_lower = st.sidebar.slider("超賣區", 10, 40, 30)
    rsi_upper = st.sidebar.slider("超買區", 60, 90, 70)

run_btn = st.sidebar.button("🚀 開始回測")


# --- 核心運算函數 ---
def calculate_adx(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where(plus_dm < 0, 0, plus_dm)
    minus_dm = np.where(minus_dm > 0, 0, minus_dm)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period).mean() / atr)
    minus_di = 100 * (abs(minus_dm).ewm(alpha=1 / period).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1 / period).mean()
    return adx


def calculate_supertrend(df, period=10, multiplier=3):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (high + low) / 2
    basic_upperband = hl2 + (multiplier * atr)
    basic_lowerband = hl2 - (multiplier * atr)
    final_upperband = pd.Series(0.0, index=df.index)
    final_lowerband = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)
    trend = pd.Series(0, index=df.index)

    for i in range(period, len(df)):
        if basic_upperband.iloc[i] < final_upperband.iloc[i - 1] or close.iloc[i - 1] > final_upperband.iloc[i - 1]:
            final_upperband.iloc[i] = basic_upperband.iloc[i]
        else:
            final_upperband.iloc[i] = final_upperband.iloc[i - 1]
        if basic_lowerband.iloc[i] > final_lowerband.iloc[i - 1] or close.iloc[i - 1] < final_lowerband.iloc[i - 1]:
            final_lowerband.iloc[i] = basic_lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = final_lowerband.iloc[i - 1]

        trend.iloc[i] = trend.iloc[i - 1]
        if trend.iloc[i - 1] == 1:
            if close.iloc[i] < final_lowerband.iloc[i - 1]:
                trend.iloc[i] = -1
        else:
            if close.iloc[i] > final_upperband.iloc[i - 1]:
                trend.iloc[i] = 1

        if trend.iloc[i] == 1:
            supertrend.iloc[i] = final_lowerband.iloc[i]
        else:
            supertrend.iloc[i] = final_upperband.iloc[i]
    return supertrend, trend


def calculate_macd(data, fast, slow, signal):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.rolling(window).mean() / loss.rolling(window).mean()
    return 100 - (100 / (1 + rs))


# --- 主程式 ---
if run_btn:
    try:
        with st.spinner('AI 正在構建三層圖表數據...'):
            stock = yf.Ticker(ticker)

            # --- 關鍵修正：解決 yfinance 少抓一天的問題 ---
            # 讓 end_date 自動 +1 天，確保包含使用者選擇的那一天
            end_date_plus = end_date + timedelta(days=1)

            df = stock.history(start=start_date, end=end_date_plus)
            df.index = df.index.tz_localize(None)

            if df.empty:
                st.error("❌ 無法抓取數據 (可能是假日、代碼錯誤或尚未開盤)")
            else:
                data = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                data['Signal'] = 0

                # 預先計算所有可能的指標
                if "ADX" in strategy_type:
                    data['ADX'] = calculate_adx(data, 14)
                if "MACD" in strategy_type:
                    data['MACD'], data['Signal_Line'] = calculate_macd(data, fast_period, slow_period, signal_period)
                if "RSI" in strategy_type:
                    data['RSI'] = calculate_rsi(data, rsi_period)

                # --- 策略訊號判斷 ---
                if "超級趨勢" in strategy_type:
                    st_line, st_trend = calculate_supertrend(data, st_period, st_multiplier)
                    data['SuperTrend'] = st_line

                    if "ADX" in strategy_type:
                        # 邏輯：SuperTrend 多頭 + ADX > 門檻
                        condition = (st_trend == 1) & (data['ADX'] > adx_threshold)
                        data.loc[condition, 'Signal'] = 1
                    else:
                        data.loc[st_trend == 1, 'Signal'] = 1

                elif "MACD" in strategy_type:
                    data.loc[data['MACD'] > data['Signal_Line'], 'Signal'] = 1

                elif "RSI" in strategy_type:
                    data.loc[data['RSI'] < rsi_lower, 'Signal'] = 1
                    data.loc[data['RSI'] > rsi_upper, 'Signal'] = 0

                # --- 績效計算 ---
                data['Position'] = data['Signal'].diff()
                data['Strategy_Ret'] = data['Close'].pct_change() * data['Signal'].shift(1)
                data['Strategy_Cum'] = (1 + data['Strategy_Ret']).cumprod() * initial_capital

                # --- 建立 3 層圖表 ---
                has_indicator = "ADX" in strategy_type or "MACD" in strategy_type or "RSI" in strategy_type

                if has_indicator:
                    fig = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.5, 0.2, 0.3],
                        subplot_titles=(f"股價 ({ticker})", "成交量", "技術指標")
                    )
                else:
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"股價 ({ticker})", "成交量")
                    )

                # --- 第一層：主圖 ---
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='股價', line=dict(color='gray', width=1)),
                              row=1, col=1)

                if "超級趨勢" in strategy_type:
                    fig.add_trace(go.Scatter(x=data.index, y=data['SuperTrend'], mode='markers', name='趨勢線',
                                             marker=dict(color='orange', size=2)), row=1, col=1)

                buy = data[data['Position'] == 1]
                sell = data[data['Position'] == -1]
                fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', name='買進',
                                         marker=dict(color='red', symbol='triangle-up', size=15)), row=1, col=1)
                fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', name='賣出',
                                         marker=dict(color='green', symbol='triangle-down', size=15)), row=1, col=1)

                # --- 第二層：成交量 ---
                vol_colors = ['red' if c >= o else 'green' for c, o in zip(data['Close'], data['Open'])]
                fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='成交量', marker_color=vol_colors), row=2,
                              col=1)

                # --- 第三層：指標 ---
                if has_indicator:
                    if "ADX" in strategy_type:
                        fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], name='ADX', line=dict(color='purple')),
                                      row=3, col=1)
                        fig.add_hline(y=adx_threshold, line_dash="dash", line_color="red", row=3, col=1)

                    elif "RSI" in strategy_type:
                        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                                      row=3, col=1)
                        fig.add_hline(y=rsi_upper, line_dash="dash", line_color="red", row=3, col=1)
                        fig.add_hline(y=rsi_lower, line_dash="dash", line_color="green", row=3, col=1)
                        fig.update_yaxes(range=[0, 100], row=3, col=1)

                    elif "MACD" in strategy_type:
                        colors = ['red' if v >= 0 else 'green' for v in (data['MACD'] - data['Signal_Line'])]
                        fig.add_trace(go.Bar(x=data.index, y=data['MACD'] - data['Signal_Line'], name='MACD柱',
                                             marker_color=colors), row=3, col=1)
                        fig.add_trace(
                            go.Scatter(x=data.index, y=data['MACD'], name='DIF', line=dict(color='blue', width=1)),
                            row=3, col=1)
                        fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], name='MACD',
                                                 line=dict(color='orange', width=1)), row=3, col=1)

                # --- 版面設定 ---
                fig.update_layout(height=800, margin=dict(l=50, r=50, t=50, b=50), hovermode="x unified")
                fig.update_xaxes(rangeslider_visible=False)

                st.plotly_chart(fig, use_container_width=True)

                total_ret = (data['Strategy_Cum'].iloc[-1] - initial_capital) / initial_capital * 100
                st.info(f"💰 回測結果：總報酬率 {total_ret:.2f}% | 最終資產 ${data['Strategy_Cum'].iloc[-1]:,.0f}")

    except Exception as e:
        st.error(f"發生錯誤: {e}")