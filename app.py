import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- 網頁設定 ---
st.set_page_config(page_title="高手指標回測系統", layout="wide")
st.title("🛡️ 終極六合一 - 超級趨勢多策略回測系統")

# --- 側邊欄：全域設定 ---
st.sidebar.header("1. 設定標的與資金")
ticker = st.sidebar.text_input("股票代碼 (台股請加 .TW)", value="3167.TW")
start_date = st.sidebar.date_input("開始日期", value=datetime.now() - timedelta(days=365 * 2))
end_date = st.sidebar.date_input("結束日期", value=datetime.now())
initial_capital = st.sidebar.number_input("初始資金 (元)", value=1000000)

# --- 側邊欄：策略選擇器 ---
st.sidebar.header("2. 選擇交易策略")
strategy_map = {
    "1. 超級趨勢 + ADX濾網": "st_adx",
    "2. 超級趨勢 (原始版)": "st_only",
    "3. 超級趨勢 + RSI 確認": "st_rsi",
    "4. 超級趨勢 + MACD 確認": "st_macd",
    "5. 超級趨勢 + 均線排列 (5/10/20MA)": "st_ma",
    "6. 超級趨勢 + KD 指標": "st_kd"
}
selected_strategy_name = st.sidebar.selectbox("請選擇策略", list(strategy_map.keys()))
strategy_code = strategy_map[selected_strategy_name]

# --- 參數設定區 ---
st.sidebar.subheader("⚙️ 參數微調")

# SuperTrend 參數
st_period = st.sidebar.slider("SuperTrend 天數 (ATR)", 5, 20, 10)
st_multiplier = st.sidebar.slider("SuperTrend 倍數", 1.0, 5.0, 3.0)

# 根據選擇顯示對應參數
if strategy_code == "st_adx":
    adx_threshold = st.sidebar.slider("ADX 門檻 (強勢才買)", 15, 40, 25)
    st.sidebar.info("💡 邏輯：SuperTrend 多頭 + ADX > 門檻")

elif strategy_code == "st_rsi":
    rsi_period = st.sidebar.slider("RSI 天數", 5, 30, 14)
    rsi_filter = st.sidebar.slider("RSI 濾網 (>此值才買)", 30, 60, 50)
    st.sidebar.info(f"💡 邏輯：SuperTrend 多頭 + RSI > {rsi_filter}")

elif strategy_code == "st_macd":
    fast_p = st.sidebar.slider("MACD 快線", 5, 20, 12)
    slow_p = st.sidebar.slider("MACD 慢線", 20, 40, 26)
    signal_p = st.sidebar.slider("MACD 訊號", 5, 15, 9)
    st.sidebar.info("💡 邏輯：SuperTrend 多頭 + MACD 金叉")

elif strategy_code == "st_ma":
    st.sidebar.info("💡 邏輯：SuperTrend 多頭 + 5MA > 10MA > 20MA (多頭排列)")

elif strategy_code == "st_kd":
    k_period = st.sidebar.slider("KD 天數 (RSV)", 5, 14, 9)
    st.sidebar.info("💡 邏輯：SuperTrend 多頭 + K值 > D值 (金叉)")

run_btn = st.sidebar.button("🚀 開始回測")


# --- 核心指標函數 ---
def calculate_supertrend(df, period=10, multiplier=3):
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)
    trend = pd.Series(0, index=df.index)

    for i in range(period, len(df)):
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        trend.iloc[i] = trend.iloc[i - 1]
        if trend.iloc[i - 1] == 1:
            if close.iloc[i] < final_lower.iloc[i - 1]: trend.iloc[i] = -1
        else:
            if close.iloc[i] > final_upper.iloc[i - 1]: trend.iloc[i] = 1

        supertrend.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]
    return supertrend, trend


def calculate_adx(df, period=14):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where(plus_dm < 0, 0, plus_dm)
    minus_dm = np.where(minus_dm > 0, 0, minus_dm)
    tr = pd.concat(
        [df['High'] - df['Low'], abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))],
        axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period).mean() / atr)
    minus_di = 100 * (abs(pd.Series(minus_dm, index=df.index)).ewm(alpha=1 / period).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    return dx.ewm(alpha=1 / period).mean()


def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    rs = gain.rolling(window).mean() / loss.rolling(window).mean()
    return 100 - (100 / (1 + rs))


def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig


def calculate_kd(data, period=9):
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    rsv = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    k = pd.Series(50.0, index=data.index)
    d = pd.Series(50.0, index=data.index)
    for i in range(period, len(data)):
        k.iloc[i] = (2 / 3) * k.iloc[i - 1] + (1 / 3) * rsv.iloc[i]
        d.iloc[i] = (2 / 3) * d.iloc[i - 1] + (1 / 3) * k.iloc[i]
    return k, d


# --- 主程式 ---
if run_btn:
    try:
        with st.spinner(f'AI 正在計算策略: {selected_strategy_name}...'):
            stock = yf.Ticker(ticker)
            end_date_plus = end_date + timedelta(days=1)
            df = stock.history(start=start_date, end=end_date_plus)
            df.index = df.index.tz_localize(None)

            if df.empty:
                st.error("❌ 無法抓取數據 (可能是假日或代碼錯誤)")
            else:
                data = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                data['Signal'] = 0

                st_line, st_trend = calculate_supertrend(data, st_period, st_multiplier)
                data['SuperTrend'] = st_line

                buy_condition = (st_trend == 1)

                if strategy_code == "st_adx":
                    data['ADX'] = calculate_adx(data, 14)
                    buy_condition = buy_condition & (data['ADX'] > adx_threshold)

                elif strategy_code == "st_rsi":
                    data['RSI'] = calculate_rsi(data, rsi_period)
                    buy_condition = buy_condition & (data['RSI'] > rsi_filter)

                elif strategy_code == "st_macd":
                    data['MACD'], data['Signal_Line'] = calculate_macd(data, fast_p, slow_p, signal_p)
                    buy_condition = buy_condition & (data['MACD'] > data['Signal_Line'])

                elif strategy_code == "st_ma":
                    data['MA5'] = data['Close'].rolling(5).mean()
                    data['MA10'] = data['Close'].rolling(10).mean()
                    data['MA20'] = data['Close'].rolling(20).mean()
                    buy_condition = buy_condition & (data['Close'] > data['MA5']) & (data['MA5'] > data['MA10']) & (
                                data['MA10'] > data['MA20'])

                elif strategy_code == "st_kd":
                    data['K'], data['D'] = calculate_kd(data, k_period)
                    buy_condition = buy_condition & (data['K'] > data['D'])

                data.loc[buy_condition, 'Signal'] = 1

                data['Position'] = data['Signal'].diff()
                data['Strategy_Ret'] = data['Close'].pct_change() * data['Signal'].shift(1)
                data['Strategy_Cum'] = (1 + data['Strategy_Ret']).cumprod() * initial_capital

                has_indicator = strategy_code not in ["st_only", "st_ma"]
                if has_indicator:
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                        row_heights=[0.5, 0.2, 0.3],
                                        subplot_titles=(f"股價 ({ticker})", "成交量", "技術指標"))
                else:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                        row_heights=[0.7, 0.3], subplot_titles=(f"股價 ({ticker})", "成交量"))

                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='股價', line=dict(color='gray', width=1)),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['SuperTrend'], mode='markers', name='趨勢線',
                                         marker=dict(color='orange', size=2)), row=1, col=1)

                if strategy_code == "st_ma":
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['MA5'], name='5MA', line=dict(color='purple', width=1)), row=1,
                        col=1)
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['MA10'], name='10MA', line=dict(color='blue', width=1)), row=1,
                        col=1)
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data['MA20'], name='20MA', line=dict(color='green', width=1)), row=1,
                        col=1)

                buy = data[data['Position'] == 1]
                sell = data[data['Position'] == -1]
                fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', name='買進',
                                         marker=dict(color='red', symbol='triangle-up', size=15)), row=1, col=1)
                fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', name='賣出',
                                         marker=dict(color='green', symbol='triangle-down', size=15)), row=1, col=1)

                vol_colors = ['red' if c >= o else 'green' for c, o in zip(data['Close'], data['Open'])]
                fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='成交量', marker_color=vol_colors), row=2,
                              col=1)

                if has_indicator:
                    row_idx = 3
                    if strategy_code == "st_adx":
                        fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], name='ADX', line=dict(color='purple')),
                                      row=row_idx, col=1)
                        fig.add_hline(y=adx_threshold, line_dash="dash", line_color="red", row=row_idx, col=1)
                    elif strategy_code == "st_rsi":
                        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                                      row=row_idx, col=1)
                        fig.add_hline(y=rsi_filter, line_dash="dash", line_color="red", row=row_idx, col=1)
                    elif strategy_code == "st_macd":
                        colors = ['red' if v >= 0 else 'green' for v in (data['MACD'] - data['Signal_Line'])]
                        fig.add_trace(go.Bar(x=data.index, y=data['MACD'] - data['Signal_Line'], name='MACD柱',
                                             marker_color=colors), row=row_idx, col=1)
                        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='DIF', line=dict(color='blue')),
                                      row=row_idx, col=1)
                        fig.add_trace(
                            go.Scatter(x=data.index, y=data['Signal_Line'], name='MACD', line=dict(color='orange')),
                            row=row_idx, col=1)
                    elif strategy_code == "st_kd":
                        fig.add_trace(go.Scatter(x=data.index, y=data['K'], name='K值', line=dict(color='blue')),
                                      row=row_idx, col=1)
                        fig.add_trace(go.Scatter(x=data.index, y=data['D'], name='D值', line=dict(color='orange')),
                                      row=row_idx, col=1)

                fig.update_layout(height=800, margin=dict(l=50, r=50, t=50, b=50), hovermode="x unified")
                fig.update_xaxes(rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                total_ret = (data['Strategy_Cum'].iloc[-1] - initial_capital) / initial_capital * 100
                st.info(f"💰 回測結果：總報酬率 {total_ret:.2f}% | 最終資產 ${data['Strategy_Cum'].iloc[-1]:,.0f}")

    except Exception as e:
        st.error(f"發生錯誤: {e}")