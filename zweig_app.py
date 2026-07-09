# -*- coding: utf-8 -*-
"""
S&P 500 版 Zweig Breadth Thrust 廣度衝刺偵測
用 500 檔成分股「每天漲/跌家數」算 10 日 EMA，抓 40→61.5 的急速翻多訊號。
本機測試：streamlit run zweig_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Zweig 廣度衝刺（S&P 500）", page_icon="📈", layout="wide")
st.title("📈 Zweig Breadth Thrust — S&P 500 版")
st.caption("用 500 檔成分股的每日漲跌家數，偵測「10 日內從 40% 急升到 61.5%」的廣度衝刺買訊")


# ---------- 資料 ----------
@st.cache_data(ttl=6 * 3600)
def get_sp500_tickers():
    # 讀打包在同資料夾的清單，不需 lxml / 不連 Wikipedia
    path = Path(__file__).parent / "sp500_tickers.csv"
    df = pd.read_csv(path)
    return [str(x).replace(".", "-").strip() for x in df["Symbol"]]


@st.cache_data(ttl=6 * 3600)
def download_closes(tickers, period):
    data = yf_download(tickers, period)
    return data


def yf_download(tickers, period):
    import yfinance as yf
    df = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all")


@st.cache_data(ttl=6 * 3600)
def download_spx(period):
    import yfinance as yf
    s = yf.download("^GSPC", period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()


# ---------- 計算 ----------
def zweig_series(closes, ema_len):
    daily = closes.diff()
    adv = (daily > 0).sum(axis=1)
    dec = (daily < 0).sum(axis=1)
    ratio = adv / (adv + dec).replace(0, np.nan)
    ema = ratio.ewm(span=ema_len, adjust=False).mean()
    return ratio, ema


def find_thrusts(ema, low, high, window):
    sigs, last_low = [], -10 ** 9
    for i, v in enumerate(ema.values):
        if v < low:
            last_low = i
        if v > high and (i - last_low) <= window:
            sigs.append(ema.index[i]); last_low = -10 ** 9
    return sigs


def forward_returns(spx, dates, horizons=(21, 63, 126, 252)):
    rows = []
    idx = spx.index
    for d in dates:
        pos = idx.searchsorted(d)
        if pos >= len(idx):
            continue
        p0 = spx.iloc[pos]
        row = {"訊號日": pd.Timestamp(d).date(), "當日收盤": round(float(p0), 0)}
        for h, name in zip(horizons, ["1個月", "3個月", "6個月", "12個月"]):
            if pos + h < len(idx):
                row[name] = round((float(spx.iloc[pos + h]) / float(p0) - 1) * 100, 1)
            else:
                row[name] = None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- 側邊欄 ----------
with st.sidebar:
    st.header("⚙️ 設定")
    period = st.selectbox("資料回溯期間", ["1y", "2y", "3y", "5y", "10y"], index=2)
    ema_len = st.number_input("EMA 天數（原版 10）", value=10, step=1)
    low = st.number_input("下緣 %（原版 40）", value=40.0, step=0.5) / 100
    high = st.number_input("上緣 %（原版 61.5）", value=61.5, step=0.5) / 100
    window = st.number_input("衝刺視窗（交易日，原版 10）", value=10, step=1)
    st.caption("⚠️ 首次抓 500 檔約需 1～3 分鐘（視 Yahoo 回應），之後快取 6 小時。")


# ---------- 主流程（抓資料放在按鈕後，避免一開頁就卡） ----------
if st.button("📥 抓取 S&P 500 並計算", type="primary"):
    try:
        with st.spinner("讀取 S&P 500 成分股清單…"):
            tickers = get_sp500_tickers()
        with st.spinner(f"下載 {len(tickers)} 檔日線資料（可能要 1～3 分鐘）…"):
            closes = download_closes(tickers, period)
            spx = download_spx(period)
        st.session_state["closes"] = closes
        st.session_state["spx"] = spx
        st.session_state["n_tickers"] = closes.shape[1]
        st.success(f"✅ 已載入 {closes.shape[1]} 檔、{len(closes)} 個交易日")
    except Exception as e:
        st.error(f"抓取失敗：{e}")
        st.caption("雲端可能被 Yahoo/Wikipedia 限流，稍後再試或改本機執行。")

if "closes" in st.session_state:
    closes = st.session_state["closes"]
    spx = st.session_state["spx"]
    ratio, ema = zweig_series(closes, int(ema_len))
    sigs = find_thrusts(ema.dropna(), low, high, int(window))

    cur = float(ema.dropna().iloc[-1]) if len(ema.dropna()) else float("nan")
    c1, c2, c3 = st.columns(3)
    c1.metric("目前廣度 10 日 EMA", f"{cur*100:.1f}%")
    c2.metric("距上緣門檻", f"{(high-cur)*100:+.1f} pt", help="正值代表還沒達標")
    c3.metric("歷史衝刺次數", f"{len(sigs)} 次")

    if sigs and (ema.index[-1] - sigs[-1]).days <= 15:
        st.success(f"🔥 最近剛觸發衝刺訊號：{pd.Timestamp(sigs[-1]).date()}")

    st.subheader("廣度 10 日 EMA 與上下緣")
    band = pd.DataFrame({
        "廣度10日EMA": ema * 100,
        f"下緣 {low*100:.0f}%": low * 100,
        f"上緣 {high*100:.1f}%": high * 100,
    })
    st.line_chart(band, color=["#0e2a47", "#3f7d5a", "#b23b3b"])

    st.subheader("S&P 500 指數")
    st.line_chart(spx.rename("S&P 500"), color="#c8a24b")

    st.subheader("歷次衝刺訊號 → 之後的 S&P 500 報酬")
    fr = forward_returns(spx, sigs)
    if len(fr):
        st.dataframe(fr, hide_index=True)
        st.subheader("勝率統計（正報酬比例）")
        stat_cols = ["1個月", "3個月", "6個月", "12個月"]
        rows = []
        for c in stat_cols:
            vals = fr[c].dropna()
            if len(vals):
                rows.append({
                    "期間": c,
                    "樣本數": len(vals),
                    "正報酬比例": f"{(vals > 0).mean()*100:.0f}%",
                    "平均報酬": f"{vals.mean():+.1f}%",
                    "中位數": f"{vals.median():+.1f}%",
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("這段期間沒有偵測到衝刺訊號（衝刺本來就很罕見，拉長回溯期間比較容易看到）。")
else:
    st.info("👆 按上方按鈕開始抓取與計算。")
