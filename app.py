import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market Pattern Scanner", layout="wide")

NIFTY50 = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV BHARTIARTL
CIPLA COALINDIA DRREDDY EICHERMOT GRASIM HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO HINDUNILVR
ICICIBANK INDUSINDBK INFY ITC JSWSTEEL KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC POWERGRID RELIANCE
SBILIFE SBIN SUNPHARMA TATACONSUM TATAMOTORS TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO""".split()
INDICES = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}


# ---------- indicators & scoring ----------
def add_signals(df, w):
    c, v = df["Close"], df["Volume"]
    df["EMA20"], df["EMA50"], df["EMA200"] = (c.ewm(span=n, adjust=False).mean() for n in (20, 50, 200))
    d = c.diff()
    rs = d.clip(lower=0).ewm(alpha=1 / 14).mean() / (-d.clip(upper=0)).ewm(alpha=1 / 14).mean()
    df["RSI"] = 100 - 100 / (1 + rs)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACDH"] = macd - macd.ewm(span=9, adjust=False).mean()
    hi, lo = df["High"].rolling(20).max().shift(1), df["Low"].rolling(20).min().shift(1)
    vr = v / v.rolling(20).mean()
    df["VOLX"] = vr
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(), (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14).mean()

    trend = 0.5 * np.sign(df["EMA20"] - df["EMA50"]) + 0.5 * np.sign(c - df["EMA200"])
    rsi = ((df["RSI"] - 50) / 20).clip(-1, 1)
    mac = np.sign(df["MACDH"])
    brk = np.where(c >= hi, 1, np.where(c <= lo, -1, 0))
    vol = np.where(vr > 1.2, np.sign(d) * (vr.clip(upper=2) / 2), 0)
    tot = sum(w.values())
    raw = (w["Trend"] * trend + w["RSI"] * rsi + w["MACD"] * mac + w["Breakout"] * brk + w["Volume"] * vol) / tot
    df["SCORE"] = (50 + 50 * raw).round(1)
    return df


def bias(s, hi, lo):
    return "CALL (CE) ▲" if s >= hi else "PUT (PE) ▼" if s <= lo else "NO TRADE"


def why(r):
    p = [("EMA20 above EMA50" if r["EMA20"] > r["EMA50"] else "EMA20 below EMA50"),
         ("price above EMA200" if r["Close"] > r["EMA200"] else "price below EMA200"),
         f"RSI {r['RSI']:.0f}", ("MACD positive" if r["MACDH"] > 0 else "MACD negative")]
    if r["VOLX"] > 1.2:
        p.append(f"volume {r['VOLX']:.1f}x average")
    return ", ".join(p)


def backtest(df, hi, lo, cost=0.0005):
    pos = pd.Series(np.where(df["SCORE"] >= hi, 1, np.where(df["SCORE"] <= lo, -1, 0)), index=df.index).shift(1).fillna(0)
    ret = df["Close"].pct_change().fillna(0)
    strat = pos * ret - pos.diff().abs().fillna(0) * cost
    eq, bh = (1 + strat).cumprod(), (1 + ret).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    trades = strat[pos.diff().abs() > 0]
    wins = (strat[pos != 0] > 0).mean() if (pos != 0).any() else 0
    return eq, bh, dict(ret=eq.iloc[-1] - 1, bh=bh.iloc[-1] - 1, dd=dd, win=wins, n=int((pos.diff().abs() > 0).sum()))


@st.cache_data(ttl=900, show_spinner=False)
def load_yf(tickers, period):
    raw = yf.download(list(tickers), period=period, group_by="ticker", auto_adjust=True, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = (raw[t] if len(tickers) > 1 else raw).dropna(subset=["Close"]).copy()
            if len(df) > 60:
                out[t] = df
        except Exception:
            pass
    return out


def load(tickers, period):
    if src.startswith("Angel"):
        import angel
        return angel.load(tickers, period)
    return load_yf(tickers, period)


# ---------- sidebar ----------
st.title("📈 Market Pattern Scanner — Nifty / Sensex")
with st.sidebar:
    src = st.radio("Data source", ["yfinance (delayed)", "Angel One (live)"])
    st.header("Settings")
    period = st.selectbox("History", ["1y", "2y", "5y"], index=1)
    hi = st.slider("CALL if score ≥", 55, 90, 65)
    lo = st.slider("PUT if score ≤", 10, 45, 35)
    st.subheader("Signal weights")
    w = {k: st.slider(k, 0.0, 1.0, v, 0.05) for k, v in
         dict(Trend=0.3, RSI=0.2, MACD=0.2, Breakout=0.15, Volume=0.15).items()}
    if sum(w.values()) == 0:
        w["Trend"] = 1.0
    st.subheader("Risk")
    capital = st.number_input("Capital (₹)", 10000, 10000000, 100000, 10000)
    risk_pct = st.slider("Risk per trade (% of capital)", 0.25, 3.0, 1.0, 0.25)
    st.caption("Educational tool. Not investment advice.")

tab1, tab2, tab3 = st.tabs(["🏆 Top picks (Nifty 50)", "🔍 Chart & backtest", "🧭 Index view"])

# ---------- tab 1: scanner ----------
with tab1:
    if st.button("Run scan", type="primary"):
        with st.spinner("Downloading data and scoring..."):
            syms = [s + ".NS" for s in NIFTY50]
            data = load(tuple(syms), period)
            rows = []
            for t, df in data.items():
                df = add_signals(df, w)
                r = df.iloc[-1]
                rows.append({"Stock": t[:-3], "Price": round(r["Close"], 2), "Score": r["SCORE"],
                             "Bias": bias(r["SCORE"], hi, lo), "RSI": round(r["RSI"], 1),
                             "Vol x avg": round(r["VOLX"], 2),
                             "vs EMA200 %": round((r["Close"] / r["EMA200"] - 1) * 100, 1),
                             "ATR": round(r["ATR"], 2), "Why": why(r)})
            st.session_state["scan"] = pd.DataFrame(rows).sort_values("Score", ascending=False)
    if "scan" in st.session_state:
        s = st.session_state["scan"]
        st.subheader("✅ Auto suggestions for today")
        picks = s[(s["Score"] >= hi) | (s["Score"] <= lo)].copy()
        picks["dist"] = (picks["Score"] - 50).abs()
        picks = picks.sort_values("dist", ascending=False).head(3)
        if picks.empty:
            st.warning("No strong setup today. Staying out is a valid trade.")
        for _, p in picks.iterrows():
            long_ = p["Score"] >= hi
            risk = 1.5 * p["ATR"]
            sl = p["Price"] - risk if long_ else p["Price"] + risk
            tg = p["Price"] + 2 * risk if long_ else p["Price"] - 2 * risk
            qty = max(int(min(capital * risk_pct / 100 / risk, capital / p["Price"])), 0)
            with st.container(border=True):
                st.markdown(f"**{p['Stock']}** → {'BUY / CALL (CE) ▲' if long_ else 'SELL / PUT (PE) ▼'} | Score {p['Score']}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Entry", f"₹{p['Price']:,.2f}")
                m2.metric("Stop-loss", f"₹{sl:,.2f}")
                m3.metric("Target (1:2)", f"₹{tg:,.2f}")
                m4.metric("Qty (shares)", qty)
                st.caption("Why: " + p["Why"])
        st.caption("Stop = 1.5 x ATR, target = 2x the risk. Qty is for cash shares; for options, size by lot and premium you can afford to lose.")
        a, b = st.columns(2)
        a.subheader("Most bullish (Call candidates)")
        a.dataframe(s.head(10), hide_index=True, use_container_width=True)
        b.subheader("Most bearish (Put candidates)")
        b.dataframe(s.tail(10).iloc[::-1], hide_index=True, use_container_width=True)
        with st.expander("Full table"):
            st.dataframe(s, hide_index=True, use_container_width=True)
    else:
        st.info("Click **Run scan** to rank all Nifty 50 stocks.")

# ---------- tab 2: chart + backtest ----------
with tab2:
    choice = st.selectbox("Symbol", list(INDICES) + NIFTY50)
    tk = INDICES.get(choice, choice + ".NS")
    d = load((tk,), period).get(tk)
    if d is None:
        st.error("No data for this symbol.")
    else:
        d = add_signals(d, w)
        last = d.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"{last['Close']:,.2f}")
        c2.metric("Score", last["SCORE"])
        c3.metric("Bias", bias(last["SCORE"], hi, lo))
        c4.metric("RSI", round(last["RSI"], 1))

        if src.startswith("Angel") and choice in ("NIFTY 50", "BANK NIFTY"):
            st.subheader("Option idea (live, from Angel One)")
            dirn = "CALL" if last["SCORE"] >= hi else "PUT" if last["SCORE"] <= lo else None
            if dirn is None:
                st.info("Score is neutral, so no option trade is suggested.")
            else:
                try:
                    import angel
                    o = angel.option_idea("NIFTY" if choice == "NIFTY 50" else "BANKNIFTY", last["Close"], dirn)
                    x1, x2, x3, x4 = st.columns(4)
                    x1.metric("Contract", o["symbol"])
                    x2.metric("Premium (LTP)", f"₹{o['ltp']:,.2f}")
                    x3.metric("1 lot cost", f"₹{o['ltp'] * o['lot']:,.0f}")
                    x4.metric("PCR (ATM±5)", "n/a" if o["pcr"] is None else round(o["pcr"], 2))
                    st.caption(f"Expiry {o['expiry']}. Rule of thumb: exit if premium falls 30% (₹{o['ltp'] * 0.7:,.2f}), "
                               f"book profit near +60% (₹{o['ltp'] * 1.6:,.2f}). Max loss is the full premium.")
                except Exception as e:
                    st.error(f"Could not load option data: {e}")

        fig = go.Figure(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Price"))
        for e in ("EMA20", "EMA50", "EMA200"):
            fig.add_scatter(x=d.index, y=d[e], name=e, line=dict(width=1))
        fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

        eq, bh, m = backtest(d, hi, lo)
        st.subheader("Backtest (trades the underlying long/short by signal, 0.05% cost per switch)")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Strategy return", f"{m['ret']:.1%}")
        k2.metric("Buy & hold", f"{m['bh']:.1%}")
        k3.metric("Max drawdown", f"{m['dd']:.1%}")
        k4.metric("Win rate (days)", f"{m['win']:.0%}")
        k5.metric("Position changes", m["n"])
        f2 = go.Figure()
        f2.add_scatter(x=eq.index, y=eq, name="Strategy")
        f2.add_scatter(x=bh.index, y=bh, name="Buy & hold")
        f2.update_layout(height=300, margin=dict(t=20))
        st.plotly_chart(f2, use_container_width=True)
        st.caption("In-sample test: tuning sliders to beat this curve is overfitting. "
                   "Options also lose to time decay and IV changes, which this test does not model.")

# ---------- tab 3: index view ----------
with tab3:
    rows = []
    idata = load(tuple(INDICES.values()), period)
    for name, t in INDICES.items():
        if t in idata:
            r = add_signals(idata[t], w).iloc[-1]
            rows.append({"Index": name, "Level": round(r["Close"], 2), "Score": r["SCORE"],
                         "Bias": bias(r["SCORE"], hi, lo), "RSI": round(r["RSI"], 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.info("For real strike/expiry selection you need option-chain data (OI, PCR, IV). "
            "Connect a broker API such as Zerodha Kite Connect or Angel One SmartAPI to add it.")
