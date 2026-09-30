"""Angel One SmartAPI data layer (read-only: no orders are ever placed)."""
import datetime as dt
import time

import pandas as pd
import pyotp
import requests
import streamlit as st
from SmartApi import SmartConnect

MASTER = "https://margincalculator.angelbroking.com/OpenAPI_Files/OpenAPIScripMaster.json"
INDEX = {"^NSEI": ("NSE", "99926000"), "^NSEBANK": ("NSE", "99926009"), "^BSESN": ("BSE", "99919000")}


@st.cache_resource(ttl=6 * 3600)
def client():
    s = st.secrets
    api = SmartConnect(s["ANGEL_API_KEY"])
    r = api.generateSession(s["ANGEL_CLIENT_ID"], s["ANGEL_PIN"], pyotp.TOTP(s["ANGEL_TOTP_SECRET"]).now())
    if not r.get("status"):
        raise RuntimeError(f"Angel One login failed: {r.get('message')}")
    return api


@st.cache_data(ttl=86400, show_spinner="Loading instrument list...")
def master():
    df = pd.DataFrame(requests.get(MASTER, timeout=120).json())
    eq = df[(df.exch_seg == "NSE") & df.symbol.str.endswith("-EQ")][["symbol", "token"]]
    op = df[(df.exch_seg == "NFO") & (df.instrumenttype == "OPTIDX") & df.name.isin(["NIFTY", "BANKNIFTY"])].copy()
    op["strike"] = op["strike"].astype(float) / 100
    op["exp"] = pd.to_datetime(op["expiry"], format="%d%b%Y")
    return eq, op[["name", "symbol", "token", "strike", "lotsize", "exp"]]


@st.cache_data(ttl=120, show_spinner=False)
def load(tickers, period):
    """Daily candles in the same shape as the yfinance loader (today's candle is live)."""
    api, (eq, _) = client(), master()
    end = dt.datetime.now()
    start = end - dt.timedelta(days={"1y": 365, "2y": 730, "5y": 1825}[period])
    out = {}
    for t in tickers:
        if t in INDEX:
            exch, tok = INDEX[t]
        else:
            row = eq[eq.symbol == t.replace(".NS", "") + "-EQ"]
            if row.empty:
                continue
            exch, tok = "NSE", row.iloc[0]["token"]
        try:
            r = api.getCandleData({"exchange": exch, "symboltoken": tok, "interval": "ONE_DAY",
                                   "fromdate": start.strftime("%Y-%m-%d %H:%M"),
                                   "todate": end.strftime("%Y-%m-%d %H:%M")})
            rows = r.get("data") or []
            if len(rows) > 60:
                df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                out[t] = df.set_index("Date")
        except Exception:
            pass
        time.sleep(0.4)  # stay under the historical-data rate limit
    return out


def option_idea(name, spot, direction, n=5):
    """Nearest-expiry ATM contract for CALL/PUT plus PCR from OI of ATM +/- n strikes."""
    api, (_, op) = client(), master()
    o = op[(op.name == name) & (op.exp >= pd.Timestamp.today().normalize())]
    exp = o.exp.min()
    o = o[o.exp == exp]
    strikes = sorted(o.strike.unique())
    atm = min(strikes, key=lambda k: abs(k - spot))
    i = strikes.index(atm)
    near = o[o.strike.isin(strikes[max(0, i - n): i + n + 1])]
    r = api.getMarketData("FULL", {"NFO": near.token.tolist()})
    q = pd.DataFrame(r["data"]["fetched"]).merge(near, left_on="symbolToken", right_on="token")
    q["type"] = q["symbol"].str[-2:]
    ce, pe = q[q.type == "CE"]["opnInterest"].sum(), q[q.type == "PE"]["opnInterest"].sum()
    pick = q[(q.type == ("CE" if direction == "CALL" else "PE")) & (q.strike == atm)].iloc[0]
    return dict(expiry=exp.date(), atm=atm, pcr=(pe / ce if ce else None), symbol=pick["symbol"],
                ltp=float(pick["ltp"]), lot=int(pick["lotsize"]))
