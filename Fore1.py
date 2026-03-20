import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import date

st.set_page_config(layout="wide", page_title="Global Economy Dashboard – Forecast", page_icon="🔮")

st.markdown("""
<style>
    section[data-testid="stSidebar"] { background-color: #ffe066 !important; }
    details summary {
        background-color: #ffcc66 !important; color: #333 !important;
        border-radius: 6px !important; padding: 8px 12px !important; font-weight: 500;
    }
    details[open] summary { background-color: #ffcc66 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Global Economy Indicators – Technical Analysis Dashboard")

# ── Nawigacja ────────────────────────────────────────────────────────────────
col_btn, col_info, _ = st.columns([2, 4, 4])
with col_btn:
    if st.button("📊 Tech Analytical Desktop", use_container_width=True, type="secondary"):
        st.switch_page("pages/Tech_Analytical_Desktop.py")
with col_info:
    st.info("Model NN trenowany na 1000 ostatnich sesjach | Prognoza 5 dni roboczych do przodu")
st.markdown("---")

# ── Tickery ──────────────────────────────────────────────────────────────────
FORE_TICKERS = {
    "^GSPC":    "SP_500",
    "^HSI":     "HANG SENG INDEX",
    "CL=F":     "Crude Oil",
    "GC=F":     "Gold",
    "^N225":    "Nikkei 225",
    "EURUSD=X": "EUR_USD",
    "JPY=X":    "USD_JPY",
}

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Ustawienia prognozy")
    hist_n  = st.slider("Sesje historyczne na wykresie", 100, 600, 250, 25)
    retrain = st.button("Przetrenuj modele od nowa", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Model: LSTM (64/32 jednostek)\nOkno: 60 sesji | Horyzont: 5 dni roboczych\nDane: Yahoo Finance")
    st.markdown("---")
    st.caption("© 2026 Michał Leśniewski")

# ── Funkcje pomocnicze ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_hist(ticker: str, n: int) -> pd.DataFrame:
    df = yf.download(ticker, period="6y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    return df[["Date", "Close"]].dropna().tail(n).reset_index(drop=True)

@st.cache_data(ttl=86400, show_spinner=False)
def get_forecast(ticker: str) -> pd.DataFrame:
    from D5_LSTM_fore import forecast_ticker
    return forecast_ticker(ticker, past=1000, retrain=False)

def get_forecast_retrain(ticker: str) -> pd.DataFrame:
    from D5_LSTM_fore import forecast_ticker
    return forecast_ticker(ticker, past=1000, retrain=True)

def build_chart(hist, fore, name, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Close"], name="Close",
        line=dict(color="#1f77b4", width=1.8),
        hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:,.4f}<extra></extra>"
    ))
    if fore is not None and not fore.empty:
        last_date  = hist["Date"].iloc[-1]
        last_price = float(hist["Close"].iloc[-1])
        fore_dates = [last_date] + list(fore["Date"])
        fore_vals  = [last_price] + list(fore["Forecast"])
        fig.add_trace(go.Scatter(
            x=fore_dates, y=fore_vals, name="Prognoza NN D+5",
            line=dict(color="#ff7f0e", width=2.5, dash="dash"),
            mode="lines+markers",
            marker=dict(size=9, symbol="circle", color="#ff7f0e",
                        line=dict(color="white", width=1.5)),
            hovertemplate="%{x}<br>Prognoza: %{y:,.4f}<extra></extra>"
        ))
        x_str = pd.Timestamp(last_date).strftime("%Y-%m-%d")
        fig.add_shape(type="line", x0=x_str, x1=x_str, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(dash="dot", color="gray", width=1.2))
        fig.add_annotation(x=x_str, y=1, xref="x", yref="paper",
                           text="Dziś", showarrow=False,
                           font=dict(color="gray", size=11),
                           xanchor="left", yanchor="top"), line_dash="dot", line_color="gray",
                      line_width=1.2, annotation_text="Dziś",
                      annotation_position="top right", annotation_font_color="gray")
    fig.update_layout(
        title=dict(text=f"{name}  ({ticker})", x=0.0, font=dict(size=15)),
        height=400, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=55, b=40),
        xaxis_rangeslider_visible=False,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"),
    )
    return fig

# ── Zakładki dla każdego tickera ─────────────────────────────────────────────
tabs = st.tabs(list(FORE_TICKERS.values()))

for i, (ticker, name) in enumerate(FORE_TICKERS.items()):
    with tabs[i]:
        hist = get_hist(ticker, hist_n)
        last_price = float(hist["Close"].iloc[-1]) if not hist.empty else float("nan")
        prev_price = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_price
        delta_pct  = (last_price - prev_price) / prev_price * 100 if prev_price else 0.0

        col_m, col_c = st.columns([1, 4])

        with col_m:
            st.metric(label=name, value=f"{last_price:,.4f}", delta=f"{delta_pct:+.2f}%")
            st.markdown("**Prognoza D+1 → D+5:**")
            fore = None
            if retrain:
                with st.spinner(f"Trening NN dla {name}..."):
                    try:
                        get_forecast.clear()
                        fore = get_forecast_retrain(ticker)
                    except Exception as e:
                        st.error(f"Błąd: {e}")
            else:
                with st.spinner(f"Ładuję model {name}..."):
                    try:
                        fore = get_forecast(ticker)
                    except Exception as e:
                        st.error(f"Błąd: {e}")
            if fore is not None:
                for _, row in fore.iterrows():
                    diff = row["Forecast"] - last_price
                    sign = "+" if diff >= 0 else ""
                    color = "green" if diff >= 0 else "red"
                    st.markdown(
                        f"`{row['Date']}` &nbsp; **{row['Forecast']:,.4f}** "
                        f"<span style='color:{color};font-size:12px'>({sign}{diff:,.4f})</span>",
                        unsafe_allow_html=True
                    )

        with col_c:
            fig = build_chart(hist, fore, name, ticker)
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.markdown("---")
st.caption("Dane © Yahoo Finance | Model NN D+5 | streamlit · plotly · yfinance")
