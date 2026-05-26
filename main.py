import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import warnings
import logging
import os
from typing import Tuple, Dict, Any
import json

# --- Konfiguration & Logging ---
st.set_page_config(
    page_title="🚀 AktienGod Pro - RSI-Divergenz & Swing Trading",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# --- Datenverwaltung ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")


def load_settings() -> dict:
    """Lädt persistente Einstellungen oder gibt Defaults zurück"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"symbols": "TSLA,NVDA,AMD,MRK,BYDDY", "period": "3y", "interval": "1d", "order": 5, "capital": 2000}


def save_settings(symbols: str) -> None:
    """Speichert die Symbole persistent"""
    settings = load_settings()
    settings["symbols"] = symbols
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)


@st.cache_data(ttl=3600)
def get_ticker_info(symbol: str) -> Dict[str, Any]:
    """Holt Währungsinformationen für das Symbol"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {'currency': info.get('currency', 'USD'), 'longName': info.get('longName', symbol)}
    except Exception as e:
        logger.warning(f"Fehler beim Abrufen von Info für {symbol}: {e}")
        return {'currency': 'USD', 'longName': symbol}


@st.cache_data(ttl=3600)
def load_cached_data(symbol: str, max_history: bool = False) -> pd.DataFrame:
    """Lädt Daten aus Parquet-Datei, falls vorhanden"""
    filename = os.path.join(DATA_DIR, f"{symbol}_data.parquet")
    try:
        if os.path.exists(filename):
            df = pd.read_parquet(filename)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            return df
    except Exception as e:
        logger.warning(f"Fehler beim Laden von {filename}: {e}")
    return None


def save_data_to_parquet(df: pd.DataFrame, symbol: str) -> None:
    """Speichert Daten persistent in Parquet-Datei"""
    filename = os.path.join(DATA_DIR, f"{symbol}_data.parquet")
    try:
        df.to_parquet(filename, index=True)
        logger.info(f"Daten für {symbol} gespeichert in {filename}")
    except Exception as e:
        logger.error(f"Fehler beim Speichern von {filename}: {e}")


@st.cache_data(ttl=3600)
def download_and_update_data(symbol: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """Lädt Daten von yfinance und aktualisiert sie"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365) if period != "max" else None

    df_existing = load_cached_data(symbol)

    if df_existing is not None:
        if df_existing.index[-1] >= end_date - timedelta(days=1):
            return df_existing
        try:
            new_data = yf.download(symbol, start=start_date, end=end_date, interval=interval,
                                   period="max" if period == "max" else None)
            if not new_data.empty:
                combined_df = pd.concat([df_existing, new_data]).drop_duplicates()
                save_data_to_parquet(combined_df, symbol)
                return combined_df
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren von {symbol}: {e}")
            if not df_existing.empty:
                return df_existing

    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval,
                         period="max" if period == "max" else None)
        save_data_to_parquet(df, symbol)
        return df
    except Exception as e:
        logger.error(f"Fehler beim Download von {symbol}: {e}")
        if df_existing is not None:
            return df_existing
        raise ValueError(f"Keine Daten für {symbol} verfügbar")


# --- Styling ---
STYLESHEET = """
<style>
    .main-header { font-size: 36px; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 20px; }
    .subheader { font-size: 24px; color: #ff9900; text-align: center; margin-bottom: 30px; }
    .section-header { font-size: 20px; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 5px; margin-top: 30px; }
    .signal-box { border-radius: 10px; padding: 15px; margin: 10px 0; text-align: center; font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .buy-signal { background-color: #d4edda; color: #155724; border: 1px solid #28a745; }
    .sell-signal { background-color: #f8d7da; color: #721c24; border: 1px solid #dc3545; }
    .hold-signal { background-color: #fff3cd; color: #856404; border: 1px solid #ffc107; }
    .divergence-signal { background-color: #e2e3e5; color: #383d40; border: 1px solid #6c757d; }
    .success-message { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin-top: 10px; }
</style>
"""
st.markdown(STYLESHEET, unsafe_allow_html=True)

# --- Hauptseite ---
st.markdown('<h1 class="main-header">🚀 AktienGod Pro - RSI-Divergenz & Swing Trading</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Professionelle technische Analyse mit automatisierter Divergenz-Erkennung</p>',
            unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Einstellungen")
    settings = load_settings()
    symbols_input = st.text_input("Aktien-Symbole (getrennt durch Komma)", value=settings["symbols"],
                                  key="symbol_input")
    save_settings(symbols_input)
    symbol_list = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    # 🕰️ Travel Back in Time Slicer
    default_end = datetime.now()
    default_start = default_end - timedelta(days=365)
    date_range = st.date_input(
        "📅 Zeitraum wählen (Travel Back in Time)",
        value=(default_start.date(), default_end.date()),
        min_value=datetime(2015, 1, 1).date(),
        max_value=datetime.now().date(),
        key="date_slicer"
    )
    start_date, end_date = date_range

    period = st.selectbox("Zeitraum", ["3y", "2y", "1y", "6m", "3m"],
                          index=["3y", "2y", "1y", "6m", "3m"].index(settings.get("period", "3y")))
    interval = st.selectbox("Intervall", ["1d", "1wk", "1mo"],
                            index=["1d", "1wk", "1mo"].index(settings.get("interval", "1d")))
    order = st.slider("Divergenz-Order", 1, 10, int(settings.get("order", 5)))
    initial_capital = st.number_input("Startkapital", value=int(settings.get("capital", 2000)), min_value=100)
    show_divergences = st.checkbox("Divergenzen anzeigen", value=True)
    show_signals = st.checkbox("Signale anzeigen", value=True)


# --- Funktionen ---
def clean_and_flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' nicht gefunden")

    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['BB_Mid'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['BB_Std'] = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()
    df.drop(columns=['TR'], inplace=True)
    return df


def detect_rsi_divergence(df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    df = df.copy()
    df['Bullish_Div'] = False
    df['Bearish_Div'] = False
    required_columns = ['Low', 'High', 'RSI']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' nicht gefunden für Divergenz-Erkennung")

    try:
        clean_df = df.dropna(subset=['Low', 'High', 'RSI'])
        low_pivot_idx = argrelextrema(clean_df.Low.values, np.less_equal, order=order)[0]
        high_pivot_idx = argrelextrema(clean_df.High.values, np.greater_equal, order=order)[0]
        df.loc[clean_df.index[low_pivot_idx], 'Low_Pivot'] = clean_df.iloc[low_pivot_idx]['Low']
        df.loc[clean_df.index[high_pivot_idx], 'High_Pivot'] = clean_df.iloc[high_pivot_idx]['High']
    except Exception as e:
        logger.warning(f"Fehler bei Pivot-Erkennung: {e}")
        return df

    low_pivots = df.dropna(subset=['Low_Pivot'])
    high_pivots = df.dropna(subset=['High_Pivot'])
    for i in range(1, len(low_pivots)):
        curr, prev = low_pivots.iloc[i], low_pivots.iloc[i - 1]
        if curr['Low'] < prev['Low'] and curr['RSI'] > prev['RSI']:
            df.loc[curr.name, 'Bullish_Div'] = True
    for i in range(1, len(high_pivots)):
        curr, prev = high_pivots.iloc[i], high_pivots.iloc[i - 1]
        if curr['High'] > prev['High'] and curr['RSI'] < prev['RSI']:
            df.loc[curr.name, 'Bearish_Div'] = True
    return df


def find_setups(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ['RSI', 'Close', 'BB_Lower', 'BB_Upper']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' nicht gefunden für Signal-Erkennung")
    df['Buy_Signal'] = (df['RSI'] < 35) & (df['Close'] <= df['BB_Lower'])
    df['Sell_Signal'] = (df['RSI'] > 65) & (df['Close'] >= df['BB_Upper'])
    df['Hold_Signal'] = (~df['Buy_Signal']) & (~df['Sell_Signal']) & (df['RSI'] > 40) & (df['RSI'] < 60)
    return df


def get_beginner_explanation(stats: dict) -> str:
    signal = stats['current_signal']
    rsi = stats['rsi']
    price = stats['price']
    currency = stats['currency']
    explanations = {
        "KAUF-SETUP": f"🟢 **Kauf-Signal erkannt!** Der Kurs ({price:.2f} {currency}) hat das untere Bollinger-Band durchbrochen, was oft auf Überverkauftheit hindeutet. Gleichzeitig ist der RSI ({rsi:.1f}) unter 35 gefallen, was bedeutet, dass der Abverkauf wahrscheinlich kurzfristig erschöpft ist. Historisch gesehen ist dies oft ein guter Zeitpunkt für eine Gegenposition.",
        "VERKAUF-SETUP": f"🔴 **Verkauf-Signal erkannt!** Der Kurs ({price:.2f} {currency}) ist über das obere Bollinger-Band gestiegen, was auf Überkauftheit schließen lässt. Der RSI ({rsi:.1f}) ist über 65, was zeigt, dass der Aufschwung stark, aber möglicherweise kurzfristig überhitzt ist. Vorsicht vor einer kurzfristigen Korrektur.",
        "BULLISCHE DIVERGENZ": f"🐂 **Bullische Divergenz!** Der Aktienkurs bildet ein neues Tief, aber der RSI-Oszillator steigt an. Das ist wie ein Auto, das bremst, bevor es die Kurve nimmt: Der Abwärtstrend verliert an Kraft, obwohl der Preis noch fällt. Oft folgt hier eine Trendwende nach oben.",
        "BÄRISCHE DIVERGENZ": f"🐻 **Bärische Divergenz!** Der Kurs erreicht ein neues Hoch, aber der RSI fällt. Das signalisiert, dass der Aufwärtstrend innerlich schwächelt. Der Markt läuft auf  Leerlaufund eine Korrektur nach unten ist wahrscheinlich.",
        "HALT (STABIL)": f"⚪ **Neutral / Halte-Zone.** Der Markt befindet sich in einer Konsolidierungsphase. Der RSI ({rsi:.1f}) ist im stabilen Mittelfeld, und der Kurs bewegt sich innerhalb der normalen Volatilität. Hier wartet man am besten auf ein klares Breakout-Signal.",
    "Neutral": f"📊 **Kein klares Signal.** Die technischen Indikatoren zeigen derzeit keine extreme Über- oder Unterversorgung. Der Markt sammelt sich. Geduld ist hier die beste Strategie."
    }
    return explanations.get(signal, explanations["Neutral"])


def plot_swing_analysis(df: pd.DataFrame, symbol: str, show_divergences: bool = True,
                        show_signals: bool = True) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f'Preis: {symbol}', 'Volumen', 'RSI'),
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]])
    fig.add_trace(
        go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Kurs',
                       increasing_line_color='green', decreasing_line_color='red'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='blue', width=1), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='red', width=2), name='EMA 200'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='cyan', dash='dot'), name='VWAP'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(173, 216, 230, 0.2)'), name='BB Oben'),
                  row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(173, 216, 230, 0.2)'), name='BB Unten'), row=1,
        col=1)
    if show_divergences:
        bull_divs = df[df['Bullish_Div']]
        bear_divs = df[df['Bearish_Div']]
        if not bull_divs.empty:
            fig.add_trace(go.Scatter(x=bull_divs.index, y=bull_divs['Low'] * 0.98, mode='markers',
                                     marker=dict(symbol='triangle-up', size=12, color='cyan'), name='Bull. Divergenz'),
                          row=1, col=1)
        if not bear_divs.empty:
            fig.add_trace(go.Scatter(x=bear_divs.index, y=bear_divs['High'] * 1.02, mode='markers',
                                     marker=dict(symbol='triangle-down', size=12, color='magenta'),
                                     name='Bear. Divergenz'), row=1, col=1)
    if show_signals:
        buy_sigs = df[df['Buy_Signal']]
        sell_sigs = df[df['Sell_Signal']]
        if not buy_sigs.empty:
            fig.add_trace(go.Scatter(x=buy_sigs.index, y=buy_sigs['Low'] * 0.96, mode='markers',
                                     marker=dict(symbol='star', size=10, color='lime'), name='Kauf-Signal'), row=1,
                          col=1)
        if not sell_sigs.empty:
            fig.add_trace(go.Scatter(x=sell_sigs.index, y=sell_sigs['High'] * 1.04, mode='markers',
                                     marker=dict(symbol='star', size=10, color='red'), name='Verkauf-Signal'), row=1,
                          col=1)
    colors = ['red' if row['Open'] > row['Close'] else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volumen'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='gold'), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.update_layout(height=1000, template='plotly_dark', xaxis_rangeslider_visible=False,
                      title=f'🚀 AktienGod Pro: {symbol} - Technische Analyse')
    return fig


def run_analysis(symbol: str, period: str = "3y", interval: str = "1d", start_date=None, end_date=None) -> Tuple[
    pd.DataFrame, Dict[str, Any]]:
    try:
        df = download_and_update_data(symbol, period, interval)
        if df.empty:
            raise ValueError(f"Keine Daten für {symbol} gefunden")
        df = clean_and_flatten_df(df)
        if start_date and end_date:
            df = df.loc[start_date:end_date]
            if len(df) < 200:
                st.warning(
                    f"⚠️ Der gewählte Zeitraum für {symbol} ist zu kurz (<200 Tage) für zuverlässige Langzeit-Indikatoren (z.B. EMA 200).")
        df = calculate_indicators(df)
        df = detect_rsi_divergence(df, order=5)
        df = find_setups(df)
        last_row = df.iloc[-1]
        current_signal = "Neutral"
        if last_row['Bullish_Div']:
            current_signal = "BULLISCHE DIVERGENZ"
        elif last_row['Bearish_Div']:
            current_signal = "BÄRISCHE DIVERGENZ"
        elif last_row['Buy_Signal']:
            current_signal = "KAUF-SETUP"
        elif last_row['Sell_Signal']:
            current_signal = "VERKAUF-SETUP"
        elif last_row['Hold_Signal']:
            current_signal = "HALT (STABIL)"
        ticker_info = get_ticker_info(symbol)
        stats = {
            'rsi': float(last_row['RSI']), 'price': float(last_row['Close']),
            'ema_20': float(last_row['EMA_20']), 'ema_50': float(last_row['EMA_50']),
            'ema_200': float(last_row['EMA_200']), 'bb_upper': float(last_row['BB_Upper']),
            'bb_lower': float(last_row['BB_Lower']), 'volume': float(last_row['Volume']),
            'current_signal': current_signal, 'currency': ticker_info['currency'], 'name': ticker_info['longName']
        }
        return df, stats
    except Exception as e:
        logger.error(f"Fehler bei Analyse von {symbol}: {e}")
        raise


def run_backtest(df_input: pd.DataFrame, initial_capital: float = 2000) -> Dict[str, Any]:
    df_bt = df_input.copy()
    position = 0
    trades = []
    equity = [initial_capital]
    for i in range(len(df_bt)):
        current_close = df_bt['Close'].iloc[i]
        if position == 0 and df_bt['Buy_Signal'].iloc[i]:
            position = 1
            entry_price = current_close
            trades.append({'type': 'BUY', 'date': df_bt.index[i], 'price': entry_price,
                           'quantity': initial_capital / entry_price})
        elif position == 1 and df_bt['Sell_Signal'].iloc[i]:
            position = 0
            exit_price = current_close
            profit = (exit_price - entry_price) / entry_price
            trades.append({'type': 'SELL', 'date': df_bt.index[i], 'price': exit_price, 'profit': profit})
        current_equity = initial_capital if position == 0 else (
            trades[-1]['quantity'] * current_close if trades else initial_capital)
        equity.append(current_equity)
    if not trades:
        return {'trades': [], 'total_return': 0, 'win_rate': 0, 'avg_profit': 0, 'num_trades': 0,
                'final_equity': initial_capital}
    results = [t['profit'] for t in trades if 'profit' in t]
    total_return = np.prod([1 + r for r in results]) - 1 if results else 0
    win_rate = len([r for r in results if r > 0]) / len(results) if results else 0
    avg_profit = np.mean(results) if results else 0
    return {'trades': trades, 'total_return': total_return, 'win_rate': win_rate, 'avg_profit': avg_profit,
            'num_trades': len(results), 'final_equity': equity[-1]}


def plot_equity_curve(df_bt: pd.DataFrame, initial_capital: float = 2000) -> go.Figure:
    df_ec = df_bt.copy()
    cash = initial_capital
    position = 0
    equity = []
    for i in range(len(df_ec)):
        current_close = df_ec['Close'].iloc[i]
        if position == 0 and df_ec['Buy_Signal'].iloc[i]:
            position = cash / current_close
            cash = 0
        elif position > 0 and df_ec['Sell_Signal'].iloc[i]:
            cash = position * current_close
            position = 0
        equity.append(cash + (position * current_close))
    df_ec['Equity'] = equity
    df_ec['Buy_and_Hold'] = initial_capital * (df_ec['Close'] / df_ec['Close'].iloc[0])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_ec.index, y=df_ec['Equity'], name='Strategie (Equity)', line=dict(color='lime', width=2)))
    fig.add_trace(
        go.Scatter(x=df_ec.index, y=df_ec['Buy_and_Hold'], name='Buy & Hold', line=dict(color='gray', dash='dash')))
    fig.update_layout(
        title=f'📈 Equity Kurve: {df_ec.index[0].strftime("%Y-%m-%d")} bis {df_ec.index[-1].strftime("%Y-%m-%d")}',
        xaxis_title='Datum', yaxis_title='Depotwert', template='plotly_dark', height=500)
    return fig


def get_trade_list(df_input: pd.DataFrame) -> pd.DataFrame:
    df_bt = df_input.copy()
    position = 0
    trades = []
    for i in range(len(df_bt)):
        current_close = df_bt['Close'].iloc[i]
        if position == 0 and df_bt['Buy_Signal'].iloc[i]:
            position = 1
            trades.append({'Aktion': '🟢 KAUF', 'Datum': df_bt.index[i].strftime('%Y-%m-%d'),
                           'Preis': round(float(current_close), 2)})
        elif position == 1 and df_bt['Sell_Signal'].iloc[i]:
            position = 0
            profit = (current_close - trades[-1]['Preis']) / trades[-1]['Preis']
            trades.append({'Aktion': '🔴 VERKAUF', 'Datum': df_bt.index[i].strftime('%Y-%m-%d'),
                           'Preis': round(float(current_close), 2), 'Profit': f'{profit:.2%}'})
    return pd.DataFrame(trades)


def generate_dashboard_summary(symbol_list: list) -> pd.DataFrame:
    """Erstellt die Benchmark-Zusammenfassungstabelle"""
    summary_data = []
    for sym in symbol_list:
        try:
            df_full = download_and_update_data(sym, period="max", interval="1d")
            df_full = clean_and_flatten_df(df_full)
            if df_full.empty: continue
            current_price = float(df_full['Close'].iloc[-1])
            currency = get_ticker_info(sym)['currency']

            returns = {}
            for y in [1, 2, 3, 5, 10]:
                days = y * 252
                if len(df_full) > days:
                    returns[f"{y}J"] = (current_price / df_full['Close'].iloc[-days]) - 1
                else:
                    returns[f"{y}J"] = 0.0

            df_analysis = df_full.tail(250)
            df_analysis = calculate_indicators(df_analysis)
            df_analysis = find_setups(df_analysis)
            signal = "Neutral"
            if df_analysis.iloc[-1]['Bullish_Div']:
                signal = "Bull. Divergenz"
            elif df_analysis.iloc[-1]['Bearish_Div']:
                signal = "Bär. Divergenz"
            elif df_analysis.iloc[-1]['Buy_Signal']:
                signal = "Kauf"
            elif df_analysis.iloc[-1]['Sell_Signal']:
                signal = "Verkauf"
            elif df_analysis.iloc[-1]['Hold_Signal']:
                signal = "Halten"

            recommendation = "🟢 KAUFEN" if signal == "Kauf" else ("🔴 VERKAUFEN" if signal == "Verkauf" else "⚪ HALTEN")

            summary_data.append({
                "Symbol": sym,
                "Aktueller Kurs": f"{current_price:.2f} {currency}",
                "Signal Heute": signal,
                "Empfehlung": recommendation,
                "1J Rendite": f"{returns['1J']:.2%}",
                "2J Rendite": f"{returns['2J']:.2%}",
                "3J Rendite": f"{returns['3J']:.2%}",
                "5J Rendite": f"{returns['5J']:.2%}",
                "10J Rendite": f"{returns['10J']:.2%}"
            })
        except Exception as e:
            logger.error(f"Fehler bei Dashboard-Zusammenfassung für {sym}: {e}")
            continue
    return pd.DataFrame(summary_data)


# --- Haupt-App ---
if not symbol_list:
    st.error("❌ Bitte geben Sie mindestens ein Aktiensymbol ein.")
else:
    # 📊 Dashboard Zusammenfassung
    st.markdown('<h2 class="section-header">📈 Dashboard Übersicht & Benchmark</h2>', unsafe_allow_html=True)
    with st.spinner("Berechne Benchmark & historische Renditen..."):
        summary_df = generate_dashboard_summary(symbol_list)
    if not summary_df.empty:
        st.dataframe(summary_df.style.highlight_max(axis=0,
                                                    subset=['1J Rendite', '2J Rendite', '3J Rendite', '5J Rendite',
                                                            '10J Rendite'], props='background-color:#d4edda'),
                     width="stretch")
    st.markdown("---")

    for symbol in symbol_list:
        with st.container():
            st.markdown(f'<h2 class="section-header">📊 Analyse für {symbol}</h2>', unsafe_allow_html=True)
            # ✅ FIX: .date() entfernt, da st.date_input bereits date-Objekte liefert
            with st.spinner(f"Lade und analysiere {symbol} für Zeitraum {start_date} bis {end_date}..."):
                try:
                    df, stats = run_analysis(symbol, period, interval, start_date, end_date)

                    st.markdown('<div class="signal-box">', unsafe_allow_html=True)
                    signal_msg = f"📊 Aktueller Status: {stats['current_signal']}"
                    if stats['current_signal'] == "KAUF-SETUP":
                        st.markdown(f'<p class="buy-signal">{signal_msg}</p>', unsafe_allow_html=True)
                    elif stats['current_signal'] == "VERKAUF-SETUP":
                        st.markdown(f'<p class="sell-signal">{signal_msg}</p>', unsafe_allow_html=True)
                    elif stats['current_signal'] == "BULLISCHE DIVERGENZ":
                        st.markdown(f'<p class="divergence-signal">{signal_msg}</p>', unsafe_allow_html=True)
                    elif stats['current_signal'] == "BÄRISCHE DIVERGENZ":
                        st.markdown(f'<p class="divergence-signal">{signal_msg}</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="hold-signal">{signal_msg}</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # 🎓 Einsteiger-Erklärung
                    with st.expander("🎓 Setup-Erklärung für Einsteiger (Aufklappen)", expanded=False):
                        st.markdown(get_beginner_explanation(stats))

                    fig = plot_swing_analysis(df, symbol, show_divergences, show_signals)
                    st.plotly_chart(fig, width="stretch")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("📈 RSI", f"{stats['rsi']:.2f}")
                    col2.metric("💰 Aktueller Kurs", f"{stats['price']:.2f} {stats['currency']}")
                    col3.metric("EMA 20", f"{stats['ema_20']:.2f}")
                    col4.metric("EMA 50", f"{stats['ema_50']:.2f}")

                    backtest_results = run_backtest(df, initial_capital)
                    st.subheader("📈 Backtest Ergebnisse")
                    if backtest_results['num_trades'] > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Anzahl Trades", backtest_results['num_trades'])
                        col2.metric("Gesamtrendite", f"{backtest_results['total_return']:.2%}")
                        col3.metric("Win-Rate", f"{backtest_results['win_rate']:.2%}")
                        col4.metric("Avg. Profit", f"{backtest_results['avg_profit']:.2%}")
                        st.subheader("📉 Equity Kurve")
                        equity_fig = plot_equity_curve(df, initial_capital)
                        st.plotly_chart(equity_fig, width="stretch")
                        trade_history = get_trade_list(df)
                        if not trade_history.empty:
                            st.subheader("📋 Trade Historie")
                            st.dataframe(trade_history, width="stretch")
                    else:
                        st.info("ℹ️ Keine Trades im gewählten Zeitraum gefunden.")
                    # ✅ FIX: .date() entfernt
                    st.caption(
                        f"📊 Stand: {datetime.now().strftime('%d.%m.%Y %H:%M')} für {symbol} | Zeitraum: {start_date} bis {end_date}")
                except Exception as e:
                    st.error(f"❌ Fehler bei Analyse von {symbol}: {str(e)}")
                    logger.error(f"Fehler bei Analyse von {symbol}: {e}")

    st.success(f"✅ Analyse abgeschlossen! {len(symbol_list)} Symbole wurden verarbeitet.")
    st.markdown('<div class="success-message">', unsafe_allow_html=True)
    st.markdown('📈 **Professionelle Aktienanalyse abgeschlossen**')
    st.markdown('</div>', unsafe_allow_html=True)
