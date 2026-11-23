import os
import json
import time
import threading
import traceback
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import yfinance as yf
import growwapi
from growwapi import GrowwAPI

import datetime
import pytz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

from engine.market_validator import MarketValidator
from engine.market_regime_engine import MarketRegimeEngine
from engine.momentum_ignition_engine import MomentumIgnitionEngine
from engine.missed_move_detector import MissedMoveDetector
from engine.strategy_learning_engine import StrategyLearningEngine
from engine.nse_data_engine import NSEDataEngine
from engine.fno_momentum_filter import FNOMomentumFilter

# Initialize NSE Institutional Data Engine
nse_engine = NSEDataEngine()

def is_market_open():
    """Returns True if the Indian market (NSE/BSE) is open right now, strictly respecting weekends and holidays"""
    return MarketValidator.is_market_open()

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

# Global Screener State
screener_state = {
    "connected": False,
    "user_profile": None,
    "margin": None,
    "fno_universe": [],
    "sectors": {},
    "bullish_momentum_watchlist": [],
    "bearish_momentum_watchlist": [],
    "last_screener_update": 0,
    "raw_market_cache": {},        # 5m OHLCV and raw technicals
    "daily_market_cache": {},      # 1mo 1d OHLC for true Wilder Daily ATR14 calculation
    "enriched_signals_cache": {},  # Full 15-strategy + AI enriched results with timestamps
    "active_signals_map": {},      # Real-time normalized cache of active trade signal IDs
    "symbol_names": {},            # Dynamic symbol to company name map from Groww API
    "stock_sector_cache": {}       # Dynamic sector mapping cache for all discovered stocks
}

# Comprehensive Groww & NSE Sector Indices (All 18 Tracked Sectors)
GROWW_SECTOR_INDICES = {
    "NIFTY AUTO": "NIFTYAUTO",
    "NIFTY PSU BANK": "NIFTYPSUBANK",
    "NIFTY FMCG": "NIFTYFMCG",
    "NIFTY CONSUMER DURABLES": "NIFTYCONSUMERDURABLES",
    "NIFTY BANK": "NIFTYBANK",
    "NIFTY PRIVATE BANK": "NIFTYPVTBANK",
    "NIFTY ENERGY": "NIFTYENERGY",
    "NIFTY OIL AND GAS": "NIFTYOILANDGAS",
    "NIFTY INFRA": "NIFTYINFRA",
    "NIFTY FINANCIAL SERVICES": "NIFTYFIN",
    "NIFTY METAL": "NIFTYMETAL",
    "NIFTY PHARMA": "NIFTYPHARMA",
    "NIFTY HEALTHCARE": "NIFTYHEALTHCARE",
    "NIFTY REALTY": "NIFTYREALTY",
    "NIFTY IT": "NIFTYIT",
    "NIFTY MEDIA": "NIFTYMEDIA",
    "NIFTY INDIA DEFENCE": "NIFTYINDIADEFENCE",
    "NIFTY INDIA RAILWAYS PSU": "NIFTYINDIARAILWAYSPSU"
}

# Mapping to live market index tickers for exact official index returns
INDEX_TICKER_MAP = {
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PSU BANK": "^CNXPSUBANK",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY CONSUMER DURABLES": "^CNXCONSUM",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY PRIVATE BANK": "NIFTY_PVT_BANK.NS",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY OIL AND GAS": "NIFTY_OIL_AND_GAS.NS",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY FINANCIAL SERVICES": "NIFTY_FIN_SERVICE.NS",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY HEALTHCARE": "NIFTY_HEALTHCARE.NS",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY IT": "^CNXIT",
    "NIFTY MEDIA": "^CNXMEDIA"
}

def auto_connect_groww():
    """Auto connect to Groww API using token in .env and fetch all F&O stocks dynamically"""
    api_token = os.getenv("GROWW_API_TOKEN")
    api_secret = os.getenv("GROWW_API_SECRET")
    if not api_token:
        return False
    try:
        token = api_token
        # Try exchange for access token if api_secret is provided; fallback to api_token if already a JWT
        if api_secret:
            try:
                token = GrowwAPI.get_access_token(api_key=api_token, secret=api_secret)
            except Exception:
                token = api_token
        client = GrowwAPI(token)

        try:
            screener_state["user_profile"] = client.get_user_profile(timeout=5)
        except Exception:
            screener_state["user_profile"] = {"ucc": "Active (Token Validated)"}

        try:
            screener_state["margin"] = client.get_available_margin_details(timeout=5)
        except Exception:
            screener_state["margin"] = {}

        # Dynamically discover all tradeable F&O stocks directly from Groww API
        fno_loaded = False
        try:
            df = client.get_all_instruments()
            fno_df = df[df['segment'] == 'FNO']
            INDEX_SYMBOLS = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'BANKEX', 'SENSEX', 'NIFTYNXT50'}
            raw_fno = [
                s for s in fno_df['underlying_symbol'].dropna().unique() 
                if s not in INDEX_SYMBOLS and 'TEST' not in s and not s.isdigit() and len(s) >= 2
            ]
            if raw_fno:
                screener_state["fno_universe"] = sorted(raw_fno)
                fno_loaded = True
                print(f"Dynamically discovered {len(raw_fno)} F&O tradeable stocks from Groww API.")

            # Dynamically extract all official company names from Groww cash segment instruments
            cash_nse = df[(df['exchange'] == 'NSE') & (df['segment'] == 'CASH')].dropna(subset=['trading_symbol', 'name'])
            symbol_to_name = dict(zip(cash_nse['trading_symbol'], cash_nse['name']))
            screener_state["symbol_names"] = symbol_to_name
            # Register dynamic company names into news engine
            news_engine.register_symbol_names(symbol_to_name)
            print(f"Dynamically mapped {len(symbol_to_name)} company names from Groww for real-time news.")
        except Exception as e:
            print("Error loading FNO universe:", e)

        # Connection is only deemed fully successful if F&O instruments loaded or verified
        screener_state["connected"] = fno_loaded
        return fno_loaded
    except Exception as e:
        screener_state["connected"] = False
        return False

def clean_dataframe(df, symbol=None):
    """Flattens MultiIndex columns from yfinance to ensure 1D Series and prevent serialization/dimension errors"""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        if symbol and symbol in df.columns.levels[1]:
            df = df.xs(symbol, axis=1, level=1)
        else:
            df.columns = df.columns.get_level_values(0)
    return df

def classify_stock_sector(sym):
    """
    Dynamically classifies ANY discovered F&O stock into its corresponding NSE Sector Index.
    Uses yfinance metadata (Sector and Industry) with persistent caching.
    ZERO hardcoded stock tickers.
    """
    clean_sym = sym.replace(".NS", "").upper()
    cached = screener_state["stock_sector_cache"].get(clean_sym)
    if cached:
        return cached

    # Query yfinance metadata for the stock
    if clean_sym in {"^NSEI", "NIFTY", "^BSESN", "SENSEX", "^NSEBANK", "BANKNIFTY"}:
        screener_state["stock_sector_cache"][clean_sym] = "NIFTY 50"
        return "NIFTY 50"

    yf_ticker = f"{clean_sym}.NS"
    sector_name = "NIFTY INFRA"  # Default fallback
    try:
        t = yf.Ticker(yf_ticker)
        yf_sec = str(t.info.get("sector", "")).strip().lower()
        yf_ind = str(t.info.get("industry", "")).strip().lower()

        if "private" in yf_ind or "private" in yf_sec:
            sector_name = "NIFTY PRIVATE BANK"
        elif "psu" in yf_ind or "public bank" in yf_ind or "psu" in yf_sec:
            sector_name = "NIFTY PSU BANK"
        elif "bank" in yf_ind or "bank" in yf_sec:
            sector_name = "NIFTY BANK"
        elif "rail" in yf_ind or "rail" in yf_sec or "locomotive" in yf_ind:
            sector_name = "NIFTY INDIA RAILWAYS PSU"
        elif "defence" in yf_ind or "aerospace" in yf_ind or "military" in yf_ind:
            sector_name = "NIFTY INDIA DEFENCE"
        elif "oil" in yf_ind or "gas" in yf_ind or "petroleum" in yf_ind:
            sector_name = "NIFTY OIL AND GAS"
        elif "consumer durable" in yf_ind or "appliances" in yf_ind or "electronics" in yf_ind:
            sector_name = "NIFTY CONSUMER DURABLES"
        elif "media" in yf_sec or "entertainment" in yf_ind or "broadcasting" in yf_ind or "publishing" in yf_ind:
            sector_name = "NIFTY MEDIA"
        elif "healthcare" in yf_sec or "hospital" in yf_ind or "diagnostic" in yf_ind:
            sector_name = "NIFTY HEALTHCARE"
        elif "pharm" in yf_ind or "biotech" in yf_ind or "drug" in yf_ind or "health" in yf_sec:
            sector_name = "NIFTY PHARMA"
        elif "financial" in yf_sec or "capital markets" in yf_ind or "credit" in yf_ind or "insurance" in yf_ind or "asset management" in yf_ind:
            sector_name = "NIFTY FINANCIAL SERVICES"
        elif "technology" in yf_sec or "software" in yf_ind or "it services" in yf_ind:
            sector_name = "NIFTY IT"
        elif "auto" in yf_ind or "auto" in yf_sec or "vehicles" in yf_ind:
            sector_name = "NIFTY AUTO"
        elif "real estate" in yf_sec or "reit" in yf_ind:
            sector_name = "NIFTY REALTY"
        elif "steel" in yf_ind or "metals" in yf_sec or "mining" in yf_sec or "aluminum" in yf_ind or "copper" in yf_ind:
            sector_name = "NIFTY METAL"
        elif "energy" in yf_sec or "power" in yf_ind or "utilities" in yf_sec or "solar" in yf_ind:
            sector_name = "NIFTY ENERGY"
        elif "consumer defensive" in yf_sec or "fmcg" in yf_ind or "food" in yf_ind or "beverage" in yf_ind or "tobacco" in yf_ind:
            sector_name = "NIFTY FMCG"
        elif "industrial" in yf_sec or "engineering" in yf_ind or "infrastructure" in yf_ind or "cement" in yf_ind or "construction" in yf_ind:
            sector_name = "NIFTY INFRA"
        else:
            sector_name = "UNCLASSIFIED"
    except Exception:
        sector_name = "UNCLASSIFIED"

    screener_state["stock_sector_cache"][clean_sym] = sector_name
    return sector_name

# Chartink Explosive Momentum Engine: Evaluates 15M Close, VWAP, EMA 9/21, RSI 14, Volume 1.5x SMA20
def evaluate_explosive_momentum(df, symbol):
    df = clean_dataframe(df, symbol)
    if df is None or df.empty or len(df) < 20:
        return None

    c = df['Close']
    o = df['Open']
    h = df['High']
    l = df['Low']
    v = df['Volume']

    # 1. EMA 9 & EMA 21
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()

    # 2. RSI 14
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, 0.0001))
    rsi = 100 - (100 / (1 + rs))

    # 3. Volume SMA 20
    vol_sma20 = v.rolling(window=20).mean()

    # 4. TRUE SESSION-RESET VWAP (Resets to 0 at 09:15 open of each day)
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
        df_copy.index = pd.to_datetime(df_copy.index)
    df_copy['Date'] = df_copy.index.date
    today_date = df_copy['Date'].iloc[-1]
    today_slice = df_copy[df_copy['Date'] == today_date]
    if today_slice.empty:
        today_slice = df_copy
    today_tp = (today_slice['High'] + today_slice['Low'] + today_slice['Close']) / 3
    today_cum_vol = today_slice['Volume'].cumsum().replace(0, 1)
    session_vwap_series = (today_tp * today_slice['Volume']).cumsum() / today_cum_vol
    curr_vwap = float(session_vwap_series.iloc[-1].item() if hasattr(session_vwap_series.iloc[-1], 'item') else session_vwap_series.iloc[-1])

    curr_c = float(c.iloc[-1].item() if hasattr(c.iloc[-1], 'item') else c.iloc[-1])
    curr_price = curr_c
    curr_o = float(o.iloc[-1].item() if hasattr(o.iloc[-1], 'item') else o.iloc[-1])
    curr_h = float(h.iloc[-1].item() if hasattr(h.iloc[-1], 'item') else h.iloc[-1])
    curr_l = float(l.iloc[-1].item() if hasattr(l.iloc[-1], 'item') else l.iloc[-1])
    curr_v = float(v.iloc[-1].item() if hasattr(v.iloc[-1], 'item') else v.iloc[-1])
    curr_ema9 = float(ema9.iloc[-1].item() if hasattr(ema9.iloc[-1], 'item') else ema9.iloc[-1])
    curr_ema21 = float(ema21.iloc[-1].item() if hasattr(ema21.iloc[-1], 'item') else ema21.iloc[-1])
    curr_rsi = float(rsi.iloc[-1].item() if hasattr(rsi.iloc[-1], 'item') else rsi.iloc[-1])
    
    last_vol_sma = vol_sma20.iloc[-1]
    is_na_val = bool(pd.isna(last_vol_sma).any()) if hasattr(pd.isna(last_vol_sma), 'any') else bool(pd.isna(last_vol_sma))
    curr_vol_sma = float(last_vol_sma.item() if hasattr(last_vol_sma, 'item') else last_vol_sma) if not is_na_val else curr_v

    vol_multiplier = round(float(curr_v / (curr_vol_sma or 1)), 2)

    # Chartink Condition 1: Early Bullish Momentum (or continuation breakout)
    # Price > VWAP, EMA9 > EMA21, healthy RSI (>= 50), and close >= EMA9 or bullish candle
    is_bullish_chartink = bool(
        curr_c > curr_vwap and
        curr_ema9 >= curr_ema21 and
        curr_rsi >= 50.0 and
        (curr_c >= curr_ema9 or curr_c >= curr_o or (float(c.iloc[-2]) < float(o.iloc[-2]) and curr_c > float(o.iloc[-2])))
    )

    # Chartink Condition 2: Early Bearish Momentum (or continuation breakdown)
    # Price < VWAP, EMA9 < EMA21, weak RSI (<= 50), and close <= EMA9 or bearish candle
    is_bearish_chartink = bool(
        curr_c < curr_vwap and
        curr_ema9 <= curr_ema21 and
        curr_rsi <= 50.0 and
        (curr_c <= curr_ema9 or curr_c <= curr_o or (float(c.iloc[-2]) > float(o.iloc[-2]) and curr_c < float(o.iloc[-2])))
    )

    # 5. SMC Order Block & Liquidity Detection
    swing_highs = []
    swing_lows = []
    for i in range(2, len(df)-2):
        if float(h.iloc[i]) == max([float(x) for x in h.iloc[i-2:i+3]]):
            swing_highs.append(float(h.iloc[i]))
        if float(l.iloc[i]) == min([float(x) for x in l.iloc[i-2:i+3]]):
            swing_lows.append(float(l.iloc[i]))

    bullish_obs = []
    bearish_obs = []
    for i in range(1, len(df)-2):
        if float(c.iloc[i]) < float(o.iloc[i]) and float(c.iloc[i+1]) > float(h.iloc[i]):
            bullish_obs.append({"top": float(h.iloc[i]), "bottom": float(l.iloc[i])})
        if float(c.iloc[i]) > float(o.iloc[i]) and float(c.iloc[i+1]) < float(l.iloc[i]):
            bearish_obs.append({"top": float(h.iloc[i]), "bottom": float(l.iloc[i])})

    nearest_demand_ob = bullish_obs[-1] if bullish_obs else None
    nearest_supply_ob = bearish_obs[-1] if bearish_obs else None

    bsl = float(max(swing_highs[-3:])) if len(swing_highs) >= 3 else float(h.max())
    ssl = float(min(swing_lows[-3:])) if len(swing_lows) >= 3 else float(l.min())

    # Order Block proximity (within 1.2% or inside)
    is_at_demand = nearest_demand_ob and (nearest_demand_ob['bottom'] * 0.995 <= curr_price <= nearest_demand_ob['top'] * 1.012)
    is_at_supply = nearest_supply_ob and (nearest_supply_ob['bottom'] * 0.988 <= curr_price <= nearest_supply_ob['top'] * 1.005)

    # Pullback to EMA9 or VWAP support
    is_pullback_demand = abs(curr_price - curr_ema9) / curr_price <= 0.012 or abs(curr_price - curr_vwap) / curr_price <= 0.015
    is_pullback_supply = abs(curr_price - curr_ema9) / curr_price <= 0.012 or abs(curr_price - curr_vwap) / curr_price <= 0.015

    action = "WAIT FOR ENTRY (NO TRADE)"
    verdict = "PRICE TRADING MID-RANGE"
    confidence = "WAIT"
    sl = None
    tp1 = None
    tp2 = None
    reasoning = ""

    if is_bullish_chartink and is_at_demand:
        action = "BUY / CALL (CE) [HIGH CONFLUENCE: DEMAND OB + MOMENTUM]"
        verdict = "DEMAND ORDER BLOCK REACTION + MOMENTUM ACTIVE"
        confidence = "A+ HIGH CONFLUENCE"
        sl = round(min(curr_l, nearest_demand_ob['bottom']) * 0.995, 2)
        risk = max(curr_price - sl, curr_price * 0.005)
        tp1 = round(curr_price + (risk * 1.5), 2)
        tp2 = round(curr_price + (risk * 2.5), 2)
        reasoning = f"Price > VWAP (₹{curr_vwap:.1f}), EMA9 > EMA21, RSI {curr_rsi:.1f}. Bouncing off Demand Order Block (₹{nearest_demand_ob['bottom']:.1f} - ₹{nearest_demand_ob['top']:.1f}). BUY Call with SL ₹{sl} targeting ₹{tp1} / ₹{tp2}."
    elif is_bullish_chartink and is_pullback_demand:
        action = "BUY / CALL (CE) [PULLBACK TO VWAP/EMA9 SUPPORT]"
        verdict = "BULLISH TREND PULLBACK ENTRY"
        confidence = "A GRADE PULLBACK"
        sl = round(min(curr_vwap, curr_ema9, curr_l) * 0.995, 2)
        risk = max(curr_price - sl, curr_price * 0.005)
        tp1 = round(curr_price + (risk * 1.5), 2)
        tp2 = round(max(bsl, curr_price + (risk * 2.5)), 2)
        reasoning = f"Price (₹{curr_price:.1f}) is riding institutional trend above VWAP (₹{curr_vwap:.1f}), EMA9 > EMA21, RSI {curr_rsi:.1f}. Pullback resting at support. Strong BUY entry with SL ₹{sl} targeting ₹{tp1}."
    elif is_bullish_chartink:
        action = "BUY / CALL (CE) [EXPLOSIVE MOMENTUM RALLY]"
        verdict = "CHARTINK MOMENTUM BREAKOUT"
        confidence = "MOMENTUM CONFIRMED"
        sl = round(curr_ema9 * 0.994, 2)
        risk = max(curr_price - sl, curr_price * 0.005)
        tp1 = round(curr_price + (risk * 1.5), 2)
        tp2 = round(max(bsl, curr_price + (risk * 2.5)), 2)
        reasoning = f"Explosive buying pressure above VWAP (₹{curr_vwap:.1f}) and EMA9 (₹{curr_ema9:.1f}), RSI {curr_rsi:.1f}. Long entry active. SL ₹{sl}, TP1 ₹{tp1}."
    elif is_bearish_chartink and is_at_supply:
        action = "SELL / PUT (PE) [HIGH CONFLUENCE: SUPPLY OB + BREAKDOWN]"
        verdict = "SUPPLY ORDER BLOCK REJECTION + BEARISH BREAKDOWN"
        confidence = "A+ HIGH CONFLUENCE"
        sl = round(max(curr_h, nearest_supply_ob['top']) * 1.005, 2)
        risk = max(sl - curr_price, curr_price * 0.005)
        tp1 = round(curr_price - (risk * 1.5), 2)
        tp2 = round(curr_price - (risk * 2.5), 2)
        reasoning = f"Price < VWAP (₹{curr_vwap:.1f}), EMA9 < EMA21, RSI {curr_rsi:.1f}. Sharp rejection from Supply Order Block (₹{nearest_supply_ob['bottom']:.1f} - ₹{nearest_supply_ob['top']:.1f}). SELL Put with SL ₹{sl} targeting ₹{tp1} / ₹{tp2}."
    elif is_bearish_chartink and is_pullback_supply:
        action = "SELL / PUT (PE) [PULLBACK TO VWAP/EMA9 RESISTANCE]"
        verdict = "BEARISH TREND PULLBACK SHORT ENTRY"
        confidence = "A GRADE PULLBACK"
        sl = round(max(curr_vwap, curr_ema9, curr_h) * 1.005, 2)
        risk = max(sl - curr_price, curr_price * 0.005)
        tp1 = round(curr_price - (risk * 1.5), 2)
        tp2 = round(min(ssl, curr_price - (risk * 2.5)), 2)
        reasoning = f"Price (₹{curr_price:.1f}) trapped below VWAP (₹{curr_vwap:.1f}), EMA9 < EMA21, RSI {curr_rsi:.1f}. Rejection at resistance. Enter Short (PE) with SL ₹{sl} targeting ₹{tp1}."
    elif is_bearish_chartink:
        action = "SELL / PUT (PE) [EXPLOSIVE DUMP BREAKDOWN]"
        verdict = "CHARTINK BREAKDOWN DUMP"
        confidence = "MOMENTUM CONFIRMED"
        sl = round(curr_ema9 * 1.006, 2)
        risk = max(sl - curr_price, curr_price * 0.005)
        tp1 = round(curr_price - (risk * 1.5), 2)
        tp2 = round(min(ssl, curr_price - (risk * 2.5)), 2)
        reasoning = f"Explosive selling pressure below VWAP (₹{curr_vwap:.1f}) and EMA9 (₹{curr_ema9:.1f}), RSI {curr_rsi:.1f}. Short entry active. SL ₹{sl}, TP1 ₹{tp1}."
    else:
        dist_to_demand = f"₹{(curr_price - nearest_demand_ob['top']):.1f}" if nearest_demand_ob else "N/A"
        dist_to_supply = f"₹{(nearest_supply_ob['bottom'] - curr_price):.1f}" if nearest_supply_ob else "N/A"
        reasoning = f"Price (₹{curr_price:.1f}) is currently mid-range. RSI is {curr_rsi:.1f}, {dist_to_demand} away from Demand OB, {dist_to_supply} away from Supply OB. Bot waiting for setup."

    return {
        "current_price": curr_price,
        "vwap": round(curr_vwap, 2),
        "ema9": round(curr_ema9, 2),
        "ema21": round(curr_ema21, 2),
        "rsi": round(curr_rsi, 1),
        "vol_multiplier": vol_multiplier,
        "is_bullish_chartink": is_bullish_chartink,
        "is_bearish_chartink": is_bearish_chartink,
        "nearest_demand_ob": nearest_demand_ob,
        "nearest_supply_ob": nearest_supply_ob,
        "bsl_liquidity": bsl,
        "ssl_liquidity": ssl,
        "action": action,
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "suggested_sl": sl,
        "suggested_tp1": tp1,
        "suggested_tp2": tp2
    }

from engine.groww_sector_engine import GrowwSectorEngine

groww_sector_engine = GrowwSectorEngine()

def calculate_dynamic_sectors():
    """
    Fetches real-time Trending Sectors directly from Groww (https://groww.in/stocks/sectors-trending).
    Matches 1D performance, gainers/losers, and filters strictly for moving F&O tradeable stocks:
    - Top 3 Bullish Sectors with Top 5 F&O Gainers
    - Top 3 Bearish Sectors with Top 5 F&O Losers (Skips non-F&O or zero-moving sectors)
    """
    fno_universe_set = set(screener_state.get("fno_universe", []))
    sectors = groww_sector_engine.fetch_groww_trending_sectors(fno_universe_set)
    return sectors

# Background Screener Daemon: dynamically scans Sectors, updates watchlists, and pre-caches market data
def background_screener_worker():
    last_outcome_check = 0
    last_market_poll = 0
    latest_dfs = {}
    fno_universe_index = 0
    while True:
        try:
            now = time.time()
            # 1. 5-minute Universal 219 F&O Momentum Refresh (Filtering out sideways/dead stocks)
            if now - screener_state["last_screener_update"] > 300 or not screener_state.get("bullish_momentum_watchlist"):
                print("[FNO SCANNER] Scanning all 219 F&O stocks: filtering sideways, selecting active momentum movers...")
                fno_list = screener_state.get("fno_universe", [])
                if not fno_list:
                    # Fallback to sector discovery if universe empty
                    sorted_sectors = calculate_dynamic_sectors()
                    screener_state["sectors"] = sorted_sectors
                    bull_watchlist = []
                    for b in [s for s in sorted_sectors.values() if s.get("is_bullish")][:3]:
                        bull_watchlist.extend(b.get("top_5", []))
                    bear_watchlist = []
                    for b in [s for s in sorted_sectors.values() if not s.get("is_bullish")][:3]:
                        bear_watchlist.extend(b.get("top_5", []))
                else:
                    top_bull, top_bear, fetched_dfs, filter_results = FNOMomentumFilter.filter_active_momentum_stocks(fno_list, max_workers=14)
                    bull_watchlist = [{"symbol": x["symbol"], "change": x["change_pct"], "velocity": x["velocity"]} for x in top_bull]
                    bear_watchlist = [{"symbol": x["symbol"], "change": x["change_pct"], "velocity": x["velocity"]} for x in top_bear]
                    latest_dfs.update(fetched_dfs)
                    
                    # Store fresh OHLCV data for EVERY successfully fetched F&O symbol (not only initial filter passes!)
                    filter_map = {r["symbol"]: r for r in filter_results}
                    for sym, cdf in fetched_dfs.items():
                        existing_entry = screener_state.setdefault("raw_market_cache", {}).get(sym, {})
                        fno_data = filter_map.get(sym) or existing_entry.get("fno_filter_data", {})
                        eval_ts = fno_data.get("evaluated_completed_candle_ts") if fno_data else str(cdf.index[-1])
                        screener_state["raw_market_cache"][sym] = {
                            "timestamp": now,
                            "raw_latest_candle_ts": str(cdf.index[-1]),
                            "evaluated_completed_candle_ts": eval_ts,
                            "rvol_candle_ts": fno_data.get("rvol_candle_ts") if fno_data else None,
                            "df": cdf,
                            "fno_filter_data": fno_data,
                            "data": existing_entry.get("data")
                        }

                screener_state["bullish_momentum_watchlist"] = bull_watchlist
                screener_state["bearish_momentum_watchlist"] = bear_watchlist
                screener_state["ranked_opportunities"] = None  # Instantly purge old rankings on 5-min refresh!
                screener_state["last_screener_update"] = now
                print(f"[FNO SCANNER] Active F&O Watchlist Ready: {len(bull_watchlist)} Bullish Movers, {len(bear_watchlist)} Bearish Movers (Filtered out dead/sideways).")

            # 2. Stock market-data polling separated from sector refresh (runs every 45-60s)
            if now - last_market_poll > 45:
                # Refresh active_signals_map directly from database
                try:
                    conn = get_db_connection()
                    cur = conn.cursor()
                    cur.execute("SELECT id, symbol FROM historical_signals WHERE outcome = 'ACTIVE'")
                    active_map = {}
                    for row in cur.fetchall():
                        s_raw = row["symbol"]
                        clean_s = s_raw.replace(".NS", "")
                        active_map[s_raw] = row["id"]
                        active_map[f"{clean_s}.NS"] = row["id"]
                        active_map[clean_s] = row["id"]
                    screener_state["active_signals_map"] = active_map
                    conn.close()
                except Exception:
                    pass

                # Poll market data across all active watchlist and active signals
                # Plus fetch NIFTY 50 benchmark synchronously for unified relative-strength calculations
                try:
                    nifty_df_raw = yf.download("^NSEI", period="2d", interval="5m", progress=False)
                    if nifty_df_raw is not None and not nifty_df_raw.empty:
                        nifty_df_clean = clean_dataframe(nifty_df_raw, "^NSEI")
                        screener_state.setdefault("raw_market_cache", {})["^NSEI"] = {
                            "timestamp": time.time(),
                            "df": nifty_df_clean
                        }
                except Exception:
                    pass

                # Build scanner list from Watchlists + Open ACTIVE Signals requiring outcome tracking
                watchlist_stks = [x["symbol"] for x in (screener_state["bullish_momentum_watchlist"] + screener_state["bearish_momentum_watchlist"])]
                current_watchlist_set = set(watchlist_stks)
                
                # Retrieve any currently open ACTIVE trade signals so they are always monitored for TP/SL outcome tracking
                active_open_symbols = list(get_active_signal_symbols())
                all_stk_symbols = list(dict.fromkeys(watchlist_stks + active_open_symbols))
                
                # Filter all evaluated candidates strictly to verified F&O instruments
                valid_fno_set = set(screener_state.get("fno_universe", []))
                norm_fno_set = {s.upper().replace(".NS", "") for s in valid_fno_set}

                candidates_for_ranking = []
                nifty_cached_df = screener_state.get("raw_market_cache", {}).get("^NSEI", {}).get("df")

                for sym in all_stk_symbols:
                    clean_sym = sym.replace(".NS", "").upper()
                    # Quality Filter: Only scan and rank stocks that are verified tradeable F&O stocks
                    if norm_fno_set and clean_sym not in norm_fno_set:
                        continue
                    try:
                        cdf = yf.download(sym, period="5d", interval="5m", progress=False)
                        if isinstance(cdf.columns, pd.MultiIndex):
                            cdf = cdf.xs(sym, axis=1, level=1) if sym in cdf.columns.levels[1] else cdf
                        res = evaluate_explosive_momentum(cdf, sym)
                        if res:
                            existing_fno = screener_state.get("raw_market_cache", {}).get(sym, {}).get("fno_filter_data")
                            screener_state["raw_market_cache"][sym] = {
                                "timestamp": time.time(),
                                "candle_ts": str(cdf.index[-1]),
                                "data": res,
                                "df": cdf,
                                "fno_filter_data": existing_fno
                            }
                            latest_dfs[sym] = cdf
                    except Exception:
                        pass
                    time.sleep(0.15)

                # Recompute unified rankings using single source of truth function
                compute_and_update_ranked_opportunities()

                last_market_poll = now

            # 3. Evaluate open signal outcomes every 60 seconds with full historical/subsequent candles
            if now - last_outcome_check > 60 and latest_dfs:
                try:
                    evaluate_open_outcomes(latest_dfs)
                    last_outcome_check = now
                except Exception as oe:
                    print("[OUTCOME TRACKER] Evaluation error:", oe)

            time.sleep(1)
        except Exception as e:
            print("Screener worker exception:", e)
            time.sleep(5)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    mkt_open = is_market_open()
    return jsonify({
        "connected": screener_state["connected"],
        "profile": screener_state["user_profile"],
        "margin": screener_state["margin"],
        "fno_count": len(screener_state["fno_universe"]),
        "fno_scanned": len(screener_state.get("bullish_momentum_watchlist", [])) + len(screener_state.get("bearish_momentum_watchlist", [])),
        "last_scan_time": screener_state.get("last_screener_update", 0),
        "is_market_open": mkt_open,
        "market_status": "LIVE MARKET OPEN" if mkt_open else "MARKET CLOSED (Using Last Session Data)"
    })

@app.route("/api/groww/sectors-dynamic")
def get_sectors_dynamic():
    """Serves 100% dynamically computed Sector Rankings and Top 5 FNO Stocks directly from Groww Trending Sectors"""
    if not screener_state["sectors"]:
        screener_state["sectors"] = calculate_dynamic_sectors()
    return jsonify({
        "success": True,
        "sectors": screener_state["sectors"]
    })

from strategies.institutional_engine import InstitutionalStrategyEngine
from engine.news_engine import NewsIntelligenceEngine
from engine.groq_client import GroqIntelligence
from engine.anomaly_engine import AnomalyDetector
from engine.pre_momentum_engine import PreMomentumEngine
from engine.opportunity_ranking_engine import OpportunityRankingEngine
from core.storage import log_signal, get_recent_signals, evaluate_open_outcomes, get_strategy_performance_stats, get_active_signal_symbols, get_db_connection

news_engine = NewsIntelligenceEngine()
groq_ai = GroqIntelligence()

@app.route("/api/market/smc-scan")
def smc_scan():
    """Returns Institutional Multi-Strategy + SMC + News Intelligence Analysis"""
    symbol = request.args.get("symbol")
    if not symbol:
        # Fallback to first stock in active watchlist or universe dynamically
        fno_list = screener_state.get("fno_universe", [])
        symbol = f"{fno_list[0]}.NS" if fno_list else "NIFTY"
    now = time.time()
    
    try:
        # Check raw market cache for pre-fetched dataframe to minimize network latency
        raw_cached = screener_state["raw_market_cache"].get(symbol)
        if raw_cached and (now - raw_cached.get("timestamp", 0) < 120) and raw_cached.get("df") is not None:
            df = raw_cached["df"]
            if "data" in raw_cached and raw_cached["data"]:
                res = raw_cached["data"].copy()
            else:
                res = evaluate_explosive_momentum(df, symbol)
                if not res:
                    return jsonify({"success": False, "error": "Insufficient market data"}), 400
                raw_cached["data"] = res
        else:
            df = yf.download(symbol, period="5d", interval="5m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs(symbol, axis=1, level=1) if symbol in df.columns.levels[1] else df
            res = evaluate_explosive_momentum(df, symbol)
            if not res:
                return jsonify({"success": False, "error": "Insufficient market data"}), 400

        # Calculate accurate previous trading session close
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
            df_copy.index = pd.to_datetime(df_copy.index)
        df_copy['Date'] = df_copy.index.date
        dates = sorted(df_copy['Date'].unique())
        
        # 1. HARDENED FEED VALIDATION VIA MARKET VALIDATOR (Single Source of Truth)
        # Stricter freshness: Max delay of 300 seconds (5m) during live market hours
        feed_val = MarketValidator.validate_feed_freshness(df, max_delay_seconds=300, candle_duration_seconds=300)
        if not feed_val["is_valid"]:
            return jsonify({
                "success": True,
                "smc": {
                    "action": "WAIT FOR ENTRY (NO TRADE)",
                    "verdict": f"FEED REJECTED: {feed_val['status']}",
                    "reasoning": feed_val["reason"],
                    "current_price": float(df['Close'].iloc[-1]),
                    "signal_status": "NEUTRAL",
                    "signal_id": None
                }
            })

        is_completed_candle = feed_val["is_completed_candle"]
        latest_candle_ts = feed_val["latest_candle_ts"]

        # 1b. Check enriched cache (TTL: 180s) strictly after validating candle freshness and completion
        # If a newer completed candle has closed since cached analysis, invalidate cache
        if symbol in screener_state["enriched_signals_cache"]:
            cached = screener_state["enriched_signals_cache"][symbol]
            cached_candle_ts = cached.get("last_candle_ts", 0)
            if (now - cached["timestamp"] < 180) and (latest_candle_ts <= cached_candle_ts):
                return jsonify({"success": True, "smc": cached["data"]})

        if len(dates) >= 2:
            prev_date = dates[-2]
            prev_day_df = df_copy[df_copy['Date'] == prev_date]
            prev_close = float(prev_day_df['Close'].iloc[-1].item() if hasattr(prev_day_df['Close'].iloc[-1], 'item') else prev_day_df['Close'].iloc[-1])
        else:
            prev_close = float(df_copy['Close'].iloc[0].item() if hasattr(df_copy['Close'].iloc[0], 'item') else df_copy['Close'].iloc[0])

        # 2. Extract Stock Sector & Live Sector Bias directly from Groww Sector Engine
        sector_bias = "NEUTRAL"
        assigned_sector = "UNCLASSIFIED"
        # Check comprehensive stock_to_sector_map populated from all Groww constituents
        groww_mapping = groww_sector_engine.stock_to_sector_map.get(symbol)
        if groww_mapping:
            assigned_sector = groww_mapping.get("sector", "UNCLASSIFIED")
            sector_bias = groww_mapping.get("bias", "NEUTRAL")
        else:
            # Fallback: check in active top-5 sector cards
            for s_name, s_data in screener_state["sectors"].items():
                stks = s_data.get("top_5", [])
                if any(x.get("symbol") == symbol for x in stks):
                    assigned_sector = s_name
                    sector_bias = "BULLISH" if s_data.get("is_bullish") else "BEARISH"
                    break
        
        if sector_bias == "NEUTRAL":
            assigned_sector = classify_stock_sector(symbol)
            sec_info = screener_state["sectors"].get(assigned_sector, {})
            sector_bias = "BULLISH" if sec_info.get("is_bullish") else ("BEARISH" if sec_info else "NEUTRAL")

        # 3. Match Verified News Catalyst (Before Strategy Evaluation)
        clean_sym = symbol.replace(".NS", "")
        news_cat = news_engine.match_catalyst_for_stock(clean_sym)
        news_headline = news_cat["title"] if news_cat else ""

        # 5. Run strategy evaluation strictly on completed candles
        if is_completed_candle:
            eval_df = df
        else:
            eval_df = df.iloc[:-1]

        if len(eval_df) < 20:
            return jsonify({
                "success": True,
                "smc": {
                    "action": "WAIT FOR ENTRY (NO TRADE)",
                    "verdict": "INSUFFICIENT COMPLETED CANDLE HISTORY",
                    "reasoning": "Waiting for at least 20 completed candles to ensure statistically verified institutional confluence.",
                    "current_price": float(df['Close'].iloc[-1]),
                    "signal_status": "NEUTRAL",
                    "signal_id": None
                }
            })

        # 4. Check Anomaly Detector strictly on completed-candle history
        anomaly_info = AnomalyDetector.detect_flat_open_breakout(df=eval_df, prev_close=prev_close)

        # 4b. Fetch or retrieve cached 1mo Daily OHLC for true Wilder Daily ATR14 calculation
        daily_df = None
        try:
            cached_daily = screener_state.get("daily_market_cache", {}).get(symbol)
            if cached_daily and (now - cached_daily.get("timestamp", 0) < 1800) and cached_daily.get("df") is not None:
                daily_df = cached_daily["df"]
            else:
                raw_daily = yf.download(symbol, period="1mo", interval="1d", progress=False)
                if raw_daily is not None and not raw_daily.empty:
                    daily_df = clean_dataframe(raw_daily, symbol)
                    screener_state.setdefault("daily_market_cache", {})[symbol] = {
                        "timestamp": now,
                        "df": daily_df
                    }
        except Exception as e:
            daily_df = None

        # 4c. Fetch NIFTY 50 Macro Regime Context
        nifty_regime = {"regime": "NEUTRAL", "bias": "NEUTRAL", "alignment_score": 50}
        try:
            nifty_cached = screener_state.get("raw_market_cache", {}).get("^NSEI")
            if nifty_cached and (now - nifty_cached.get("timestamp", 0) < 300):
                nifty_df = nifty_cached.get("df")
            else:
                nifty_df = yf.download("^NSEI", period="2d", interval="5m", progress=False)
                if nifty_df is not None and not nifty_df.empty:
                    nifty_df = clean_dataframe(nifty_df, "^NSEI")
                    screener_state.setdefault("raw_market_cache", {})["^NSEI"] = {
                        "timestamp": now,
                        "df": nifty_df
                    }
            nifty_regime = MarketRegimeEngine.evaluate_benchmark_regime(nifty_df)
        except Exception:
            pass

        # 6. Evaluate Institutional 15-Strategy Multi-Layer Engine WITH Sector, News, Anomaly, Daily DF & Macro Market Regime
        inst_res = InstitutionalStrategyEngine.evaluate_strategies(
            df=eval_df,
            prev_close=prev_close,
            sector_bias=sector_bias,
            catalyst_info=news_cat,
            anomaly_info=anomaly_info,
            daily_df=daily_df,
            market_regime_info=nifty_regime
        )

        # 6a. Blend Official NSE Live Institutional Flows & Large Deals (FII/DII + Block Deals)
        nse_inst = nse_engine.get_institutional_bias(symbol)
        if nse_inst.get("score_modifier"):
            inst_res["composite_score"] = min(100, max(0, inst_res["composite_score"] + nse_inst["score_modifier"]))
            if nse_inst["score_modifier"] > 0:
                inst_res["triggered_strategies"].append(f"NSE Large Deals (Net Buying)")
            elif nse_inst["score_modifier"] < 0:
                inst_res["triggered_strategies"].append(f"NSE Large Deals (Net Selling)")
        res["nse_institutional"] = nse_inst

        # Apply Daily ATR Fallback Confidence Penalty (15 pts if unmeasured baseline 1.8% was used)
        atr_penalty = inst_res.get("confidence_penalty", 0)
        if atr_penalty > 0:
            inst_res["composite_score"] = max(0, inst_res["composite_score"] - atr_penalty)
            inst_res["probability_pct"] = max(35.0, round(inst_res["probability_pct"] - (atr_penalty * 0.5), 1))

        # 6b. Audit for Unsignaled Explosive Moves (Missed Move Detection)
        MissedMoveDetector.audit_stock_move(
            symbol=symbol,
            df=eval_df,
            active_signal_id=screener_state.get("active_signals_map", {}).get(symbol),
            opp_eval=inst_res
        )

        # 6c. Run Dedicated Pre-Momentum Prediction Engine
        pre_mom = PreMomentumEngine.classify_stage(
            df=eval_df,
            sector_bias=sector_bias,
            vwap=inst_res["vwap"],
            rvol=inst_res["rvol"],
            composite_score=inst_res["composite_score"],
            opp_eval=inst_res
        )
        res["pre_momentum_stage"] = pre_mom["stage"]
        res["pre_momentum_score"] = pre_mom["stage_score"]
        res["pre_momentum_indicators"] = pre_mom["indicators"]
        res["pre_momentum_details"] = pre_mom["details"]

        # 7. Groq AI Institutional Summary
        ai_intel = groq_ai.analyze_signal(
            symbol=clean_sym,
            direction=inst_res["direction"],
            strategy=inst_res["primary_strategy"],
            price=inst_res["current_price"],
            metrics={"score": inst_res["composite_score"], "rvol": inst_res["rvol"], "rsi": inst_res["rsi"]},
            news_headline=news_headline
        )
        
        # Merge metrics into response (Using Session VWAP from Institutional Engine)
        res["vwap"] = inst_res["vwap"]
        res["composite_score"] = inst_res["composite_score"]
        res["probability_pct"] = inst_res["probability_pct"]
        res["triggered_strategies"] = inst_res["triggered_strategies"]
        res["news_headline"] = news_headline
        res["ai_summary"] = ai_intel.get("ai_summary", "")
        res["catalyst_impact"] = ai_intel.get("catalyst_impact", "")
        res["risk_rating"] = ai_intel.get("risk_rating", "MODERATE")
        res["anomaly_info"] = anomaly_info
        res["is_viable"] = inst_res.get("is_viable", False)
        res["suggested_sl"] = inst_res.get("suggested_sl")
        res["suggested_tp1"] = inst_res.get("suggested_tp1")
        res["suggested_tp2"] = inst_res.get("suggested_tp2")
        res["potential_pct"] = inst_res.get("potential_pct", 0.0)
        res["potential_tp2_pct"] = inst_res.get("potential_tp2_pct", 0.0)
        res["risk_reward_ratio"] = inst_res.get("risk_reward_ratio", 0.0)
        res["nearest_obstacle"] = inst_res.get("nearest_obstacle", "None")
        res["exhaustion_risk"] = inst_res.get("exhaustion_risk", "LOW")
        res["daily_atr_source"] = inst_res.get("daily_atr_source", "DEDICATED_DAILY_WILDER_ATR14")
        res["confidence_penalty"] = atr_penalty
        
        # Enforce strict institutional action: if institutional engine is NEUTRAL or MARKET IS CLOSED, action MUST be NO TRADE
        mkt_is_open = is_market_open()
        if not mkt_is_open:
            res["action"] = "MARKET CLOSED (WAIT FOR NEXT LIVE SESSION)"
            res["verdict"] = f"SCORE {inst_res['composite_score']}/100 | {inst_res.get('primary_strategy', 'OFFLINE')}"
            res["suggested_sl"] = None
            res["suggested_tp1"] = None
            res["suggested_tp2"] = None
            res["reasoning"] = f"NSE/BSE Indian Market is currently closed. Showing end-of-day closing structure for {clean_sym}. Real-time trade signals will activate at 09:15 AM IST."
            res["signal_id"] = None
            res["signal_status"] = "MARKET CLOSED"
        elif inst_res["direction"] != "NEUTRAL":
            res["action"] = f"{inst_res['direction']} / {'CALL (CE)' if inst_res['direction'] == 'BUY' else 'PUT (PE)'} [{inst_res['primary_strategy'].upper()}]"
            res["verdict"] = f"SCORE {inst_res['composite_score']}/100 | WIN PROB {inst_res['probability_pct']}%"
            res["suggested_sl"] = inst_res["suggested_sl"]
            res["suggested_tp1"] = inst_res["suggested_tp1"]
            res["suggested_tp2"] = inst_res["suggested_tp2"]
            res["reasoning"] = f"{inst_res['primary_strategy']} active. {ai_intel.get('ai_summary', '')}"
            res["signal_status"] = "CONFIRMED (Completed Candle)" if is_completed_candle else "PROVISIONAL (Live Candle)"
            
            # Persist ONLY CONFIRMED signals generated during live market hours to historical signals database
            if is_completed_candle and mkt_is_open:
                signal_id = log_signal({
                    "symbol": symbol,
                    "direction": inst_res["direction"],
                    "strategy": inst_res["primary_strategy"],
                    "entry_price": inst_res["current_price"],
                    "stop_loss": inst_res["suggested_sl"],
                    "target_1": inst_res["suggested_tp1"],
                    "target_2": inst_res["suggested_tp2"],
                    "score": inst_res["composite_score"],
                    "confidence": f"{inst_res['probability_pct']}%",
                    "catalyst": news_headline or "Institutional Order Flow",
                    "features": {"rvol": inst_res["rvol"], "rsi": inst_res["rsi"], "vwap": inst_res["vwap"]},
                    "all_triggered": inst_res["triggered_strategies"]
                })
                res["signal_id"] = signal_id
                # Instantly cache to active_signals_map
                screener_state.setdefault("active_signals_map", {})[symbol] = signal_id
                screener_state["active_signals_map"][clean_sym] = signal_id
                screener_state["active_signals_map"][f"{clean_sym}.NS"] = signal_id
            else:
                res["signal_id"] = None
        else:
            # Institutional engine is NEUTRAL: clear any momentum-only buy/sell action
            opp_reason = inst_res.get("opportunity_reason") or "Institutional confluence is neutral"
            res["action"] = "WAIT FOR ENTRY (NO TRADE)"
            res["verdict"] = f"SCORE {inst_res['composite_score']}/100 | {opp_reason}"
            res["suggested_sl"] = None
            res["suggested_tp1"] = None
            res["suggested_tp2"] = None
            res["reasoning"] = f"{opp_reason}. Score {inst_res['composite_score']}/100. Bot waiting for confirmed setup."
            res["signal_id"] = None
            res["signal_status"] = "NEUTRAL"

        # Save to enriched signals cache with TTL and latest candle timestamp
        screener_state["enriched_signals_cache"][symbol] = {
            "timestamp": now,
            "last_candle_ts": latest_candle_ts,
            "data": res
        }
        return jsonify({"success": True, "smc": res})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

def compute_and_update_ranked_opportunities(allowed_sector_symbols=None):
    """
    Unified ranking pipeline used identically by background worker and API endpoint.
    Implements true Top-3 Bullish & Top-3 Bearish Sector-First Selection:
    1. Evaluates live sectors from Groww.
    2. Identifies Top 3 Bullish sectors and Top 3 Bearish sectors.
    3. Evaluates constituent F&O stocks with strict completed 5-min candle synchronization & liquidity.
    4. Ranks candidates within those sector groups into ENTRY READY, IGNITION, PREPARING, and directional WATCHLISTS.
    """
    now = time.time()
    
    # Determine latest completed candle timestamp from NIFTY
    nifty_cached = screener_state.get("raw_market_cache", {}).get("^NSEI", {})
    nifty_df = nifty_cached.get("df")
    current_candle_ts = None
    eval_nifty = None
    if nifty_df is not None and not nifty_df.empty:
        nifty_val = MarketValidator.validate_feed_freshness(nifty_df, max_delay_seconds=300, candle_duration_seconds=300)
        is_nifty_comp = nifty_val.get("is_completed_candle", False)
        eval_nifty = nifty_df if is_nifty_comp else nifty_df.iloc[:-1]
        if not eval_nifty.empty:
            current_candle_ts = str(eval_nifty.index[-1])
    
    if not current_candle_ts:
        # If NIFTY data is unavailable, market-wide synchronization cannot be established
        return {
            "top_bullish": [], "top_bearish": [], "entry_ready": [], "ignition": [], "preparing": []
        }, None

    candidates = []
    valid_fno_set = set(screener_state.get("fno_universe", []))
    norm_fno_set = {s.upper().replace(".NS", "") for s in valid_fno_set}

    # Sector-First Hierarchy:
    # 1. Select Top 3 Bullish and Top 3 Bearish sectors by explicit live performance sorting
    sectors = screener_state.get("sectors", {})
    if not sectors:
        sectors = calculate_dynamic_sectors()
        screener_state["sectors"] = sectors

    # Explicitly sort sectors by performance
    sorted_bull_sectors = sorted(
        [(name, data) for name, data in sectors.items() if data.get("is_bullish") and data.get("change", 0.0) > 0.0],
        key=lambda x: x[1].get("change", 0.0),
        reverse=True
    )[:3]
    sorted_bear_sectors = sorted(
        [(name, data) for name, data in sectors.items() if not data.get("is_bullish") and data.get("change", 0.0) < 0.0],
        key=lambda x: x[1].get("change", 0.0)
    )[:3]
    
    priority_sectors = set([s[0] for s in sorted_bull_sectors] + [s[0] for s in sorted_bear_sectors])

    # Gather constituent symbols from top sectors and validate against F&O universe
    sector_constituents = set()
    for s_name in priority_sectors:
        stks = sectors[s_name].get("top_5") or sectors[s_name].get("top_5_stocks") or []
        for stk in stks:
            sym_clean = stk["symbol"] if isinstance(stk, dict) else str(stk)
            clean_bare = sym_clean.replace(".NS", "").upper()
            if not norm_fno_set or clean_bare in norm_fno_set:
                sector_constituents.add(sym_clean)
                sector_constituents.add(f"{clean_bare}.NS")

    raw_items = screener_state.get("raw_market_cache", {})
    for sym, item in raw_items.items():
        if sym == "^NSEI" or item.get("df") is None:
            continue
        clean_sym = sym.replace(".NS", "").upper()
        if norm_fno_set and clean_sym not in norm_fno_set:
            continue
        
        # Sector-First Priority Filter: If priority sector constituents are known, prioritize them
        if sector_constituents and sym not in sector_constituents and f"{clean_sym}.NS" not in sector_constituents:
            # Still scan if allowed_sector_symbols specifically requests, else prioritize top sectors
            if not allowed_sector_symbols or sym not in allowed_sector_symbols:
                continue

        df_sym = item["df"]
        
        # Enforce unified 5m feed validation
        f_val = MarketValidator.validate_feed_freshness(df_sym, max_delay_seconds=300, candle_duration_seconds=300)
        if not f_val.get("is_valid", False):
            continue
        is_comp = f_val.get("is_completed_candle", False)
        eval_df_sym = df_sym if is_comp else df_sym.iloc[:-1]
        if len(eval_df_sym) < 15:
            continue

        stock_candle_ts = str(eval_df_sym.index[-1])

        # Strict Stock-NIFTY Completed Candle Synchronization (P0):
        # Reject candidates whose latest completed candle is misaligned with the market benchmark
        if stock_candle_ts != current_candle_ts:
            continue

        sec_info = groww_sector_engine.stock_to_sector_map.get(sym, {})
        sec_name = sec_info.get("sector")
        sec_chg = sec_info.get("sector_change", 0.0)
        if not sec_name:
            sec_name = classify_stock_sector(sym)
            sec_data = screener_state.get("sectors", {}).get(sec_name, {})
            sec_chg = sec_data.get("change", 0.0)
        
        # Retrieve enriched cache ONLY if its evaluated completed candle timestamp matches exactly
        enriched_entry = screener_state.get("enriched_signals_cache", {}).get(sym, {})
        enriched_candle_ts = str(enriched_entry.get("last_candle_ts", ""))
        enriched = enriched_entry.get("data", {}) if (enriched_candle_ts and enriched_candle_ts == stock_candle_ts) else {}
        
        opp_eval = {
            "is_viable": enriched.get("is_viable", False),
            "potential_pct": enriched.get("potential_pct", 0.0),
            "risk_reward_ratio": enriched.get("risk_reward_ratio", 0.0),
            "stop_loss": enriched.get("suggested_sl"),
            "tp1": enriched.get("suggested_tp1"),
            "tp2": enriched.get("suggested_tp2"),
            "exhaustion_risk": enriched.get("exhaustion_risk", "LOW")
        }

        # Validate RVOL freshness strictly against evaluated completed candle timestamp (P0)
        fno_data = item.get("fno_filter_data", {})
        rvol_candle_ts = item.get("rvol_candle_ts") or (fno_data.get("rvol_candle_ts") if fno_data else None)
        time_slot_rvol = None
        rvol_obs = 0
        if rvol_candle_ts and rvol_candle_ts == stock_candle_ts and fno_data:
            time_slot_rvol = fno_data.get("rvol")
            rvol_obs = fno_data.get("rvol_obs", 0)

        eval_res = OpportunityRankingEngine.evaluate_stock_opportunity(
            stock_df=eval_df_sym,
            symbol=sym,
            sector_name=sec_name,
            sector_change=sec_chg,
            nifty_df=eval_nifty,
            vwap=enriched.get("vwap") or (fno_data.get("vwap") if fno_data else None),
            opp_eval=opp_eval,
            time_slot_rvol=time_slot_rvol,
            rvol_obs=rvol_obs,
            is_liquid=True
        )
        if eval_res:
            candidates.append(eval_res)

    if not candidates:
        ranked = {
            "top_bullish": [], "top_bearish": [], "entry_ready": [], "ignition": [], "preparing": []
        }
    else:
        ranked = OpportunityRankingEngine.rank_top_opportunities(candidates, top_n=3)

    screener_state["ranked_opportunities"] = ranked
    screener_state["ranked_opportunities_ts"] = now
    screener_state["ranked_candle_ts"] = current_candle_ts
    return ranked, current_candle_ts

@app.route("/api/opportunities/ranked")
def get_ranked_opportunities_endpoint():
    """
    Returns dynamically computed Top 3 Bullish and Top 3 Bearish F&O opportunities 
    across the active watchlist using multi-factor RS acceleration, breakout proximity and early stages.
    Ties cache freshness directly to the exact completed 5-minute candle timestamp.
    """
    try:
        now = time.time()
        
        # Determine latest completed candle timestamp from NIFTY
        nifty_cached = screener_state.get("raw_market_cache", {}).get("^NSEI", {})
        nifty_df = nifty_cached.get("df")
        current_candle_ts = None
        if nifty_df is not None and not nifty_df.empty:
            nifty_val = MarketValidator.validate_feed_freshness(nifty_df, max_delay_seconds=300, candle_duration_seconds=300)
            is_nifty_comp = nifty_val.get("is_completed_candle", False)
            eval_nifty = nifty_df if is_nifty_comp else nifty_df.iloc[:-1]
            if not eval_nifty.empty:
                current_candle_ts = str(eval_nifty.index[-1])
        
        cached_ranked = screener_state.get("ranked_opportunities")
        cached_candle_ts = screener_state.get("ranked_candle_ts")
        cached_wall_ts = screener_state.get("ranked_opportunities_ts", 0)

        # Cache is valid ONLY if the evaluated completed candle timestamp matches AND wall-clock age < 300s
        if cached_ranked and (current_candle_ts and cached_candle_ts == current_candle_ts) and (now - cached_wall_ts < 300):
            return jsonify({
                "success": True, 
                "ranked": {
                    "top_bullish": cached_ranked.get("top_bullish", []), 
                    "top_bearish": cached_ranked.get("top_bearish", []),
                    "entry_ready": cached_ranked.get("entry_ready", [])[:2],
                    "ignition": cached_ranked.get("ignition", [])[:3],
                    "preparing": cached_ranked.get("preparing", [])[:5]
                }
            })

        ranked, current_candle_ts = compute_and_update_ranked_opportunities()
        if not ranked or not any(ranked.values()):
            return jsonify({
                "success": True,
                "status": "NO_TRADE",
                "ranked": {
                    "top_bullish": [], "top_bearish": [], "entry_ready": [], "ignition": [], "preparing": []
                }
            })

        return jsonify({"success": True, "ranked": ranked})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "ranked": {"top_bullish": [], "top_bearish": []}})

@app.route("/api/signals/performance")
def get_performance_stats():
    """Returns actual win rate, profit factor, and outcome stats per strategy"""
    return jsonify({"success": True, "stats": get_strategy_performance_stats()})

@app.route("/api/signals/missed")
def get_missed_moves_endpoint():
    """Returns missed opportunities detected during market hours for audit and strategy retraining"""
    try:
        from core.storage import get_db_connection
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM missed_moves ORDER BY id DESC LIMIT 50")
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return jsonify({"success": True, "missed_moves": rows})
    except Exception as e:
        return jsonify({"success": False, "missed_moves": [], "error": str(e)})

@app.route("/api/signals/calibrations")
def get_strategy_calibrations_endpoint():
    """Calibrates strategy weights based on verified closed outcomes (N >= 15) and returns current multipliers"""
    calibs = StrategyLearningEngine.calibrate_weights()
    return jsonify({"success": True, "calibrations": calibs})

@app.route("/api/signals/active")
def get_active_signals_endpoint():
    """Returns all open ACTIVE trades so frontend can synchronize top confirmed banner"""
    try:
        from core.storage import get_db_connection, evaluate_open_outcomes
        from engine.market_validator import MarketValidator
        
        # If market is closed past 15:30 IST or non-trading day, run an EOD squareoff check
        mkt_open = MarketValidator.is_market_open()
        if not mkt_open:
            evaluate_open_outcomes({}, force_eod=True)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM historical_signals WHERE outcome = 'ACTIVE' ORDER BY id DESC")
        active = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return jsonify({"success": True, "active_signals": active, "is_market_open": mkt_open})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "active_signals": []})

@app.route("/api/signals/log-confirmed", methods=["POST"])
def log_confirmed_signal_endpoint():
    """Logs a confirmed trade signal generated by user or ranking engine into historical signals DB"""
    try:
        from engine.market_validator import MarketValidator
        if not MarketValidator.is_market_open():
            return jsonify({"success": False, "message": "Market is closed. No live signals logged."})

        data = request.get_json() or {}
        sym = data.get("symbol")
        if not sym:
            return jsonify({"success": False, "error": "No symbol provided"}), 400

        action = data.get("action", "")
        direction = "BUY" if "BUY" in action else ("SELL" if "SELL" in action else "NEUTRAL")
        if direction == "NEUTRAL":
            return jsonify({"success": False, "error": "Direction is neutral"}), 400

        sig_id = log_signal({
            "symbol": sym,
            "direction": direction,
            "strategy": data.get("primary_strategy", "SMC Structure"),
            "entry_price": data.get("entry_price"),
            "stop_loss": data.get("suggested_sl"),
            "target_1": data.get("suggested_tp1"),
            "target_2": data.get("suggested_tp2"),
            "score": data.get("composite_score", 75),
            "confidence": f"{data.get('composite_score', 75)}%",
            "catalyst": "SMC Order Flow Ignition",
            "features": {},
            "all_triggered": data.get("triggered_strategies", [])
        })

        # Cache in memory
        screener_state.setdefault("active_signals_map", {})[sym] = sig_id

        return jsonify({"success": True, "signal_id": sig_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/signals/close", methods=["POST"])
def close_signal_endpoint():
    """Manually or auto-closes an active signal at market close / user dismissal"""
    try:
        data = request.get_json() or {}
        symbol = data.get("symbol")
        signal_id = data.get("signal_id")
        exit_price = data.get("exit_price")
        reason = data.get("reason", "CLOSED_MANUAL")

        from core.storage import get_db_connection
        conn = get_db_connection()
        c = conn.cursor()

        if signal_id:
            c.execute("UPDATE historical_signals SET outcome = ?, closed_at = ? WHERE id = ?", (reason, time.time(), signal_id))
        elif symbol:
            c.execute("UPDATE historical_signals SET outcome = ?, closed_at = ? WHERE symbol = ? AND outcome = 'ACTIVE'", (reason, time.time(), symbol))
        
        conn.commit()
        conn.close()

        # Update in-memory active signals map
        if symbol:
            clean = symbol.replace(".NS", "")
            screener_state.get("active_signals_map", {}).pop(symbol, None)
            screener_state.get("active_signals_map", {}).pop(clean, None)
            screener_state.get("active_signals_map", {}).pop(f"{clean}.NS", None)

        return jsonify({"success": True, "message": f"Signal for {symbol or signal_id} closed."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/signals/recent")
def get_recent_logged_signals():
    """Returns persistent learning signals and trade history"""
    return jsonify({"success": True, "signals": get_recent_signals(limit=50)})

@app.route("/api/news")
def get_live_news():
    """Returns curated live market and catalyst headlines"""
    return jsonify({"success": True, "news": news_engine.fetch_market_news()})

@app.route("/api/news/stock")
def get_stock_specific_news():
    """Returns targeted live news affecting the currently scanned F&O stock"""
    symbol = request.args.get("symbol", "")
    if not symbol:
        return jsonify({"success": False, "news": []})
    clean = symbol.replace(".NS", "")
    articles = news_engine.fetch_news_for_stock(clean)
    return jsonify({"success": True, "symbol": clean, "news": articles})

@app.route("/api/fno/momentum-watchlists")
def get_fno_momentum_watchlists_endpoint():
    """Returns dynamically scanned Top Bullish & Bearish momentum movers from all 219 F&O stocks"""
    bull = screener_state.get("bullish_momentum_watchlist", [])
    bear = screener_state.get("bearish_momentum_watchlist", [])
    return jsonify({
        "success": True,
        "bullish": bull,
        "bearish": bear,
        "fno_total": len(screener_state.get("fno_universe", []))
    })

@app.route("/api/nse/institutional")
def get_nse_institutional_data():
    """Returns official NSE India live FII/DII activity and Bulk/Block deals"""
    try:
        symbol = request.args.get("symbol")
        fii_dii = nse_engine.get_fii_dii_activity()
        bias = nse_engine.get_institutional_bias(symbol)
        large_deals = nse_engine.get_large_deals(symbol)
        return jsonify({
            "success": True,
            "fii_dii": fii_dii,
            "bias": bias,
            "large_deals": large_deals[:20]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    print("Auto-connecting to Groww API...")
    auto_connect_groww()
    
    # Run initial scan immediately so UI loads without waiting
    print("[INIT] Running initial dynamic scan...")
    try:
        sorted_sectors = calculate_dynamic_sectors()
        screener_state["sectors"] = sorted_sectors

        valid_sectors = [s for s in sorted_sectors.values() if (s.get("top_5") and len(s.get("top_5")) >= 1)]
        bull_sectors = [s for s in valid_sectors if s["is_bullish"]][:3]
        bear_sectors = sorted([s for s in valid_sectors if not s["is_bullish"]], key=lambda x: x["change"])[:3]

        bull_watchlist = []
        for b in bull_sectors:
            for stk in b.get("top_5", []):
                bull_watchlist.append({
                    "symbol": stk["symbol"],
                    "change": stk["change"],
                    "velocity": stk.get("velocity", round(max(1.0, abs(stk["change"]) * 1.3), 1))
                })

        bear_watchlist = []
        for b in bear_sectors:
            for stk in b.get("top_5", []):
                bear_watchlist.append({
                    "symbol": stk["symbol"],
                    "change": stk["change"],
                    "velocity": stk.get("velocity", round(max(1.0, abs(stk["change"]) * 1.3), 1))
                })

        screener_state["bullish_momentum_watchlist"] = bull_watchlist
        screener_state["bearish_momentum_watchlist"] = bear_watchlist
        screener_state["last_screener_update"] = time.time()
        print(f"[INIT] Dynamic scan ready! Top Bullish: {[s['groww_code'] for s in bull_sectors]}, Top Bearish: {[s['groww_code'] for s in bear_sectors]}")
    except Exception as e:
        print("[INIT] Error during initial scan:", e)

    # Launch background worker
    worker = threading.Thread(target=background_screener_worker, daemon=True)
    worker.start()
    print("Groww Chartink Momentum + SMC Screener running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
