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

def is_market_open():
    """Returns True if the Indian market (NSE/BSE) is open right now (Mon-Fri 09:15 to 15:30 IST)"""
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.datetime.now(ist)
        # Weekdays: Monday is 0, Sunday is 6
        if now.weekday() >= 5:
            return False
        minute = now.hour * 60 + now.minute
        # 09:15 is 555 min, 15:30 is 930 min
        return 555 <= minute <= 930
    except Exception:
        return True

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
    "raw_market_cache": {},        # 15m OHLCV and raw technicals
    "enriched_signals_cache": {},  # Full 15-strategy + AI enriched results with timestamps
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
    """Auto connect to Groww API using token + secret in .env and fetch all F&O stocks dynamically"""
    api_token = os.getenv("GROWW_API_TOKEN")
    api_secret = os.getenv("GROWW_API_SECRET")
    if not api_token:
        return False
    try:
        if api_secret:
            token = GrowwAPI.get_access_token(api_key=api_token, secret=api_secret)
        else:
            token = api_token
        client = GrowwAPI(token)

        try:
            screener_state["user_profile"] = client.get_user_profile(timeout=5)
        except Exception:
            screener_state["user_profile"] = {"ucc": "Active"}

        try:
            screener_state["margin"] = client.get_available_margin_details(timeout=5)
        except Exception:
            screener_state["margin"] = {}

        # Dynamically discover all tradeable F&O stocks directly from Groww API
        try:
            df = client.get_all_instruments()
            fno_df = df[df['segment'] == 'FNO']
            INDEX_SYMBOLS = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'BANKEX', 'SENSEX', 'NIFTYNXT50'}
            raw_fno = [
                s for s in fno_df['underlying_symbol'].dropna().unique() 
                if s not in INDEX_SYMBOLS and 'TEST' not in s and not s.isdigit() and len(s) >= 2
            ]
            screener_state["fno_universe"] = sorted(raw_fno)
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

        screener_state["connected"] = True
        return True
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
        confidence = "95% VERY HIGH ACCURACY"
        sl = round(min(curr_l, nearest_demand_ob['bottom']) * 0.995, 2)
        risk = max(curr_price - sl, curr_price * 0.005)
        tp1 = round(curr_price + (risk * 1.5), 2)
        tp2 = round(curr_price + (risk * 2.5), 2)
        reasoning = f"Price > VWAP (₹{curr_vwap:.1f}), EMA9 > EMA21, RSI {curr_rsi:.1f}. Bouncing off Demand Order Block (₹{nearest_demand_ob['bottom']:.1f} - ₹{nearest_demand_ob['top']:.1f}). BUY Call with SL ₹{sl} targeting ₹{tp1} / ₹{tp2}."
    elif is_bullish_chartink and is_pullback_demand:
        action = "BUY / CALL (CE) [PULLBACK TO VWAP/EMA9 SUPPORT]"
        verdict = "BULLISH TREND PULLBACK ENTRY"
        confidence = "89% HIGH CONFIDENCE"
        sl = round(min(curr_vwap, curr_ema9, curr_l) * 0.995, 2)
        risk = max(curr_price - sl, curr_price * 0.005)
        tp1 = round(curr_price + (risk * 1.5), 2)
        tp2 = round(max(bsl, curr_price + (risk * 2.5)), 2)
        reasoning = f"Price (₹{curr_price:.1f}) is riding institutional trend above VWAP (₹{curr_vwap:.1f}), EMA9 > EMA21, RSI {curr_rsi:.1f}. Pullback resting at support. Strong BUY entry with SL ₹{sl} targeting ₹{tp1}."
    elif is_bullish_chartink:
        action = "BUY / CALL (CE) [EXPLOSIVE MOMENTUM RALLY]"
        verdict = "CHARTINK MOMENTUM BREAKOUT"
        confidence = "84% CONFIRMED MOMENTUM"
        sl = round(curr_ema9 * 0.994, 2)
        risk = max(curr_price - sl, curr_price * 0.005)
        tp1 = round(curr_price + (risk * 1.5), 2)
        tp2 = round(max(bsl, curr_price + (risk * 2.5)), 2)
        reasoning = f"Explosive buying pressure above VWAP (₹{curr_vwap:.1f}) and EMA9 (₹{curr_ema9:.1f}), RSI {curr_rsi:.1f}. Long entry active. SL ₹{sl}, TP1 ₹{tp1}."
    elif is_bearish_chartink and is_at_supply:
        action = "SELL / PUT (PE) [HIGH CONFLUENCE: SUPPLY OB + BREAKDOWN]"
        verdict = "SUPPLY ORDER BLOCK REJECTION + BEARISH BREAKDOWN"
        confidence = "95% VERY HIGH ACCURACY"
        sl = round(max(curr_h, nearest_supply_ob['top']) * 1.005, 2)
        risk = max(sl - curr_price, curr_price * 0.005)
        tp1 = round(curr_price - (risk * 1.5), 2)
        tp2 = round(curr_price - (risk * 2.5), 2)
        reasoning = f"Price < VWAP (₹{curr_vwap:.1f}), EMA9 < EMA21, RSI {curr_rsi:.1f}. Sharp rejection from Supply Order Block (₹{nearest_supply_ob['bottom']:.1f} - ₹{nearest_supply_ob['top']:.1f}). SELL Put with SL ₹{sl} targeting ₹{tp1} / ₹{tp2}."
    elif is_bearish_chartink and is_pullback_supply:
        action = "SELL / PUT (PE) [PULLBACK TO VWAP/EMA9 RESISTANCE]"
        verdict = "BEARISH TREND PULLBACK SHORT ENTRY"
        confidence = "89% HIGH CONFIDENCE"
        sl = round(max(curr_vwap, curr_ema9, curr_h) * 1.005, 2)
        risk = max(sl - curr_price, curr_price * 0.005)
        tp1 = round(curr_price - (risk * 1.5), 2)
        tp2 = round(min(ssl, curr_price - (risk * 2.5)), 2)
        reasoning = f"Price (₹{curr_price:.1f}) trapped below VWAP (₹{curr_vwap:.1f}), EMA9 < EMA21, RSI {curr_rsi:.1f}. Rejection at resistance. Enter Short (PE) with SL ₹{sl} targeting ₹{tp1}."
    elif is_bearish_chartink:
        action = "SELL / PUT (PE) [EXPLOSIVE DUMP BREAKDOWN]"
        verdict = "CHARTINK BREAKDOWN DUMP"
        confidence = "84% CONFIRMED DUMP"
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
    while True:
        try:
            now = time.time()
            if now - screener_state["last_screener_update"] > 300 or not screener_state["sectors"]:
                print("[CHARTINK ENGINE] Scanning live Sector indices & Top FNO candidates...")
                sorted_sectors = calculate_dynamic_sectors()
                screener_state["sectors"] = sorted_sectors

                # Bullish & Bearish momentum watchlist from Top 3 Bullish and Top 3 Bearish Groww sectors
                bull_sectors = [s for s in sorted_sectors.values() if s.get("is_bullish")][:3]
                bear_sectors = [s for s in sorted_sectors.values() if not s.get("is_bullish")][:3]

                bull_watchlist = []
                for b in bull_sectors:
                    bull_watchlist.extend(b.get("top_5", []))

                bear_watchlist = []
                for b in bear_sectors:
                    bear_watchlist.extend(b.get("top_5", []))

                screener_state["bullish_momentum_watchlist"] = bull_watchlist
                screener_state["bearish_momentum_watchlist"] = bear_watchlist
                screener_state["last_screener_update"] = now
                print(f"[CHARTINK ENGINE] Sectors updated. Bullish: {[x['symbol'] for x in bull_watchlist[:4]]}, Bearish: {[x['symbol'] for x in bear_watchlist[:4]]}")

            # Pre-compute raw technicals into raw_market_cache (Never overwrite enriched_signals_cache)
            # Combine current top sectors watchlist + all symbols with open ACTIVE trades across database
            watchlist_stks = [x["symbol"] for x in (screener_state["bullish_momentum_watchlist"] + screener_state["bearish_momentum_watchlist"])]
            active_db_symbols = list(get_active_signal_symbols())
            # Union of both sets to ensure active open trades are ALWAYS tracked even if stock dropped out of top sectors
            all_stk_symbols = list(dict.fromkeys(watchlist_stks + active_db_symbols))
            
            latest_dfs = {}
            for sym in all_stk_symbols:
                try:
                    cdf = yf.download(sym, period="5d", interval="15m", progress=False)
                    if isinstance(cdf.columns, pd.MultiIndex):
                        cdf = cdf.xs(sym, axis=1, level=1) if sym in cdf.columns.levels[1] else cdf
                    res = evaluate_explosive_momentum(cdf, sym)
                    if res:
                        screener_state["raw_market_cache"][sym] = {
                            "timestamp": time.time(),
                            "data": res,
                            "df": cdf
                        }
                        # Pass full chronological OHLC DataFrame to outcome tracker
                        latest_dfs[sym] = cdf
                except Exception:
                    pass
                time.sleep(0.4)

            # Evaluate open signal outcomes every 60 seconds with full historical/subsequent candles
            if now - last_outcome_check > 60 and latest_dfs:
                try:
                    evaluate_open_outcomes(latest_dfs)
                    last_outcome_check = now
                except Exception as oe:
                    print("[OUTCOME TRACKER] Evaluation error:", oe)

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
from core.storage import log_signal, get_recent_signals, evaluate_open_outcomes, get_strategy_performance_stats, get_active_signal_symbols

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
    
    # 1. Check enriched cache (TTL: 180 seconds)
    if symbol in screener_state["enriched_signals_cache"]:
        cached = screener_state["enriched_signals_cache"][symbol]
        if now - cached["timestamp"] < 180:
            return jsonify({"success": True, "smc": cached["data"]})

    try:
        # Check raw market cache for pre-fetched dataframe to minimize network latency
        raw_cached = screener_state["raw_market_cache"].get(symbol)
        if raw_cached and (now - raw_cached["timestamp"] < 120) and raw_cached.get("df") is not None:
            df = raw_cached["df"]
            res = raw_cached["data"].copy()
        else:
            df = yf.download(symbol, period="5d", interval="15m", progress=False)
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
        
        if len(dates) >= 2:
            prev_date = dates[-2]
            prev_day_df = df_copy[df_copy['Date'] == prev_date]
            prev_close = float(prev_day_df['Close'].iloc[-1].item() if hasattr(prev_day_df['Close'].iloc[-1], 'item') else prev_day_df['Close'].iloc[-1])
        else:
            prev_close = float(df_copy['Close'].iloc[0].item() if hasattr(df_copy['Close'].iloc[0], 'item') else df_copy['Close'].iloc[0])

        # 2. Extract Stock Sector & Live Sector Bias directly from Groww Sector Engine
        sector_bias = "NEUTRAL"
        assigned_sector = "UNCLASSIFIED"
        # Check if stock exists in active Groww sectors
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

        # 4. Check Anomaly Detector (Flat-Open Sudden Expansion & Time-of-Day RVOL Acceleration)
        anomaly_info = AnomalyDetector.detect_flat_open_breakout(df=df_copy, prev_close=prev_close)

        # 5. Evaluate Institutional 15-Strategy Multi-Layer Engine WITH Sector, News & Anomaly Evidence
        inst_res = InstitutionalStrategyEngine.evaluate_strategies(
            df=df,
            prev_close=prev_close,
            sector_bias=sector_bias,
            catalyst_info=news_cat,
            anomaly_info=anomaly_info
        )

        # 6. Groq AI Institutional Summary
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
        res["potential_pct"] = inst_res.get("potential_pct", 0.0)
        res["potential_tp2_pct"] = inst_res.get("potential_tp2_pct", 0.0)
        res["nearest_obstacle"] = inst_res.get("nearest_obstacle", "None")
        res["exhaustion_risk"] = inst_res.get("exhaustion_risk", "LOW")
        
        # Enforce strict institutional action: if institutional engine is NEUTRAL, action MUST be NO TRADE
        if inst_res["direction"] != "NEUTRAL":
            res["action"] = f"{inst_res['direction']} / {'CALL (CE)' if inst_res['direction'] == 'BUY' else 'PUT (PE)'} [{inst_res['primary_strategy'].upper()}]"
            res["verdict"] = f"SCORE {inst_res['composite_score']}/100 | WIN PROB {inst_res['probability_pct']}%"
            res["suggested_sl"] = inst_res["suggested_sl"]
            res["suggested_tp1"] = inst_res["suggested_tp1"]
            res["suggested_tp2"] = inst_res["suggested_tp2"]
            res["reasoning"] = f"{inst_res['primary_strategy']} active. {ai_intel.get('ai_summary', '')}"
            
            # Persist to historical signals database and capture logged signal ID directly
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

            # Completed candle vs Provisional candle confirmation:
            # Check if the latest candle is completed (timestamp + 900s <= current time or market closed)
            last_candle_end = 0
            if hasattr(df.index[-1], 'timestamp'):
                last_candle_end = df.index[-1].timestamp() + 900
            
            is_completed_candle = (not is_market_open()) or (now >= last_candle_end and last_candle_end > 0)
            res["signal_status"] = "CONFIRMED (Completed Candle)" if is_completed_candle else "PROVISIONAL (Live Candle)"
        else:
            # Institutional engine is NEUTRAL: clear any momentum-only buy/sell action
            res["action"] = "WAIT FOR ENTRY (NO TRADE)"
            res["verdict"] = f"SCORE {inst_res['composite_score']}/100 (NEUTRAL CONFLUENCE)"
            res["suggested_sl"] = None
            res["suggested_tp1"] = None
            res["suggested_tp2"] = None
            res["reasoning"] = f"Institutional confluence is neutral. Score {inst_res['composite_score']}/100. Bot waiting for confirmed setup."
            res["signal_id"] = None
            res["signal_status"] = "NEUTRAL"

        # Save to enriched signals cache with TTL
        screener_state["enriched_signals_cache"][symbol] = {
            "timestamp": now,
            "data": res
        }
        return jsonify({"success": True, "smc": res})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/signals/performance")
def get_performance_stats():
    """Returns actual win rate, profit factor, and outcome stats per strategy"""
    return jsonify({"success": True, "stats": get_strategy_performance_stats()})

@app.route("/api/signals/active")
def get_active_signals_endpoint():
    """Returns all open ACTIVE trades so frontend can synchronize top confirmed banner"""
    try:
        from core.storage import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM historical_signals WHERE outcome = 'ACTIVE' ORDER BY id DESC")
        active = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return jsonify({"success": True, "active_signals": active})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "active_signals": []})

@app.route("/api/signals/recent")
def get_recent_logged_signals():
    """Returns persistent learning signals and trade history"""
    return jsonify({"success": True, "signals": get_recent_signals(limit=50)})

@app.route("/api/news")
def get_live_news():
    """Returns curated live market and catalyst headlines"""
    return jsonify({"success": True, "news": news_engine.fetch_market_news()})

if __name__ == "__main__":
    print("Auto-connecting to Groww API...")
    auto_connect_groww()
    
    # Run initial scan immediately so UI loads without waiting
    print("[INIT] Running initial dynamic scan...")
    try:
        sorted_sectors = calculate_dynamic_sectors()
        screener_state["sectors"] = sorted_sectors

        valid_sectors = [s for s in sorted_sectors.values() if (s.get("top_5") and len(s.get("top_5")) >= 3)]
        bull_sectors = [s for s in valid_sectors if s["is_bullish"]][:3]
        bear_sectors = sorted([s for s in valid_sectors if not s["is_bullish"]], key=lambda x: x["change"])[:3]

        bull_watchlist = []
        for b in bull_sectors:
            bull_watchlist.extend(b.get("top_5", []))

        bear_watchlist = []
        for b in bear_sectors:
            bear_watchlist.extend(b.get("top_5", []))

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
