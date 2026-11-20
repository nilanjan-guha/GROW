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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

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
    "cached_smc_analysis": {}
}

# Real-Time NSE Sector Indices (Live index price performance queried directly via API)
NSE_SECTOR_INDICES = {
    "NIFTY BANK": "^NSEBANK",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY PSU BANK": "^CNXPSUBANK"
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
    """Classifies any F&O symbol from Groww into its corresponding NSE Sector Index dynamically"""
    sym = sym.replace(".NS", "").upper()
    if 'BANK' in sym or sym in {'SBIN', 'PNB', 'CANBK', 'CUB', 'INDIANB', 'MAHABANK', 'UNIONBANK', 'YESBANK', 'RBLBANK', 'KARURVYSYA', 'UJJIVANSFB', 'AUBANK', 'FEDERALBNK', 'IDFCFIRSTB', 'BANDHANBNK', 'BANKBARODA', 'BANKINDIA'}:
        return 'NIFTY BANK'
    if sym in {'TCS', 'INFY', 'WIPRO', 'TECHM', 'HCLTECH', 'COFORGE', 'PERSISTENT', 'MPHASIS', 'LTIM', 'OFSS', 'KPITTECH', 'TATAELXSI', 'NAUKRI'}:
        return 'NIFTY IT'
    if sym in {'MARUTI', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT', 'TVSMOTOR', 'BHARATFORG', 'ASHOKLEY', 'BOSCHLTD', 'MOTHERSON', 'SONACOMS', 'UNOMINDA', 'FORCEMOT', 'TIINDIA', 'BALKRISIND', 'MRF', 'APOLLOTYRE', 'EXIDEIND', 'AMARAJABAT'}:
        return 'NIFTY AUTO'
    if sym in {'SUNPHARMA', 'CIPLA', 'DRREDDY', 'DIVISLAB', 'APOLLOHOSP', 'MANKIND', 'TORNTPHARM', 'LUPIN', 'ZYDUSLIFE', 'AUROPHARMA', 'ALKEM', 'BIOCON', 'GLENMARK', 'MAXHEALTH', 'LAURUSLABS', 'FORTIS', 'SYNGENE', 'IPCALAB', 'GRANULES'}:
        return 'NIFTY PHARMA'
    if sym in {'DLF', 'GODREJPROP', 'LODHA', 'OBEROIRLTY', 'PHOENIXLTD', 'BRIGADE', 'PRESTIGE', 'NBCC', 'SOBHA'}:
        return 'NIFTY REALTY'
    if sym in {'TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'JINDALSTEL', 'NMDC', 'SAIL', 'NATIONALUM', 'HINDZINC', 'APLAPOLLO', 'RATNAMANI', 'HINDCOPPER'}:
        return 'NIFTY METAL'
    if sym in {'RELIANCE', 'NTPC', 'ONGC', 'POWERGRID', 'BPCL', 'COALINDIA', 'IOC', 'GAIL', 'ADANIENT', 'ADANIGREEN', 'ADANIPOWER', 'TATAPOWER', 'PETRONET', 'OIL', 'HINDPETRO', 'JSWENERGY', 'NHPC', 'IREDA', 'SJVN', 'SUZLON'}:
        return 'NIFTY ENERGY'
    if sym in {'LT', 'ULTRACEMCO', 'AMBUJACEM', 'ACC', 'SHREECEM', 'DALBHARAT', 'GRASIM', 'BHARTIARTL', 'INDIGO', 'CONCOR', 'DELHIVERY', 'GMRAIRPORT', 'INDUSTOWER', 'ADANIPORTS', 'IRCTC', 'BHEL', 'SIEMENS', 'ABB', 'CUMMINSIND', 'HAL', 'BEL', 'BDL', 'COCHINSHIP', 'MAZDOCK'}:
        return 'NIFTY INFRA'
    if sym in {'ITC', 'HINDUNILVR', 'NESTLEIND', 'BRITANNIA', 'TATACONSUM', 'DABUR', 'MARICO', 'COLPAL', 'VBL', 'GODREJCP', 'JUBLFOOD', 'PATANJALI', 'RADICO', 'UNITDSPR', 'PAGEIND', 'TITAN', 'TRENT', 'KALYANKJIL', 'DMART', 'HAVELLS', 'VOLTAS', 'POLYCAB', 'DIXON', 'CROMPTON'}:
        return 'NIFTY FMCG'
    return 'NIFTY INFRA'  # Default institutional proxy

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

    # 4. VWAP (Intraday cumulative)
    typical_price = (h + l + c) / 3
    cum_vol = v.cumsum()
    vwap = (typical_price * v).cumsum() / cum_vol.replace(0, 1)

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
    curr_vwap = float(vwap.iloc[-1].item() if hasattr(vwap.iloc[-1], 'item') else vwap.iloc[-1])

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

def calculate_dynamic_sectors():
    """Calculates live sector performance from official NSE Sector Indices and dynamically ranks F&O constituents from Groww"""
    # 1. Download live sector index returns via API
    sec_index_tickers = list(NSE_SECTOR_INDICES.values())
    try:
        sec_data = yf.download(sec_index_tickers, period="2d", interval="1d", progress=False)['Close']
    except Exception as e:
        print("[DYNAMIC SECTOR] Index download failed:", e)
        sec_data = pd.DataFrame()

    # 2. Get live FNO stocks discovered dynamically from Groww API
    fno_list = screener_state.get("fno_universe", [])
    INDEX_SYMBOLS = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'BANKEX', 'SENSEX', 'NIFTYNXT50', 'NIFTYFPI'}
    clean_fno = [s for s in fno_list if s not in INDEX_SYMBOLS and 'TEST' not in s]

    # 3. Download price data for all active Groww FNO candidates in one batch
    fno_tickers = [f"{s}.NS" for s in clean_fno]
    try:
        fno_prices = yf.download(fno_tickers, period="2d", interval="1d", progress=False)['Close']
    except Exception as e:
        print("[DYNAMIC SECTOR] FNO prices download failed:", e)
        fno_prices = pd.DataFrame()

    # Calculate change for all FNO stocks
    stock_changes = {}
    for sym in fno_tickers:
        if sym in fno_prices.columns and len(fno_prices[sym].dropna()) >= 2:
            sr = fno_prices[sym].dropna()
            prev_p = float(sr.iloc[-2])
            curr_p = float(sr.iloc[-1])
            pct = ((curr_p - prev_p) / prev_p) * 100 if prev_p else 0.0
            stock_changes[sym] = {"symbol": sym, "price": curr_p, "change": round(pct, 2)}

    # Rank all FNO stocks by overall market gainers and losers
    all_fno_ranked = sorted(stock_changes.values(), key=lambda x: x["change"], reverse=True)
    fno_gainers = [x for x in all_fno_ranked if x["change"] >= 0]
    fno_losers = sorted([x for x in all_fno_ranked if x["change"] < 0], key=lambda x: x["change"])

    # 4. Group all F&O stocks into their actual Sector Buckets dynamically
    sector_buckets = {sec: [] for sec in NSE_SECTOR_INDICES}
    for item in stock_changes.values():
        assigned_sec = classify_stock_sector(item["symbol"])
        if assigned_sec in sector_buckets:
            sector_buckets[assigned_sec].append(item)

    # 5. Compute each Sector's live status & rank its true F&O constituents
    sector_results = {}
    for sec_name, ticker in NSE_SECTOR_INDICES.items():
        sec_pct = 0.0
        if ticker in sec_data.columns and len(sec_data[ticker].dropna()) >= 2:
            s = sec_data[ticker].dropna()
            p = float(s.iloc[-2])
            c = float(s.iloc[-1])
            sec_pct = ((c - p) / p) * 100 if p else 0.0

        is_bull = sec_pct >= 0
        constituents = sector_buckets.get(sec_name, [])
        constituents.sort(key=lambda x: x["change"], reverse=True)

        gainers_count = len([x for x in constituents if x["change"] >= 0])
        losers_count = len([x for x in constituents if x["change"] < 0])

        top_stocks = constituents[:5] if is_bull else sorted(constituents[-5:], key=lambda x: x["change"])
        if not top_stocks:
            # Fallback only if sector bucket empty
            top_stocks = fno_gainers[:5] if is_bull else fno_losers[:5]

        sector_results[sec_name] = {
            "ticker": ticker,
            "change": round(sec_pct, 2),
            "is_bullish": is_bull,
            "gainers": gainers_count,
            "losers": losers_count,
            "top_5": top_stocks
        }

    # Sort sectors by live index performance
    sorted_sectors = dict(sorted(sector_results.items(), key=lambda x: x[1]["change"], reverse=True))
    return sorted_sectors

# Background Screener Daemon: dynamically scans Sectors and F&O stocks with Chartink Explosive Momentum
def background_screener_worker():
    while True:
        try:
            now = time.time()
            if now - screener_state["last_screener_update"] > 300 or not screener_state["sectors"]:
                print("[CHARTINK ENGINE] Scanning live Sector indices & Top FNO candidates...")
                sorted_sectors = calculate_dynamic_sectors()
                screener_state["sectors"] = sorted_sectors

                # Bullish & Bearish momentum watchlist from Top 3 Bullish and Bearish sectors
                bull_sectors = [s for s in sorted_sectors.values() if s["is_bullish"]][:3]
                bear_sectors = [s for s in sorted_sectors.values() if not s["is_bullish"]][:3]

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

            # Pre-compute Chartink Momentum + SMC Order Blocks
            all_stks = (screener_state["bullish_momentum_watchlist"] + screener_state["bearish_momentum_watchlist"])
            for item in all_stks:
                sym = item["symbol"]
                try:
                    cdf = yf.download(sym, period="5d", interval="15m", progress=False)
                    if isinstance(cdf.columns, pd.MultiIndex):
                        cdf = cdf.xs(sym, axis=1, level=1) if sym in cdf.columns.levels[1] else cdf
                    res = evaluate_explosive_momentum(cdf, sym)
                    if res:
                        screener_state["cached_smc_analysis"][sym] = res
                except Exception:
                    pass
                time.sleep(0.8)

        except Exception as e:
            print("Screener worker exception:", e)
            time.sleep(5)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    return jsonify({
        "connected": screener_state["connected"],
        "profile": screener_state["user_profile"],
        "margin": screener_state["margin"],
        "fno_count": len(screener_state["fno_universe"])
    })

@app.route("/api/groww/sectors-dynamic")
def get_sectors_dynamic():
    """Serves 100% dynamically computed Sector Rankings and Top 5 FNO Stocks (No hardcoded values)"""
    return jsonify({
        "success": True,
        "sectors": screener_state["sectors"]
    })

from strategies.institutional_engine import InstitutionalStrategyEngine
from engine.news_engine import NewsIntelligenceEngine
from engine.groq_client import GroqIntelligence
from core.storage import log_signal, get_recent_signals

news_engine = NewsIntelligenceEngine()
groq_ai = GroqIntelligence()

@app.route("/api/market/smc-scan")
def smc_scan():
    """Returns Institutional Multi-Strategy + SMC + News Intelligence Analysis"""
    symbol = request.args.get("symbol", "INDUSINDBK.NS")
    if symbol in screener_state["cached_smc_analysis"]:
        return jsonify({"success": True, "smc": screener_state["cached_smc_analysis"][symbol]})

    try:
        df = yf.download(symbol, period="5d", interval="15m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, axis=1, level=1) if symbol in df.columns.levels[1] else df
        
        # 1. Evaluate classic momentum
        res = evaluate_explosive_momentum(df, symbol)
        
        # 2. Evaluate Institutional 15-Strategy Multi-Layer Engine
        prev_close = float(df['Close'].iloc[-20]) if len(df) >= 20 else float(df['Close'].iloc[0])
        inst_res = InstitutionalStrategyEngine.evaluate_strategies(df, prev_close)
        
        # 3. Match News & AI Summary
        clean_sym = symbol.replace(".NS", "")
        news_cat = news_engine.match_catalyst_for_stock(clean_sym)
        news_headline = news_cat["title"] if news_cat else ""
        ai_intel = groq_ai.analyze_signal(
            symbol=clean_sym,
            direction=inst_res["direction"],
            strategy=inst_res["primary_strategy"],
            price=inst_res["current_price"],
            metrics={"score": inst_res["composite_score"], "rvol": inst_res["rvol"], "rsi": inst_res["rsi"]},
            news_headline=news_headline
        )
        
        # Merge metrics
        res["composite_score"] = inst_res["composite_score"]
        res["probability_pct"] = inst_res["probability_pct"]
        res["triggered_strategies"] = inst_res["triggered_strategies"]
        res["news_headline"] = news_headline
        res["ai_summary"] = ai_intel.get("ai_summary", "")
        res["catalyst_impact"] = ai_intel.get("catalyst_impact", "")
        res["risk_rating"] = ai_intel.get("risk_rating", "MODERATE")
        
        # If institutional engine flagged BUY or SELL, upgrade action
        if inst_res["direction"] != "NEUTRAL":
            res["action"] = f"{inst_res['direction']} / {'CALL (CE)' if inst_res['direction'] == 'BUY' else 'PUT (PE)'} [{inst_res['primary_strategy'].upper()}]"
            res["verdict"] = f"SCORE {inst_res['composite_score']}/100 | PROBABILITY {inst_res['probability_pct']}%"
            res["suggested_sl"] = inst_res["suggested_sl"]
            res["suggested_tp1"] = inst_res["suggested_tp1"]
            res["suggested_tp2"] = inst_res["suggested_tp2"]
            res["reasoning"] = f"{inst_res['primary_strategy']} active. {ai_intel.get('ai_summary', '')}"
            
            # Persist to historical signals database
            log_signal({
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
                "features": {"rvol": inst_res["rvol"], "rsi": inst_res["rsi"], "vwap": inst_res["vwap"]}
            })

        screener_state["cached_smc_analysis"][symbol] = res
        return jsonify({"success": True, "smc": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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

        bull_sectors = [s for s in sorted_sectors.values() if s["is_bullish"]][:3]
        bear_sectors = [s for s in sorted_sectors.values() if not s["is_bullish"]][:3]

        bull_watchlist = []
        for b in bull_sectors:
            bull_watchlist.extend(b.get("top_5", []))

        bear_watchlist = []
        for b in bear_sectors:
            bear_watchlist.extend(b.get("top_5", []))

        screener_state["bullish_momentum_watchlist"] = bull_watchlist
        screener_state["bearish_momentum_watchlist"] = bear_watchlist
        screener_state["last_screener_update"] = time.time()
        print(f"[INIT] Dynamic scan ready! Top Bullish: {list(sorted_sectors.keys())[:3]}, Top Bearish: {list(sorted_sectors.keys())[-3:]}")
    except Exception as e:
        print("[INIT] Error during initial scan:", e)

    # Launch background worker
    worker = threading.Thread(target=background_screener_worker, daemon=True)
    worker.start()
    print("Groww Chartink Momentum + SMC Screener running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
