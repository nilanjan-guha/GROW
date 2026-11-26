import os
import json
import time
import threading
import traceback
import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import yfinance as yf
from yfinance.utils import get_yf_logger

# Suppress yfinance delisted/failed download console spam
_yf_logger = get_yf_logger()
_yf_logger.setLevel(logging.CRITICAL)
for _h in _yf_logger.handlers:
    _h.setLevel(logging.CRITICAL)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
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
from core.groww_data_provider import GrowwDataProvider

# Initialize Groww Native Data Provider
groww_data_provider = GrowwDataProvider()

# Initialize NSE Institutional Data Engine
nse_engine = NSEDataEngine()

def is_market_open():
    """Returns True if the Indian market (NSE/BSE) is open right now, strictly respecting weekends and holidays"""
    return MarketValidator.is_market_open()

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

# Global Screener State
# Seed fallback F&O universe from authoritative taxonomy so offline or cold starts immediately have universe
from engine.market.groww_sector_engine import GrowwSectorEngine
_taxonomy_map = GrowwSectorEngine().build_complete_fno_sector_universe_mapping(set())
_default_fno_list = sorted(list({s for s in _taxonomy_map.keys() if not s.endswith(".NS") and s != "UNMAPPED"}))

screener_state = {
    "connected": False,
    "user_profile": None,
    "margin": None,
    "fno_universe": _default_fno_list,
    "sectors": {},
    "bullish_momentum_watchlist": [],
    "bearish_momentum_watchlist": [],
    "last_screener_update": 0,
    "persistent_preparing_stocks": {}, # {symbol: {"direction": "...", "trigger_price": 123.4, "first_detected_date": "2026-11-25", "candle_ts": "..."}}
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
        screener_state["groww_client"] = client
        groww_data_provider.set_client(client)

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
            INDEX_SYMBOLS = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'BANKEX', 'SENSEX', 'NIFTYNXT50', 'NIFTYFPI'}
            raw_fno = [
                s for s in fno_df['underlying_symbol'].dropna().unique() 
                if s not in INDEX_SYMBOLS and 'TEST' not in s and not s.startswith('NIFTY') and not s.isdigit() and len(s) >= 2
            ]
            if raw_fno:
                screener_state["fno_universe"] = sorted(raw_fno)
                fno_loaded = True
                print(f"Dynamically discovered {len(raw_fno)} F&O tradeable stocks from Groww API.")

                # Register discovered F&O lot sizes into ContractMaster dynamically via authorized adapter secret
                try:
                    from core.contract_master import ContractMaster
                    contract_map = {}
                    today_str = str(datetime.date.today())
                    today_dt = datetime.date.today()
                    if 'lot_size' in fno_df.columns:
                        # Group rows by underlying symbol to systematically sort by nearest active expiry
                        grouped_rows = {}
                        for _, row in fno_df.dropna(subset=['underlying_symbol', 'lot_size']).iterrows():
                            u_sym = str(row['underlying_symbol']).strip().upper()
                            raw_lot = row['lot_size']
                            # Enforce genuine integer lot size
                            try:
                                if isinstance(raw_lot, float) and not raw_lot.is_integer():
                                    continue
                                lot_val = int(raw_lot)
                                if lot_val <= 0:
                                    continue
                            except Exception:
                                continue

                            # Broker instrument master metadata: Validate authoritative broker instrument ID!
                            # Trading symbol alone is NOT an instrument ID. Require authentic broker identifier
                            # such as exchange_token or numeric/alpha instrument_id from Groww instrument master.
                            raw_inst = (
                                row.get('exchange_token') 
                                or row.get('instrument_id') 
                                or row.get('groww_symbol')
                            )
                            if not raw_inst:
                                continue
                            inst_id = str(raw_inst).strip()
                            # Reject missing, non-authoritative trading symbol proxy, synthetic or malformed IDs
                            if not inst_id or ContractMaster.is_synthetic_instrument_id(inst_id):
                                continue
                            # Ensure instrument ID is not merely an unverified plain trading symbol string like RELIANCE_FUT
                            u_clean = u_sym.replace(".NS", "").strip()
                            if inst_id.upper() in (u_clean, f"{u_clean}_FUT", f"NSE_{u_clean}_FUT"):
                                continue

                            raw_exp = row.get('expiry_date') or row.get('expiry')
                            if not raw_exp:
                                continue
                            try:
                                exp_parsed = pd.to_datetime(raw_exp).date()
                                if exp_parsed < today_dt:
                                    continue  # Skip already expired contracts
                            except Exception:
                                continue

                            exp_val = str(exp_parsed)
                            grouped_rows.setdefault(u_sym, []).append({
                                "exp_dt": exp_parsed,
                                "exp_val": exp_val,
                                "lot_size": lot_val,
                                "instrument_id": inst_id
                            })

                        # Register every expiry with composite key; choose earliest active expiry as symbol alias
                        for u_sym, contracts in grouped_rows.items():
                            # Sort by nearest expiry date
                            contracts.sort(key=lambda x: x["exp_dt"])
                            for idx, c in enumerate(contracts):
                                spec_dict = {
                                    "lot_size": c["lot_size"],
                                    "expiry": c["exp_val"],
                                    "instrument_id": c["instrument_id"],
                                    "source": "GROWW_API",
                                    "update_date": today_str
                                }
                                contract_map[f"{u_sym}:{c['exp_val']}"] = spec_dict
                                if idx == 0:
                                    # Nearest active expiry becomes default symbol alias
                                    contract_map[u_sym] = spec_dict

                    if contract_map:
                        auth_secret = os.getenv("BROKER_ADAPTER_AUTH_SECRET")
                        if not auth_secret:
                            print("[CONTRACT_MASTER] Refusing dynamic contract registration: BROKER_ADAPTER_AUTH_SECRET environment variable is missing (fail closed).")
                        else:
                            ContractMaster.register_dynamic_contracts(contract_map, adapter_secret=auth_secret)
                            print(f"Registered {len(contract_map)} dynamic F&O contract lot sizes into ContractMaster via authorized Groww adapter.")
                except Exception as c_err:
                    print("ContractMaster dynamic registration notice:", c_err)

            # Dynamically extract all official company names from Groww cash segment instruments
            cash_nse = df[(df['exchange'] == 'NSE') & (df['segment'] == 'CASH')].dropna(subset=['trading_symbol', 'name'])
            symbol_to_name = dict(zip(cash_nse['trading_symbol'], cash_nse['name']))
            screener_state["symbol_names"] = symbol_to_name
            # Register dynamic company names into news engine
            news_engine.register_symbol_names(symbol_to_name)
            print(f"Dynamically mapped {len(symbol_to_name)} company names from Groww for real-time news.")

            # Dynamically map BSE exchange tokens to high-res Groww CDN logos (GSTK<bse_scrip_code>.png)
            bse_cash = df[(df['exchange'] == 'BSE') & (df['segment'] == 'CASH')].dropna(subset=['trading_symbol', 'exchange_token'])
            logo_map = {}
            for _, r in bse_cash.iterrows():
                s_sym = str(r['trading_symbol']).upper().strip()
                tok = str(r['exchange_token']).strip()
                if tok and tok.isdigit():
                    logo_map[s_sym] = f"https://assets-netstorage.groww.in/stock-assets/logos/GSTK{tok}.png"
            screener_state["stock_logos"] = logo_map
            print(f"Dynamically mapped {len(logo_map)} official Groww company logo CDN paths.")
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

    # First query authoritative groww_sector_engine universe mapping (zero HTTP / Yahoo Crumb overhead!)
    mapped_sec = groww_sector_engine.stock_to_sector_map.get(f"{clean_sym}.NS", {}).get("sector") or groww_sector_engine.stock_to_sector_map.get(clean_sym, {}).get("sector")
    if mapped_sec:
        screener_state["stock_sector_cache"][clean_sym] = mapped_sec
        return mapped_sec

    yf_ticker = f"{clean_sym}.NS"
    sector_name = "UNMAPPED"  # Fail-closed: Never default to NIFTY INFRA for unmapped stocks
    try:
        t = yf.Ticker(yf_ticker)
        # Handle Yahoo Finance 401 Unauthorized / Invalid Crumb gracefully
        fast_info = getattr(t, "fast_info", None)
        info_dict = getattr(t, "info", {}) if hasattr(t, "info") else {}
        yf_sec = str(info_dict.get("sector", "")).strip().lower()
        yf_ind = str(info_dict.get("industry", "")).strip().lower()

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
            sector_name = "UNMAPPED"
    except Exception:
        sector_name = "UNMAPPED"

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
            # 1. 5-minute Universal 219 F&O Momentum Refresh & Dynamic Sector Recalibration
            if now - screener_state["last_screener_update"] > 300 or not screener_state.get("bullish_momentum_watchlist"):
                print("[FNO SCANNER] Scanning all 219 F&O stocks & refreshing Groww trending sectors...")
                
                # Periodically refresh sectors and atomic mappings every 5 minutes
                try:
                    screener_state["sectors"] = calculate_dynamic_sectors()
                except Exception as se:
                    print("[SECTOR REFRESH ERROR]:", se)

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
                        # Fix P1 Issue 1: If current filter run returns no result for sym, clear/invalidate stale fno_filter_data!
                        # Do not fallback to existing_entry's old fno_filter_data, which causes stale setups to persist.
                        fno_data = filter_map.get(sym)
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

                # Build scanner list from:
                # 1. Watchlists + Open ACTIVE Signals (high priority)
                # 2. Persistent PREPARING (pre-breakout coil) stocks discovered today - tracked continuously so breakout is never missed!
                # 3. Rotating batch of 25 stocks from the full F&O universe (ensuring every F&O stock refreshes continuously)
                today_str = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d')
                
                # Daily purge: reset persistent preparing tracking on a new trading date
                prep_dict = screener_state.setdefault("persistent_preparing_stocks", {})
                for p_sym in list(prep_dict.keys()):
                    if prep_dict[p_sym].get("first_detected_date") != today_str:
                        del prep_dict[p_sym]

                preparing_tracked_symbols = list(prep_dict.keys())

                watchlist_stks = [x["symbol"] for x in (screener_state["bullish_momentum_watchlist"] + screener_state["bearish_momentum_watchlist"])]
                active_open_symbols = list(get_active_signal_symbols())
                
                fno_full_list = screener_state.get("fno_universe", [])
                rotating_batch = []
                if fno_full_list:
                    batch_size = 25
                    start_idx = fno_universe_index % len(fno_full_list)
                    end_idx = start_idx + batch_size
                    if end_idx <= len(fno_full_list):
                        rotating_batch = [f"{s}.NS" if not s.endswith(".NS") else s for s in fno_full_list[start_idx:end_idx]]
                    else:
                        rotating_batch = [f"{s}.NS" if not s.endswith(".NS") else s for s in (fno_full_list[start_idx:] + fno_full_list[:end_idx % len(fno_full_list)])]
                    fno_universe_index = (fno_universe_index + batch_size) % len(fno_full_list)

                all_stk_symbols = list(dict.fromkeys(watchlist_stks + active_open_symbols + preparing_tracked_symbols + rotating_batch))
                
                # Filter all evaluated candidates strictly to verified F&O instruments
                valid_fno_set = set(screener_state.get("fno_universe", []))
                norm_fno_set = {s.upper().replace(".NS", "") for s in valid_fno_set}

                candidates_for_ranking = []
                raw_cache = screener_state.get("raw_market_cache") or {}
                nifty_cached_entry = raw_cache.get("^NSEI") if isinstance(raw_cache, dict) else {}
                nifty_cached_df = nifty_cached_entry.get("df") if isinstance(nifty_cached_entry, dict) else None

                for sym in all_stk_symbols:
                    clean_sym = sym.replace(".NS", "").upper()
                    # Quality Filter: Only scan and rank stocks that are verified tradeable F&O stocks
                    if norm_fno_set and clean_sym not in norm_fno_set:
                        continue

                    try:
                        # Unified Groww Data Provider (single source of truth)
                        cdf, cdf_src = groww_data_provider.fetch_5m_candles(clean_sym, days_back=5)
                        
                        # Always cache raw OHLCV DataFrame if valid and non-empty (do not discard if momentum res is None!)
                        if cdf is not None and not cdf.empty and len(cdf) >= 15:
                            existing_entry = screener_state.get("raw_market_cache", {}).get(sym, {})
                            curr_stock_candle_ts = str(cdf.index[-1])
                            
                            # CRITICAL: Determine completed candle timestamp vs raw forming candle
                            feed_meta = MarketValidator.validate_feed_freshness(cdf, max_delay_seconds=300, candle_duration_seconds=300)
                            if not feed_meta.get("is_valid", False):
                                continue  # Reject invalid feed immediately (P0)

                            is_cdf_completed = feed_meta.get("is_completed_candle", False)
                            completed_candle_ts = curr_stock_candle_ts if is_cdf_completed else str(cdf.index[-2])

                            # Completed-Candle Freshness (P1): If stock's completed candle hasn't changed, skip to cycle to other stocks!
                            if sym not in active_open_symbols and sym not in preparing_tracked_symbols:
                                from engine.learning.historical_stock_learner import HistoricalStockLearner
                                if HistoricalStockLearner.should_skip_scan(clean_sym, current_completed_candle_ts=completed_candle_ts):
                                    continue
                            
                            # Invalidate stale fno_filter_data if it does NOT match the completed candle timestamp!
                            # Do NOT compare against forming candle ts!
                            existing_fno = existing_entry.get("fno_filter_data")
                            if existing_fno:
                                fno_eval_ts = existing_fno.get("evaluated_completed_candle_ts")
                                if fno_eval_ts and fno_eval_ts != completed_candle_ts:
                                    existing_fno = None  # Stale setup expired; clear to prevent false standalone bypass

                            res = evaluate_explosive_momentum(cdf, sym)
                            curr_p = float(cdf['Close'].squeeze().iloc[-1])
                            print(f"[SCANNING] {clean_sym:<12} | Price: ₹{curr_p:<8.2f} | Completed Candle: {completed_candle_ts}")
                            
                            screener_state.setdefault("raw_market_cache", {})[sym] = {
                                "timestamp": time.time(),
                                "raw_latest_candle_ts": curr_stock_candle_ts,
                                "evaluated_completed_candle_ts": completed_candle_ts,
                                "rvol_candle_ts": existing_fno.get("rvol_candle_ts") if existing_fno else None,
                                "data": res if res else existing_entry.get("data"),
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
            traceback.print_exc()
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

@app.route("/api/learning/diagnostics")
def get_learning_diagnostics():
    """
    Returns real-time diagnostics of the dedicated Continuous Tracker and
    Adaptive Learning post-mortem history.
    """
    try:
        from engine.adaptive_trade_learner import AdaptiveTradeLearner
        recent_post_mortems = AdaptiveTradeLearner.get_recent_learnings(limit=10)
        tracker_instance = screener_state.get("continuous_tracker_instance")
        tracker_stats = tracker_instance.tracked_stats if tracker_instance else {
            "active_prep_count": len(screener_state.get("persistent_preparing_stocks", {})),
            "open_signals_count": 0,
            "status": "ONLINE_ACTIVE"
        }
        return jsonify({
            "success": True,
            "tracker_stats": tracker_stats,
            "post_mortems": recent_post_mortems,
            "preparing_stocks": list(screener_state.get("persistent_preparing_stocks", {}).values())
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/groww/sectors-dynamic")
def get_sectors_dynamic():
    """Serves 100% dynamically computed Sector Rankings and Top 5 FNO Stocks directly from Groww Trending Sectors"""
    if not screener_state["sectors"]:
        screener_state["sectors"] = calculate_dynamic_sectors()
    return jsonify({
        "success": True,
        "sectors": screener_state["sectors"],
        "diagnostics": getattr(groww_sector_engine, "sector_diagnostics", {})
    })

@app.route("/api/groww/sectors-diagnostics")
def get_sectors_diagnostics():
    """Returns granular coverage ratio, expected universe comparison, and failed sector fetch reporting"""
    diag = getattr(groww_sector_engine, "sector_diagnostics", {})
    return jsonify({
        "success": True,
        "diagnostics": diag,
        "is_stale": getattr(groww_sector_engine, "is_stale", False),
        "is_usable": getattr(groww_sector_engine, "is_usable", False),
        "last_error": getattr(groww_sector_engine, "last_error", None)
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
    elif not symbol.endswith(".NS") and not symbol.startswith("^") and symbol != "NIFTY":
        symbol = f"{symbol}.NS"
    now = time.time()
    
    try:
        # Check raw market cache for pre-fetched dataframe to minimize network latency
        clean_s = symbol.replace(".NS", "").replace("^NSEI", "NIFTY").upper()
        raw_cache = screener_state.get("raw_market_cache") or {}
        raw_cached = raw_cache.get(symbol) or raw_cache.get(f"{clean_s}.NS") or raw_cache.get(clean_s)
        
        # Check fast memory cache for already computed smc analysis (TTL 120s)
        enriched_cached = screener_state.get("enriched_signals_cache", {}).get(symbol) or screener_state.get("enriched_signals_cache", {}).get(clean_s)
        if enriched_cached and (now - enriched_cached.get("timestamp", 0) < 120):
            return jsonify({"success": True, "smc": enriched_cached["data"]})

        if raw_cached and (now - raw_cached.get("timestamp", 0) < 300) and raw_cached.get("df") is not None:
            df = raw_cached["df"]
            if "data" in raw_cached and raw_cached["data"]:
                res = raw_cached["data"].copy()
            else:
                res = evaluate_explosive_momentum(df, symbol)
                if not res:
                    return jsonify({"success": False, "error": "Insufficient market data"}), 400
                raw_cached["data"] = res
        else:
            df, _ = groww_data_provider.fetch_5m_candles(clean_s, days_back=2)
            if df is None or df.empty:
                return jsonify({"success": False, "error": "Insufficient market data"}), 400
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
            # Even when feed is rejected for active signal generation, populate basic technicals for the UI
            reject_smc = res.copy() if res else {}
            
            # Safe extraction of current price with strict finite check
            curr_c = None
            try:
                raw_last_c = df['Close'].iloc[-1]
                val_last_c = float(raw_last_c)
                if math.isfinite(val_last_c) and val_last_c > 0:
                    curr_c = val_last_c
            except Exception:
                curr_c = None

            reject_smc.update({
                "action": "WAIT FOR ENTRY (NO TRADE)",
                "verdict": f"FEED REJECTED: {feed_val['status']}",
                "reasoning": feed_val["reason"],
                "current_price": curr_c,
                "signal_status": "NEUTRAL",
                "signal_id": None,
                "composite_score": 0,
                "probability_pct": 0
            })
            return jsonify({
                "success": True,
                "smc": reject_smc
            })

        is_completed_candle = feed_val["is_completed_candle"]
        latest_candle_ts = feed_val["latest_candle_ts"]

        # 1b. Check enriched cache (TTL: 180s) strictly after validating candle freshness and completion
        # If a newer completed candle has closed since cached analysis, invalidate cache
        cached = screener_state.get("enriched_signals_cache", {}).get(symbol) or screener_state.get("enriched_signals_cache", {}).get(clean_s)
        if cached:
            cached_candle_ts = cached.get("last_candle_ts", 0)
            if (now - cached.get("timestamp", 0) < 180) and (latest_candle_ts <= cached_candle_ts):
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
            nifty_cached = screener_state.get("raw_market_cache", {}).get("^NSEI") if isinstance(screener_state.get("raw_market_cache"), dict) else None
            if nifty_cached and isinstance(nifty_cached, dict) and (now - nifty_cached.get("timestamp", 0) < 300):
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
        
        # Check if candidate is in ranked opportunities to use its exact conviction score
        ranked_cache = screener_state.get("ranked_opportunities") or {}
        all_ranked_cands = (ranked_cache.get("top_bullish", []) + 
                            ranked_cache.get("top_bearish", []) + 
                            ranked_cache.get("entry_ready", []) + 
                            ranked_cache.get("ignition", []) + 
                            ranked_cache.get("preparing", []) + 
                            ranked_cache.get("observing", []))
        
        matching_cand = next((c for c in all_ranked_cands if c.get("symbol") in (symbol, clean_sym, f"{clean_sym}.NS")), None)
        if matching_cand and matching_cand.get("conviction_score") is not None:
            final_display_score = matching_cand["conviction_score"]
        else:
            final_display_score = inst_res["composite_score"]

        # Merge metrics into response (Using Session VWAP from Institutional Engine)
        res["vwap"] = inst_res["vwap"]
        res["rvol"] = inst_res["rvol"]
        res["vol_multiplier"] = inst_res["rvol"] if inst_res.get("rvol") else res.get("vol_multiplier", 1.0)
        res["composite_score"] = final_display_score
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
        res["has_eql_inducement_sweep"] = inst_res.get("has_eql_inducement_sweep", False)
        res["has_eqh_inducement_sweep"] = inst_res.get("has_eqh_inducement_sweep", False)
        res["nearest_bull_breaker"] = inst_res.get("nearest_bull_breaker")
        res["nearest_bear_breaker"] = inst_res.get("nearest_bear_breaker")
        
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
            # Check if this candidate is in ranked opportunities to verify entry-readiness / actionability
            ranked_cache = screener_state.get("ranked_opportunities") or {}
            all_ranked_cands = (ranked_cache.get("top_bullish", []) + 
                                ranked_cache.get("top_bearish", []) + 
                                ranked_cache.get("entry_ready", []) + 
                                ranked_cache.get("ignition", []) + 
                                ranked_cache.get("preparing", []) + 
                                ranked_cache.get("observing", []))
            
            matching_cand = next((c for c in all_ranked_cands if c.get("symbol") == symbol or c.get("symbol") == clean_sym), None)
            
            # FAIL-CLOSED ACTIONABILITY ENFORCEMENT (P1):
            # If candidate is missing from ranked cache or not explicitly actionable, reject trade generation!
            is_candidate_actionable = False
            cand_stage = "UNRANKED"
            if matching_cand is not None:
                is_candidate_actionable = matching_cand.get("is_actionable", False)
                cand_stage = matching_cand.get("stage", "WATCHLIST")
            
            # Fail closed: must be explicitly actionable (ENTRY READY / IGNITION) AND strictly completed 5m candle
            if not is_candidate_actionable or not is_completed_candle:
                res["action"] = f"MONITORING ({cand_stage}) - WAIT FOR TRIGGER"
                res["verdict"] = f"STAGE: {cand_stage} | SCORE {inst_res['composite_score']}/100"
                res["suggested_sl"] = None
                res["suggested_tp1"] = None
                res["suggested_tp2"] = None
                res["reasoning"] = f"Candidate is in stage {cand_stage} (watch-only). Strictly awaiting confirmed completed 5m breakout trigger."
                res["signal_id"] = None
                res["signal_status"] = "WATCH_ONLY"
                res["is_actionable"] = False
                
                screener_state.setdefault("enriched_signals_cache", {})[symbol] = {
                    "timestamp": now,
                    "last_candle_ts": latest_candle_ts,
                    "data": res
                }
                screener_state["enriched_signals_cache"][clean_s] = screener_state["enriched_signals_cache"][symbol]
                return jsonify({"success": True, "smc": res})

            res["is_actionable"] = True
            res["action"] = f"{inst_res['direction']} / {'CALL (CE)' if inst_res['direction'] == 'BUY' else 'PUT (PE)'} [{inst_res['primary_strategy'].upper()}]"
            res["verdict"] = f"SCORE {inst_res['composite_score']}/100 | WIN PROB {inst_res['probability_pct']}%"
            res["suggested_sl"] = inst_res["suggested_sl"]
            res["suggested_tp1"] = inst_res["suggested_tp1"]
            res["suggested_tp2"] = inst_res["suggested_tp2"]
            res["reasoning"] = f"{inst_res['primary_strategy']} active. {ai_intel.get('ai_summary', '')}"
            res["signal_status"] = "CONFIRMED (Completed Candle)" if is_completed_candle else "PROVISIONAL (Live Candle)"
            
            # Persist ONLY CONFIRMED signals generated during live market hours to historical signals database
            if is_completed_candle and mkt_is_open:
                sig_close_price = inst_res["current_price"]
                # For 5-minute intraday execution, expected entry at next candle open incorporates conservative slippage
                slippage_pct = 0.0005  # 0.05% slippage
                norm_dir = {"BULLISH": "BUY", "BEARISH": "SELL", "BUY": "BUY", "SELL": "SELL"}.get(str(inst_res["direction"]).upper())
                if not norm_dir:
                    print(f"[SIGNAL GENERATION ERROR] Unknown direction {inst_res['direction']} for {symbol}")
                    res["signal_id"] = None
                else:
                    exp_entry = round(sig_close_price * (1.0 + slippage_pct) if norm_dir == "BUY" else sig_close_price * (1.0 - slippage_pct), 2)
                    
                    # Compute candle start and close timestamps from validated feed
                    # In 5-minute market feeds, latest_candle_ts is the candle start timestamp
                    c_start_ts = float(latest_candle_ts)
                    c_close_ts = c_start_ts + 300.0  # Completed 5m candle close timestamp
                    
                    # Fetch contract lot size if F&O equity futures
                    from core.contract_master import ContractMaster
                    contract_spec = ContractMaster.get_contract_spec(clean_sym)
                    intended_qty = contract_spec["lot_size"] if contract_spec else 100
                    intended_inst = "FUTURES" if contract_spec else "EQUITY_INTRADAY"
                    
                    signal_id = log_signal({
                        "symbol": symbol,
                        "direction": norm_dir,
                        "strategy": inst_res["primary_strategy"],
                        "entry_price": exp_entry,
                        "stop_loss": inst_res["suggested_sl"],
                        "target_1": inst_res["suggested_tp1"],
                        "target_2": inst_res["suggested_tp2"],
                        "score": inst_res["composite_score"],
                        "confidence": f"{inst_res['probability_pct']}%",
                        "catalyst": news_headline or "Institutional Order Flow",
                        "features": {"rvol": inst_res["rvol"], "rsi": inst_res["rsi"], "vwap": inst_res["vwap"]},
                        "all_triggered": inst_res["triggered_strategies"],
                        "signal_price": sig_close_price,
                        "expected_entry_price": exp_entry,
                        "actual_entry_price": None,
                        "entry_timestamp": None,
                        "execution_status": "SIGNAL_GENERATED",
                        "signal_candle_start": c_start_ts,
                        "signal_candle_close_time": c_close_ts,
                        "signal_generated_at": now,
                        "quantity": intended_qty,
                        "instrument_type": intended_inst
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
        screener_state.setdefault("enriched_signals_cache", {})[symbol] = {
            "timestamp": now,
            "last_candle_ts": latest_candle_ts,
            "data": res
        }
        screener_state["enriched_signals_cache"][clean_s] = screener_state["enriched_signals_cache"][symbol]
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
    
    # Determine latest completed candle timestamp from NIFTY (Fail-Closed Benchmark Gate)
    raw_market = screener_state.get("raw_market_cache", {})
    nifty_cached = raw_market.get("^NSEI") if isinstance(raw_market, dict) else None
    nifty_df = nifty_cached.get("df") if isinstance(nifty_cached, dict) else None
    current_candle_ts = None
    eval_nifty = None
    if nifty_df is not None and not nifty_df.empty:
        nifty_val = MarketValidator.validate_feed_freshness(nifty_df, max_delay_seconds=300, candle_duration_seconds=300)
        # P1 Fail-Closed Gate: If NIFTY benchmark validation fails, reject immediately! Never construct eval_nifty from corrupted feed.
        if not nifty_val.get("is_valid", False):
            print(f"[BENCHMARK GATE] NIFTY validation rejected: {nifty_val.get('reason')}. Fail-closed: producing no actionable rankings.")
            return {
                "top_bullish": [], "top_bearish": [], "entry_ready": [], "ignition": [], "preparing": [], "observing": [],
                "benchmark_status": "INVALID_BENCHMARK",
                "benchmark_reason": nifty_val.get("reason")
            }, None

        is_nifty_comp = nifty_val.get("is_completed_candle", False)
        eval_nifty = nifty_df if is_nifty_comp else nifty_df.iloc[:-1]
        if not eval_nifty.empty:
            current_candle_ts = str(eval_nifty.index[-1])
    
    if not current_candle_ts or eval_nifty is None or eval_nifty.empty:
        # If NIFTY data is unavailable or corrupted, market-wide synchronization cannot be established
        return {
            "top_bullish": [], "top_bearish": [], "entry_ready": [], "ignition": [], "preparing": [], "observing": [],
            "benchmark_status": "BENCHMARK_UNAVAILABLE"
        }, None

    candidates = []
    valid_fno_set = set(screener_state.get("fno_universe", []))
    norm_fno_set = {s.upper().replace(".NS", "") for s in valid_fno_set}

    # Sector-First Hierarchy:
    # 1. Select Top 3 Bullish, Top 3 Bearish, and Neutral sectors by explicit live performance sorting
    # Enforce sector cache freshness and coverage
    sectors = screener_state.get("sectors", {})
    if not sectors or getattr(groww_sector_engine, "is_stale", False):
        fresh_sectors = calculate_dynamic_sectors()
        if fresh_sectors:
            sectors = fresh_sectors
            screener_state["sectors"] = sectors

    # Explicitly sort sectors by performance
    sorted_bull_sectors = sorted(
        [(name, data) for name, data in sectors.items() if data.get("is_bullish") and not data.get("is_neutral") and data.get("change", 0.0) > 0.0],
        key=lambda x: x[1].get("change", 0.0),
        reverse=True
    )[:3]
    sorted_bear_sectors = sorted(
        [(name, data) for name, data in sectors.items() if not data.get("is_bullish") and not data.get("is_neutral") and data.get("change", 0.0) < 0.0],
        key=lambda x: x[1].get("change", 0.0)
    )[:3]
    # Include Neutral / Sideways sectors for stock-level discovery
    sorted_neutral_sectors = [
        (name, data) for name, data in sectors.items() if data.get("is_neutral")
    ][:2]
    
    priority_sectors = set([s[0] for s in sorted_bull_sectors] + [s[0] for s in sorted_bear_sectors] + [s[0] for s in sorted_neutral_sectors])

    # Gather all F&O constituent symbols from top sectors (not just top 5)
    sector_constituents = set()
    for s_name in priority_sectors:
        stks = sectors[s_name].get("all_constituents") or sectors[s_name].get("top_5") or sectors[s_name].get("top_5_stocks") or []
        for stk in stks:
            sym_clean = stk["symbol"] if isinstance(stk, dict) else str(stk)
            clean_bare = sym_clean.replace(".NS", "").upper()
            if not norm_fno_set or clean_bare in norm_fno_set:
                sector_constituents.add(sym_clean)
                sector_constituents.add(f"{clean_bare}.NS")

    raw_items = screener_state.get("raw_market_cache", {})
    for sym, item in raw_items.items():
        if not item or not isinstance(item, dict) or sym == "^NSEI" or item.get("df") is None:
            continue
        clean_sym = sym.replace(".NS", "").upper()
        if norm_fno_set and clean_sym not in norm_fno_set:
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

        # Sector-First Priority with Stock-Level Discovery Exception:
        # Prioritize top sector constituents, BUT allow stocks with exceptional 5m momentum/coils (pre_coil or high RVOL)
        # CRITICAL: Verify that the standalone setup data matches the EXACT current completed candle timestamp!
        # Both evaluated_completed_candle_ts and rvol_candle_ts must match stock_candle_ts.
        item_dict = item if isinstance(item, dict) else {}
        fno_data_raw = item_dict.get("fno_filter_data")
        fno_data_item = fno_data_raw if isinstance(fno_data_raw, dict) else {}
        fno_eval_ts = fno_data_item.get("evaluated_completed_candle_ts")
        fno_rvol_ts = fno_data_item.get("rvol_candle_ts")
        
        # Mandatory freshness: must have valid evaluated candle timestamp equal to current completed candle
        is_fno_fresh = bool(fno_eval_ts and fno_eval_ts == stock_candle_ts and (fno_rvol_ts is None or fno_rvol_ts == stock_candle_ts))

        is_persistent_preparing = bool(sym in screener_state.get("persistent_preparing_stocks", {}) or f"{clean_sym}.NS" in screener_state.get("persistent_preparing_stocks", {}))

        has_standalone_setup = bool(
            is_persistent_preparing or (
                is_fno_fresh and (
                    fno_data_item.get("is_pre_coil", False) or 
                    fno_data_item.get("rvol", 0.0) >= 1.5 or 
                    fno_data_item.get("velocity", 0.0) >= 2.0
                )
            )
        )

        if sector_constituents and sym not in sector_constituents and f"{clean_sym}.NS" not in sector_constituents:
            # If outside top sectors, only evaluate if it has a validated, FRESH standalone pre-breakout setup
            if not has_standalone_setup and (not allowed_sector_symbols or sym not in allowed_sector_symbols):
                continue

        sec_info = groww_sector_engine.stock_to_sector_map.get(sym, {})
        sec_name = sec_info.get("sector")
        sec_chg = sec_info.get("sector_change", 0.0)
        if not sec_name:
            sec_name = classify_stock_sector(sym)
            sec_data = screener_state.get("sectors", {}).get(sec_name, {})
            sec_chg = sec_data.get("change", 0.0)
        
        # Sector benchmark DataFrame if available
        sec_ticker = groww_sector_engine.get_sector_index_ticker(sec_name)
        sec_df = screener_state.get("raw_market_cache", {}).get(sec_ticker, {}).get("df") if sec_ticker else None

        # Fetch or retrieve cached 30-day daily dataset for genuine daily learning & ATR (P0 Fix)
        daily_cached = screener_state.get("daily_market_cache", {}).get(clean_sym)
        daily_stock_df = None
        if daily_cached and (now - daily_cached.get("timestamp", 0) < 3600):
            daily_stock_df = daily_cached.get("df")
        else:
            d_df, _ = groww_data_provider.fetch_daily_candles(clean_sym, days_back=45)
            if d_df is not None and not d_df.empty and len(d_df) >= 15:
                daily_stock_df = d_df
                screener_state.setdefault("daily_market_cache", {})[clean_sym] = {
                    "timestamp": now,
                    "df": d_df
                }

        # Extract time_slot_rvol and observation count from fno_filter_data if present
        time_slot_rvol = fno_data_item.get("rvol") if is_fno_fresh else None
        rvol_obs = fno_data_item.get("rvol_obs", 0) if is_fno_fresh else 0

        # Execute unified analysis via ScreenerPipeline (P0: ensures identical pipeline for live & replay)
        from engine.intelligence.screener_pipeline import ScreenerPipeline
        pipe_res = ScreenerPipeline.evaluate_candidate_unified(
            stock_df=eval_df_sym,
            symbol=sym,
            sector_name=sec_name,
            sector_change=sec_chg,
            sector_bias=screener_state.get("sectors", {}).get(sec_name, {}).get("bias", "NEUTRAL"),
            nifty_df=eval_nifty,
            daily_df=daily_stock_df,
            prev_close=None,
            time_slot_rvol=time_slot_rvol,
            rvol_obs=rvol_obs,
            is_replay=False,
            sector_df=sec_df
        )

        if pipe_res and pipe_res.get("opp_res"):
            candidates.append(pipe_res["opp_res"])

    if not candidates:
        ranked = {
            "top_bullish": [], "top_bearish": [], "entry_ready": [], "ignition": [], "preparing": [], "observing": []
        }
    else:
        ranked = OpportunityRankingEngine.rank_top_opportunities(candidates, top_n=3)

    # Strictly sort preparing opportunities by conviction score descending
    if ranked.get("preparing"):
        ranked["preparing"].sort(key=lambda x: x.get("conviction_score", 0), reverse=True)

    # Intraday Persistent PREPARING Tracker:
    # Any stock identified in PREPARING (pre-breakout coil) is added to intraday persistent tracking
    # so the background worker continues to scan it on every cycle until breakout or market close.
    today_str = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d')
    persistent_prep = screener_state.setdefault("persistent_preparing_stocks", {})
    for prep_item in ranked.get("preparing", []):
        p_sym = prep_item.get("symbol")
        if p_sym:
            persistent_prep[p_sym] = {
                "symbol": p_sym,
                "direction": prep_item.get("direction"),
                "entry_trigger": prep_item.get("entry_trigger"),
                "change_pct": prep_item.get("change_pct", 0.0),
                "conviction_score": prep_item.get("conviction_score", 70),
                "current_price": prep_item.get("current_price"),
                "first_detected_date": today_str,
                "first_detected_ts": now,
                "last_seen_candle_ts": current_candle_ts
            }

    # If a persistently tracked stock has broken out or triggered IGNITION/CONFIRMED,
    # ensure it is promoted directly into entry_ready so the trader never misses the move:
    for cand in candidates:
        c_sym = cand.get("symbol")
        if c_sym in persistent_prep:
            # Check if this preparing stock broke out or ignited
            if cand.get("stage") in ("CONFIRMED", "IGNITION") and cand.get("is_actionable"):
                if cand not in ranked.get("entry_ready", []):
                    ranked["entry_ready"].insert(0, cand)
                    ranked["entry_ready"] = ranked["entry_ready"][:2]

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
        raw_market = screener_state.get("raw_market_cache", {})
        nifty_cached = raw_market.get("^NSEI") if isinstance(raw_market, dict) else None
        nifty_df = nifty_cached.get("df") if isinstance(nifty_cached, dict) else None
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
                    "entry_ready": cached_ranked.get("entry_ready", []),
                    "ignition": cached_ranked.get("ignition", []),
                    "preparing": cached_ranked.get("preparing", []),
                    "observing": cached_ranked.get("observing", [])
                }
            })

        ranked, current_candle_ts = compute_and_update_ranked_opportunities()
        if not ranked or not any(ranked.values()):
            return jsonify({
                "success": True,
                "status": "NO_TRADE",
                "ranked": {
                    "top_bullish": [], "top_bearish": [], "entry_ready": [], "ignition": [], "preparing": [], "observing": []
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
        # Also return the latest closed/tracked trades (e.g. TARGET HIT, SL HIT) with clear status tags
        cursor.execute("SELECT * FROM historical_signals ORDER BY id DESC LIMIT 15")
        recent_all = [dict(r) for r in cursor.fetchall()]
        conn.close()

        # Enrich active and recent signals with live price, day change, and lot size from ContractMaster/raw_cache
        from core.contract_master import ContractMaster
        raw_cache = screener_state.get("raw_market_cache") or {}
        for sig_list in (active, recent_all):
            for s in sig_list:
                sym = s.get("symbol", "")
                clean_sym = sym.replace(".NS", "").upper()
                # Lot size
                spec = ContractMaster.get_contract_spec(clean_sym)
                if spec:
                    s["lot_size"] = spec.get("lot_size") if isinstance(spec, dict) else getattr(spec, "lot_size", s.get("quantity") or 100)
                else:
                    s["lot_size"] = s.get("quantity") or 100
                # Live price & day change
                cache_entry = raw_cache.get(sym) or raw_cache.get(clean_sym) or raw_cache.get(f"{clean_sym}.NS")
                if cache_entry and isinstance(cache_entry, dict) and cache_entry.get("df") is not None:
                    cdf = cache_entry["df"]
                    if not cdf.empty:
                        close_col = "close" if "close" in cdf.columns else ("Close" if "Close" in cdf.columns else None)
                        open_col = "open" if "open" in cdf.columns else ("Open" if "Open" in cdf.columns else None)
                        if close_col:
                            raw_val = cdf[close_col].squeeze()
                            s["live_price"] = round(float(raw_val.iloc[-1] if hasattr(raw_val, 'iloc') else raw_val), 2)
                        if open_col and close_col:
                            raw_open = cdf[open_col].squeeze()
                            first_o = float(raw_open.iloc[0] if hasattr(raw_open, 'iloc') else raw_open)
                            if first_o > 0 and "live_price" in s:
                                s["change_pct"] = round(((s["live_price"] - first_o) / first_o) * 100.0, 2)
                if "live_price" not in s or not s["live_price"]:
                    s["live_price"] = s.get("signal_price") or s.get("entry_price") or 0.0

        return jsonify({"success": True, "active_signals": active, "recent_tracked": recent_all, "is_market_open": mkt_open})
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

        clean_s = sym.replace(".NS", "")
        # SERVER-SIDE INDEPENDENT SIGNAL VALIDATION (P1):
        # Never trust client-provided actionability as authorization to log a trade!
        # Revalidate server-side against current ranked opportunities, exact candle timestamp, direction, and price limits
        ranked_cache = screener_state.get("ranked_opportunities") or {}
        all_ranked_cands = (ranked_cache.get("top_bullish", []) + 
                            ranked_cache.get("top_bearish", []) + 
                            ranked_cache.get("entry_ready", []) + 
                            ranked_cache.get("ignition", []) + 
                            ranked_cache.get("preparing", []) + 
                            ranked_cache.get("observing", []))
        
        matching_cand = next((c for c in all_ranked_cands if c.get("symbol") == sym or c.get("symbol") == clean_s), None)
        if matching_cand is None or not matching_cand.get("is_actionable", False):
            return jsonify({"success": False, "error": "Server rejection: Candidate is not an active, actionable trade setup."}), 403

        # 1. Reject logging if candidate stage is watch-only or non-actionable
        cand_stage = matching_cand.get("stage", "")
        if cand_stage not in ("CONFIRMED", "IGNITION", "ENTRY READY"):
            return jsonify({"success": False, "error": f"Server rejection: Stage {cand_stage} is strictly watch-only."}), 403

        # 2. Revalidate completed 5m candle freshness: must match the authoritative latest completed market candle
        # Reject signal logging if the authoritative latest completed market candle cannot be established!
        raw_market = screener_state.get("raw_market_cache", {})
        nifty_cached = raw_market.get("^NSEI") if isinstance(raw_market, dict) else None
        nifty_df = nifty_cached.get("df") if isinstance(nifty_cached, dict) else None
        latest_market_completed_ts = None
        if nifty_df is not None and not nifty_df.empty:
            nifty_val = MarketValidator.validate_feed_freshness(nifty_df, max_delay_seconds=300, candle_duration_seconds=300)
            if nifty_val.get("is_valid", False):
                is_nifty_comp = nifty_val.get("is_completed_candle", False)
                eval_nifty = nifty_df if is_nifty_comp else nifty_df.iloc[:-1]
                if not eval_nifty.empty:
                    latest_market_completed_ts = str(eval_nifty.index[-1])

        server_ranked_ts = screener_state.get("ranked_candle_ts") or ranked_cache.get("ranked_candle_ts")
        cand_ts = matching_cand.get("candle_ts")
        client_candle_ts = data.get("candle_ts")

        # Authoritative live market baseline MUST exist. NO unsafe fallback to unverified candidate timestamp!
        # Both latest completed NIFTY benchmark timestamp AND fresh server ranked cache timestamp must agree.
        if not latest_market_completed_ts:
            return jsonify({"success": False, "error": "Server rejection: Live market completed candle baseline (NIFTY) cannot be verified."}), 409

        authoritative_ts = latest_market_completed_ts

        # Reject if the ranked cache is older or missing
        if not server_ranked_ts or str(server_ranked_ts).strip() != str(authoritative_ts).strip():
            return jsonify({"success": False, "error": f"Server rejection: Ranked candidate cache ({server_ranked_ts}) is stale compared to live market baseline ({authoritative_ts})."}), 409

        # Mandatory Candidate Timestamp Enforcement (P1):
        if not cand_ts or str(cand_ts).strip() == "":
            return jsonify({"success": False, "error": "Server rejection: Missing candle timestamp on ranked candidate."}), 409

        if str(cand_ts).strip() != str(authoritative_ts).strip():
            return jsonify({"success": False, "error": f"Server rejection: Candidate candle ({cand_ts}) is stale compared to authoritative completed candle ({authoritative_ts})."}), 409

        # Independent Stock-Feed Freshness & Timestamp Synchronization Revalidation (P0):
        # Missing or empty stock data is a hard rejection. Stock's completed candle must match market baseline!
        stock_cached_entry = raw_market.get(sym) or raw_market.get(f"{clean_s}.NS") or raw_market.get(clean_s)
        if not isinstance(stock_cached_entry, dict) or stock_cached_entry.get("df") is None:
            return jsonify({"success": False, "error": f"Server rejection: Live market feed data for {sym} is missing from cache."}), 409

        stk_df = stock_cached_entry["df"]
        if stk_df.empty:
            return jsonify({"success": False, "error": f"Server rejection: Live market feed for {sym} is empty."}), 409

        stk_val = MarketValidator.validate_feed_freshness(stk_df, max_delay_seconds=300, candle_duration_seconds=300)
        if not stk_val.get("is_valid", False):
            return jsonify({"success": False, "error": f"Server rejection: Stock feed {sym} failed freshness revalidation ({stk_val.get('reason')})."}), 409

        # Verify stock latest completed candle equals authoritative NIFTY baseline
        is_stk_comp = stk_val.get("is_completed_candle", False)
        eval_stk = stk_df if is_stk_comp else stk_df.iloc[:-1]
        if eval_stk.empty or str(eval_stk.index[-1]).strip() != str(authoritative_ts).strip():
            stk_last_ts = str(eval_stk.index[-1]) if not eval_stk.empty else "NONE"
            return jsonify({"success": False, "error": f"Server rejection: Stock {sym} completed candle ({stk_last_ts}) is out of sync with market baseline ({authoritative_ts})."}), 409

        # Contract Master Validation:
        # Verify the underlying contract is recognized in the F&O contract master with valid positive lot size
        from core.contract_master import ContractMaster
        contract_spec = ContractMaster.get_contract_spec(clean_s)
        if not contract_spec or not contract_spec.get("lot_size") or int(contract_spec.get("lot_size")) <= 0:
            return jsonify({"success": False, "error": f"Server rejection: No valid F&O contract specification found for {clean_s}."}), 403

        if not client_candle_ts or str(client_candle_ts).strip() == "":
            return jsonify({"success": False, "error": "Server rejection: Missing candle timestamp in trade submission."}), 400
        
        if str(client_candle_ts).strip() != str(authoritative_ts).strip():
            return jsonify({"success": False, "error": f"Server rejection: Stale candle timestamp (client: '{client_candle_ts}' vs server: '{authoritative_ts}')."}), 409

        # 3. Validate directional alignment against the server's evaluated direction
        # Strict enum token parsing: Reject conflicting directions or ambiguous strings!
        raw_server_dir = (matching_cand.get("direction") or "").upper().strip()
        if raw_server_dir in ("BUY", "CALL", "BULLISH", "LONG"):
            server_direction = "BULLISH"
        elif raw_server_dir in ("SELL", "PUT", "BEARISH", "SHORT"):
            server_direction = "BEARISH"
        else:
            return jsonify({"success": False, "error": f"Server rejection: Invalid server candidate direction '{raw_server_dir}'."}), 400

        # Tokenize client submission
        import re
        tokens = set(re.findall(r'[A-Za-z]+', f"{data.get('action', '')} {data.get('direction', '')}".upper()))
        
        is_client_bull = bool(tokens & {"BUY", "CALL", "BULLISH", "LONG"})
        is_client_bear = bool(tokens & {"SELL", "PUT", "BEARISH", "SHORT"})
        is_client_wait = bool(tokens & {"WAIT", "MONITORING", "NEUTRAL", "HOLD", "OBSERVING"})

        if is_client_wait or (is_client_bull and is_client_bear):
            return jsonify({"success": False, "error": "Server rejection: Ambiguous, conflicting, or non-actionable direction tokens in client submission."}), 400
        
        if is_client_bull:
            normalized_client_dir = "BULLISH"
        elif is_client_bear:
            normalized_client_dir = "BEARISH"
        else:
            return jsonify({"success": False, "error": "Server rejection: No valid directional tokens found in client submission."}), 400

        if normalized_client_dir != server_direction:
            return jsonify({"success": False, "error": f"Server rejection: Direction mismatch ({normalized_client_dir} vs server {server_direction})."}), 400

        # 4. Strict Server-Side Price & Risk Authorization (Zero client fallbacks allowed):
        # Do not use client-supplied entry, SL or target when server-calculated values are missing. Reject the signal instead.
        server_entry = matching_cand.get("current_price")
        server_sl = matching_cand.get("suggested_sl")
        server_tp1 = matching_cand.get("suggested_tp1")
        server_tp2 = matching_cand.get("suggested_tp2")

        if server_entry is None or server_sl is None or server_tp1 is None:
            return jsonify({"success": False, "error": "Server rejection: Candidate lacks server-evaluated execution prices or risk levels."}), 400

        try:
            entry_val = float(server_entry)
            sl_val = float(server_sl)
            tp1_val = float(server_tp1)
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": "Server rejection: Non-numeric server execution prices."}), 400

        import math
        if not (math.isfinite(entry_val) and math.isfinite(sl_val) and math.isfinite(tp1_val)) or entry_val <= 0 or sl_val <= 0 or tp1_val <= 0:
            return jsonify({"success": False, "error": "Server rejection: Invalid non-finite or non-positive execution prices."}), 400

        # Strict TP2 Validation (Item 8):
        # If client or server provides TP2, it MUST be numeric, finite, positive, and directionally valid. Never silently convert invalid TP2 to None!
        validated_tp2 = None
        raw_tp2 = server_tp2 if server_tp2 is not None else data.get("suggested_tp2") or data.get("target_2")
        if raw_tp2 is not None and str(raw_tp2).strip() != "":
            try:
                tp2_num = float(raw_tp2)
                if not math.isfinite(tp2_num) or tp2_num <= 0:
                    return jsonify({"success": False, "error": f"Server rejection: Provided TP2 ({raw_tp2}) is non-positive or non-finite."}), 400
                validated_tp2 = tp2_num
            except (ValueError, TypeError):
                return jsonify({"success": False, "error": f"Server rejection: Provided TP2 ({raw_tp2}) is not a valid number."}), 400

        if server_direction == "BULLISH":
            if not (sl_val < entry_val and tp1_val > entry_val):
                return jsonify({"success": False, "error": f"Server rejection: Structural boundary violation for BULLISH (SL {sl_val} must be < Entry {entry_val} < TP1 {tp1_val})."}), 400
            if validated_tp2 is not None and validated_tp2 <= tp1_val:
                return jsonify({"success": False, "error": f"Server rejection: Target 2 ({validated_tp2}) must be strictly higher than Target 1 ({tp1_val}) for BULLISH."}), 400
            risk = entry_val - sl_val
            reward = tp1_val - entry_val
        elif server_direction == "BEARISH":
            if not (sl_val > entry_val and tp1_val < entry_val):
                return jsonify({"success": False, "error": f"Server rejection: Structural boundary violation for BEARISH (SL {sl_val} must be > Entry {entry_val} > TP1 {tp1_val})."}), 400
            if validated_tp2 is not None and validated_tp2 >= tp1_val:
                return jsonify({"success": False, "error": f"Server rejection: Target 2 ({validated_tp2}) must be strictly lower than Target 1 ({tp1_val}) for BEARISH."}), 400
            risk = sl_val - entry_val
            reward = entry_val - tp1_val
        else:
            return jsonify({"success": False, "error": "Server rejection: Unknown directional state."}), 400

        if risk <= 0:
            return jsonify({"success": False, "error": "Server rejection: Calculated risk is zero or negative."}), 400

        rr_ratio = reward / risk
        if rr_ratio < 1.20:
            return jsonify({"success": False, "error": f"Server rejection: Minimum risk/reward threshold not met (RR 1:{rr_ratio:.2f} < 1:1.20)."}), 400

        slippage_pct = 0.0005
        normalized_direction = {
            "BULLISH": "BUY",
            "BEARISH": "SELL",
            "BUY": "BUY",
            "SELL": "SELL"
        }.get(server_direction.upper())
        if not normalized_direction:
            return jsonify({"success": False, "error": f"Server rejection: Direction '{server_direction}' cannot be normalized to BUY or SELL."}), 400

        exp_entry = round(entry_val * (1.0 + slippage_pct) if normalized_direction == "BUY" else entry_val * (1.0 - slippage_pct), 2)

        # Derive authoritative candle timestamps without guessing
        # authoritative_ts is the start timestamp of the completed 5m candle
        try:
            if hasattr(authoritative_ts, 'timestamp'):
                c_start_ts = float(authoritative_ts.timestamp())
            elif isinstance(authoritative_ts, (int, float)):
                c_start_ts = float(authoritative_ts)
            elif isinstance(authoritative_ts, str) and authoritative_ts.replace('.', '', 1).isdigit():
                c_start_ts = float(authoritative_ts)
            else:
                parsed_dt = pd.to_datetime(authoritative_ts)
                c_start_ts = float(parsed_dt.timestamp())
        except Exception as ts_err:
            return jsonify({"success": False, "error": f"Server rejection: Unparseable authoritative candle timestamp '{authoritative_ts}': {ts_err}"}), 400

        c_close_ts = c_start_ts + 300.0

        # Determine intended quantity and instrument type from contract master or client request
        from core.contract_master import ContractMaster
        clean_cand_sym = sym.replace(".NS", "").upper()
        req_inst = (data.get("instrument_type") or "").upper()
        req_expiry = data.get("expiry") or data.get("contract_expiry")

        if req_inst == "OPTIONS":
            # Reject OPTIONS until dedicated option chain contracts (strike, expiry, type) are integrated
            return jsonify({"success": False, "error": "Server rejection: OPTIONS trading mode requires live contract-level option chain and strikes (currently disabled until option chain feed is active)."}), 400

        # Query contract specification (with explicit expiry if requested)
        contract_spec = ContractMaster.get_contract_spec(clean_cand_sym, expiry=req_expiry)

        # STRICT FUTURES IDENTITY ENFORCEMENT (P0):
        # A request for FUTURES mode MUST have an authoritative broker contract spec with exact instrument_id, expiry, and lot_size.
        # It must NEVER fall back to quantity 100 or proceed with missing contract identity!
        if req_inst == "FUTURES":
            if not contract_spec or not contract_spec.get("instrument_id") or not contract_spec.get("expiry") or not contract_spec.get("lot_size"):
                return jsonify({
                    "success": False,
                    "error": f"Server rejection: FUTURES trading mode requires a valid registered broker contract with exact instrument_id, expiry, and lot_size for {clean_cand_sym}."
                }), 400
            intended_inst = "FUTURES"
            intended_qty = int(contract_spec["lot_size"])
        elif req_inst == "EQUITY_INTRADAY":
            intended_inst = "EQUITY_INTRADAY"
            req_qty = data.get("quantity")
            intended_qty = int(req_qty) if (req_qty and int(req_qty) > 0) else 100
        else:
            # Default determination: if contract_spec exists, default to FUTURES; otherwise EQUITY_INTRADAY
            if contract_spec:
                intended_inst = "FUTURES"
                intended_qty = int(contract_spec["lot_size"])
            else:
                intended_inst = "EQUITY_INTRADAY"
                intended_qty = 100

        sig_id = log_signal({
            "symbol": sym,
            "direction": normalized_direction,
            "strategy": matching_cand.get("stage_desc") or "SMC Structure",
            "entry_price": exp_entry,
            "stop_loss": sl_val,
            "target_1": tp1_val,
            "target_2": validated_tp2,
            "score": matching_cand.get("conviction_score", 75),
            "confidence": f"{matching_cand.get('confirmation_score', 75)}%",
            "catalyst": "SMC Order Flow Ignition",
            "features": {"rvol": matching_cand.get("rvol", 1.0), "rs_nifty": matching_cand.get("rs_nifty", 0.0), "candle_ts": str(authoritative_ts)},
            "all_triggered": matching_cand.get("reasons", []),
            "signal_price": entry_val,
            "expected_entry_price": exp_entry,
            "actual_entry_price": exp_entry,
            "entry_timestamp": time.time(),
            "execution_status": "PAPER_FILLED",
            "signal_candle_start": c_start_ts,
            "signal_candle_close_time": c_close_ts,
            "signal_generated_at": time.time(),
            "quantity": intended_qty,
            "instrument_type": intended_inst,
            "instrument_id": contract_spec.get("instrument_id") if contract_spec else None,
            "contract_expiry": str(contract_spec.get("expiry")) if contract_spec and contract_spec.get("expiry") else None,
            "contract_source": contract_spec.get("source") if contract_spec else None
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

@app.route("/api/stock-logos")
def get_stock_logos_endpoint():
    """Returns mapping of ticker symbols to official Groww CDN logo URLs"""
    return jsonify({"success": True, "logos": screener_state.get("stock_logos", {})})

@app.route("/api/fno/momentum-watchlists")
def get_fno_momentum_watchlists_endpoint():
    """Returns dynamically scanned Top Bullish & Bearish momentum movers from all 219 F&O stocks, including PREPARING stocks"""
    bull = list(screener_state.get("bullish_momentum_watchlist", []))
    bear = list(screener_state.get("bearish_momentum_watchlist", []))

    # Seamlessly merge all PREPARING stocks into the appropriate Bullish or Bearish list
    ranked = screener_state.get("ranked_opportunities") or {}
    prep_list = ranked.get("preparing", []) + list(screener_state.get("persistent_preparing_stocks", {}).values())

    seen_bull = {s.get("symbol") for s in bull}
    seen_bear = {s.get("symbol") for s in bear}
    raw_cache = screener_state.get("raw_market_cache") or {}

    for p in prep_list:
        p_sym = p.get("symbol")
        if not p_sym:
            continue
        p_dir = (p.get("direction") or "BULLISH").upper()
        
        # Calculate or lookup actual day change percentage (avoid 0.0% default)
        calc_chg = p.get("change_pct")
        if calc_chg is None or float(calc_chg) == 0.0:
            clean_p = p_sym.replace(".NS", "").upper()
            c_entry = raw_cache.get(p_sym) or raw_cache.get(f"{clean_p}.NS") or raw_cache.get(clean_p)
            if c_entry and isinstance(c_entry, dict) and c_entry.get("df") is not None:
                cdf = c_entry["df"]
                if len(cdf) >= 2:
                    try:
                        curr_c = float(cdf['Close'].squeeze().iloc[-1] if hasattr(cdf['Close'].squeeze(), 'iloc') else cdf['Close'].squeeze())
                        # Look for session open or previous session close
                        df_d = cdf.copy()
                        if not pd.api.types.is_datetime64_any_dtype(df_d.index):
                            df_d.index = pd.to_datetime(df_d.index)
                        df_d['Date'] = df_d.index.date
                        dates = sorted(df_d['Date'].unique())
                        if len(dates) >= 2:
                            prev_d = dates[-2]
                            prev_series = df_d[df_d['Date'] == prev_d]['Close'].squeeze()
                            prev_c = float(prev_series.iloc[-1] if hasattr(prev_series, 'iloc') else prev_series)
                        else:
                            open_series = df_d['Open'].squeeze()
                            prev_c = float(open_series.iloc[0] if hasattr(open_series, 'iloc') else open_series)
                        if prev_c > 0:
                            calc_chg = round(((curr_c - prev_c) / prev_c) * 100.0, 2)
                    except Exception:
                        pass

        item = {
            "symbol": p_sym,
            "change": float(calc_chg) if calc_chg is not None else 0.0,
            "velocity": "COIL",
            "is_preparing": True,
            "score": p.get("conviction_score", 70)
        }
        if p_dir in ("BULLISH", "BUY"):
            if p_sym not in seen_bull:
                bull.append(item)
                seen_bull.add(p_sym)
        else:
            if p_sym not in seen_bear:
                bear.append(item)
                seen_bear.add(p_sym)

    # Enrich all watchlist stocks with cached live price, score, rvol, and compression for zero-latency client rendering
    for stk_list in (bull, bear):
        for s in stk_list:
            s_sym = s.get("symbol", "")
            clean_s = s_sym.replace(".NS", "").upper()
            c_entry = raw_cache.get(s_sym) or raw_cache.get(f"{clean_s}.NS") or raw_cache.get(clean_s)
            rvol = 1.0
            compression = False
            proximity_score = 0.0
            if c_entry and isinstance(c_entry, dict) and c_entry.get("df") is not None:
                cdf = c_entry["df"]
                if not cdf.empty:
                    close_col = "Close" if "Close" in cdf.columns else ("close" if "close" in cdf.columns else None)
                    if close_col:
                        c_series = cdf[close_col].squeeze()
                        s["price"] = round(float(c_series.iloc[-1] if hasattr(c_series, 'iloc') else c_series), 2)
                fno_data = c_entry.get("fno_filter_data") or {}
                if isinstance(fno_data, dict):
                    rvol = float(fno_data.get("rvol", 1.0) or 1.0)
                    is_bb = bool(fno_data.get("is_bb_squeeze", False))
                    is_nr4 = bool(fno_data.get("is_nr4", False))
                    is_nr7 = bool(fno_data.get("is_nr7", False))
                    compression = is_bb or is_nr4 or is_nr7
                    s["is_bb_squeeze"] = is_bb
                    s["is_nr4"] = is_nr4
                    s["is_nr7"] = is_nr7
                s["rvol"] = rvol
                s["is_compressed"] = compression
            # P2 Fix: Compute trigger proximity and structural invalidation score
            proximity_score = 0.0
            is_invalidated = False
            cand = next((c for c in prep_list if c.get("symbol") in (s_sym, clean_s)), None)
            if cand:
                if not s.get("score"):
                    s["score"] = cand.get("conviction_score", 70)
                if "price" not in s or not s["price"]:
                    if cand.get("current_price"):
                        s["price"] = round(float(cand["current_price"]), 2)

                trig_str = str(cand.get("entry_trigger") or "")
                direction = str(cand.get("direction") or "BULLISH").upper()
                curr_price = float(s.get("price") or 0.0)
                if curr_price > 0 and trig_str:
                    try:
                        lvl = float(trig_str.split("₹")[-1].strip())
                        if lvl > 0:
                            # P2 Fix: Direction-aware trigger proximity & structural invalidation
                            # Bullish setup: price coiling below level (or up to 0.5% above on breakout)
                            if "Enter above" in trig_str or direction in ("BULLISH", "BUY"):
                                if curr_price < lvl:
                                    diff_pct = ((lvl - curr_price) / lvl) * 100.0
                                    if diff_pct <= 0.35:
                                        proximity_score = 30.0 # On the verge of breakout!
                                    elif diff_pct <= 0.85:
                                        proximity_score = 20.0
                                    elif diff_pct <= 1.50:
                                        proximity_score = 10.0
                                    elif diff_pct > 4.0:
                                        is_invalidated = True # Dumped too far below setup structure
                                elif curr_price >= lvl:
                                    # Already at or slightly above level
                                    overshoot_pct = ((curr_price - lvl) / lvl) * 100.0
                                    if overshoot_pct <= 0.50:
                                        proximity_score = 35.0 # Right at breakout trigger!
                                    elif overshoot_pct <= 1.50:
                                        proximity_score = 15.0 # Acceptable breakout follow-through
                                    else:
                                        # P2 Fix: Overextended > 1.5% - 2.0% past trigger is marked invalidated for early entry
                                        is_invalidated = True
                            # Bearish setup: price coiling above breakdown level
                            elif "Enter below" in trig_str or direction in ("BEARISH", "SELL"):
                                if curr_price > lvl:
                                    diff_pct = ((curr_price - lvl) / lvl) * 100.0
                                    if diff_pct <= 0.35:
                                        proximity_score = 30.0 # On the verge of breakdown!
                                    elif diff_pct <= 0.85:
                                        proximity_score = 20.0
                                    elif diff_pct <= 1.50:
                                        proximity_score = 10.0
                                    elif diff_pct > 4.0:
                                        is_invalidated = True # Rallied too far above setup structure
                                elif curr_price <= lvl:
                                    undershoot_pct = ((lvl - curr_price) / lvl) * 100.0
                                    if undershoot_pct <= 0.50:
                                        proximity_score = 35.0 # Right at breakdown trigger!
                                    elif undershoot_pct <= 1.50:
                                        proximity_score = 15.0 # Acceptable breakdown follow-through
                                    else:
                                        # P2 Fix: Overextended dump > 1.5% - 2.0% past trigger is marked invalidated for early entry
                                        is_invalidated = True
                    except Exception:
                        pass
            s["proximity_score"] = proximity_score
            s["is_structurally_invalidated"] = is_invalidated

    # Multi-factor Watchlist Prioritization (P1/P2):
    # Rather than solely sorting by percentage change (which pushes already-exhausted stocks to top),
    # prioritize:
    # 1. Setup Quality (conviction score / stage confidence)
    # 2. Coiling / Pre-breakout Compression (imminent moves before the chase, using BB squeeze, NR4, NR7)
    # 3. Trigger Proximity (stocks within 0.35% - 1.5% of breakout level get high execution priority)
    # 4. Relative Volume (RVOL >= 1.25x indicating institutional footprint)
    # 5. Orderly session change % (reward early stage momentum over extreme overextensions)
    def compute_watchlist_rank_priority(item):
        if item.get("is_structurally_invalidated"):
            return -100.0 # Demote setups that broke their structure

        score = float(item.get("score") or 50.0)
        is_prep = 1.0 if item.get("is_preparing") else 0.0
        is_bb = 1.0 if item.get("is_bb_squeeze") else 0.0
        is_nr4 = 1.0 if item.get("is_nr4") else 0.0
        is_nr7 = 1.0 if item.get("is_nr7") else 0.0
        compression_bonus = (is_bb * 20.0) + (is_nr7 * 25.0) + (is_nr4 * 15.0)
        proximity_bonus = float(item.get("proximity_score") or 0.0)
        rvol = min(4.0, float(item.get("rvol") or 1.0))
        chg = abs(float(item.get("change") or 0.0))
        
        # Diminishing returns for overextended percentage moves (> 6% intraday often mean exhausted climax)
        orderly_chg = min(4.0, chg) if chg <= 5.0 else max(1.0, 5.0 - (chg - 5.0) * 0.5)
        
        # Composite early momentum priority
        priority = (
            score * 1.5 +                # Base setup quality
            (is_prep * 35.0) +           # Priority for active pre-breakout coils
            proximity_bonus +            # Distance to trigger level bonus
            compression_bonus +          # Bonus for volatility contraction (BB Squeeze, NR4, NR7)
            (rvol * 20.0) +              # Institutional volume accumulation
            (orderly_chg * 10.0)         # Controlled directional momentum
        )
        return priority

    bull.sort(key=compute_watchlist_rank_priority, reverse=True)
    bear.sort(key=compute_watchlist_rank_priority, reverse=True)

    return jsonify({
        "success": True,
        "bullish": bull,
        "bearish": bear,
        "bullish_watchlist": bull,
        "bearish_watchlist": bear,
        "fno_total": len(screener_state.get("fno_universe", []))
    })

@app.route("/api/groww/option-chain")
def get_groww_option_chain():
    """Fetch complete option chain and Greeks using native Groww API with fail-closed structure"""
    try:
        symbol = request.args.get("symbol", "NIFTY").upper().replace(".NS", "")
        expiry = request.args.get("expiry")
        chain_data = groww_data_provider.get_option_chain_data(underlying=symbol, expiry_date=expiry)
        if chain_data:
            return jsonify({"success": True, "source": "GROWW_API", "data": chain_data})
        return jsonify({"success": False, "source": "FALLBACK", "message": "Option chain not available for symbol/token roles"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/groww/ltp-batch")
def get_groww_ltp_batch():
    """Fetch real-time LTP for a batch of instruments using native Groww API"""
    try:
        symbols_str = request.args.get("symbols", "")
        if not symbols_str:
            symbols = screener_state.get("fno_universe", [])[:50]
        else:
            symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
        ltp_map = groww_data_provider.get_ltp_batch(symbols)
        return jsonify({"success": True, "count": len(ltp_map), "data": ltp_map})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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

    # Launch Background Screener Worker (Handles 5-min 219 F&O momentum & sector scanning)
    worker = threading.Thread(target=background_screener_worker, daemon=True, name="BackgroundScreenerWorker")
    worker.start()

    # Launch Dedicated Continuous Background Tracker & Learning Daemon (Runs side-by-side without blocking the screener)
    # Passes authenticated groww_data_provider directly to ensure unified broker data flow (P0)
    from engine.background_continuous_tracker import BackgroundContinuousTracker
    continuous_tracker = BackgroundContinuousTracker(
        screener_state,
        compute_ranking_fn=compute_and_update_ranked_opportunities,
        data_provider=groww_data_provider
    )
    continuous_tracker.start()
    screener_state["continuous_tracker_instance"] = continuous_tracker

    print("Groww Chartink Momentum + SMC Screener running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
