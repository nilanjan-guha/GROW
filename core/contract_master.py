"""
Dated NSE F&O Contract Master & Margin Engine
Provides historical and active contract lot sizes and estimated baseline SPAN+Exposure margins.
Tracks contract specifications by calendar effective dates with explicit NSE circular reference tracking.
"""
import datetime
import pandas as pd

# Dated NSE F&O Contract Master:
# Format: Symbol -> list of dicts:
#   - "effective_date": datetime.date
#   - "lot_size": int
#   - "estimated_margin_pct": float (SPAN + Exposure baseline estimate)
#   - "circular_ref": str (NSE Circular or Gazette reference)
#
# IMPORTANT: Margin percentages are explicitly labelled as ESTIMATES (simulated baseline parameter).
# Contract prices, basis, roll-over spreads, and daily changing SPAN margins are not identical to actual futures contracts.
NSE_CONTRACT_SPECS_HISTORY = {
    "RELIANCE": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 250, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/46880"},
        {"effective_date": datetime.date(2024, 11, 29), "lot_size": 250, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/64875"}
    ],
    "TCS": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 175, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/46880"}
    ],
    "HDFCBANK": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 550, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/46880"}
    ],
    "INFY": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 400, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/46880"}
    ],
    "ICICIBANK": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 700, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/46880"}
    ],
    "SBIN": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 750, "estimated_margin_pct": 0.21, "circular_ref": "NSE/FAOP/46880"}
    ],
    "BHARTIARTL": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 475, "estimated_margin_pct": 0.21, "circular_ref": "NSE/FAOP/46880"}
    ],
    "ITC": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 1600, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/46880"}
    ],
    "LT": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 150, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/46880"}
    ],
    "AXISBANK": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 625, "estimated_margin_pct": 0.21, "circular_ref": "NSE/FAOP/46880"}
    ],
    "KOTAKBANK": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 400, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/46880"}
    ],
    "HINDUNILVR": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 300, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/46880"}
    ],
    "BAJFINANCE": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 125, "estimated_margin_pct": 0.22, "circular_ref": "NSE/FAOP/46880"}
    ],
    "MARUTI": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 50, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/46880"}
    ],
    "TATAMOTORS": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 1425, "estimated_margin_pct": 0.22, "circular_ref": "NSE/FAOP/46880"}
    ],
    "TATASTEEL": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 5500, "estimated_margin_pct": 0.22, "circular_ref": "NSE/FAOP/46880"}
    ],
    "SUNPHARMA": [
        {"effective_date": datetime.date(2021, 1, 1), "lot_size": 700, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/46880"}
    ]
}

class ContractMaster:
    """
    Validates symbol F&O eligibility, dated lot sizes, and estimated baseline exchange margin rates.
    Supports dated historical circular tables as well as dynamic live ingestion from broker/exchange instrument masters.
    """
    _DYNAMIC_LOT_SIZES = {}

    @classmethod
    def register_dynamic_contracts(cls, lot_size_map: dict):
        """
        Dynamically registers live contract specifications ingested from broker or exchange instrument masters.
        Strictly validates:
        - Symbol must be non-empty string.
        - Lot size must be strictly positive int.
        - Margin percentage must be between 0.05 (5%) and 1.0 (100%).
        - Expiry date, instrument token/identifier, source, and ingestion timestamp are tracked when provided.
        """
        if not isinstance(lot_size_map, dict):
            return
        import math
        today_date = datetime.date.today()
        for sym, val in lot_size_map.items():
            if not sym or not isinstance(sym, str):
                continue
            clean = sym.replace(".NS", "").strip().upper()
            if not clean:
                continue

            if isinstance(val, int):
                if val > 0:
                    cls._DYNAMIC_LOT_SIZES[clean] = {
                        "lot_size": val,
                        "estimated_margin_pct": 0.20,
                        "expiry": None,
                        "instrument_id": f"FUT-{clean}-CURRENT",
                        "source": "BROKER_FEED",
                        "update_date": today_date,
                        "circular_ref": "NSE/FAOP/LIVE"
                    }
            elif isinstance(val, dict):
                try:
                    raw_lot = val.get("lot_size")
                    if raw_lot is None:
                        continue
                    lot_size = int(raw_lot)
                    if lot_size <= 0:
                        continue

                    raw_margin = val.get("estimated_margin_pct", 0.20)
                    margin_pct = float(raw_margin) if raw_margin is not None else 0.20
                    if not (math.isfinite(margin_pct) and 0.05 <= margin_pct <= 1.0):
                        margin_pct = 0.20

                    expiry = val.get("expiry")
                    if expiry is not None:
                        try:
                            expiry = pd.to_datetime(expiry).date()
                        except Exception:
                            expiry = None

                    inst_id = str(val.get("instrument_id", f"FUT-{clean}-{expiry or 'CURRENT'}"))
                    source = str(val.get("source", "BROKER_FEED"))
                    update_dt = val.get("update_date", today_date)

                    cls._DYNAMIC_LOT_SIZES[clean] = {
                        "lot_size": lot_size,
                        "estimated_margin_pct": margin_pct,
                        "expiry": expiry,
                        "instrument_id": inst_id,
                        "source": source,
                        "update_date": update_dt,
                        "circular_ref": str(val.get("circular_ref", "NSE/FAOP/LIVE"))
                    }
                except (ValueError, TypeError):
                    continue

    @classmethod
    def get_contract_spec(cls, symbol, as_of_date=None):
        """
        Retrieves the exact dated lot size and estimated baseline margin percentage for a symbol as of a specific date.
        
        Clear Source-of-Truth Precedence Policy:
        1. When evaluating for the CURRENT trading date (as_of_date is None or today's date):
           - Ingested dynamic live broker instrument master (_DYNAMIC_LOT_SIZES) takes precedence.
           - Falls back to the latest entry in the historical circular table.
        2. When evaluating for a HISTORICAL date (as_of_date < today's date):
           - Strict historical circular table (NSE_CONTRACT_SPECS_HISTORY) on or before as_of_date is the sole source of truth.
           - Dynamic live values are NOT applied to past historical dates to prevent forward-looking data leakage.
        3. Returns None if neither source can establish a valid contract specification.
        """
        if not symbol or not isinstance(symbol, str):
            return None

        clean_sym = symbol.replace(".NS", "").strip().upper()
        history = NSE_CONTRACT_SPECS_HISTORY.get(clean_sym)
        dyn = cls._DYNAMIC_LOT_SIZES.get(clean_sym)

        # Parse and normalize as_of_date
        today = datetime.date.today()
        parsed_date = None
        if as_of_date is not None:
            if isinstance(as_of_date, datetime.date) and not isinstance(as_of_date, datetime.datetime):
                parsed_date = as_of_date
            elif isinstance(as_of_date, (datetime.datetime, pd.Timestamp)):
                parsed_date = as_of_date.date()
            elif isinstance(as_of_date, str):
                try:
                    parsed_date = pd.to_datetime(as_of_date).date()
                except Exception:
                    return None
            else:
                return None

        is_current_date = (parsed_date is None or parsed_date >= today)

        # 1. Current trading session: Dynamic live broker table takes precedence if present
        if is_current_date and dyn:
            return {
                "symbol": clean_sym,
                "lot_size": dyn["lot_size"],
                "estimated_margin_pct": dyn["estimated_margin_pct"],
                "margin_pct": dyn["estimated_margin_pct"],
                "expiry": dyn.get("expiry"),
                "instrument_id": dyn.get("instrument_id"),
                "source": dyn.get("source", "BROKER_FEED"),
                "update_date": dyn.get("update_date", today),
                "effective_date": today,
                "circular_ref": dyn.get("circular_ref", "NSE/FAOP/LIVE")
            }

        # 2. Historical replay or fallback to circular table
        if not history:
            return None

        sorted_history = sorted(history, key=lambda x: x["effective_date"])

        if parsed_date is None:
            latest = sorted_history[-1]
            return {
                "symbol": clean_sym,
                "lot_size": latest["lot_size"],
                "estimated_margin_pct": latest["estimated_margin_pct"],
                "margin_pct": latest["estimated_margin_pct"],
                "effective_date": latest["effective_date"],
                "circular_ref": latest.get("circular_ref", "NSE/FAOP/STANDARD")
            }

        # Find latest effective circular specification on or before parsed_date
        valid_specs = [spec for spec in sorted_history if spec["effective_date"] <= parsed_date]
        if not valid_specs:
            return None

        chosen = valid_specs[-1]
        return {
            "symbol": clean_sym,
            "lot_size": chosen["lot_size"],
            "estimated_margin_pct": chosen["estimated_margin_pct"],
            "margin_pct": chosen["estimated_margin_pct"],
            "effective_date": chosen["effective_date"],
            "circular_ref": chosen.get("circular_ref", "NSE/FAOP/STANDARD")
        }
