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
        lot_size_map: dict mapping symbol (e.g. 'RELIANCE') to lot_size (int) or spec dict.
        """
        if not isinstance(lot_size_map, dict):
            return
        for sym, val in lot_size_map.items():
            clean = sym.replace(".NS", "").strip().upper()
            if isinstance(val, int) and val > 0:
                cls._DYNAMIC_LOT_SIZES[clean] = {"lot_size": val, "estimated_margin_pct": 0.20, "circular_ref": "NSE/FAOP/LIVE"}
            elif isinstance(val, dict) and "lot_size" in val:
                cls._DYNAMIC_LOT_SIZES[clean] = {
                    "lot_size": int(val["lot_size"]),
                    "estimated_margin_pct": float(val.get("estimated_margin_pct", 0.20)),
                    "circular_ref": str(val.get("circular_ref", "NSE/FAOP/LIVE"))
                }

    @classmethod
    def get_contract_spec(cls, symbol, as_of_date=None):
        """
        Retrieves the exact dated lot size and estimated baseline margin percentage for a symbol as of a specific date.
        
        Args:
            symbol (str): Equity or Futures symbol (e.g. "RELIANCE.NS" or "RELIANCE")
            as_of_date (str, datetime.date, pd.Timestamp, optional): Evaluation date.
        
        Returns:
            dict or None: Contract specification dictionary containing:
                - symbol: Normalized uppercase ticker
                - lot_size: int
                - estimated_margin_pct: float (explicitly labeled as estimate)
                - margin_pct: float (alias for backward compatibility)
                - effective_date: datetime.date
                - circular_ref: str
            Returns None if symbol is invalid, unknown, or date precedes earliest contract inception.
        """
        if not symbol or not isinstance(symbol, str):
            return None

        clean_sym = symbol.replace(".NS", "").strip().upper()
        history = NSE_CONTRACT_SPECS_HISTORY.get(clean_sym)
        if not history:
            # Check dynamic live ingested contracts if available
            dyn = cls._DYNAMIC_LOT_SIZES.get(clean_sym)
            if dyn:
                return {
                    "symbol": clean_sym,
                    "lot_size": dyn["lot_size"],
                    "estimated_margin_pct": dyn["estimated_margin_pct"],
                    "margin_pct": dyn["estimated_margin_pct"],
                    "effective_date": datetime.date.today(),
                    "circular_ref": dyn.get("circular_ref", "NSE/FAOP/LIVE")
                }
            return None

        # Sort history strictly by effective_date to guard against unordered entries
        sorted_history = sorted(history, key=lambda x: x["effective_date"])

        if as_of_date is None:
            latest = sorted_history[-1]
            return {
                "symbol": clean_sym,
                "lot_size": latest["lot_size"],
                "estimated_margin_pct": latest["estimated_margin_pct"],
                "margin_pct": latest["estimated_margin_pct"],
                "effective_date": latest["effective_date"],
                "circular_ref": latest.get("circular_ref", "NSE/FAOP/STANDARD")
            }

        # Safe date normalization
        parsed_date = None
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

        # Find the latest effective circular specification on or before parsed_date
        valid_specs = [spec for spec in sorted_history if spec["effective_date"] <= parsed_date]
        if not valid_specs:
            # Date precedes earliest verified NSE contract inception
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
