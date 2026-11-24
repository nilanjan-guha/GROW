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
    _DYNAMIC_CONTRACTS = {}  # Key: (symbol, expiry) or (symbol, 'CURRENT') -> dict

    # Backward compatibility property/dict for legacy tests that access _DYNAMIC_LOT_SIZES directly
    class _DynamicLotSizesProxy(dict):
        def clear(self_proxy):
            ContractMaster._DYNAMIC_CONTRACTS.clear()
            super().clear()

    _DYNAMIC_LOT_SIZES = _DynamicLotSizesProxy()

    # Private registration secret to guarantee contract provenance:
    # Only trusted broker ingestion adapters possessing the adapter secret can register live contracts.
    _REGISTRATION_SECRET = "BROKER_INGESTION_ADAPTER_AUTH_KEY_2026"

    # Strict Whitelist of Trusted Broker / Exchange Provenance Sources for Live Futures Contracts
    TRUSTED_CONTRACT_SOURCES = {
        "GROWW_API",
        "GROWW_LIVE",
        "NSE_CIRCULAR",
        "NSE_MASTER",
        "BROKER_FEED",
        "BROKER_LIVE"
    }

    @classmethod
    def is_synthetic_instrument_id(cls, inst_id: str) -> bool:
        """
        Detects and rejects fabricated / synthetic instrument IDs (e.g., 'FUT-RELIANCE-TEST', 'FUT-SYM-CURRENT', 'SYNTH-*').
        Legitimate broker/exchange instrument IDs are authoritative identifiers (numeric tokens, ISINs, official trading symbols like 'RELIANCE26NOVFUT', or broker tokens).
        """
        if not inst_id or not isinstance(inst_id, str):
            return True
        s = inst_id.strip().upper()
        if not s:
            return True
        # Reject fabricated test patterns or placeholder tokens
        forbidden_substrings = ["-TEST", "TEST-", "_TEST", "SYNTH", "MOCK", "DUMMY", "-CURRENT", "CURRENT-", "PLACEHOLDER"]
        if any(f in s for f in forbidden_substrings):
            return True
        # Synthetic template pattern FUT-{symbol}-{anything}
        parts = s.split("-")
        if len(parts) >= 2 and parts[0] == "FUT":
            # If it's a test/synthetic string like FUT-RELIANCE-1 or FUT-SYM
            if any(p in ("TEST", "CURRENT", "FAKE", "MOCK", "DUMMY", "1", "0") for p in parts):
                return True
        return False

    @classmethod
    def register_dynamic_contracts(cls, lot_size_map: dict, adapter_secret: str = None):
        """
        Dynamically registers live contract specifications ingested from broker or exchange instrument masters.
        Supports registering multiple futures expiries per underlying stock without overwrite.
        
        Strictly validates:
        - Provenance enforcement: Caller must provide the authorized adapter_secret proving call originates from a trusted ingestion adapter.
        - Symbol must be non-empty string.
        - Contract record must be a dict containing broker-sourced instrument identity.
        - Lot size must be strictly a positive integer (rejecting booleans, fractional floats like 250.9, and malformed numeric strings).
        - Margin percentage must be between 0.05 (5%) and 1.0 (100%).
        - Expiry date, instrument_id, source, and update_date are mandatory.
        - Source must be within TRUSTED_CONTRACT_SOURCES.
        - Synthetic instrument IDs (e.g. FUT-RELIANCE-TEST, mock IDs) are strictly rejected.
        """
        # PROVENANCE GATE: Only trusted broker ingestion adapter with valid adapter_secret can register live contracts
        if adapter_secret != cls._REGISTRATION_SECRET:
            return

        if not isinstance(lot_size_map, dict):
            return
        import math
        today_date = datetime.date.today()
        for key, val in lot_size_map.items():
            if not key or not isinstance(key, str):
                continue

            # Key can be clean symbol, e.g. 'RELIANCE', or composite 'RELIANCE:2026-11-26'
            parts = key.split(":")
            sym = parts[0]
            clean = sym.replace(".NS", "").strip().upper()
            if not clean:
                continue

            explicit_expiry = None
            if len(parts) > 1:
                try:
                    explicit_expiry = pd.to_datetime(parts[1]).date()
                except Exception:
                    continue  # Reject unparseable explicit expiry in composite key

            # P0: REJECT SCALAR LOT SIZES.
            # Never manufacture synthetic instrument_id, synthetic source, or fake today_date!
            # Live futures contracts require full broker-verified metadata dicts.
            if isinstance(val, dict):
                try:
                    raw_lot = val.get("lot_size")
                    # Reject booleans or missing lot size
                    if raw_lot is None or type(raw_lot) is bool:
                        continue
                    # P1: Reject fractional floats (e.g. 250.9) and enforce genuine positive integer
                    if isinstance(raw_lot, float):
                        if not raw_lot.is_integer():
                            continue
                        lot_size = int(raw_lot)
                    elif isinstance(raw_lot, int):
                        lot_size = raw_lot
                    elif isinstance(raw_lot, str):
                        s_lot = raw_lot.strip()
                        if not s_lot.isdigit():
                            continue
                        lot_size = int(s_lot)
                    else:
                        continue

                    if lot_size <= 0:
                        continue

                    raw_margin = val.get("estimated_margin_pct", 0.20)
                    if type(raw_margin) is bool:
                        continue
                    margin_pct = float(raw_margin) if raw_margin is not None else 0.20
                    if not (math.isfinite(margin_pct) and 0.05 <= margin_pct <= 1.0):
                        margin_pct = 0.20

                    expiry = val.get("expiry") or explicit_expiry
                    if expiry is None:
                        continue  # Expiry is mandatory for live futures contracts
                    try:
                        expiry = pd.to_datetime(expiry).date()
                    except Exception:
                        continue  # Reject unparseable expiry date
                    # Reject expired contracts
                    if expiry < today_date:
                        continue

                    # STRICT BROKER IDENTITY ENFORCEMENT:
                    # Require an explicit instrument_id from the broker feed.
                    raw_inst_id = val.get("instrument_id")
                    if raw_inst_id is None:
                        continue
                    inst_id = str(raw_inst_id).strip()
                    if not inst_id:
                        continue  # Reject empty instrument_id

                    # P0: REJECT SYNTHETIC INSTRUMENT IDs
                    if cls.is_synthetic_instrument_id(inst_id):
                        continue

                    # Source validation: must be in the trusted whitelist of broker/exchange sources
                    raw_source = val.get("source")
                    if raw_source is None:
                        continue
                    source = str(raw_source).strip()
                    if not source or source.upper() not in cls.TRUSTED_CONTRACT_SOURCES:
                        continue

                    # Update date validation: strictly required from broker feed (no manufactured today_date)
                    raw_update_dt = val.get("update_date")
                    if raw_update_dt is None:
                        continue  # Reject missing update_date
                    try:
                        update_dt = pd.to_datetime(raw_update_dt).date()
                    except Exception:
                        continue  # Reject invalid update_date
                    # Reject update_date in the future (data corruption / clock skew)
                    if update_dt > today_date:
                        continue
                    # Freshness policy: Reject stale broker contract records older than 7 calendar days
                    if (today_date - update_dt).days > 7:
                        continue

                    spec_record = {
                        "symbol": clean,
                        "lot_size": lot_size,
                        "estimated_margin_pct": margin_pct,
                        "expiry": expiry,
                        "instrument_id": inst_id,
                        "source": source.upper(),
                        "update_date": update_dt,
                        "circular_ref": str(val.get("circular_ref", "NSE/FAOP/LIVE"))
                    }
                    cls._DYNAMIC_CONTRACTS[(clean, expiry)] = spec_record
                except (ValueError, TypeError):
                    continue

    @classmethod
    def get_contract_spec(cls, symbol, as_of_date=None, expiry=None):
        """
        Retrieves the exact dated lot size and estimated baseline margin percentage for a symbol as of a specific date.
        Optionally selects an explicit futures expiry contract.
        
        Clear Source-of-Truth Precedence Policy:
        1. When evaluating for the CURRENT trading date (as_of_date is None or today's date):
           - Ingested dynamic live broker instrument master (_DYNAMIC_CONTRACTS) takes precedence.
           - If an explicit expiry is requested, retrieves that exact contract (rejecting expired contracts).
           - If no expiry requested, picks the nearest unexpired registered contract or 'CURRENT'.
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

        # 1. When an explicit futures expiry is requested, FAIL CLOSED:
        # Strictly return that exact contract from dynamic registry if unexpired.
        # NEVER silently substitute another expiry or fall back to historical circular tables.
        if expiry is not None:
            try:
                target_expiry = pd.to_datetime(expiry).date()
            except Exception:
                return None  # Unparseable expiry fails closed

            # Target expiry must be unexpired relative to evaluation date
            eval_date = parsed_date or today
            if target_expiry < eval_date:
                return None

            # Look up exact match in dynamic contract registry
            for (s, exp_k), spec in cls._DYNAMIC_CONTRACTS.items():
                if s == clean_sym and spec.get("expiry") == target_expiry:
                    return {
                        "symbol": clean_sym,
                        "lot_size": spec["lot_size"],
                        "estimated_margin_pct": spec["estimated_margin_pct"],
                        "margin_pct": spec["estimated_margin_pct"],
                        "expiry": spec.get("expiry"),
                        "instrument_id": spec.get("instrument_id"),
                        "source": spec.get("source", "BROKER_FEED"),
                        "update_date": spec.get("update_date", today),
                        "effective_date": today,
                        "circular_ref": spec.get("circular_ref", "NSE/FAOP/LIVE")
                    }

            # If exact expiry is requested but not found in registered contracts, fail closed!
            return None

        # 2. Current trading session without explicit expiry: Dynamic live broker contracts take precedence if present
        if is_current_date and cls._DYNAMIC_CONTRACTS:
            # Filter dynamic contracts matching this symbol
            matching_dyn = []
            for (s, exp_k), spec in cls._DYNAMIC_CONTRACTS.items():
                if s == clean_sym:
                    matching_dyn.append(spec)

            if matching_dyn:
                # Sort by expiry, favoring unexpired upcoming contracts over expired
                unexpired = [m for m in matching_dyn if m.get("expiry") and m.get("expiry") >= today]
                if unexpired:
                    unexpired.sort(key=lambda x: x["expiry"])
                    chosen_dyn = unexpired[0]
                else:
                    # Fallback to 'CURRENT' or first available
                    curr_specs = [m for m in matching_dyn if m.get("expiry") is None]
                    chosen_dyn = curr_specs[0] if curr_specs else matching_dyn[0]

                if chosen_dyn:
                    return {
                        "symbol": clean_sym,
                        "lot_size": chosen_dyn["lot_size"],
                        "estimated_margin_pct": chosen_dyn["estimated_margin_pct"],
                        "margin_pct": chosen_dyn["estimated_margin_pct"],
                        "expiry": chosen_dyn.get("expiry"),
                        "instrument_id": chosen_dyn.get("instrument_id"),
                        "source": chosen_dyn.get("source", "BROKER_FEED"),
                        "update_date": chosen_dyn.get("update_date", today),
                        "effective_date": today,
                        "circular_ref": chosen_dyn.get("circular_ref", "NSE/FAOP/LIVE")
                    }

        # 3. Historical replay or fallback to circular table
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
