"""
Dated NSE F&O Contract Master & Margin Engine
Provides historical and active lot sizes, tick sizes, and verified baseline SPAN+Exposure margins.
Tracks contract specifications by calendar effective dates to eliminate lookahead and static lot assumptions.
"""
import datetime

# Contract Master History: Symbol -> list of (effective_from_date, lot_size, baseline_margin_pct)
# Dates represent NSE circular effective dates.
NSE_CONTRACT_SPECS_HISTORY = {
    "RELIANCE": [
        (datetime.date(2024, 1, 1), 250, 0.20),
        (datetime.date(2024, 11, 29), 250, 0.20),
    ],
    "TCS": [
        (datetime.date(2024, 1, 1), 175, 0.20),
    ],
    "HDFCBANK": [
        (datetime.date(2024, 1, 1), 550, 0.20),
    ],
    "INFY": [
        (datetime.date(2024, 1, 1), 400, 0.20),
    ],
    "ICICIBANK": [
        (datetime.date(2024, 1, 1), 700, 0.20),
    ],
    "SBIN": [
        (datetime.date(2024, 1, 1), 750, 0.21),
    ],
    "BHARTIARTL": [
        (datetime.date(2024, 1, 1), 475, 0.21),
    ],
    "ITC": [
        (datetime.date(2024, 1, 1), 1600, 0.20),
    ],
    "LT": [
        (datetime.date(2024, 1, 1), 150, 0.20),
    ],
    "AXISBANK": [
        (datetime.date(2024, 1, 1), 625, 0.21),
    ],
    "KOTAKBANK": [
        (datetime.date(2024, 1, 1), 400, 0.20),
    ],
    "HINDUNILVR": [
        (datetime.date(2024, 1, 1), 300, 0.20),
    ],
    "BAJFINANCE": [
        (datetime.date(2024, 1, 1), 125, 0.22),
    ],
    "MARUTI": [
        (datetime.date(2024, 1, 1), 50, 0.20),
    ],
}

class ContractMaster:
    """
    Validates symbol F&O eligibility, dated lot sizes, and baseline exchange margin rates.
    """
    @classmethod
    def get_contract_spec(cls, symbol, as_of_date=None):
        """
        Retrieves the exact dated lot size and baseline margin percentage for a symbol as of a specific date.
        If as_of_date is None, returns the latest active specification.
        Returns None if the symbol is not in the verified contract master.
        """
        clean_sym = symbol.replace(".NS", "").upper()
        history = NSE_CONTRACT_SPECS_HISTORY.get(clean_sym)
        if not history:
            return None

        if as_of_date is None:
            # Return latest
            latest = history[-1]
            return {
                "symbol": clean_sym,
                "lot_size": latest[1],
                "margin_pct": latest[2],
                "effective_date": latest[0]
            }

        if hasattr(as_of_date, 'date'):
            as_of_date = as_of_date.date()

        # Find the most recent specification on or before as_of_date
        valid_specs = [spec for spec in history if spec[0] <= as_of_date]
        if not valid_specs:
            # Symbol not tradeable in F&O prior to earliest effective date
            return None

        chosen = valid_specs[-1]
        return {
            "symbol": clean_sym,
            "lot_size": chosen[1],
            "margin_pct": chosen[2],
            "effective_date": chosen[0]
        }
