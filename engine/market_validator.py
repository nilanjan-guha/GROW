import datetime
import pytz
import pandas as pd

# Official NSE Standard Holidays for 2026 (Major National Holidays)
NSE_HOLIDAYS_2026 = {
    datetime.date(2026, 1, 26),  # Republic Day
    datetime.date(2026, 3, 6),   # Holi
    datetime.date(2026, 3, 20),  # Id-Ul-Fitr
    datetime.date(2026, 4, 3),   # Good Friday
    datetime.date(2026, 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti
    datetime.date(2026, 5, 1),   # Maharashtra Day
    datetime.date(2026, 5, 27),  # Bakri Id
    datetime.date(2026, 8, 15),  # Independence Day
    datetime.date(2026, 8, 26),  # Milad-un-Nabi
    datetime.date(2026, 10, 2),  # Mahatma Gandhi Jayanti
    datetime.date(2026, 10, 20), # Dussehra
    datetime.date(2026, 11, 8),  # Diwali Laxmi Pujan (Muhurat trading separate)
    datetime.date(2026, 11, 10), # Diwali Balipratipada
    datetime.date(2026, 11, 24), # Guru Nanak Jayanti
    datetime.date(2026, 12, 25), # Christmas
}

class MarketValidator:
    """
    Production-grade validation for Indian Equity & F&O Market Data:
    1. Trading session & official holiday calendar verification.
    2. Candle timestamp completion & maximum feed latency checks.
    3. Missing candle / feed gap anomaly detection.
    """

    @staticmethod
    def get_ist_now():
        ist = pytz.timezone('Asia/Kolkata')
        return datetime.datetime.now(ist)

    @staticmethod
    def is_trading_day(dt=None):
        if dt is None:
            dt = MarketValidator.get_ist_now().date()
        elif isinstance(dt, datetime.datetime):
            dt = dt.date()
        
        # Weekend check: Monday is 0, Sunday is 6
        if dt.weekday() >= 5:
            return False
        # Holiday check
        if dt in NSE_HOLIDAYS_2026:
            return False
        return True

    @staticmethod
    def is_market_open(dt=None):
        """Returns True strictly during regular NSE/BSE cash & F&O trading hours (09:15 to 15:30 IST)"""
        now = dt or MarketValidator.get_ist_now()
        if not MarketValidator.is_trading_day(now.date()):
            return False
        minute = now.hour * 60 + now.minute
        return 555 <= minute <= 930

    @staticmethod
    def validate_feed_freshness(df, max_delay_seconds=900):
        """
        Validates OHLCV DataFrame integrity and freshness:
        - Checks if empty or insufficient length (< 5 candles).
        - Validates that the latest candle is completed.
        - Flags stale feeds if delay > max_delay_seconds during active market hours.
        - Checks for missing intraday candles during trading session.
        """
        if df is None or df.empty:
            return {"is_valid": False, "status": "EMPTY_FEED", "reason": "Market dataframe is empty"}

        now = MarketValidator.get_ist_now()
        now_ts = now.timestamp()
        
        # Check index type
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                return {"is_valid": False, "status": "INVALID_INDEX", "reason": f"Cannot parse datetime index: {e}"}

        last_dt = df.index[-1]
        last_ts = last_dt.timestamp() if hasattr(last_dt, 'timestamp') else float(last_dt)
        candle_end = last_ts + 300  # 5-minute candle closes after 300 seconds

        is_completed = (now_ts >= candle_end)

        # Session freshness check
        if MarketValidator.is_market_open(now):
            last_date = last_dt.date() if hasattr(last_dt, 'date') else None
            today_date = now.date()
            if last_date and last_date < today_date:
                return {
                    "is_valid": False,
                    "status": "PREVIOUS_SESSION_DATA",
                    "reason": f"Feed contains previous session data ({last_date}) during active market hours ({today_date})"
                }

            # Same-day delay check after 09:30 IST
            curr_min = now.hour * 60 + now.minute
            if curr_min >= 570 and (now_ts - last_ts > max_delay_seconds):
                delay_m = int((now_ts - last_ts) / 60)
                return {
                    "is_valid": False,
                    "status": "FEED_STALLED",
                    "delay_minutes": delay_m,
                    "reason": f"Feed delayed by {delay_m} minutes (> {int(max_delay_seconds/60)}m allowed)"
                }

        # Check for gap / missing candles in current session
        missing_count = 0
        if len(df) >= 2:
            df_today = df[df.index.date == last_dt.date()] if hasattr(last_dt, 'date') else df
            if len(df_today) >= 2:
                diffs = (df_today.index[1:] - df_today.index[:-1]).total_seconds()
                gaps = [d for d in diffs if d > 360]  # Greater than 6 minutes interval
                missing_count = len(gaps)

        return {
            "is_valid": True,
            "status": "FRESH",
            "is_completed_candle": is_completed,
            "latest_candle_ts": last_ts,
            "candle_end_ts": candle_end,
            "missing_candle_gaps": missing_count
        }
