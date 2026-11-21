import datetime
import pytz
import pandas as pd

# Multi-Year Indian NSE/BSE Market Holidays (2025 - 2027)
# Supports special/extended calendar lookup with configurable overrides
OFFICIAL_NSE_HOLIDAYS = {
    # 2025
    datetime.date(2025, 1, 26), datetime.date(2025, 2, 26), datetime.date(2025, 3, 14),
    datetime.date(2025, 3, 31), datetime.date(2025, 4, 10), datetime.date(2025, 4, 14),
    datetime.date(2025, 4, 18), datetime.date(2025, 5, 1), datetime.date(2025, 8, 15),
    datetime.date(2025, 8, 27), datetime.date(2025, 10, 2), datetime.date(2025, 10, 21),
    datetime.date(2025, 11, 1), datetime.date(2025, 11, 5), datetime.date(2025, 12, 25),
    # 2026
    datetime.date(2026, 1, 26), datetime.date(2026, 3, 6), datetime.date(2026, 3, 20),
    datetime.date(2026, 4, 3), datetime.date(2026, 4, 14), datetime.date(2026, 5, 1),
    datetime.date(2026, 5, 27), datetime.date(2026, 8, 15), datetime.date(2026, 8, 26),
    datetime.date(2026, 10, 2), datetime.date(2026, 10, 20), datetime.date(2026, 11, 8),
    datetime.date(2026, 11, 10), datetime.date(2026, 11, 24), datetime.date(2026, 12, 25),
    # 2027
    datetime.date(2027, 1, 26), datetime.date(2027, 3, 22), datetime.date(2027, 4, 14),
    datetime.date(2027, 5, 1), datetime.date(2027, 8, 15), datetime.date(2027, 10, 2),
    datetime.date(2027, 12, 25)
}

# Configurable special trading sessions (e.g. Diwali Muhurat trading, disaster recovery live tests)
SPECIAL_TRADING_SESSIONS = {
    datetime.date(2026, 11, 8): {"start_minute": 1095, "end_minute": 1155} # Example 18:15 to 19:15 IST
}

class MarketValidator:
    """
    Production-grade validation for Indian Equity & F&O Market Data:
    1. Trading session & official holiday calendar verification (Multi-year + special sessions).
    2. Candle timestamp completion & maximum feed latency checks.
    3. Missing candle / feed gap anomaly detection & feed integrity enforcement.
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
        
        # Check special trading session whitelist first
        if dt in SPECIAL_TRADING_SESSIONS:
            return True

        # Weekend check: Monday is 0, Sunday is 6
        if dt.weekday() >= 5:
            return False
        # Holiday check
        if dt in OFFICIAL_NSE_HOLIDAYS:
            return False
        return True

    @staticmethod
    def is_market_open(dt=None):
        """Returns True strictly during regular NSE/BSE cash & F&O trading hours (09:15 to 15:30 IST) or special sessions"""
        now = dt or MarketValidator.get_ist_now()
        cur_date = now.date()
        minute = now.hour * 60 + now.minute

        # Special session timing check
        if cur_date in SPECIAL_TRADING_SESSIONS:
            cfg = SPECIAL_TRADING_SESSIONS[cur_date]
            return cfg["start_minute"] <= minute <= cfg["end_minute"]

        if not MarketValidator.is_trading_day(cur_date):
            return False
        return 555 <= minute <= 930

    @staticmethod
    def validate_feed_freshness(df, max_delay_seconds=900, max_allowed_missing_gaps=2, candle_duration_seconds=300):
        """
        Validates OHLCV DataFrame integrity and freshness as single source of truth:
        - Rejects empty feeds or feeds with insufficient candle history (< 5 candles).
        - Enforces candle completion (`now >= last_ts + candle_duration_seconds`).
        - Rejects stale previous session feeds during active market hours.
        - Rejects stalled feeds (> max_delay_seconds delay).
        - Rejects feeds with excessive missing candle gaps (> max_allowed_missing_gaps).
        """
        if df is None or df.empty:
            return {"is_valid": False, "status": "EMPTY_FEED", "reason": "Market dataframe is empty"}

        if len(df) < 5:
            return {"is_valid": False, "status": "INSUFFICIENT_HISTORY", "reason": f"Only {len(df)} candles available (< 5 required)"}

        df_eval = df.copy()
        ist = pytz.timezone('Asia/Kolkata')
        now = MarketValidator.get_ist_now()
        now_ts = now.timestamp()
        
        # Check index type and ensure timezone awareness in Asia/Kolkata
        if not pd.api.types.is_datetime64_any_dtype(df_eval.index):
            try:
                df_eval.index = pd.to_datetime(df_eval.index)
            except Exception as e:
                return {"is_valid": False, "status": "INVALID_INDEX", "reason": f"Cannot parse datetime index: {e}"}

        # Normalize naive datetime index to Asia/Kolkata
        if df_eval.index.tz is None:
            df_eval.index = df_eval.index.tz_localize(ist)
        elif str(df_eval.index.tz) != 'Asia/Kolkata':
            df_eval.index = df_eval.index.tz_convert(ist)

        last_dt = df_eval.index[-1]
        last_ts = last_dt.timestamp() if hasattr(last_dt, 'timestamp') else float(last_dt)
        candle_end = last_ts + candle_duration_seconds  # Candle closes after configured duration

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
        if len(df_eval) >= 2:
            df_today = df_eval[df_eval.index.date == last_dt.date()] if hasattr(last_dt, 'date') else df_eval
            if len(df_today) >= 2:
                diffs = (df_today.index[1:] - df_today.index[:-1]).total_seconds()
                gaps = [d for d in diffs if d > 360]  # Greater than 6 minutes interval
                missing_count = len(gaps)

        # Feed integrity gate: reject feeds with severe missing candle gaps (> 2 gaps)
        if missing_count > max_allowed_missing_gaps:
            return {
                "is_valid": False,
                "status": "CORRUPT_FEED_GAPS",
                "missing_candle_gaps": missing_count,
                "reason": f"Feed integrity compromised: detected {missing_count} missing candle gaps during session."
            }

        return {
            "is_valid": True,
            "status": "FRESH",
            "is_completed_candle": is_completed,
            "latest_candle_ts": last_ts,
            "candle_end_ts": candle_end,
            "missing_candle_gaps": missing_count
        }
