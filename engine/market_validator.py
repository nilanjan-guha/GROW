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

    @staticmethod
    def validate_historical_replay_data(df, candle_duration_seconds=300):
        """
        Validates historical data integrity for backtesting/replay (NO current wall-clock checks):
        - Enforces strictly unique, monotonic datetime index ordering.
        - Validates all OHLCV values: rejects NaN, infinite, or negative values.
        - Enforces strict mathematical OHLC relationships:
          High >= Open, High >= Close, Low <= Open, Low <= Close, High >= Low.
        - Validates volume is non-negative and finite.
        """
        if df is None or df.empty:
            return {"is_valid": False, "status": "EMPTY_FEED", "reason": "Historical dataframe is empty"}

        if len(df) < 5:
            return {"is_valid": False, "status": "INSUFFICIENT_HISTORY", "reason": f"Only {len(df)} candles available (< 5 required)"}

        df_eval = df.copy()
        if isinstance(df_eval.columns, pd.MultiIndex):
            df_eval.columns = df_eval.columns.get_level_values(0)

        if not pd.api.types.is_datetime64_any_dtype(df_eval.index):
            try:
                df_eval.index = pd.to_datetime(df_eval.index)
            except Exception as e:
                return {"is_valid": False, "status": "INVALID_INDEX", "reason": f"Cannot parse datetime index: {e}"}

        # Check unique chronological ordering
        if df_eval.index.has_duplicates:
            return {"is_valid": False, "status": "DUPLICATE_TIMESTAMPS", "reason": "Duplicate candle timestamps detected in historical slice"}

        if not df_eval.index.is_monotonic_increasing:
            return {"is_valid": False, "status": "NON_MONOTONIC_INDEX", "reason": "Historical candles not strictly chronological"}

        # Validate required columns
        req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df_eval.columns for col in req_cols):
            return {"is_valid": False, "status": "MISSING_COLUMNS", "reason": f"Required OHLCV columns missing: {[c for c in req_cols if c not in df_eval.columns]}"}

        c = df_eval['Close'].values
        h = df_eval['High'].values
        l = df_eval['Low'].values
        o = df_eval['Open'].values
        v = df_eval['Volume'].values

        import numpy as np
        # Finite number check (no NaN, no Inf)
        for name, arr in [('Open', o), ('High', h), ('Low', l), ('Close', c), ('Volume', v)]:
            if not np.all(np.isfinite(arr)):
                return {"is_valid": False, "status": "NON_FINITE_DATA", "reason": f"NaN or Inf values detected in {name} array"}

        # Price values must be strictly positive
        if np.any(c <= 0) or np.any(h <= 0) or np.any(l <= 0) or np.any(o <= 0):
            return {"is_valid": False, "status": "INVALID_PRICES", "reason": "Non-positive price values detected in historical slice"}

        # Volume must be non-negative
        if np.any(v < 0):
            return {"is_valid": False, "status": "NEGATIVE_VOLUME", "reason": "Negative volume values detected in historical slice"}

        # Strict OHLC logical relationships
        # High must be the absolute highest: High >= Low, High >= Open, High >= Close
        if np.any(h < l) or np.any(h < o) or np.any(h < c):
            return {"is_valid": False, "status": "CORRUPT_OHLC_HIGH", "reason": "High price violated boundary (High < Low, High < Open, or High < Close)"}

        # Low must be the absolute lowest: Low <= High, Low <= Open, Low <= Close
        if np.any(l > o) or np.any(l > c):
            return {"is_valid": False, "status": "CORRUPT_OHLC_LOW", "reason": "Low price violated boundary (Low > Open or Low > Close)"}

        # Session-aware Intraday Continuity & Missing Candle Validation (Item 6):
        # Distinguish normal overnight/weekend session gaps from missing intraday candles.
        # Normal 5-minute session has 75 candles (09:15 to 15:30 IST).
        df_eval['Date'] = df_eval.index.date
        session_dates = df_eval['Date'].unique()
        session_stats = []
        has_intraday_data_corruption = False
        corrupt_session_reason = ""

        for s_date in session_dates:
            s_slice = df_eval[df_eval['Date'] == s_date]
            n_bars = len(s_slice)
            
            # Check timestamps alignment to 5-minute boundaries (minutes % 5 == 0 or 15)
            s_minutes = [t.minute for t in s_slice.index]
            unaligned_bars = sum(1 for m in s_minutes if m % 5 != 0)
            if unaligned_bars > 0:
                has_intraday_data_corruption = True
                corrupt_session_reason = f"Session {s_date} contains {unaligned_bars} non-5-minute aligned candle timestamps"
                break

            # Evaluate intraday intervals between consecutive candles within the SAME session
            if n_bars >= 2:
                sec_diffs = (s_slice.index[1:] - s_slice.index[:-1]).total_seconds()
                # Missing candles inside trading session (interval > 360 seconds / 6 minutes)
                intraday_gaps = [d for d in sec_diffs if d > 360]
                # Reject sessions with missing intraday gaps (> 1 gap or total missing duration > 15 mins)
                total_missing_gap_secs = sum(d - 300 for d in intraday_gaps)
                if len(intraday_gaps) > 1 or total_missing_gap_secs > 900:
                    has_intraday_data_corruption = True
                    corrupt_session_reason = f"Session {s_date} has {len(intraday_gaps)} missing intraday candle gaps ({int(total_missing_gap_secs/60)}m missing)"
                    break

            session_stats.append({
                "date": str(s_date),
                "candles_count": n_bars
            })

        if has_intraday_data_corruption:
            return {
                "is_valid": False,
                "status": "CORRUPT_INTRADAY_SESSION",
                "reason": corrupt_session_reason
            }

        last_dt = df_eval.index[-1]
        last_ts = last_dt.timestamp() if hasattr(last_dt, 'timestamp') else float(last_dt)

        return {
            "is_valid": True,
            "status": "VALID_HISTORICAL",
            "is_completed_candle": True,
            "latest_candle_ts": last_ts,
            "candle_end_ts": last_ts + candle_duration_seconds,
            "total_sessions": len(session_dates),
            "session_stats": session_stats
        }
