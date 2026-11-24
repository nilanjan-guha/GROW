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
    def validate_3way_alignment(stock_df, nifty_df=None, sector_df=None):
        """
        Validates strict point-in-time completed candle timestamp equality across:
        1. Stock completed candle
        2. NIFTY benchmark completed candle
        3. Sector index completed candle
        If any required benchmark is missing, unaligned, or stale, marks that benchmark
        explicitly UNAVAILABLE rather than silently using stale data or misaligned bars.
        """
        if stock_df is None or stock_df.empty:
            return {"is_aligned": False, "stock_ts": None, "nifty_aligned": False, "sector_aligned": False}

        stock_ts = stock_df.index[-1]
        
        nifty_aligned = False
        nifty_ts = None
        if nifty_df is not None and not nifty_df.empty:
            nifty_ts = nifty_df.index[-1]
            nifty_aligned = (stock_ts == nifty_ts)

        sector_aligned = False
        sector_ts = None
        if sector_df is not None and not sector_df.empty:
            sector_ts = sector_df.index[-1]
            sector_aligned = (stock_ts == sector_ts)

        return {
            "stock_ts": stock_ts,
            "nifty_ts": nifty_ts,
            "sector_ts": sector_ts,
            "nifty_aligned": nifty_aligned,
            "sector_aligned": sector_aligned,
            "is_aligned": nifty_aligned and (sector_df is None or sector_aligned)
        }

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

        # STRICT 5-MINUTE TIMEFRAME & SESSION GRID ENFORCEMENT (Live P0):
        # 1. Reject 1-minute, 3-minute, 15-minute or other invalid candle intervals
        # 2. Enforce seconds == 0 and microseconds == 0
        # 3. Enforce minutes % 5 == 0
        if len(df_eval) >= 2:
            recent_diffs = (df_eval.index[1:] - df_eval.index[:-1]).total_seconds()
            # If median interval is less than 240 seconds (e.g. 60s for 1m feed) or greater than 360 seconds (15m feed), reject!
            median_interval = float(np.median(recent_diffs[-10:] if len(recent_diffs) >= 10 else recent_diffs))
            if abs(median_interval - candle_duration_seconds) > 60:
                return {
                    "is_valid": False,
                    "status": "INVALID_TIMEFRAME",
                    "reason": f"Feed candle interval {int(median_interval)}s does not match expected {candle_duration_seconds}s (5-minute required; 1-minute strictly rejected)"
                }

        # Check timestamp alignment on recent candles
        for t in df_eval.index[-5:]:
            if t.second != 0 or t.microsecond != 0 or (t.minute % 5 != 0):
                return {
                    "is_valid": False,
                    "status": "UNALIGNED_CANDLE_TIMESTAMPS",
                    "reason": f"Candle timestamp {t.strftime('%Y-%m-%d %H:%M:%S')} is not aligned to exact 5-minute boundary"
                }

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
    def validate_historical_replay_data(df, candle_duration_seconds=300, allow_partial_final_session=True, requested_start_dt=None, requested_end_dt=None):
        """
        Validates historical data integrity for backtesting/replay (NO current wall-clock checks):
        - Enforces strictly unique, monotonic datetime index ordering.
        - Validates all OHLCV values: rejects NaN, infinite, or negative values.
        - Enforces strict mathematical OHLC relationships:
          High >= Open, High >= Close, Low <= Open, Low <= Close, High >= Low.
        - Validates volume is non-negative and finite.
        - Rigorous Session Grid Comparison:
          Generates the exact expected timestamp list for each completed session and validates that:
          1. Every candle timestamp matches the exact expected 5m session grid.
          2. Exactly 75 bars exist for completed full trading days (or the exact count for special sessions).
          3. Zero missing internal candle gaps exist.
          4. When requested_start_dt is provided, partial opening session is validated against requested window;
             otherwise, full sessions require market open (09:15 IST).
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
        if np.any(h < l) or np.any(h < o) or np.any(h < c):
            return {"is_valid": False, "status": "CORRUPT_OHLC_HIGH", "reason": "High price violated boundary (High < Low, High < Open, or High < Close)"}

        if np.any(l > o) or np.any(l > c):
            return {"is_valid": False, "status": "CORRUPT_OHLC_LOW", "reason": "Low price violated boundary (Low > Open or Low > Close)"}

        # Exact Expected Session Grid Comparison:
        df_eval['Date'] = df_eval.index.date
        session_dates = list(df_eval['Date'].unique())
        session_stats = []
        has_intraday_data_corruption = False
        corrupt_session_reason = ""

        ist = pytz.timezone('Asia/Kolkata')

        for idx_date, s_date in enumerate(session_dates):
            s_slice = df_eval[df_eval['Date'] == s_date]
            n_bars = len(s_slice)
            is_first_session = (idx_date == 0)
            is_final_session = (idx_date == len(session_dates) - 1)

            # Generate exact expected timestamp list for this date
            if s_date in SPECIAL_TRADING_SESSIONS:
                spec = SPECIAL_TRADING_SESSIONS[s_date]
                start_m = spec.get("start_minute", 1095)
                end_m = spec.get("end_minute", 1155)
            else:
                start_m = 555   # 09:15 IST
                end_m = 925     # 15:25 IST start of bar

            expected_minutes = list(range(start_m, end_m + 1, 5))
            expected_count = len(expected_minutes)

            # Check for any timestamps not matching expected session minute grid
            actual_minutes = [t.hour * 60 + t.minute for t in s_slice.index]
            off_grid_bars = sum(1 for m in actual_minutes if m not in expected_minutes)
            if off_grid_bars > 0:
                has_intraday_data_corruption = True
                corrupt_session_reason = f"Session {s_date} contains {off_grid_bars} candles outside the official NSE 5-minute schedule"
                break

            # Internal gaps check: all consecutive bars must have exactly 300s spacing
            if n_bars >= 2:
                sec_diffs = (s_slice.index[1:] - s_slice.index[:-1]).total_seconds()
                if any(d != 300 for d in sec_diffs):
                    has_intraday_data_corruption = True
                    corrupt_session_reason = f"Session {s_date} has internal missing candle gaps"
                    break

            # Window boundary awareness:
            # If intermediate completed session, must strictly have all expected bars
            if not is_first_session and not is_final_session:
                if n_bars != expected_count:
                    has_intraday_data_corruption = True
                    corrupt_session_reason = f"Session {s_date} incomplete: expected {expected_count} bars, found {n_bars} (missing {expected_count - n_bars})"
                    break
            elif is_first_session:
                if requested_start_dt is not None:
                    # Strict window start boundary validation:
                    req_start = pd.to_datetime(requested_start_dt)
                    if hasattr(req_start, 'tz') and req_start.tz is not None and df_eval.index.tz is None:
                        req_start = req_start.tz_localize(None)
                    elif hasattr(req_start, 'tz') and req_start.tz is None and df_eval.index.tz is not None:
                        req_start = req_start.tz_localize(df_eval.index.tz)
                    if df_eval.index[0] != req_start:
                        has_intraday_data_corruption = True
                        corrupt_session_reason = f"First candle {df_eval.index[0]} does not match requested_start_dt {req_start}"
                        break
                elif len(session_dates) > 1 and actual_minutes[0] != start_m:
                    # If requested_start_dt not specified and intermediate sessions follow, must start at 09:15
                    has_intraday_data_corruption = True
                    corrupt_session_reason = f"First session {s_date} missing market open (starts at {s_slice.index[0].strftime('%H:%M')})"
                    break
            elif is_final_session:
                if requested_end_dt is not None:
                    # Strict window end boundary validation:
                    req_end = pd.to_datetime(requested_end_dt)
                    if hasattr(req_end, 'tz') and req_end.tz is not None and df_eval.index.tz is None:
                        req_end = req_end.tz_localize(None)
                    elif hasattr(req_end, 'tz') and req_end.tz is None and df_eval.index.tz is not None:
                        req_end = req_end.tz_localize(df_eval.index.tz)
                    if df_eval.index[-1] != req_end:
                        has_intraday_data_corruption = True
                        corrupt_session_reason = f"Final candle {df_eval.index[-1]} does not match requested_end_dt {req_end}"
                        break
                elif not allow_partial_final_session:
                    if n_bars != expected_count:
                        has_intraday_data_corruption = True
                        corrupt_session_reason = f"Final session {s_date} incomplete: expected {expected_count} bars, found {n_bars}"
                        break

            session_stats.append({
                "date": str(s_date),
                "candles_count": n_bars,
                "expected_count": expected_count
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
