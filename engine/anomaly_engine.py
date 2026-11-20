import numpy as np
import pandas as pd

class AnomalyDetector:
    """
    Detects abnormal volume acceleration, price velocity, range expansion, 
    and relative volume (RVOL) anomalies in real-time.
    Supports Time-of-Day RVOL across historical completed intraday sessions.
    """
    @staticmethod
    def calculate_time_of_day_rvol(df, current_candle_idx=-1):
        """
        Computes true Time-of-Day RVOL: compares current 15m candle volume
        with the average volume of the exact same time slot across prior sessions.
        """
        if df is None or df.empty or len(df) < 10:
            return 1.0

        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
            df_copy.index = pd.to_datetime(df_copy.index)

        df_copy['Time'] = df_copy.index.time
        df_copy['Date'] = df_copy.index.date
        
        target_time = df_copy['Time'].iloc[current_candle_idx]
        current_date = df_copy['Date'].iloc[current_candle_idx]
        curr_vol = float(df_copy['Volume'].iloc[current_candle_idx])

        # Historical candles matching the same time of day from previous sessions
        same_time_bars = df_copy[(df_copy['Time'] == target_time) & (df_copy['Date'] < current_date)]
        if not same_time_bars.empty:
            avg_hist_vol = float(same_time_bars['Volume'].mean()) or 1.0
            return round(curr_vol / avg_hist_vol, 2)

        # Fallback to 20-candle rolling volume if insufficient historical same-time bars
        v = df_copy['Volume']
        prior_v = v.iloc[:-1] if len(v) > 1 else v
        avg_vol = float(prior_v.tail(20).mean()) or 1.0
        return round(curr_vol / avg_vol, 2)

    @staticmethod
    def calculate_rvol(v_series, window=20):
        if len(v_series) < 2:
            return 1.0
        curr_vol = float(v_series.iloc[-1])
        prior = v_series.iloc[:-1]
        avg_vol = float(prior.tail(window).mean()) or 1.0
        return round(curr_vol / avg_vol, 2)

    @staticmethod
    def detect_flat_open_breakout(df, prev_close=None):
        """
        Detects Scenario C: Stocks opening flat (|gap| < 0.65%) that experience
        sudden time-of-day volume displacement and explosive range expansion.
        """
        if df is None or len(df) < 2:
            return {"is_flat_open": False, "is_sudden_expansion": False, "rvol": 1.0, "range_expansion": 1.0, "direction": "NEUTRAL"}

        o = df['Open']
        h = df['High']
        l = df['Low']
        c = df['Close']
        v = df['Volume']

        # Isolate today's session
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
            df_copy.index = pd.to_datetime(df_copy.index)
        df_copy['Date'] = df_copy.index.date
        today_date = df_copy['Date'].iloc[-1]
        today_df = df_copy[df_copy['Date'] == today_date]
        if today_df.empty:
            today_df = df_copy

        open_price = float(today_df['Open'].iloc[0])
        if prev_close and prev_close > 0:
            gap_pct = abs((open_price - prev_close) / prev_close) * 100
        else:
            gap_pct = 0.0

        is_flat_open = gap_pct <= 0.65

        # Current candle velocity: compare current candle against completed prior candles
        curr_c = float(c.iloc[-1])
        curr_o = float(o.iloc[-1])
        curr_range = float(h.iloc[-1] - l.iloc[-1])
        prior_ranges = (h - l).iloc[:-1]
        avg_range = float(prior_ranges.tail(15).mean()) if len(prior_ranges) > 0 else (curr_range or 0.0001)
        avg_range = avg_range or 0.0001

        range_expansion = curr_range / avg_range
        # True time-of-day RVOL
        tod_rvol = AnomalyDetector.calculate_time_of_day_rvol(df)

        is_sudden_expansion = bool(tod_rvol >= 1.5 and range_expansion >= 1.4)
        direction = "BULLISH" if curr_c >= curr_o else "BEARISH"

        return {
            "is_flat_open": is_flat_open,
            "gap_pct": round(gap_pct, 2),
            "is_sudden_expansion": is_sudden_expansion,
            "rvol": tod_rvol,
            "range_expansion": round(range_expansion, 2),
            "direction": direction
        }
