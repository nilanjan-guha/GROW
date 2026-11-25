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
        Requires at least 2 distinct historical sessions to prevent single-candle distortions.
        """
        if df is None or df.empty or len(df) < 10:
            return 1.0

        df_copy = df.copy()
        if isinstance(df_copy.columns, pd.MultiIndex):
            df_copy.columns = df_copy.columns.get_level_values(0)

        if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
            df_copy.index = pd.to_datetime(df_copy.index)

        df_copy['Time'] = df_copy.index.time
        df_copy['Date'] = df_copy.index.date
        
        target_time = df_copy['Time'].iloc[current_candle_idx]
        current_date = df_copy['Date'].iloc[current_candle_idx]
        raw_vol = df_copy['Volume'].iloc[current_candle_idx]
        curr_vol = float(raw_vol.iloc[0] if isinstance(raw_vol, pd.Series) else (raw_vol.item() if hasattr(raw_vol, 'item') else raw_vol))

        # Historical candles matching the same time of day from previous sessions
        same_time_bars = df_copy[(df_copy['Time'] == target_time) & (df_copy['Date'] < current_date)]
        if len(same_time_bars) >= 2:
            avg_hist_vol = float(same_time_bars['Volume'].values.mean()) or 1.0
            return round(curr_vol / avg_hist_vol, 2)

        # Fallback to 20-candle rolling volume across prior completed bars
        v = df_copy['Volume']
        prior_v = v.iloc[:-1] if len(v) > 1 else v
        avg_vol = float(prior_v.tail(20).values.mean()) or 1.0
        return round(curr_vol / avg_vol, 2)

    @staticmethod
    def calculate_rvol(v_series, window=20):
        if len(v_series) < 2:
            return 1.0
        raw_v = v_series.iloc[-1]
        curr_vol = float(raw_v.iloc[0] if isinstance(raw_v, pd.Series) else (raw_v.item() if hasattr(raw_v, 'item') else raw_v))
        prior = v_series.iloc[:-1]
        avg_vol = float(prior.tail(window).values.mean()) or 1.0
        return round(curr_vol / avg_vol, 2)

    @staticmethod
    def detect_flat_open_breakout(df, prev_close=None):
        """
        Detects Scenario C: Stocks opening flat (|gap| < 0.65%) that experience
        sudden time-of-day volume displacement and explosive range expansion.
        """
        if df is None or len(df) < 2:
            return {"is_flat_open": False, "is_sudden_expansion": False, "rvol": 1.0, "range_expansion": 1.0, "direction": "NEUTRAL"}

        df_copy = df.copy()
        if isinstance(df_copy.columns, pd.MultiIndex):
            df_copy.columns = df_copy.columns.get_level_values(0)

        o = df_copy['Open']
        h = df_copy['High']
        l = df_copy['Low']
        c = df_copy['Close']
        v = df_copy['Volume']

        # Isolate today's session
        if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
            df_copy.index = pd.to_datetime(df_copy.index)
        df_copy['Date'] = df_copy.index.date
        today_date = df_copy['Date'].iloc[-1]
        today_df = df_copy[df_copy['Date'] == today_date]
        if today_df.empty:
            today_df = df_copy

        raw_open = today_df['Open'].iloc[0]
        open_price = float(raw_open.iloc[0] if isinstance(raw_open, pd.Series) else (raw_open.item() if hasattr(raw_open, 'item') else raw_open))
        if prev_close and prev_close > 0:
            gap_pct = abs((open_price - prev_close) / prev_close) * 100
        else:
            gap_pct = 0.0

        is_flat_open = gap_pct <= 0.65

        # Current candle velocity: compare current candle against completed prior candles
        raw_c = c.iloc[-1]
        raw_o = o.iloc[-1]
        raw_h = h.iloc[-1]
        raw_l = l.iloc[-1]

        curr_c = float(raw_c.iloc[0] if isinstance(raw_c, pd.Series) else (raw_c.item() if hasattr(raw_c, 'item') else raw_c))
        curr_o = float(raw_o.iloc[0] if isinstance(raw_o, pd.Series) else (raw_o.item() if hasattr(raw_o, 'item') else raw_o))
        val_h = float(raw_h.iloc[0] if isinstance(raw_h, pd.Series) else (raw_h.item() if hasattr(raw_h, 'item') else raw_h))
        val_l = float(raw_l.iloc[0] if isinstance(raw_l, pd.Series) else (raw_l.item() if hasattr(raw_l, 'item') else raw_l))
        curr_range = float(val_h - val_l)

        prior_ranges = (h - l).iloc[:-1]
        avg_range = float(prior_ranges.tail(15).values.mean()) if len(prior_ranges) > 0 else (curr_range or 0.0001)
        avg_range = avg_range or 0.0001

        range_expansion = curr_range / avg_range
        # True time-of-day RVOL
        tod_rvol = AnomalyDetector.calculate_time_of_day_rvol(df_copy)

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
