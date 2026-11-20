import numpy as np
import pandas as pd

class AnomalyDetector:
    """
    Detects abnormal volume acceleration, price velocity, range expansion, 
    and relative volume (RVOL) anomalies in real-time.
    """
    @staticmethod
    def calculate_rvol(v_series, window=20):
        if len(v_series) < window:
            return 1.0
        curr_vol = float(v_series.iloc[-1])
        avg_vol = float(v_series.iloc[-window:-1].mean()) or 1.0
        return round(curr_vol / avg_vol, 2)

    @staticmethod
    def calculate_zscore(series, window=20):
        if len(series) < window:
            return 0.0
        recent = series.iloc[-window:-1]
        mean = float(recent.mean())
        std = float(recent.std()) or 0.0001
        curr = float(series.iloc[-1])
        return round((curr - mean) / std, 2)

    @staticmethod
    def detect_flat_open_breakout(o, h, l, c, v, prev_close=None):
        """
        Detects Scenario C: Stocks opening flat (|gap| < 0.65%) that experience
        sudden volume displacement and explosive range expansion.
        """
        if len(o) < 2:
            return {"is_flat_open": False, "is_sudden_expansion": False, "rvol": 1.0, "range_expansion": 1.0, "direction": "NEUTRAL"}

        # Calculate actual today open vs previous close
        open_price = float(o.iloc[0])
        if prev_close and prev_close > 0:
            gap_pct = abs((open_price - prev_close) / prev_close) * 100
        else:
            gap_pct = 0.0
        
        # Check if open was flat (gap <= 0.65%)
        is_flat_open = gap_pct <= 0.65
        
        # Current candle velocity
        curr_c = float(c.iloc[-1])
        curr_o = float(o.iloc[-1])
        curr_range = float(h.iloc[-1] - l.iloc[-1])
        avg_range = float((h - l).rolling(min(len(h), 15)).mean().iloc[-1]) or 0.0001
        
        range_expansion = curr_range / avg_range
        rvol = AnomalyDetector.calculate_rvol(v)
        
        is_sudden_expansion = bool(rvol >= 1.5 and range_expansion >= 1.4)
        
        direction = "BULLISH" if curr_c >= curr_o else "BEARISH"
        return {
            "is_flat_open": is_flat_open,
            "gap_pct": round(gap_pct, 2),
            "is_sudden_expansion": is_sudden_expansion,
            "rvol": rvol,
            "range_expansion": round(range_expansion, 2),
            "direction": direction
        }
