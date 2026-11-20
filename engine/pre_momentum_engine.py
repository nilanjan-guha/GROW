import pandas as pd
import numpy as np

class PreMomentumEngine:
    """
    Dedicated Pre-Momentum Prediction Engine:
    
    Scans F&O candidates to detect emerging setups BEFORE the major move becomes obvious.
    Separates market state into 3 distinct operational stages:
    1. PREPARING: Early structural compression, RVOL accumulation, or sector-led divergence.
    2. IGNITION: Emerging momentum, range expansion, or structural VWAP reclaim.
    3. CONFIRMED: Completed-candle institutional confluence with validated entry, structural SL and RR >= 1.5.
    """

    @staticmethod
    def classify_stage(df, sector_bias="NEUTRAL", vwap=None, rvol=1.0, composite_score=0, opp_eval=None) -> dict:
        """
        Classifies stock into PREPARING, IGNITION, or CONFIRMED.
        """
        if df is None or df.empty or len(df) < 10:
            return {
                "stage": "INACTIVE",
                "stage_score": 0,
                "bias": "NEUTRAL",
                "indicators": ["Insufficient data"],
                "is_tradeable": False
            }

        c = df['Close']
        o = df['Open']
        h = df['High']
        l = df['Low']
        v = df['Volume']

        curr_c = float(c.iloc[-1])
        curr_o = float(o.iloc[-1])
        curr_h = float(h.iloc[-1])
        curr_l = float(l.iloc[-1])

        # 1. 14-period ATR
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1]) or (curr_c * 0.01)

        # 2. Volatility Compression (Bollinger Bandwidth / NR4 / NR7)
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bandwidth = float(((std20 * 2) / sma20).iloc[-1]) if len(c) >= 20 else 0.05
        is_compressed = bandwidth < 0.025  # Tight squeeze

        # 3. Micro Range Compression (Prior 3 candles smaller than 0.75x ATR)
        recent_ranges = [(float(h.iloc[k]) - float(l.iloc[k])) for k in range(-4, -1)]
        is_range_squeezed = all(r < (atr14 * 0.85) for r in recent_ranges)

        # 4. Range Expansion on latest candle
        curr_range = curr_h - curr_l
        is_range_expanding = curr_range >= (atr14 * 1.5)

        # 5. Volume Accumulation / Surge
        is_vol_accumulating = (rvol >= 1.25)
        is_vol_surge = (rvol >= 1.8)

        # 6. VWAP Position
        above_vwap = (curr_c > vwap) if vwap else False
        below_vwap = (curr_c < vwap) if vwap else False

        # Directional Tendency
        bull_bias = (sector_bias == "BULLISH") or (above_vwap and curr_c > curr_o)
        bear_bias = (sector_bias == "BEARISH") or (below_vwap and curr_c < curr_o)
        direction = "BULLISH" if bull_bias and not bear_bias else ("BEARISH" if bear_bias and not bull_bias else "NEUTRAL")

        indicators = []
        if is_compressed or is_range_squeezed:
            indicators.append("Volatility Squeeze (Energy Coil)")
        if is_vol_accumulating:
            indicators.append(f"Volume Accumulation ({rvol:.1f}x)")
        if is_range_expanding:
            indicators.append(f"Range Expansion ({curr_range/atr14:.1f}x ATR)")
        if above_vwap:
            indicators.append("Holding Above VWAP")
        elif below_vwap:
            indicators.append("Trading Below VWAP")

        # Determine Stage
        # Stage 3: CONFIRMED
        if opp_eval and opp_eval.get("is_viable") and composite_score >= 50:
            return {
                "stage": "CONFIRMED",
                "stage_score": composite_score,
                "bias": direction,
                "indicators": indicators + [f"Verified Structural Targets (RR 1:{opp_eval.get('risk_reward_ratio', 1.5)})"],
                "is_tradeable": True,
                "details": "Institutional Confluence + Strict Opportunity Validated"
            }

        # Stage 2: IGNITION
        elif (is_range_expanding or is_vol_surge) and (composite_score >= 40 or rvol >= 1.75):
            return {
                "stage": "IGNITION",
                "stage_score": max(composite_score, int(rvol * 30)),
                "bias": direction,
                "indicators": indicators + ["Momentum Surge in Progress"],
                "is_tradeable": False,
                "details": "Momentum Ignited: Awaiting completed-candle closure and structural risk-reward check."
            }

        # Stage 1: PREPARING
        elif (is_compressed or is_range_squeezed or is_vol_accumulating) and (sector_bias != "NEUTRAL" or rvol >= 1.2):
            prep_score = int(min(90, 30 + (rvol * 20) + (15 if is_compressed else 0) + (15 if sector_bias != "NEUTRAL" else 0)))
            return {
                "stage": "PREPARING",
                "stage_score": prep_score,
                "bias": direction,
                "indicators": indicators + [f"Sector Momentum Alignment ({sector_bias})"],
                "is_tradeable": False,
                "details": "Early Compression/Coil detected. Order flow loading before breakout."
            }

        return {
            "stage": "INACTIVE",
            "stage_score": composite_score,
            "bias": "NEUTRAL",
            "indicators": indicators if indicators else ["Normal Sideways Fluctuation"],
            "is_tradeable": False,
            "details": "No pre-momentum or ignition pattern active."
        }
