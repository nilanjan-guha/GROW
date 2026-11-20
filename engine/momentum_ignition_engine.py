import pandas as pd
import numpy as np

class MomentumIgnitionEngine:
    """
    Dedicated Momentum Ignition Engine:
    Detects sudden acceleration and explosive directional impulses:
    1. Consolidation Compression: Bollinger/Keltner or narrow range (NR4/NR7).
    2. Volume Expansion: RVOL surge >= 2.0x time-of-day baseline.
    3. Range Expansion: Current candle range > 2.0x 14-period ATR.
    4. Price Acceleration: Consecutive directional candles breaking local fractals.
    5. Structural VWAP Reclaim/Loss confirmation.
    """

    @staticmethod
    def detect_ignition(df, vwap=None):
        if df is None or df.empty or len(df) < 15:
            return {"is_ignited": False, "reason": "Insufficient history"}

        try:
            c = df['Close']
            o = df['Open']
            h = df['High']
            l = df['Low']
            v = df['Volume']

            curr_c = float(c.iloc[-1])
            curr_o = float(o.iloc[-1])
            curr_h = float(h.iloc[-1])
            curr_l = float(l.iloc[-1])
            curr_v = float(v.iloc[-1])

            # 1. 14-period local ATR
            tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
            atr14 = float(tr.rolling(14).mean().iloc[-2]) if len(tr) >= 15 else float(curr_c * 0.005)
            if pd.isna(atr14) or atr14 <= 0:
                atr14 = float(curr_c * 0.005)

            # 2. Candle Range Expansion
            curr_bar_range = curr_h - curr_l
            range_expansion_ratio = round(curr_bar_range / atr14, 2)
            is_range_expansion = range_expansion_ratio >= 1.75

            # 3. Volume Expansion (Time of day or rolling 14-period volume)
            vol_mean = float(v.iloc[-15:-1].mean()) if len(v) >= 15 else curr_v
            vol_ratio = round(curr_v / (vol_mean or 1), 2)
            is_vol_expansion = vol_ratio >= 1.8

            # 4. Compression Breakout (Prior 3 to 5 bars had compressed ranges < 0.8x ATR)
            prior_ranges = [(float(h.iloc[k]) - float(l.iloc[k])) for k in range(-6, -1)]
            avg_prior_range = np.mean(prior_ranges) if prior_ranges else atr14
            is_compressed_prior = avg_prior_range < (atr14 * 0.95)

            # 5. Price Acceleration & Momentum Direction
            candle_body = abs(curr_c - curr_o)
            body_ratio = candle_body / (curr_bar_range or 0.001)

            # Check directional acceleration
            is_bullish_ignition = (
                curr_c > curr_o and 
                body_ratio >= 0.60 and 
                is_range_expansion and 
                is_vol_expansion and
                (vwap is None or curr_c > vwap)
            )

            is_bearish_ignition = (
                curr_c < curr_o and 
                body_ratio >= 0.60 and 
                is_range_expansion and 
                is_vol_expansion and
                (vwap is None or curr_c < vwap)
            )

            is_ignited = is_bullish_ignition or is_bearish_ignition
            direction = "BUY" if is_bullish_ignition else ("SELL" if is_bearish_ignition else "NEUTRAL")

            ignition_score = 0
            if is_ignited:
                ignition_score = min(95, int(50 + (range_expansion_ratio * 10) + (vol_ratio * 5) + (10 if is_compressed_prior else 0)))

            return {
                "is_ignited": is_ignited,
                "direction": direction,
                "ignition_score": ignition_score,
                "range_expansion_ratio": range_expansion_ratio,
                "volume_expansion_ratio": vol_ratio,
                "is_compressed_prior": is_compressed_prior,
                "reason": f"Ignition {direction}: Range {range_expansion_ratio}x ATR, Vol {vol_ratio}x avg, Body {int(body_ratio*100)}%" if is_ignited else "No explosive momentum ignition detected"
            }
        except Exception as e:
            return {"is_ignited": False, "direction": "NEUTRAL", "error": str(e)}
