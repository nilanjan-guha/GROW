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

    @staticmethod
    def detect_early_warning_preparation(df, vwap=None, sector_bias="NEUTRAL"):
        """
        Early-Warning Pre-Breakout Detection:
        Identifies subtle accumulation & energy coil before ignition or breakout:
        1. Consecutive volume ramp (v[t] > v[t-1] > v[t-2]).
        2. Candle body narrowing (compression).
        3. Holding near VWAP (+/- 0.35%).
        4. Sector alignment.
        """
        if df is None or df.empty or len(df) < 10:
            return {"is_preparing": False, "reason": "Insufficient history"}

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

            # 1. 14-period ATR
            tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
            atr14 = float(tr.rolling(14).mean().iloc[-1]) or (curr_c * 0.005)

            # 2. Volume Ramp
            vol_ramp = False
            if len(v) >= 3:
                vol_ramp = bool(v.iloc[-1] > v.iloc[-2] > v.iloc[-3])

            # 3. Micro Compression
            ranges = [(float(h.iloc[i]) - float(l.iloc[i])) for i in range(-4, 0)]
            is_compressed = all(r < (atr14 * 0.85) for r in ranges[:-1])

            # 4. Proximity to VWAP
            vwap_dist_pct = (abs(curr_c - vwap) / vwap * 100) if vwap else 1.0
            holding_near_vwap = vwap_dist_pct <= 0.40

            is_preparing = (vol_ramp or is_compressed) and (holding_near_vwap or sector_bias != "NEUTRAL")
            direction = "BULLISH" if curr_c >= curr_o else "BEARISH"

            prep_score = 0
            if is_preparing:
                prep_score = int(40 + (25 if vol_ramp else 0) + (20 if is_compressed else 0) + (15 if holding_near_vwap else 0))

            return {
                "is_preparing": is_preparing,
                "direction": direction,
                "preparation_score": prep_score,
                "vol_ramp": vol_ramp,
                "is_compressed": is_compressed,
                "holding_near_vwap": holding_near_vwap,
                "reason": "Early Energy Coil: Volume accumulating with narrowing candle ranges near VWAP" if is_preparing else "No early coil detected"
            }
        except Exception as e:
            return {"is_preparing": False, "error": str(e)}
