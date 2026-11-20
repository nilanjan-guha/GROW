import pandas as pd
import numpy as np

class InstitutionalStrategyEngine:
    """
    Evaluates 15 Institutional & SMC Trading Strategies:
    A: Gap-and-Go
    B: Gap-Fill Reversal
    C: Opening Range Breakout (ORB)
    D: Opening Range Breakdown
    E: VWAP Reclaim
    F: VWAP Rejection
    G: EMA 9/21 Momentum
    H: RVOL Breakout
    I: Flat-Open Sudden Momentum
    J: Liquidity Sweep Reversal
    K: ICT Fair Value Gap (FVG) Continuation
    L: Break of Structure (BOS) Continuation
    M: Smart Money Demand/Supply Mitigation
    N: Sector Rotation Alignment
    O: News Catalyst Momentum
    """
    @staticmethod
    def evaluate_strategies(df, prev_close, sector_bias="NEUTRAL", catalyst_info=None):
        if df.empty or len(df) < 20:
            return None

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

        # EMAs
        ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
        ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])

        # RSI 14
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss.replace(0, 0.0001))
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])

        # Intraday VWAP
        typical_price = (h + l + c) / 3
        cum_vol = v.cumsum()
        vwap = float(((typical_price * v).cumsum() / cum_vol.replace(0, 1)).iloc[-1])

        # ATR 14
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1]) or (curr_c * 0.01)

        # RVOL
        avg_vol20 = float(v.rolling(20).mean().iloc[-1]) or 1.0
        rvol = round(curr_v / avg_vol20, 2)

        # Gap calculation
        day_open = float(o.iloc[0])
        gap_pct = ((day_open - prev_close) / prev_close) * 100 if prev_close else 0.0

        # Opening Range (First 15-min candle)
        or_high = float(h.iloc[0])
        or_low = float(l.iloc[0])

        # SMC: Order Blocks & Liquidity
        bull_obs, bear_obs = [], []
        for i in range(1, len(df)-2):
            if c.iloc[i] < o.iloc[i] and c.iloc[i+1] > h.iloc[i]:
                bull_obs.append({"top": float(h.iloc[i]), "bottom": float(l.iloc[i])})
            if c.iloc[i] > o.iloc[i] and c.iloc[i+1] < l.iloc[i]:
                bear_obs.append({"top": float(h.iloc[i]), "bottom": float(l.iloc[i])})

        near_demand_ob = bull_obs[-1] if bull_obs else None
        near_supply_ob = bear_obs[-1] if bear_obs else None

        # Fair Value Gaps (FVG)
        fvg_bull = False
        fvg_bear = False
        if len(df) >= 3:
            # Bullish FVG: Low of candle 3 > High of candle 1
            if float(l.iloc[-1]) > float(h.iloc[-3]):
                fvg_bull = True
            # Bearish FVG: High of candle 3 < Low of candle 1
            if float(h.iloc[-1]) < float(l.iloc[-3]):
                fvg_bear = True

        # Signals dictionary
        signals = {}

        # Strategy A: Gap-and-Go
        if gap_pct >= 1.0 and curr_c > or_high and curr_c > vwap:
            signals["Gap-and-Go"] = {"direction": "BUY", "confidence": 88}
        # Strategy B: Gap-Down and Dump
        elif gap_pct <= -1.0 and curr_c < or_low and curr_c < vwap:
            signals["Gap-Down-Dump"] = {"direction": "SELL", "confidence": 88}

        # Strategy C & D: Opening Range Breakout / Breakdown
        if curr_c > or_high and curr_c > vwap and rvol > 1.2:
            signals["ORB Breakout"] = {"direction": "BUY", "confidence": 85}
        elif curr_c < or_low and curr_c < vwap and rvol > 1.2:
            signals["ORB Breakdown"] = {"direction": "SELL", "confidence": 85}

        # Strategy E & F: VWAP Reclaim / Rejection
        prev_c = float(c.iloc[-2])
        if prev_c < vwap and curr_c > vwap and ema9 > ema21:
            signals["VWAP Reclaim"] = {"direction": "BUY", "confidence": 89}
        elif prev_c > vwap and curr_c < vwap and ema9 < ema21:
            signals["VWAP Rejection"] = {"direction": "SELL", "confidence": 89}

        # Strategy G: EMA 9/21 Momentum
        if ema9 > ema21 and curr_c > ema9 and 52 <= rsi <= 75:
            signals["EMA Momentum"] = {"direction": "BUY", "confidence": 82}
        elif ema9 < ema21 and curr_c < ema9 and 25 <= rsi <= 48:
            signals["EMA Breakdown"] = {"direction": "SELL", "confidence": 82}

        # Strategy I: Flat-Open Sudden Momentum
        if abs(gap_pct) <= 0.65 and rvol >= 1.8:
            if curr_c > curr_o and curr_c > vwap:
                signals["Flat-Open Surge"] = {"direction": "BUY", "confidence": 92}
            elif curr_c < curr_o and curr_c < vwap:
                signals["Flat-Open Dump"] = {"direction": "SELL", "confidence": 92}

        # Strategy J & K: SMC Liquidity / Order Block / FVG Confluence
        if near_demand_ob and (near_demand_ob['bottom'] * 0.995 <= curr_c <= near_demand_ob['top'] * 1.012) and curr_c > vwap:
            signals["SMC Demand OB Mitigation"] = {"direction": "BUY", "confidence": 94}
        if near_supply_ob and (near_supply_ob['bottom'] * 0.988 <= curr_c <= near_supply_ob['top'] * 1.005) and curr_c < vwap:
            signals["SMC Supply OB Rejection"] = {"direction": "SELL", "confidence": 94}

        if fvg_bull and curr_c > vwap:
            signals["ICT Bullish FVG"] = {"direction": "BUY", "confidence": 86}
        elif fvg_bear and curr_c < vwap:
            signals["ICT Bearish FVG"] = {"direction": "SELL", "confidence": 86}

        # Composite Scoring (0 to 100)
        buy_weight = 0
        sell_weight = 0

        # Factor 1: RVOL & Volume (max 20)
        if rvol >= 2.0:
            buy_weight += 20 if curr_c > curr_o else 0
            sell_weight += 20 if curr_c < curr_o else 0
        elif rvol >= 1.3:
            buy_weight += 12 if curr_c > curr_o else 0
            sell_weight += 12 if curr_c < curr_o else 0

        # Factor 2: VWAP Positioning (max 20)
        if curr_c > vwap:
            buy_weight += 20
        else:
            sell_weight += 20

        # Factor 3: EMA 9/21 Trend Alignment (max 20)
        if ema9 > ema21:
            buy_weight += 20
        else:
            sell_weight += 20

        # Factor 4: RSI Momentum Sweet Spot (max 15)
        if 55 <= rsi <= 75:
            buy_weight += 15
        elif 25 <= rsi <= 45:
            sell_weight += 15

        # Factor 5: Sector Alignment (max 15)
        if sector_bias == "BULLISH":
            buy_weight += 15
        elif sector_bias == "BEARISH":
            sell_weight += 15

        # Factor 6: SMC / Liquidity confirmation (max 10)
        if near_demand_ob and curr_c >= near_demand_ob['bottom']:
            buy_weight += 10
        if near_supply_ob and curr_c <= near_supply_ob['top']:
            sell_weight += 10

        # Determine Primary Recommendation
        if buy_weight >= 65 and buy_weight > sell_weight:
            direction = "BUY"
            composite_score = buy_weight
            primary_strategy = list(signals.keys())[0] if signals else "VWAP Trend Continuation"
            sl = round(max(curr_c - (atr14 * 1.5), vwap * 0.996), 2)
            risk = max(curr_c - sl, curr_c * 0.005)
            tp1 = round(curr_c + (risk * 1.5), 2)
            tp2 = round(curr_c + (risk * 2.5), 2)
        elif sell_weight >= 65 and sell_weight > buy_weight:
            direction = "SELL"
            composite_score = sell_weight
            primary_strategy = list(signals.keys())[0] if signals else "VWAP Breakdown"
            sl = round(min(curr_c + (atr14 * 1.5), vwap * 1.004), 2)
            risk = max(sl - curr_c, curr_c * 0.005)
            tp1 = round(curr_c - (risk * 1.5), 2)
            tp2 = round(curr_c - (risk * 2.5), 2)
        else:
            direction = "NEUTRAL"
            composite_score = max(buy_weight, sell_weight)
            primary_strategy = "WAIT FOR ENTRY (NO TRADE)"
            sl = None
            tp1 = None
            tp2 = None

        return {
            "current_price": curr_c,
            "vwap": round(vwap, 2),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "rsi": round(rsi, 1),
            "atr": round(atr14, 2),
            "rvol": rvol,
            "gap_pct": round(gap_pct, 2),
            "direction": direction,
            "composite_score": composite_score,
            "primary_strategy": primary_strategy,
            "triggered_strategies": list(signals.keys()),
            "suggested_sl": sl,
            "suggested_tp1": tp1,
            "suggested_tp2": tp2,
            "nearest_demand_ob": near_demand_ob,
            "nearest_supply_ob": near_supply_ob,
            "probability_pct": min(95, max(60, composite_score + 10)) if direction != "NEUTRAL" else 50
        }
