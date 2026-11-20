import pandas as pd
import numpy as np

class InstitutionalStrategyEngine:
    """
    15-Strategy Institutional Confluence Engine:
    A: Gap-and-Go Momentum
    B: Gap-Fill Reversal
    C: Opening Range Breakout (ORB)
    D: Opening Range Breakdown
    E: Intraday VWAP Reclaim
    F: Intraday VWAP Rejection
    G: EMA 9/21 Trend Momentum
    H: Relative Volume (RVOL) Institutional Anomaly
    I: Flat-Open Sudden Momentum Surge / Dump
    J: Liquidity Sweep & Stop Hunt Reversal
    K: ICT Fair Value Gap (FVG) Continuation
    L: Break of Structure (BOS) Continuation
    M: Smart Money Demand/Supply Order Block Mitigation
    N: Sector Leadership & Rotation Alignment
    O: Corporate & News Catalyst Momentum
    """
    @staticmethod
    def evaluate_strategies(df, prev_close=None, sector_bias="NEUTRAL", catalyst_info=None):
        if df is None or df.empty or len(df) < 5:
            return None

        # Ensure single-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 1. Date isolation for session-accurate intraday calculations
        df['Date'] = df.index.date
        unique_dates = sorted(df['Date'].unique())
        today = unique_dates[-1]
        prev_date = unique_dates[-2] if len(unique_dates) >= 2 else None

        today_df = df[df['Date'] == today]
        if today_df.empty:
            today_df = df

        # True Previous Day Close
        if prev_close is None:
            if prev_date is not None:
                prev_close = float(df[df['Date'] == prev_date]['Close'].iloc[-1])
            else:
                prev_close = float(today_df['Open'].iloc[0])

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

        # 2. TRUE SESSION-RESET INTRADAY VWAP (Resets at 09:15 open every session)
        today_tp = (today_df['High'] + today_df['Low'] + today_df['Close']) / 3
        today_cum_vol = today_df['Volume'].cumsum().replace(0, 1)
        session_vwap_series = (today_tp * today_df['Volume']).cumsum() / today_cum_vol
        vwap = float(session_vwap_series.iloc[-1])

        # 3. TRUE OPENING RANGE & GAP FROM TODAY'S FIRST CANDLE
        today_open = float(today_df['Open'].iloc[0])
        or_high = float(today_df['High'].iloc[0])
        or_low = float(today_df['Low'].iloc[0])
        gap_pct = ((today_open - prev_close) / prev_close) * 100 if prev_close else 0.0

        # Technical Indicators across full series
        ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
        ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])

        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss.replace(0, 0.0001))
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])

        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1]) or (curr_c * 0.01)

        # RVOL (Time-of-day comparable or 20-candle rolling volume)
        avg_vol20 = float(v.rolling(20).mean().iloc[-1]) or 1.0
        rvol = round(curr_v / avg_vol20, 2)

        # SMC: Order Blocks & Liquidity Sweeps
        bull_obs, bear_obs = [], []
        swing_highs, swing_lows = [], []
        for i in range(2, len(df)-2):
            if float(h.iloc[i]) == max([float(x) for x in h.iloc[i-2:i+3]]):
                swing_highs.append(float(h.iloc[i]))
            if float(l.iloc[i]) == min([float(x) for x in l.iloc[i-2:i+3]]):
                swing_lows.append(float(l.iloc[i]))

        for i in range(1, len(df)-2):
            if float(c.iloc[i]) < float(o.iloc[i]) and float(c.iloc[i+1]) > float(h.iloc[i]):
                bull_obs.append({"top": float(h.iloc[i]), "bottom": float(l.iloc[i])})
            if float(c.iloc[i]) > float(o.iloc[i]) and float(c.iloc[i+1]) < float(l.iloc[i]):
                bear_obs.append({"top": float(h.iloc[i]), "bottom": float(l.iloc[i])})

        near_demand_ob = bull_obs[-1] if bull_obs else None
        near_supply_ob = bear_obs[-1] if bear_obs else None

        # Fair Value Gaps (FVG)
        fvg_bull = len(df) >= 3 and float(l.iloc[-1]) > float(h.iloc[-3])
        fvg_bear = len(df) >= 3 and float(h.iloc[-1]) < float(l.iloc[-3])

        # Break of Structure (BOS)
        prev_swing_high = max(swing_highs[-3:]) if len(swing_highs) >= 3 else float(h.max())
        prev_swing_low = min(swing_lows[-3:]) if len(swing_lows) >= 3 else float(l.min())
        bos_bullish = curr_c > prev_swing_high and curr_c > vwap
        bos_bearish = curr_c < prev_swing_low and curr_c < vwap

        # Liquidity Sweep & Stop Hunt Reversal
        sweep_bullish = float(l.iloc[-1]) < prev_swing_low and curr_c > prev_swing_low and curr_c > curr_o
        sweep_bearish = float(h.iloc[-1]) > prev_swing_high and curr_c < prev_swing_high and curr_c < curr_o

        # Strategy Evaluations & Evidence Scoring
        evaluated_signals = []

        # Strategy A: Gap-and-Go
        if gap_pct >= 0.8 and curr_c > or_high and curr_c > vwap:
            evaluated_signals.append({"strategy": "Gap-and-Go", "direction": "BUY", "weight": 22})
        # Strategy B: Gap-Fill Reversal
        elif gap_pct >= 1.5 and curr_c < vwap and curr_c < curr_o:
            evaluated_signals.append({"strategy": "Gap-Fill Reversal", "direction": "SELL", "weight": 20})
        elif gap_pct <= -1.5 and curr_c > vwap and curr_c > curr_o:
            evaluated_signals.append({"strategy": "Gap-Fill Reversal", "direction": "BUY", "weight": 20})

        # Strategy C & D: ORB Breakout / Breakdown
        if curr_c > or_high and curr_c > vwap and rvol >= 1.2:
            evaluated_signals.append({"strategy": "Opening Range Breakout", "direction": "BUY", "weight": 24})
        elif curr_c < or_low and curr_c < vwap and rvol >= 1.2:
            evaluated_signals.append({"strategy": "Opening Range Breakdown", "direction": "SELL", "weight": 24})

        # Strategy E & F: VWAP Reclaim / Rejection
        prev_c = float(c.iloc[-2]) if len(c) >= 2 else curr_c
        if prev_c < vwap and curr_c > vwap and ema9 >= ema21:
            evaluated_signals.append({"strategy": "VWAP Reclaim", "direction": "BUY", "weight": 22})
        elif prev_c > vwap and curr_c < vwap and ema9 <= ema21:
            evaluated_signals.append({"strategy": "VWAP Rejection", "direction": "SELL", "weight": 22})

        # Strategy G: EMA 9/21 Trend Momentum
        if ema9 > ema21 and curr_c > ema9 and 52 <= rsi <= 76:
            evaluated_signals.append({"strategy": "EMA Momentum", "direction": "BUY", "weight": 18})
        elif ema9 < ema21 and curr_c < ema9 and 24 <= rsi <= 48:
            evaluated_signals.append({"strategy": "EMA Breakdown", "direction": "SELL", "weight": 18})

        # Strategy H: RVOL Institutional Breakout
        if rvol >= 2.0 and abs(curr_c - curr_o) / curr_c >= 0.008:
            evaluated_signals.append({"strategy": "RVOL Anomaly Breakout", "direction": "BUY" if curr_c > curr_o else "SELL", "weight": 25})

        # Strategy I: Flat-Open Sudden Momentum
        if abs(gap_pct) <= 0.65 and rvol >= 1.6:
            if curr_c > curr_o and curr_c > vwap:
                evaluated_signals.append({"strategy": "Flat-Open Surge", "direction": "BUY", "weight": 25})
            elif curr_c < curr_o and curr_c < vwap:
                evaluated_signals.append({"strategy": "Flat-Open Dump", "direction": "SELL", "weight": 25})

        # Strategy J: Liquidity Sweep & Stop Hunt Reversal
        if sweep_bullish:
            evaluated_signals.append({"strategy": "Liquidity Sweep Reversal", "direction": "BUY", "weight": 24})
        elif sweep_bearish:
            evaluated_signals.append({"strategy": "Liquidity Sweep Reversal", "direction": "SELL", "weight": 24})

        # Strategy K: ICT Fair Value Gap (FVG) Continuation
        if fvg_bull and curr_c > vwap:
            evaluated_signals.append({"strategy": "ICT Bullish FVG", "direction": "BUY", "weight": 19})
        elif fvg_bear and curr_c < vwap:
            evaluated_signals.append({"strategy": "ICT Bearish FVG", "direction": "SELL", "weight": 19})

        # Strategy L: Break of Structure (BOS)
        if bos_bullish:
            evaluated_signals.append({"strategy": "Break of Structure", "direction": "BUY", "weight": 21})
        elif bos_bearish:
            evaluated_signals.append({"strategy": "Break of Structure", "direction": "SELL", "weight": 21})

        # Strategy M: Order Block Mitigation
        if near_demand_ob and (near_demand_ob['bottom'] * 0.995 <= curr_c <= near_demand_ob['top'] * 1.012) and curr_c > vwap:
            evaluated_signals.append({"strategy": "Demand OB Mitigation", "direction": "BUY", "weight": 23})
        elif near_supply_ob and (near_supply_ob['bottom'] * 0.988 <= curr_c <= near_supply_ob['top'] * 1.005) and curr_c < vwap:
            evaluated_signals.append({"strategy": "Supply OB Rejection", "direction": "SELL", "weight": 23})

        # Strategy N: Sector Alignment
        if sector_bias == "BULLISH" and curr_c > vwap:
            evaluated_signals.append({"strategy": "Sector Leadership Momentum", "direction": "BUY", "weight": 15})
        elif sector_bias == "BEARISH" and curr_c < vwap:
            evaluated_signals.append({"strategy": "Sector Weakness Breakdown", "direction": "SELL", "weight": 15})

        # Strategy O: News Catalyst Momentum
        if catalyst_info and catalyst_info.get("sentiment") == "BULLISH" and curr_c > vwap:
            evaluated_signals.append({"strategy": "News Catalyst Momentum", "direction": "BUY", "weight": 20})
        elif catalyst_info and catalyst_info.get("sentiment") == "BEARISH" and curr_c < vwap:
            evaluated_signals.append({"strategy": "News Catalyst Breakdown", "direction": "SELL", "weight": 20})

        # Rank strategies by weight/evidence
        buy_signals = [s for s in evaluated_signals if s["direction"] == "BUY"]
        sell_signals = [s for s in evaluated_signals if s["direction"] == "SELL"]

        buy_score = min(100, sum(s["weight"] for s in buy_signals))
        sell_score = min(100, sum(s["weight"] for s in sell_signals))

        # Sort strategies by weight descending so strongest strategy is primary
        buy_signals.sort(key=lambda x: x["weight"], reverse=True)
        sell_signals.sort(key=lambda x: x["weight"], reverse=True)

        if buy_score >= 50 and buy_score > sell_score:
            direction = "BUY"
            composite_score = buy_score
            primary_strategy = buy_signals[0]["strategy"]
            triggered = [s["strategy"] for s in buy_signals]
            sl = round(max(curr_c - (atr14 * 1.5), vwap * 0.996), 2)
            risk = max(curr_c - sl, curr_c * 0.005)
            tp1 = round(curr_c + (risk * 1.5), 2)
            tp2 = round(curr_c + (risk * 2.5), 2)
        elif sell_score >= 50 and sell_score > buy_score:
            direction = "SELL"
            composite_score = sell_score
            primary_strategy = sell_signals[0]["strategy"]
            triggered = [s["strategy"] for s in sell_signals]
            sl = round(min(curr_c + (atr14 * 1.5), vwap * 1.004), 2)
            risk = max(sl - curr_c, curr_c * 0.005)
            tp1 = round(curr_c - (risk * 1.5), 2)
            tp2 = round(curr_c - (risk * 2.5), 2)
        else:
            direction = "NEUTRAL"
            composite_score = max(buy_score, sell_score)
            primary_strategy = "WAIT FOR ENTRY (NO TRADE)"
            triggered = [s["strategy"] for s in (buy_signals + sell_signals)]
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
            "triggered_strategies": triggered,
            "suggested_sl": sl,
            "suggested_tp1": tp1,
            "suggested_tp2": tp2,
            "nearest_demand_ob": near_demand_ob,
            "nearest_supply_ob": near_supply_ob,
            "or_high": or_high,
            "or_low": or_low,
            "prev_close": prev_close
        }
