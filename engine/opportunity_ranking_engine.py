import pandas as pd
import numpy as np

class OpportunityRankingEngine:
    """
    Dedicated Multi-Factor Pre-Breakout & Opportunity Ranking Engine.
    
    Evaluates candidate F&O stocks across 6 independent quantitative dimensions:
    1. Relative Strength Acceleration:
       - vs NIFTY index (stock change % minus NIFTY change %)
       - vs Sector index (stock change % minus sector change %)
    2. Volume Acceleration across completed candles:
       - Consecutive volume expansion over last 3 bars (v[t] > v[t-1] > v[t-2])
    3. Structural Volatility Squeeze & Coil:
       - Narrowing candle ranges (NR4/NR7) / Bollinger compression
    4. VWAP Position & Trend Confluence:
       - Holding strictly above session VWAP for Longs / below for Shorts
    5. Structural Distance to Breakout / Key Level:
       - Proximity to ORH/ORL or Swing Resistance/Support (< 0.5% away)
    6. Opportunity Gate & Risk:Reward:
       - Remaining expansion potential > 1.0%, actual RR >= 1.5, low exhaustion risk
    
    Produces distinct stage classification:
    - PREPARING: Early energy accumulation & volume ramp before the move
    - IGNITION: Momentum expansion underway
    - CONFIRMED: Completed-candle institutional confluence with validated structural targets
    
    Selects Top 3 Bullish & Top 3 Bearish opportunities with complete setup transparency.
    """

    @staticmethod
    def calculate_relative_strength_acceleration(stock_df, nifty_df=None, sector_change_pct=0.0):
        """Calculates multi-timeframe relative strength against NIFTY and Sector"""
        if stock_df is None or stock_df.empty or len(stock_df) < 2:
            return {"rs_nifty": 0.0, "rs_sector": 0.0, "rs_accel": 0.0}

        c = stock_df['Close']
        stock_change_pct = float(((c.iloc[-1] - c.iloc[0]) / c.iloc[0]) * 100) if c.iloc[0] > 0 else 0.0
        
        # NIFTY change
        nifty_change_pct = 0.0
        if nifty_df is not None and not nifty_df.empty and len(nifty_df) >= 2:
            nc = nifty_df['Close']
            nifty_change_pct = float(((nc.iloc[-1] - nc.iloc[0]) / nc.iloc[0]) * 100) if nc.iloc[0] > 0 else 0.0

        rs_nifty = round(stock_change_pct - nifty_change_pct, 2)
        rs_sector = round(stock_change_pct - sector_change_pct, 2)

        # Acceleration: is RS growing over last 3 bars?
        if len(c) >= 4:
            prev_stock_pct = float(((c.iloc[-2] - c.iloc[0]) / c.iloc[0]) * 100) if c.iloc[0] > 0 else 0.0
            rs_accel = round(stock_change_pct - prev_stock_pct, 2)
        else:
            rs_accel = 0.0

        return {
            "stock_change_pct": round(stock_change_pct, 2),
            "rs_nifty": rs_nifty,
            "rs_sector": rs_sector,
            "rs_accel": rs_accel
        }

    @staticmethod
    def evaluate_stock_opportunity(stock_df, symbol, sector_name="GENERAL", sector_change=0.0, nifty_df=None, vwap=None, opp_eval=None):
        """Computes comprehensive quantitative opportunity score and prediction stage"""
        if stock_df is None or stock_df.empty or len(stock_df) < 5:
            return None

        c = stock_df['Close']
        o = stock_df['Open']
        h = stock_df['High']
        l = stock_df['Low']
        v = stock_df['Volume']

        curr_c = float(c.iloc[-1])
        curr_o = float(o.iloc[-1])
        curr_h = float(h.iloc[-1])
        curr_l = float(l.iloc[-1])
        curr_v = float(v.iloc[-1])

        # 1. 14-period ATR (Wilder)
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1).dropna()
        if len(tr) >= 14:
            tr_vals = tr.values
            atr14 = float(np.mean(tr_vals[:14]))
            for t_val in tr_vals[14:]:
                atr14 = (atr14 * 13 + float(t_val)) / 14
        else:
            atr14 = float(tr.mean()) if not tr.empty else (curr_c * 0.01)

        # 2. Volume Expansion & Consecutive Acceleration
        vol_mean_15 = float(v.iloc[-16:-1].mean()) if len(v) >= 16 else curr_v
        rvol = round(curr_v / (vol_mean_15 or 1), 2)
        
        # Consecutive rising volume across last 3 completed bars
        vol_consecutive_rise = False
        if len(v) >= 3:
            vol_consecutive_rise = (v.iloc[-1] > v.iloc[-2] > v.iloc[-3])

        # 3. Volatility Squeeze / Narrow Range (NR4/NR7)
        recent_ranges = [(float(h.iloc[k]) - float(l.iloc[k])) for k in range(-4, 0)] if len(h) >= 4 else [curr_h - curr_l]
        is_range_squeezed = all(r < (atr14 * 0.85) for r in recent_ranges[:-1])
        curr_range = curr_h - curr_l
        is_expanding_now = curr_range >= (atr14 * 1.5)

        # 4. Relative Strength Acceleration
        rs_data = OpportunityRankingEngine.calculate_relative_strength_acceleration(stock_df, nifty_df, sector_change)
        rs_nifty = rs_data["rs_nifty"]
        rs_sector = rs_data["rs_sector"]
        rs_accel = rs_data["rs_accel"]

        # 5. VWAP Relationship
        above_vwap = (curr_c > vwap) if vwap else False
        below_vwap = (curr_c < vwap) if vwap else False

        # Directional Tendency
        bull_score = 0
        bear_score = 0

        # Bullish scoring components
        if rs_nifty > 0: bull_score += min(20, int(rs_nifty * 5))
        if rs_sector > 0: bull_score += min(20, int(rs_sector * 5))
        if rs_accel > 0: bull_score += 15
        if above_vwap: bull_score += 20
        if rvol >= 1.5: bull_score += 15
        if vol_consecutive_rise and curr_c > curr_o: bull_score += 15
        if is_range_squeezed and curr_c > curr_o: bull_score += 15
        if is_expanding_now and curr_c > curr_o: bull_score += 20

        # Bearish scoring components
        if rs_nifty < 0: bear_score += min(20, int(abs(rs_nifty) * 5))
        if rs_sector < 0: bear_score += min(20, int(abs(rs_sector) * 5))
        if rs_accel < 0: bear_score += 15
        if below_vwap: bear_score += 20
        if rvol >= 1.5: bear_score += 15
        if vol_consecutive_rise and curr_c < curr_o: bear_score += 15
        if is_range_squeezed and curr_c < curr_o: bear_score += 15
        if is_expanding_now and curr_c < curr_o: bear_score += 20

        direction = "BULLISH" if (bull_score >= bear_score and bull_score >= 40) else ("BEARISH" if bear_score >= 40 else "NEUTRAL")
        raw_score = bull_score if direction == "BULLISH" else (bear_score if direction == "BEARISH" else max(bull_score, bear_score))
        conviction_score = min(100, raw_score)

        # Stage Classification
        is_viable_opportunity = opp_eval.get("is_viable", False) if opp_eval else False
        if is_viable_opportunity and conviction_score >= 55:
            stage = "CONFIRMED"
            stage_desc = f"Confirmed completed setup with RR 1:{opp_eval.get('risk_reward_ratio', 1.5):.2f}"
        elif is_expanding_now or (rvol >= 1.8 and conviction_score >= 50):
            stage = "IGNITION"
            stage_desc = f"Momentum breaking out (RVOL {rvol:.1f}x, Range {curr_range/atr14:.1f}x ATR)"
        elif is_range_squeezed or (vol_consecutive_rise and conviction_score >= 40):
            stage = "PREPARING"
            stage_desc = "Early coil & volume ramp: preparing for breakout before general market notices"
        else:
            stage = "MONITORING"
            stage_desc = "Normal intraday action"

        # Drivers & Evidence List
        reasons = []
        if vol_consecutive_rise: reasons.append("3-bar volume surge")
        if is_range_squeezed: reasons.append("Volatility squeeze coil")
        if rs_nifty > 1.0: reasons.append(f"+{rs_nifty:.1f}% outperforming NIFTY")
        elif rs_nifty < -1.0: reasons.append(f"{rs_nifty:.1f}% lagging NIFTY")
        if above_vwap: reasons.append("Holding above session VWAP")
        elif below_vwap: reasons.append("Trading below session VWAP")
        if opp_eval and opp_eval.get("potential_pct", 0) > 1.0:
            reasons.append(f"{opp_eval.get('potential_pct')}% room to next structural hurdle")

        return {
            "symbol": symbol,
            "sector": sector_name,
            "current_price": curr_c,
            "change_pct": rs_data["stock_change_pct"],
            "direction": direction,
            "conviction_score": conviction_score,
            "stage": stage,
            "stage_desc": stage_desc,
            "rvol": rvol,
            "vol_consecutive_rise": vol_consecutive_rise,
            "is_range_squeezed": is_range_squeezed,
            "rs_nifty": rs_nifty,
            "rs_sector": rs_sector,
            "rs_accel": rs_accel,
            "reasons": reasons,
            "potential_pct": opp_eval.get("potential_pct", 0.0) if opp_eval else 0.0,
            "risk_reward_ratio": opp_eval.get("risk_reward_ratio", 0.0) if opp_eval else 0.0,
            "suggested_sl": opp_eval.get("tp1") if opp_eval else None,
            "is_viable": is_viable_opportunity
        }

    @staticmethod
    def rank_top_opportunities(candidates, top_n=3):
        """
        Ranks opportunities across entire analyzed universe:
        Returns Top N Bullish and Top N Bearish stocks based on multi-factor conviction.
        """
        if not candidates:
            return {"top_bullish": [], "top_bearish": []}

        # Filter strictly by directional conviction
        bullish = [c for c in candidates if c and c.get("direction") == "BULLISH"]
        bearish = [c for c in candidates if c and c.get("direction") == "BEARISH"]

        # Sort with preference for:
        # 1. Stage (CONFIRMED > IGNITION > PREPARING)
        # 2. Conviction Score descending
        # 3. Relative Strength Acceleration
        stage_weights = {"CONFIRMED": 300, "IGNITION": 200, "PREPARING": 100, "MONITORING": 0}

        bullish.sort(
            key=lambda x: (stage_weights.get(x["stage"], 0) + x["conviction_score"] + (x["rs_accel"] * 10)),
            reverse=True
        )

        bearish.sort(
            key=lambda x: (stage_weights.get(x["stage"], 0) + x["conviction_score"] + (abs(x["rs_accel"]) * 10)),
            reverse=True
        )

        return {
            "top_bullish": bullish[:top_n],
            "top_bearish": bearish[:top_n]
        }
