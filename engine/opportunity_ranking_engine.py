import pandas as pd
import numpy as np

class OpportunityRankingEngine:
    """
    Dedicated Multi-Factor Pre-Breakout & Opportunity Ranking Engine.
    
    Evaluates candidate F&O stocks across 6 independent quantitative dimensions:
    1. Synchronized Session Relative Strength Acceleration:
       - vs NIFTY index: synchronized strictly over today's intraday session
       - vs Sector index: synchronized with live sector performance
       - Multi-window acceleration: 3-candle, 6-candle, and session RS delta
    2. Time-of-Day Relative Volume & Volume Acceleration:
       - Consecutive rising volume across completed bars (v[t] > v[t-1] > v[t-2])
       - High volume expansion (RVOL >= 1.5x)
    3. Structural Volatility Squeeze & Energy Coil:
       - Narrowing candle ranges (NR4/NR7) / Bollinger bandwidth squeeze
    4. Structural Breakout Proximity:
       - Distance to Opening Range High (ORH) / Opening Range Low (ORL)
       - Distance to Previous Day High (PDH) / Previous Day Low (PDL)
       - Detects consolidation directly beneath key breakout triggers (< 0.5% away)
    5. VWAP Position & Trend Confluence:
       - Holding strictly above session VWAP for Longs / below for Shorts
    6. Strict Opportunity & Risk:Reward Gates:
       - For CONFIRMED: Raw Potential > 1.0%, actual RR >= 1.50, low exhaustion risk
       - PREPARING and IGNITION: early warnings before breakout
    
    Selects Top 3 Bullish & Top 3 Bearish opportunities with complete setup transparency.
    """

    @staticmethod
    def calculate_relative_strength_acceleration(stock_df, nifty_df=None, sector_change_pct=0.0):
        """
        Calculates multi-window relative strength against NIFTY and Sector
        strictly synchronized to the current trading session.
        """
        if stock_df is None or stock_df.empty or len(stock_df) < 2:
            return {"rs_nifty": 0.0, "rs_sector": 0.0, "rs_accel_3": 0.0, "rs_accel_6": 0.0, "stock_change_pct": 0.0}

        # Date isolation for current session synchronization
        s_df = stock_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(s_df.index):
            s_df.index = pd.to_datetime(s_df.index)
        s_dates = sorted(s_df.index.date)
        today_date = s_dates[-1] if s_dates else None

        today_stock = s_df[s_df.index.date == today_date] if today_date else s_df
        if today_stock.empty or len(today_stock) < 2:
            today_stock = s_df

        sc = today_stock['Close']
        stock_open = float(today_stock['Open'].iloc[0])
        stock_change_pct = float(((sc.iloc[-1] - stock_open) / stock_open) * 100) if stock_open > 0 else 0.0

        # Synchronize NIFTY strictly to today's session
        nifty_change_pct = 0.0
        if nifty_df is not None and not nifty_df.empty:
            n_df = nifty_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(n_df.index):
                n_df.index = pd.to_datetime(n_df.index)
            today_nifty = n_df[n_df.index.date == today_date] if today_date else n_df
            if not today_nifty.empty and len(today_nifty) >= 2:
                n_open = float(today_nifty['Open'].iloc[0])
                n_close = float(today_nifty['Close'].iloc[-1])
                nifty_change_pct = float(((n_close - n_open) / n_open) * 100) if n_open > 0 else 0.0

        rs_nifty = round(stock_change_pct - nifty_change_pct, 2)
        rs_sector = round(stock_change_pct - sector_change_pct, 2)

        # Multi-window RS acceleration: 3-candle and 6-candle delta relative to NIFTY
        rs_accel_3 = 0.0
        rs_accel_6 = 0.0
        
        # Calculate stock return over last 3 and 6 candles
        stock_ret_3 = 0.0
        if len(sc) >= 4:
            c_3 = float(sc.iloc[-3])
            stock_ret_3 = float(((sc.iloc[-1] - c_3) / c_3) * 100) if c_3 > 0 else 0.0
            
        stock_ret_6 = 0.0
        if len(sc) >= 7:
            c_6 = float(sc.iloc[-6])
            stock_ret_6 = float(((sc.iloc[-1] - c_6) / c_6) * 100) if c_6 > 0 else 0.0

        # Calculate NIFTY return over matching 3 and 6 candles if available
        nifty_ret_3 = 0.0
        nifty_ret_6 = 0.0
        if nifty_df is not None and not nifty_df.empty:
            n_df = nifty_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(n_df.index):
                n_df.index = pd.to_datetime(n_df.index)
            today_nifty = n_df[n_df.index.date == today_date] if today_date else n_df
            if not today_nifty.empty:
                nc = today_nifty['Close']
                if len(nc) >= 4:
                    nc_3 = float(nc.iloc[-3])
                    nifty_ret_3 = float(((nc.iloc[-1] - nc_3) / nc_3) * 100) if nc_3 > 0 else 0.0
                if len(nc) >= 7:
                    nc_6 = float(nc.iloc[-6])
                    nifty_ret_6 = float(((nc.iloc[-1] - nc_6) / nc_6) * 100) if nc_6 > 0 else 0.0

        rs_accel_3 = round(stock_ret_3 - nifty_ret_3, 2)
        rs_accel_6 = round(stock_ret_6 - nifty_ret_6, 2)

        return {
            "stock_change_pct": round(stock_change_pct, 2),
            "rs_nifty": rs_nifty,
            "rs_sector": rs_sector,
            "rs_accel_3": rs_accel_3,
            "rs_accel_6": rs_accel_6,
            "rs_accel": rs_accel_3
        }

    @staticmethod
    def evaluate_stock_opportunity(stock_df, symbol, sector_name="GENERAL", sector_change=0.0, nifty_df=None, vwap=None, opp_eval=None):
        """
        Computes comprehensive quantitative opportunity score, breakout proximity, and prediction stage.
        """
        if stock_df is None or stock_df.empty or len(stock_df) < 5:
            return None

        # Ensure single-level columns
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.get_level_values(0)

        df_copy = stock_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
            df_copy.index = pd.to_datetime(df_copy.index)
        
        dates = sorted(df_copy.index.date)
        today_date = dates[-1] if dates else None
        prev_date = dates[-2] if len(dates) >= 2 else None

        today_df = df_copy[df_copy.index.date == today_date] if today_date else df_copy
        if today_df.empty or len(today_df) < 2:
            today_df = df_copy

        c = df_copy['Close']
        o = df_copy['Open']
        h = df_copy['High']
        l = df_copy['Low']
        v = df_copy['Volume']

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
        if pd.isna(atr14) or atr14 <= 0:
            atr14 = curr_c * 0.01

        # 2. Time-of-day & Consecutive Volume Acceleration
        vol_mean_15 = float(v.iloc[-16:-1].mean()) if len(v) >= 16 else curr_v
        rvol = round(curr_v / (vol_mean_15 or 1), 2)
        
        # 3-bar consecutive rising volume
        vol_consecutive_rise = False
        if len(v) >= 3:
            vol_consecutive_rise = bool(v.iloc[-1] > v.iloc[-2] > v.iloc[-3])

        # 3. Volatility Squeeze / Narrow Range (NR4/NR7)
        recent_ranges = [(float(h.iloc[k]) - float(l.iloc[k])) for k in range(-4, 0)] if len(h) >= 4 else [curr_h - curr_l]
        is_range_squeezed = all(r < (atr14 * 0.85) for r in recent_ranges[:-1])
        curr_range = curr_h - curr_l
        is_expanding_now = curr_range >= (atr14 * 1.5)

        # 4. Opening Range & Key Structural Breakout Levels
        orh = float(today_df['High'].iloc[:3].max()) if len(today_df) >= 3 else float(today_df['High'].iloc[0])
        orl = float(today_df['Low'].iloc[:3].min()) if len(today_df) >= 3 else float(today_df['Low'].iloc[0])
        
        # Previous Day High / Low
        pdh = None
        pdl = None
        if prev_date is not None:
            prev_day_df = df_copy[df_copy.index.date == prev_date]
            if not prev_day_df.empty:
                pdh = float(prev_day_df['High'].max())
                pdl = float(prev_day_df['Low'].min())

        # Breakout Proximity (< 0.5% away from ORH/ORL or PDH/PDL)
        dist_to_orh_pct = ((orh - curr_c) / curr_c * 100) if orh else 999.0
        dist_to_orl_pct = ((curr_c - orl) / curr_c * 100) if orl else 999.0
        is_coiling_near_orh = (0.0 <= dist_to_orh_pct <= 0.50)
        is_coiling_near_orl = (0.0 <= dist_to_orl_pct <= 0.50)
        is_broken_above_orh = (curr_c > orh)
        is_broken_below_orl = (curr_c < orl)

        # 5. Synchronized Relative Strength Acceleration
        rs_data = OpportunityRankingEngine.calculate_relative_strength_acceleration(stock_df, nifty_df, sector_change)
        rs_nifty = rs_data["rs_nifty"]
        rs_sector = rs_data["rs_sector"]
        rs_accel = rs_data["rs_accel"]

        # 6. VWAP Relationship
        above_vwap = (curr_c > vwap) if vwap else False
        below_vwap = (curr_c < vwap) if vwap else False

        # Directional Tendency & Scoring
        bull_score = 0
        bear_score = 0

        # Bullish factors
        if rs_nifty > 0: bull_score += min(20, int(rs_nifty * 5))
        if rs_sector > 0: bull_score += min(20, int(rs_sector * 5))
        if rs_accel > 0: bull_score += 15
        if above_vwap: bull_score += 20
        if rvol >= 1.5: bull_score += 15
        if vol_consecutive_rise and curr_c > curr_o: bull_score += 15
        if is_range_squeezed and curr_c > curr_o: bull_score += 15
        if is_coiling_near_orh: bull_score += 20
        if is_broken_above_orh: bull_score += 20
        if is_expanding_now and curr_c > curr_o: bull_score += 15

        # Bearish factors
        if rs_nifty < 0: bear_score += min(20, int(abs(rs_nifty) * 5))
        if rs_sector < 0: bear_score += min(20, int(abs(rs_sector) * 5))
        if rs_accel < 0: bear_score += 15
        if below_vwap: bear_score += 20
        if rvol >= 1.5: bear_score += 15
        if vol_consecutive_rise and curr_c < curr_o: bear_score += 15
        if is_range_squeezed and curr_c < curr_o: bear_score += 15
        if is_coiling_near_orl: bear_score += 20
        if is_broken_below_orl: bear_score += 20
        if is_expanding_now and curr_c < curr_o: bear_score += 15

        direction = "BULLISH" if (bull_score >= bear_score and bull_score >= 40) else ("BEARISH" if bear_score >= 40 else "NEUTRAL")
        raw_score = bull_score if direction == "BULLISH" else (bear_score if direction == "BEARISH" else max(bull_score, bear_score))
        conviction_score = min(100, raw_score)

        # Correct Stop-Loss & Targets from Opportunity Potential Engine
        sl = opp_eval.get("stop_loss") or opp_eval.get("suggested_sl") if opp_eval else None
        tp1 = opp_eval.get("tp1") or opp_eval.get("suggested_tp1") if opp_eval else None
        tp2 = opp_eval.get("tp2") or opp_eval.get("suggested_tp2") if opp_eval else None
        potential_pct = float(opp_eval.get("potential_pct", 0.0)) if opp_eval else 0.0
        rr_ratio = float(opp_eval.get("risk_reward_ratio", 0.0)) if opp_eval else 0.0
        exhaustion_risk = opp_eval.get("exhaustion_risk", "LOW") if opp_eval else "LOW"

        # Quality Gates for CONFIRMED:
        # Strict Potential > 1.0%, actual RR >= 1.50, exhaustion != HIGH, is_viable == True
        is_viable_opportunity = (
            opp_eval.get("is_viable", False) and 
            potential_pct > 1.0 and 
            rr_ratio >= 1.50 and 
            exhaustion_risk != "HIGH"
        ) if opp_eval else False

        # Stage Classification
        if is_viable_opportunity and conviction_score >= 55:
            stage = "CONFIRMED"
            stage_desc = f"Verified Completed Trade: RR 1:{rr_ratio:.2f}, +{potential_pct:.1f}% Room"
        elif is_expanding_now or (rvol >= 1.75 and conviction_score >= 50) or is_broken_above_orh or is_broken_below_orl:
            stage = "IGNITION"
            stage_desc = f"Momentum Breaking Out: RVOL {rvol:.1f}x, Range {curr_range/atr14:.1f}x ATR"
        elif is_coiling_near_orh or is_coiling_near_orl or is_range_squeezed or (vol_consecutive_rise and conviction_score >= 35):
            stage = "PREPARING"
            stage_desc = "Early Energy Coil: Volume accumulating below breakout level before the move"
        else:
            stage = "MONITORING"
            stage_desc = "Normal intraday action"

        # Setup Evidence & Drivers
        reasons = []
        if vol_consecutive_rise: reasons.append("3-bar volume surge")
        if is_coiling_near_orh: reasons.append(f"Coiling {dist_to_orh_pct:.2f}% under ORH")
        elif is_broken_above_orh: reasons.append("Broken above 15m ORH")
        if is_coiling_near_orl: reasons.append(f"Coiling {dist_to_orl_pct:.2f}% above ORL")
        elif is_broken_below_orl: reasons.append("Broken below 15m ORL")
        if is_range_squeezed: reasons.append("Volatility squeeze coil")
        if rs_nifty > 1.0: reasons.append(f"+{rs_nifty:.1f}% vs NIFTY")
        elif rs_nifty < -1.0: reasons.append(f"{rs_nifty:.1f}% vs NIFTY")
        if above_vwap: reasons.append("Above VWAP")
        elif below_vwap: reasons.append("Below VWAP")
        if potential_pct > 1.0: reasons.append(f"+{potential_pct}% room")

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
            "orh": orh,
            "orl": orl,
            "reasons": reasons,
            "potential_pct": potential_pct,
            "risk_reward_ratio": rr_ratio,
            "suggested_sl": sl,
            "suggested_tp1": tp1,
            "suggested_tp2": tp2,
            "is_viable": is_viable_opportunity
        }

    @staticmethod
    def rank_top_opportunities(candidates, top_n=3):
        """
        Ranks opportunities across entire analyzed universe:
        Returns Top N Bullish and Top N Bearish stocks enforcing quality gates.
        """
        if not candidates:
            return {"top_bullish": [], "top_bearish": []}

        # Filter strictly by directional conviction
        bullish = [c for c in candidates if c and c.get("direction") == "BULLISH"]
        bearish = [c for c in candidates if c and c.get("direction") == "BEARISH"]

        # Sort priority:
        # 1. Stage (CONFIRMED > IGNITION > PREPARING > MONITORING)
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
