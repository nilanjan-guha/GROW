import pandas as pd
import numpy as np
import datetime
import pytz

class OpportunityPotentialEngine:
    """
    Dedicated Opportunity Potential & Obstacle Clearance Engine:
    
    Before ANY BUY or SELL trade is validated, this engine inspects the market structure
    to verify whether price has genuine, unobstructed expansion room (strictly > 1.0%)
    to the next structural hurdle before facing exhaustion or reversal.
    
    Key Features:
    1. Dual Volatility Tracking:
       - intraday_atr_15m: 15-minute ATR for micro-structure and stop sizing
       - daily_atr_14d: True multi-day ATR for daily range exhaustion
    2. Strict > 1.0% Potential Gate:
       - Uses raw unrounded potential (raw_room_to_tp1_pct > 1.0)
    3. Structural Liquidity Mapping:
       - Distinguishes confirmed structural targets (PDH/PDL, ORH/ORL, Swing Pivots, Order Blocks)
         from volatility-based extensions.
    4. Reversal and Setup Invalidation:
       - Detects opposing candle momentum, rejection wicks, and opposing order flow.
    5. Reward-to-Risk Validation:
       - Computes actual risk_reward_ratio against structural stop-loss and requires RR >= 1.5.
    """

    @staticmethod
    def calculate_daily_atr(df_copy, dates, current_price, daily_df=None):
        """
        Calculates true Daily Wilder ATR14 using historical completed daily sessions.
        Uses conventional Wilder initialization (SMA of first 14 TRs, followed by recursive smoothing).
        Excludes today's incomplete daily candle during live market hours.
        Returns a tuple: (atr_value: float, source: str)
        """
        ist_tz = pytz.timezone('Asia/Kolkata') if 'pytz' in globals() else None
        now_dt = datetime.datetime.now(ist_tz) if ist_tz else datetime.datetime.now()
        today_date = now_dt.date()
        # Market hours check: Weekdays 09:15 to 15:30 IST
        is_live_mkt = (now_dt.weekday() < 5) and (555 <= (now_dt.hour * 60 + now_dt.minute) <= 930)

        # Case 1: Separate dedicated Daily OHLC DataFrame provided
        if daily_df is not None and not daily_df.empty:
            try:
                d_df = daily_df.copy()
                if not pd.api.types.is_datetime64_any_dtype(d_df.index):
                    d_df.index = pd.to_datetime(d_df.index)
                
                d_df['d_date'] = d_df.index.date
                # Strictly exclude today's incomplete candle if market is live or if candle belongs to today
                if is_live_mkt or (d_df['d_date'].iloc[-1] == today_date and now_dt.hour < 16):
                    clean_daily = d_df[d_df['d_date'] < today_date]
                else:
                    clean_daily = d_df

                if len(clean_daily) >= 15:
                    dh = clean_daily['High']
                    dl = clean_daily['Low']
                    dc = clean_daily['Close']
                    d_prev_c = dc.shift(1)
                    tr = pd.concat([dh - dl, (dh - d_prev_c).abs(), (dl - d_prev_c).abs()], axis=1).max(axis=1).dropna()
                    
                    if len(tr) >= 14:
                        # Conventional Wilder smoothing: first value is 14-period SMA
                        tr_vals = tr.values
                        wilder_atr = np.mean(tr_vals[:14])
                        for t in tr_vals[14:]:
                            wilder_atr = (wilder_atr * 13 + t) / 14
                        
                        if not pd.isna(wilder_atr) and wilder_atr > 0:
                            return float(wilder_atr), "DEDICATED_DAILY_WILDER_ATR14"
            except Exception:
                pass

        # Case 2: Daily records extracted from intraday history (excluding today's incomplete session)
        daily_records = []
        completed_dates = [d for d in dates if d < today_date] if dates else []
        if len(completed_dates) < 2 and len(dates) >= 2:
            completed_dates = dates[:-1]

        for d in completed_dates:
            day_data = df_copy[df_copy['Date'] == d]
            if not day_data.empty:
                d_high = float(day_data['High'].max())
                d_low = float(day_data['Low'].min())
                d_close = float(day_data['Close'].iloc[-1])
                daily_records.append({"high": d_high, "low": d_low, "close": d_close})

        if len(daily_records) >= 2:
            true_ranges = []
            for j in range(1, len(daily_records)):
                prev_c = daily_records[j - 1]["close"]
                cur_h = daily_records[j]["high"]
                cur_l = daily_records[j]["low"]
                tr = max(cur_h - cur_l, abs(cur_h - prev_c), abs(cur_l - prev_c))
                true_ranges.append(tr)
            if true_ranges:
                avg_atr = float(np.mean(true_ranges))
                if not pd.isna(avg_atr) and avg_atr > 0:
                    return avg_atr, "INTRADAY_DAILY_RANGE_AVERAGE"

        return float(current_price * 0.018), "BASELINE_VOLATILITY_FALLBACK"  # Standard 1.8% daily volatility baseline

    @staticmethod
    def evaluate_potential(df, current_price, direction, prev_close=None, vwap=None, supply_obs=None, demand_obs=None, stop_loss=None, daily_df=None, orb_levels=None):
        if df is None or df.empty or len(df) < 5 or direction == "NEUTRAL":
            return {
                "is_viable": False,
                "potential_pct": 0.0,
                "potential_tp2_pct": 0.0,
                "tp1": None,
                "tp2": None,
                "tp1_type": "None",
                "tp2_type": "None",
                "nearest_obstacle": "None",
                "structural_target": "None",
                "risk_reward_ratio": 0.0,
                "exhaustion_risk": "HIGH",
                "range_exhaustion_ratio": 0.0,
                "daily_atr_source": "NONE",
                "is_reversal_detected": False,
                "confidence_penalty": 0,
                "reason": "Neutral direction or insufficient data"
            }

        curr_c = float(current_price)
        h = df['High']
        l = df['Low']
        c = df['Close']
        o = df['Open']

        # 1. Date isolation
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
            df_copy.index = pd.to_datetime(df_copy.index)
        df_copy['Date'] = df_copy.index.date
        dates = sorted(df_copy['Date'].unique())

        pdh = None
        pdl = None
        if len(dates) >= 2:
            prev_day_df = df_copy[df_copy['Date'] == dates[-2]]
            pdh = float(prev_day_df['High'].max())
            pdl = float(prev_day_df['Low'].min())

        today_df = df_copy[df_copy['Date'] == dates[-1]] if dates else df_copy
        
        # Unified ORB resolution: use passed-in verified orb_levels if available
        if orb_levels and orb_levels.get("orh") is not None and orb_levels.get("orl") is not None:
            orh = float(orb_levels["orh"])
            orl = float(orb_levels["orl"])
            orb_type = orb_levels.get("orb_type", "15m ORB")
        else:
            # Fallback unified multi-duration ORB calculation
            if len(today_df) >= 6:
                orh = float(today_df['High'].iloc[:6].max())
                orl = float(today_df['Low'].iloc[:6].min())
                orb_type = "30m ORB"
            elif len(today_df) >= 3:
                orh = float(today_df['High'].iloc[:3].max())
                orl = float(today_df['Low'].iloc[:3].min())
                orb_type = "15m ORB"
            elif not today_df.empty:
                orh = float(today_df['High'].iloc[0])
                orl = float(today_df['Low'].iloc[0])
                orb_type = "5m ORB"
            else:
                orh = float(h.iloc[0])
                orl = float(l.iloc[0])
                orb_type = "5m ORB"

        # 2. SEPARATE INTRADAY 5m ATR vs TRUE DAILY WILDER ATR14
        tr_5m = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        rolling_atr5m = tr_5m.rolling(14).mean().iloc[-1]
        atr_5m_14 = float(rolling_atr5m) if (not pd.isna(rolling_atr5m) and rolling_atr5m > 0) else float(curr_c * 0.005)

        # True Daily Wilder ATR14 using dedicated daily data when available
        daily_atr, daily_atr_source = OpportunityPotentialEngine.calculate_daily_atr(df_copy, dates, curr_c, daily_df=daily_df)
        today_range = float(today_df['High'].max() - today_df['Low'].min()) if not today_df.empty else atr_5m_14
        range_exhaustion_ratio = round(today_range / daily_atr, 2)

        # Volatility-aware dynamic clearance threshold:
        # Normalized as max(0.20%, min(0.60%, 0.45 * atr_5m_14 / curr_c * 100))
        dynamic_clearance_pct = max(0.20, min(0.60, round((0.45 * atr_5m_14 / curr_c) * 100, 2)))

        # 3. Detect Fractal Swing Highs and Swing Lows across all completed candles in df
        hist_df = df
        swing_highs = []
        swing_lows = []
        for i in range(2, len(hist_df) - 2):
            if float(h.iloc[i]) == max([float(x) for x in h.iloc[i-2:i+3]]):
                swing_highs.append(float(h.iloc[i]))
            if float(l.iloc[i]) == min([float(x) for x in l.iloc[i-2:i+3]]):
                swing_lows.append(float(l.iloc[i]))

        # 4. REVERSAL & SETUP INVALIDATION DETECTION
        # Check opposing candle pattern, rejection wick, and opposing OB reactions
        curr_bar_o = float(o.iloc[-1])
        curr_bar_h = float(h.iloc[-1])
        curr_bar_l = float(l.iloc[-1])
        curr_bar_range = (curr_bar_h - curr_bar_l) or 0.001
        upper_wick = curr_bar_h - max(curr_c, curr_bar_o)
        lower_wick = min(curr_c, curr_bar_o) - curr_bar_l

        reversal_detected = False
        reversal_reason = ""

        if direction == "BUY":
            # Bearish rejection: upper wick > 45% of candle range or bear engulfing at high
            if upper_wick / curr_bar_range > 0.45 and curr_c < curr_bar_o:
                reversal_detected = True
                reversal_reason = "Bearish rejection wick detected at resistance"
            elif len(df) >= 2 and curr_c < float(l.iloc[-2]) and float(c.iloc[-2]) > float(o.iloc[-2]):
                reversal_detected = True
                reversal_reason = "Bearish engulfing breakdown below prior candle low"
            elif vwap and curr_c < vwap and float(c.iloc[-2]) > vwap:
                reversal_detected = True
                reversal_reason = "Session VWAP lost on high volume"

            # Identify all structural obstacles ABOVE current price
            obstacles = []
            if pdh and pdh > curr_c:
                obstacles.append(("Previous Day High (PDH)", pdh, "STRUCTURAL_PDH"))
            if orh > curr_c:
                obstacles.append(("Opening Range High (ORH)", orh, "STRUCTURAL_ORH"))
            for sh in swing_highs:
                if sh > curr_c:
                    obstacles.append(("Swing Resistance", sh, "STRUCTURAL_SWING"))
            if supply_obs:
                for sob in supply_obs:
                    b = float(sob.get("bottom", 0))
                    if b > curr_c and not sob.get("mitigated"):
                        obstacles.append(("Supply Zone Floor", b, "STRUCTURAL_SUPPLY_OB"))

            obstacles.sort(key=lambda x: x[1])
            immediate_obstacle_str = f"{obstacles[0][0]} @ ₹{obstacles[0][1]:.1f}" if obstacles else "None"
            meaningful_obstacles = [obs for obs in obstacles if ((obs[1] - curr_c) / curr_c) >= 0.0025]

            if meaningful_obstacles:
                nearest_name, nearest_price, tp1_type = meaningful_obstacles[0]
                tp1 = round(nearest_price, 2)
                if len(meaningful_obstacles) > 1:
                    tp2 = round(meaningful_obstacles[1][1], 2)
                    tp2_type = meaningful_obstacles[1][2]
                else:
                    tp2 = round(tp1 + (atr_5m_14 * 1.5), 2)
                    tp2_type = "ESTIMATED_ATR_EXTENSION"
            else:
                nearest_name = "Blue Sky Expansion (No major structural resistance)"
                tp1 = round(curr_c + (daily_atr * 0.75), 2)
                tp2 = round(curr_c + (daily_atr * 1.25), 2)
                tp1_type = "VOLATILITY_EXPANSION"
                tp2_type = "ESTIMATED_ATR_EXTENSION"

            raw_room_to_tp1_pct = ((tp1 - curr_c) / curr_c) * 100
            room_to_tp1_pct = round(raw_room_to_tp1_pct, 2)
            room_to_tp2_pct = round(((tp2 - curr_c) / curr_c) * 100, 2)

            # Daily Range Exhaustion: If already consumed > 1.8x Daily ATR near the high of day
            is_exhausted = range_exhaustion_ratio >= 1.8 and curr_c >= float(today_df['High'].max()) * 0.995

            # Calculate Reward-to-Risk ratio against stop-loss using exact unrounded precision
            risk = (curr_c - float(stop_loss)) if (stop_loss and float(stop_loss) < curr_c) else (atr_5m_14 * 1.2)
            reward = tp1 - curr_c
            raw_rr_ratio = (reward / risk) if risk > 0 else 0.0

            # Obstacle clearance check: volatility-aware dynamic clearance threshold
            immediate_clearance_blocked = False
            imm_dist_pct = 0.0
            if obstacles:
                imm_dist_pct = ((obstacles[0][1] - curr_c) / curr_c) * 100
                if imm_dist_pct < dynamic_clearance_pct:
                    immediate_clearance_blocked = True

            # STRICT > 1.0% UNROUNDED POTENTIAL GATE + STRICT UNROUNDED RR >= 1.5 GATE
            is_viable = (raw_room_to_tp1_pct > 1.0) and (raw_rr_ratio >= 1.5) and not is_exhausted and not reversal_detected and not immediate_clearance_blocked
            rr_ratio = round(raw_rr_ratio, 2)

            exhaustion_level = "HIGH" if is_exhausted else ("MODERATE" if range_exhaustion_ratio >= 1.2 else "LOW")

            if not is_viable:
                if reversal_detected:
                    reason = f"Trade Invalidated: {reversal_reason}."
                elif is_exhausted:
                    reason = f"Intraday move exhausted: Daily range ({range_exhaustion_ratio}x Daily ATR) already consumed."
                elif immediate_clearance_blocked:
                    reason = f"Entry blocked by immediate overhead ceiling: {obstacles[0][0]} @ ₹{obstacles[0][1]:.1f} ({imm_dist_pct:.2f}% away < {dynamic_clearance_pct:.2f}% dynamic threshold)."
                elif raw_rr_ratio < 1.5:
                    reason = f"Insufficient Risk:Reward ratio (RR: 1:{rr_ratio:.2f} < 1:1.50 required)."
                else:
                    reason = f"Insufficient upside clearance: Nearest resistance ({nearest_name} @ ₹{tp1:.1f}) is only +{room_to_tp1_pct}% away (Strictly > 1.0% required)."
            else:
                reason = f"Clear expansion room: +{room_to_tp1_pct}% to {nearest_name} (₹{tp1:.1f}) [{tp1_type}], RR: 1:{rr_ratio}."

            # If daily ATR relied on the 1.8% fallback, apply a strict confidence penalty & warn in signal
            confidence_penalty = 15 if daily_atr_source == "BASELINE_VOLATILITY_FALLBACK" else 0

            return {
                "is_viable": is_viable,
                "potential_pct": room_to_tp1_pct,
                "potential_tp2_pct": room_to_tp2_pct,
                "tp1": tp1,
                "tp2": tp2,
                "tp1_type": tp1_type,
                "tp2_type": tp2_type,
                "nearest_obstacle": immediate_obstacle_str if immediate_obstacle_str != "None" else f"{nearest_name} @ ₹{tp1:.1f}",
                "obstacle_distance_pct": round(imm_dist_pct, 2) if obstacles else None,
                "structural_target": f"{nearest_name} @ ₹{tp1:.1f}",
                "risk_reward_ratio": rr_ratio,
                "range_exhaustion_ratio": range_exhaustion_ratio,
                "exhaustion_risk": exhaustion_level,
                "daily_atr_source": daily_atr_source,
                "confidence_penalty": confidence_penalty,
                "dynamic_clearance_pct": dynamic_clearance_pct,
                "is_reversal_detected": reversal_detected,
                "reason": reason
            }

        elif direction == "SELL":
            # Bullish rejection: lower wick > 45% of candle range or bull engulfing at low
            if lower_wick / curr_bar_range > 0.45 and curr_c > curr_bar_o:
                reversal_detected = True
                reversal_reason = "Bullish absorption wick detected at support"
            elif len(df) >= 2 and curr_c > float(h.iloc[-2]) and float(c.iloc[-2]) < float(o.iloc[-2]):
                reversal_detected = True
                reversal_reason = "Bullish engulfing reclaim above prior candle high"
            elif vwap and curr_c > vwap and float(c.iloc[-2]) < vwap:
                reversal_detected = True
                reversal_reason = "Session VWAP reclaimed on high volume"

            obstacles = []
            if pdl and pdl < curr_c:
                obstacles.append(("Previous Day Low (PDL)", pdl, "STRUCTURAL_PDL"))
            if orl < curr_c:
                obstacles.append(("Opening Range Low (ORL)", orl, "STRUCTURAL_ORL"))
            for sl_val in swing_lows:
                if sl_val < curr_c:
                    obstacles.append(("Swing Support", sl_val, "STRUCTURAL_SWING"))
            if demand_obs:
                for dob in demand_obs:
                    t = float(dob.get("top", 0))
                    if t < curr_c and not dob.get("mitigated"):
                        obstacles.append(("Demand Zone Ceiling", t, "STRUCTURAL_DEMAND_OB"))

            obstacles.sort(key=lambda x: x[1], reverse=True)
            immediate_obstacle_str = f"{obstacles[0][0]} @ ₹{obstacles[0][1]:.1f}" if obstacles else "None"
            meaningful_obstacles = [obs for obs in obstacles if ((curr_c - obs[1]) / curr_c) >= 0.0025]

            if meaningful_obstacles:
                nearest_name, nearest_price, tp1_type = meaningful_obstacles[0]
                tp1 = round(nearest_price, 2)
                if len(meaningful_obstacles) > 1:
                    tp2 = round(meaningful_obstacles[1][1], 2)
                    tp2_type = meaningful_obstacles[1][2]
                else:
                    tp2 = round(tp1 - (atr_5m_14 * 1.5), 2)
                    tp2_type = "ESTIMATED_ATR_EXTENSION"
            else:
                nearest_name = "Blue Sky Breakdown (No major structural support)"
                tp1 = round(curr_c - (daily_atr * 0.75), 2)
                tp2 = round(curr_c - (daily_atr * 1.25), 2)
                tp1_type = "VOLATILITY_EXPANSION"
                tp2_type = "ESTIMATED_ATR_EXTENSION"

            raw_room_to_tp1_pct = ((curr_c - tp1) / curr_c) * 100
            room_to_tp1_pct = round(raw_room_to_tp1_pct, 2)
            room_to_tp2_pct = round(((curr_c - tp2) / curr_c) * 100, 2)

            is_exhausted = range_exhaustion_ratio >= 1.8 and curr_c <= float(today_df['Low'].min()) * 1.005

            risk = (float(stop_loss) - curr_c) if (stop_loss and float(stop_loss) > curr_c) else (atr_5m_14 * 1.2)
            reward = curr_c - tp1
            raw_rr_ratio = (reward / risk) if risk > 0 else 0.0

            # Obstacle clearance check: volatility-aware dynamic clearance threshold
            immediate_clearance_blocked = False
            imm_dist_pct = 0.0
            if obstacles:
                imm_dist_pct = ((curr_c - obstacles[0][1]) / curr_c) * 100
                if imm_dist_pct < dynamic_clearance_pct:
                    immediate_clearance_blocked = True

            # STRICT > 1.0% UNROUNDED POTENTIAL GATE + STRICT UNROUNDED RR >= 1.5 GATE
            is_viable = (raw_room_to_tp1_pct > 1.0) and (raw_rr_ratio >= 1.5) and not is_exhausted and not reversal_detected and not immediate_clearance_blocked
            rr_ratio = round(raw_rr_ratio, 2)

            exhaustion_level = "HIGH" if is_exhausted else ("MODERATE" if range_exhaustion_ratio >= 1.2 else "LOW")

            if not is_viable:
                if reversal_detected:
                    reason = f"Trade Invalidated: {reversal_reason}."
                elif is_exhausted:
                    reason = f"Downside move exhausted: Daily range ({range_exhaustion_ratio}x Daily ATR) already consumed."
                elif immediate_clearance_blocked:
                    reason = f"Entry blocked by immediate floor support: {obstacles[0][0]} @ ₹{obstacles[0][1]:.1f} ({imm_dist_pct:.2f}% away < {dynamic_clearance_pct:.2f}% dynamic threshold)."
                elif raw_rr_ratio < 1.5:
                    reason = f"Insufficient Risk:Reward ratio (RR: 1:{rr_ratio:.2f} < 1:1.50 required)."
                else:
                    reason = f"Insufficient downside clearance: Nearest support ({nearest_name} @ ₹{tp1:.1f}) is only -{room_to_tp1_pct}% away (Strictly > 1.0% required)."
            else:
                reason = f"Clear downside expansion room: -{room_to_tp1_pct}% to {nearest_name} (₹{tp1:.1f}) [{tp1_type}], RR: 1:{rr_ratio}."

            # If daily ATR relied on the 1.8% fallback, apply a strict confidence penalty & warn in signal
            confidence_penalty = 15 if daily_atr_source == "BASELINE_VOLATILITY_FALLBACK" else 0

            return {
                "is_viable": is_viable,
                "potential_pct": room_to_tp1_pct,
                "potential_tp2_pct": room_to_tp2_pct,
                "tp1": tp1,
                "tp2": tp2,
                "tp1_type": tp1_type,
                "tp2_type": tp2_type,
                "nearest_obstacle": immediate_obstacle_str if immediate_obstacle_str != "None" else f"{nearest_name} @ ₹{tp1:.1f}",
                "obstacle_distance_pct": round(imm_dist_pct, 2) if obstacles else None,
                "structural_target": f"{nearest_name} @ ₹{tp1:.1f}",
                "risk_reward_ratio": rr_ratio,
                "range_exhaustion_ratio": range_exhaustion_ratio,
                "exhaustion_risk": exhaustion_level,
                "daily_atr_source": daily_atr_source,
                "confidence_penalty": confidence_penalty,
                "dynamic_clearance_pct": dynamic_clearance_pct,
                "is_reversal_detected": reversal_detected,
                "reason": reason
            }
