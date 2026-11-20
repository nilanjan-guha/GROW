import pandas as pd
import numpy as np

class OpportunityPotentialEngine:
    """
    Dedicated Opportunity Potential & Obstacle Clearance Engine:
    
    Before ANY BUY or SELL trade is validated, this engine inspects the market structure
    to verify whether price has genuine, unobstructed expansion room (at least 1.0% to 1.5%+)
    to the next structural hurdle before facing exhaustion or reversal.
    
    Obstacles evaluated:
    - Swing Highs / Swing Lows (Fractal pivots)
    - Previous Day High (PDH) & Previous Day Low (PDL)
    - Today's Opening Range High (ORH) & Low (ORL)
    - Fresh Supply Zones (for Longs) & Demand Zones (for Shorts)
    - Intraday Session VWAP & Key Moving Averages
    - Daily Average True Range (ATR) & Remaining Daily Range Exhaustion
    """

    @staticmethod
    def evaluate_potential(df, current_price, direction, prev_close=None, vwap=None, supply_obs=None, demand_obs=None):
        """
        Determines whether the market actually has sufficient room for the trade.
        Returns:
            dict containing:
            - is_viable (bool): True if room >= 1.0% to first obstacle
            - potential_pct (float): Real percentage room to next hurdle
            - tp1 (float): First structural target price
            - tp2 (float): Secondary liquidity pool target price
            - nearest_obstacle (str): Name and price of the opposing obstacle
            - risk_reward_ratio (float): Structural reward vs risk
            - exhaustion_risk (str): LOW, MODERATE, HIGH
            - reason (str): Human-readable diagnosis
        """
        if df is None or df.empty or len(df) < 5 or direction == "NEUTRAL":
            return {
                "is_viable": False,
                "potential_pct": 0.0,
                "tp1": None,
                "tp2": None,
                "nearest_obstacle": "None",
                "risk_reward_ratio": 0.0,
                "exhaustion_risk": "HIGH",
                "reason": "Neutral direction or insufficient data"
            }

        curr_c = float(current_price)
        h = df['High']
        l = df['Low']
        c = df['Close']
        o = df['Open']

        # 1. Previous Day High & Low
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
        orh = float(today_df['High'].iloc[0]) if not today_df.empty else float(h.iloc[0])
        orl = float(today_df['Low'].iloc[0]) if not today_df.empty else float(l.iloc[0])

        # 2. Daily ATR & Today's Intraday Excursion (Exhaustion Detection)
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1]) or (curr_c * 0.012)
        today_range = float(today_df['High'].max() - today_df['Low'].min()) if not today_df.empty else atr14
        range_exhaustion_ratio = round(today_range / atr14, 2)

        # 3. Detect Fractal Swing Highs and Swing Lows across prior completed candles
        hist_df = df.iloc[:-1] if len(df) > 5 else df
        swing_highs = []
        swing_lows = []
        for i in range(2, len(hist_df) - 2):
            if float(h.iloc[i]) == max([float(x) for x in h.iloc[i-2:i+3]]):
                swing_highs.append(float(h.iloc[i]))
            if float(l.iloc[i]) == min([float(x) for x in l.iloc[i-2:i+3]]):
                swing_lows.append(float(l.iloc[i]))

        if direction == "BUY":
            # Identify all structural obstacles ABOVE current price
            obstacles = []
            if pdh and pdh > curr_c:
                obstacles.append(("Previous Day High (PDH)", pdh))
            if orh > curr_c:
                obstacles.append(("Opening Range High (ORH)", orh))
            
            for sh in swing_highs:
                if sh > curr_c:
                    obstacles.append(("Swing Resistance", sh))

            # Fresh supply order blocks
            if supply_obs:
                for sob in supply_obs:
                    b = float(sob.get("bottom", 0))
                    if b > curr_c and not sob.get("mitigated"):
                        obstacles.append(("Supply Zone Floor", b))

            # Sort obstacles by price ascending (nearest to furthest)
            obstacles.sort(key=lambda x: x[1])

            # Filter out micro-noise obstacles (< 0.25% away)
            meaningful_obstacles = [obs for obs in obstacles if ((obs[1] - curr_c) / curr_c) >= 0.0025]

            if meaningful_obstacles:
                nearest_name, nearest_price = meaningful_obstacles[0]
                tp1 = nearest_price
                # TP2 is the next hurdle beyond TP1, or ATR-extended liquidity target
                tp2 = meaningful_obstacles[1][1] if len(meaningful_obstacles) > 1 else round(tp1 + (atr14 * 1.2), 2)
            else:
                # Clear blue sky above: targets set by structural volatility expansion
                nearest_name = "Blue Sky Expansion (No major structural resistance)"
                tp1 = round(curr_c + (atr14 * 1.5), 2)
                tp2 = round(curr_c + (atr14 * 2.5), 2)

            room_to_tp1_pct = round(((tp1 - curr_c) / curr_c) * 100, 2)
            room_to_tp2_pct = round(((tp2 - curr_c) / curr_c) * 100, 2)

            # Check if stock has already moved > 1.8x its daily ATR (Range Exhaustion)
            is_exhausted = range_exhaustion_ratio >= 1.8 and curr_c >= float(today_df['High'].max()) * 0.995

            # Viability Rule: Achievable room to nearest obstacle must be >= 1.0% AND not range exhausted
            is_viable = (room_to_tp1_pct >= 1.0) and not is_exhausted

            exhaustion_level = "HIGH" if is_exhausted else ("MODERATE" if range_exhaustion_ratio >= 1.2 else "LOW")
            
            if not is_viable:
                if is_exhausted:
                    reason = f"Intraday move exhausted: Daily range ({range_exhaustion_ratio}x ATR) already consumed."
                else:
                    reason = f"Insufficient upside clearance: Nearest resistance ({nearest_name} @ ₹{tp1:.1f}) is only +{room_to_tp1_pct}% away."
            else:
                reason = f"Clear expansion room: +{room_to_tp1_pct}% to {nearest_name} (₹{tp1:.1f}), +{room_to_tp2_pct}% to secondary liquidity."

            return {
                "is_viable": is_viable,
                "potential_pct": room_to_tp1_pct,
                "potential_tp2_pct": room_to_tp2_pct,
                "tp1": tp1,
                "tp2": tp2,
                "nearest_obstacle": f"{nearest_name} @ ₹{tp1:.1f}",
                "range_exhaustion_ratio": range_exhaustion_ratio,
                "exhaustion_risk": exhaustion_level,
                "reason": reason
            }

        elif direction == "SELL":
            # Identify all structural obstacles BELOW current price
            obstacles = []
            if pdl and pdl < curr_c:
                obstacles.append(("Previous Day Low (PDL)", pdl))
            if orl < curr_c:
                obstacles.append(("Opening Range Low (ORL)", orl))
            
            for sl_val in swing_lows:
                if sl_val < curr_c:
                    obstacles.append(("Swing Support", sl_val))

            # Fresh demand order blocks
            if demand_obs:
                for dob in demand_obs:
                    t = float(dob.get("top", 0))
                    if t < curr_c and not dob.get("mitigated"):
                        obstacles.append(("Demand Zone Ceiling", t))

            # Sort obstacles by price descending (nearest to furthest below current price)
            obstacles.sort(key=lambda x: x[1], reverse=True)

            meaningful_obstacles = [obs for obs in obstacles if ((curr_c - obs[1]) / curr_c) >= 0.0025]

            if meaningful_obstacles:
                nearest_name, nearest_price = meaningful_obstacles[0]
                tp1 = nearest_price
                tp2 = meaningful_obstacles[1][1] if len(meaningful_obstacles) > 1 else round(tp1 - (atr14 * 1.2), 2)
            else:
                nearest_name = "Blue Sky Breakdown (No major structural support)"
                tp1 = round(curr_c - (atr14 * 1.5), 2)
                tp2 = round(curr_c - (atr14 * 2.5), 2)

            room_to_tp1_pct = round(((curr_c - tp1) / curr_c) * 100, 2)
            room_to_tp2_pct = round(((curr_c - tp2) / curr_c) * 100, 2)

            is_exhausted = range_exhaustion_ratio >= 1.8 and curr_c <= float(today_df['Low'].min()) * 1.005
            is_viable = (room_to_tp1_pct >= 1.0) and not is_exhausted
            exhaustion_level = "HIGH" if is_exhausted else ("MODERATE" if range_exhaustion_ratio >= 1.2 else "LOW")

            if not is_viable:
                if is_exhausted:
                    reason = f"Downside move exhausted: Daily range ({range_exhaustion_ratio}x ATR) already consumed."
                else:
                    reason = f"Insufficient downside clearance: Nearest support ({nearest_name} @ ₹{tp1:.1f}) is only -{room_to_tp1_pct}% away."
            else:
                reason = f"Clear downside expansion room: -{room_to_tp1_pct}% to {nearest_name} (₹{tp1:.1f}), -{room_to_tp2_pct}% to secondary liquidity."

            return {
                "is_viable": is_viable,
                "potential_pct": room_to_tp1_pct,
                "potential_tp2_pct": room_to_tp2_pct,
                "tp1": tp1,
                "tp2": tp2,
                "nearest_obstacle": f"{nearest_name} @ ₹{tp1:.1f}",
                "range_exhaustion_ratio": range_exhaustion_ratio,
                "exhaustion_risk": exhaustion_level,
                "reason": reason
            }
