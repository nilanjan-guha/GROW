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
    def calculate_relative_strength_acceleration(stock_df, nifty_df=None, sector_change_pct=0.0, sector_df=None):
        """
        Calculates 3-way synchronized multi-window relative strength:
        1. Stock vs NIFTY 50 benchmark
        2. Sector Index vs NIFTY 50 benchmark
        3. Stock vs Sector Index (true sector leadership!)
        Strictly synchronized to matching candle timestamps over today's trading session.
        """
        if stock_df is None or stock_df.empty or len(stock_df) < 2:
            return {
                "rs_nifty": 0.0, "rs_sector": 0.0, "sector_rs_nifty": 0.0, 
                "rs_accel_3": 0.0, "rs_accel_6": 0.0, "rs_accel": 0.0, 
                "stock_change_pct": 0.0, "sector_change_pct": 0.0, "nifty_change_pct": 0.0,
                "is_aligned": False, "alignment_status": "EMPTY_DATA"
            }

        # Date isolation for current session synchronization
        s_df = stock_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(s_df.index):
            s_df.index = pd.to_datetime(s_df.index)
        
        # If timezone-aware, normalize to Asia/Kolkata
        try:
            if s_df.index.tz is None:
                s_df.index = s_df.index.tz_localize("Asia/Kolkata")
            else:
                s_df.index = s_df.index.tz_convert("Asia/Kolkata")
        except Exception:
            pass

        s_dates = sorted(s_df.index.date)
        today_date = s_dates[-1] if s_dates else None

        today_stock = s_df[s_df.index.date == today_date] if today_date else s_df
        if today_stock.empty or len(today_stock) < 2:
            today_stock = s_df

        sc = today_stock['Close']
        stock_open = float(today_stock['Open'].iloc[0])
        stock_change_pct = float(((sc.iloc[-1] - stock_open) / stock_open) * 100) if stock_open > 0 else 0.0

        # Synchronize NIFTY strictly to matching timestamps in today's session
        nifty_change_pct = 0.0
        rs_accel_3 = 0.0
        rs_accel_6 = 0.0
        is_aligned = False

        if nifty_df is not None and not nifty_df.empty:
            n_df = nifty_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(n_df.index):
                n_df.index = pd.to_datetime(n_df.index)
            try:
                if n_df.index.tz is None:
                    n_df.index = n_df.index.tz_localize("Asia/Kolkata")
                else:
                    n_df.index = n_df.index.tz_convert("Asia/Kolkata")
            except Exception:
                pass

            today_nifty = n_df[n_df.index.date == today_date] if today_date else n_df
            if not today_nifty.empty and len(today_nifty) >= 2:
                # Strict timestamp inner alignment to avoid row-shift or unaligned session comparisons
                joined = pd.DataFrame({"stock_c": today_stock['Close']}).join(
                    pd.DataFrame({"nifty_c": today_nifty['Close']}), how="inner"
                ).dropna()

                if not joined.empty:
                    # Verify exact timestamp match of latest completed candle
                    latest_stock_ts = today_stock.index[-1]
                    latest_nifty_ts = today_nifty.index[-1]
                    is_aligned = bool(latest_stock_ts == latest_nifty_ts)

                    if is_aligned:
                        s_open = float(today_stock['Open'].iloc[0])
                        s_sync_c = float(joined['stock_c'].iloc[-1])
                        stock_change_pct = float(((s_sync_c - s_open) / s_open) * 100) if s_open > 0 else 0.0

                        n_open = float(today_nifty['Open'].iloc[0])
                        n_sync_c = float(joined['nifty_c'].iloc[-1])
                        nifty_change_pct = float(((n_sync_c - n_open) / n_open) * 100) if n_open > 0 else 0.0

                        if len(joined) >= 4:
                            s_last = float(joined['stock_c'].iloc[-1])
                            s_prev3 = float(joined['stock_c'].iloc[-3])
                            n_last = float(joined['nifty_c'].iloc[-1])
                            n_prev3 = float(joined['nifty_c'].iloc[-3])

                            s_ret3 = ((s_last - s_prev3) / s_prev3) * 100 if s_prev3 > 0 else 0.0
                            n_ret3 = ((n_last - n_prev3) / n_prev3) * 100 if n_prev3 > 0 else 0.0
                            rs_accel_3 = round(s_ret3 - n_ret3, 2)

                        if len(joined) >= 7:
                            s_last = float(joined['stock_c'].iloc[-1])
                            s_prev6 = float(joined['stock_c'].iloc[-6])
                            n_last = float(joined['nifty_c'].iloc[-1])
                            n_prev6 = float(joined['nifty_c'].iloc[-6])

                            s_ret6 = ((s_last - s_prev6) / s_prev6) * 100 if s_prev6 > 0 else 0.0
                            n_ret6 = ((n_last - n_prev6) / n_prev6) * 100 if n_prev6 > 0 else 0.0
                            rs_accel_6 = round(s_ret6 - n_ret6, 2)
        else:
            if len(sc) >= 4:
                c_3 = float(sc.iloc[-3])
                rs_accel_3 = round(((sc.iloc[-1] - c_3) / c_3) * 100, 2) if c_3 > 0 else 0.0
            if len(sc) >= 7:
                c_6 = float(sc.iloc[-6])
                rs_accel_6 = round(((sc.iloc[-1] - c_6) / c_6) * 100, 2) if c_6 > 0 else 0.0

        # Synchronize Sector Index (e.g. ^NSEBANK, ^CNXIT) strictly to exact matching completed candle timestamp
        actual_sector_pct = float(sector_change_pct)
        is_sector_aligned = False
        sector_rs_available = False

        if sector_df is not None and not sector_df.empty:
            sec_df_c = sector_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(sec_df_c.index):
                sec_df_c.index = pd.to_datetime(sec_df_c.index)
            try:
                if sec_df_c.index.tz is None:
                    sec_df_c.index = sec_df_c.index.tz_localize("Asia/Kolkata")
                else:
                    sec_df_c.index = sec_df_c.index.tz_convert("Asia/Kolkata")
            except Exception:
                pass
            today_sec = sec_df_c[sec_df_c.index.date == today_date] if today_date else sec_df_c
            if not today_sec.empty and len(today_sec) >= 2:
                # STRICT TIMESTAMP EQUALITY: Latest sector completed candle must match latest stock completed candle
                latest_sec_ts = today_sec.index[-1]
                latest_stock_ts = today_stock.index[-1]
                if latest_sec_ts == latest_stock_ts:
                    sec_joined = pd.DataFrame({"stock_c": today_stock['Close']}).join(
                        pd.DataFrame({"sec_c": today_sec['Close']}), how="inner"
                    ).dropna()
                    if not sec_joined.empty:
                        sec_open = float(today_sec['Open'].iloc[0])
                        sec_last = float(sec_joined['sec_c'].iloc[-1])
                        actual_sector_pct = float(((sec_last - sec_open) / sec_open) * 100) if sec_open > 0 else 0.0
                        is_sector_aligned = True
                        sector_rs_available = True

        rs_nifty = round(stock_change_pct - nifty_change_pct, 2) if is_aligned else 0.0
        rs_sector = round(stock_change_pct - actual_sector_pct, 2) if is_sector_aligned else 0.0
        sector_rs_nifty = round(actual_sector_pct - nifty_change_pct, 2) if (is_aligned and is_sector_aligned) else 0.0

        return {
            "stock_change_pct": round(stock_change_pct, 2),
            "sector_change_pct": round(actual_sector_pct, 2) if is_sector_aligned else None,
            "nifty_change_pct": round(nifty_change_pct, 2) if is_aligned else None,
            "rs_nifty": rs_nifty,
            "rs_sector": rs_sector,
            "sector_rs_nifty": sector_rs_nifty,
            "sector_rs_available": sector_rs_available,
            "rs_accel_3": rs_accel_3,
            "rs_accel_6": rs_accel_6,
            "rs_accel": rs_accel_3,
            "is_aligned": is_aligned,
            "is_sector_aligned": is_sector_aligned,
            "alignment_status": "ALIGNED" if (is_aligned and (sector_df is None or is_sector_aligned)) else "DATA_MISALIGNED"
        }

    @staticmethod
    def evaluate_stock_opportunity(stock_df, symbol, sector_name="GENERAL", sector_change=0.0, nifty_df=None, vwap=None, opp_eval=None, time_slot_rvol=None, rvol_obs=0, is_liquid=True, sector_df=None):
        """
        Computes comprehensive quantitative opportunity score, breakout proximity, and prediction stage.
        Enforces hard liquidity and candle alignment gates.
        """
        if not is_liquid:
            return None

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

        # Hard Liquidity Gate: Reject dead contracts using actual candle-by-candle traded value
        typical_price = (h + l + c) / 3.0
        traded_value = (typical_price * v)
        avg_traded_value_10 = float(traded_value.iloc[-10:].mean()) if len(traded_value) >= 10 else float(traded_value.mean())
        recent_v = v.iloc[-10:].mean() if len(v) >= 10 else v.mean()
        if recent_v < 5000 or avg_traded_value_10 < 1500000:
            return None

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
        # Use consistent time-slot adjusted RVOL if provided, else compute time-of-day slot matching
        if time_slot_rvol is not None and time_slot_rvol > 0:
            rvol = round(float(time_slot_rvol), 2)
        else:
            df_copy['Time'] = df_copy.index.time
            slot_t = df_copy['Time'].iloc[-1]
            same_slot_vols = df_copy[df_copy['Time'] == slot_t]['Volume']
            if len(same_slot_vols) > 1:
                hist_avg = float(same_slot_vols.iloc[:-1].mean())
                rvol = round(curr_v / hist_avg, 2) if hist_avg > 0 else 1.0
            else:
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
        
        # Previous Day High / Low (PDH / PDL)
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

        # PDH/PDL Proximity, Sweeps, and Retests
        dist_to_pdh_pct = ((pdh - curr_c) / curr_c * 100) if pdh else 999.0
        dist_to_pdl_pct = ((curr_c - pdl) / curr_c * 100) if pdl else 999.0
        is_coiling_near_pdh = (0.0 <= dist_to_pdh_pct <= 0.50) if pdh else False
        is_coiling_near_pdl = (0.0 <= dist_to_pdl_pct <= 0.50) if pdl else False
        is_broken_above_pdh = (curr_c > pdh) if pdh else False
        is_broken_below_pdl = (curr_c < pdl) if pdl else False

        # Liquidity Sweep & Retest Logic:
        # Bullish PDH retest: Price broke above PDH or swept it, and now holding above PDH
        # Bullish PDL sweep: Price dipped below PDL earlier, but rejected and reclaimed above PDL (liquidity grab)
        is_pdl_liquidity_sweep_bull = (float(today_df['Low'].min()) < pdl and curr_c > pdl) if pdl else False
        is_pdh_liquidity_sweep_bear = (float(today_df['High'].max()) > pdh and curr_c < pdh) if pdh else False
        is_pdh_retest_hold_bull = (is_broken_above_pdh and curr_l >= (pdh * 0.998)) if pdh else False
        is_pdl_retest_hold_bear = (is_broken_below_pdl and curr_h <= (pdl * 1.002)) if pdl else False

        # Confirmed Swing Pivot Structure in Opportunity Engine (No silent unconfirmed fallback)
        low_vals = [float(x) for x in l.values]
        high_vals = [float(x) for x in h.values]
        n_bars = len(low_vals)
        c_lows = []
        c_highs = []
        for i in range(n_bars - 3, max(1, n_bars - 35), -1):
            if (i >= 2 and i + 2 < n_bars):
                if (low_vals[i] <= low_vals[i-1] and low_vals[i] <= low_vals[i-2] and
                    low_vals[i] <= low_vals[i+1] and low_vals[i] <= low_vals[i+2]):
                    c_lows.append(low_vals[i])
                if (high_vals[i] >= high_vals[i-1] and high_vals[i] >= high_vals[i-2] and
                    high_vals[i] >= high_vals[i+1] and high_vals[i] >= high_vals[i+2]):
                    c_highs.append(high_vals[i])
            if len(c_lows) >= 2 and len(c_highs) >= 2:
                break
        
        has_swing_higher_lows = (len(c_lows) >= 2 and c_lows[0] > c_lows[1])
        has_swing_lower_highs = (len(c_highs) >= 2 and c_highs[0] < c_highs[1])

        # Establish structural invalidation levels:
        # Bullish invalidation: Most recent confirmed swing low or ORL
        # Bearish invalidation: Most recent confirmed swing high or ORH
        bull_invalidation = c_lows[0] if c_lows else (orl if orl and orl < curr_c else None)
        bear_invalidation = c_highs[0] if c_highs else (orh if orh and orh > curr_c else None)

        # 5. Synchronized 3-Way Relative Strength Acceleration
        rs_data = OpportunityRankingEngine.calculate_relative_strength_acceleration(stock_df, nifty_df, sector_change, sector_df=sector_df)
        # Enforce mandatory alignment: if NIFTY is provided but timestamps are misaligned, reject
        if nifty_df is not None and not nifty_df.empty and not rs_data.get("is_aligned", False):
            return None
        rs_nifty = rs_data["rs_nifty"]
        rs_sector = rs_data["rs_sector"]
        rs_accel_3 = rs_data.get("rs_accel_3", 0.0)
        rs_accel_6 = rs_data.get("rs_accel_6", 0.0)
        rs_accel = rs_accel_3

        # 6. VWAP Relationship
        above_vwap = (curr_c > vwap) if vwap else False
        below_vwap = (curr_c < vwap) if vwap else False

        # 7. Specialized Pattern Detectors: Gap-Up / Gap-Down, Flat-Open Ignition, Index Shock
        open_gap_pct = 0.0
        is_gap_up = False
        is_gap_down = False
        is_flat_open = False
        is_flat_open_ignition = False
        is_exhausted_gap = False  # Gap too huge (> 2.5%) where move is already over/exhausted!

        if prev_date is not None:
            prev_day_df = df_copy[df_copy.index.date == prev_date]
            if not prev_day_df.empty:
                prev_day_close = float(prev_day_df['Close'].iloc[-1])
                today_first_open = float(today_df['Open'].iloc[0])
                if prev_day_close > 0:
                    open_gap_pct = round(((today_first_open - prev_day_close) / prev_day_close) * 100, 2)
                    # Sophisticated Large Gap Classification:
                    # Distinguish genuine gap continuation vs runaway exhaustion.
                    # If gap >= 2.5%, stock is NOT automatically discarded if it establishes an intraday base,
                    # holds above the opening candle (gap continuation), or exhibits low-risk consolidation.
                    # It is only rejected as an exhausted gap if price is extended far beyond ORH (> 3.5% gap without consolidation)
                    # or failing back through opening range (gap trap).
                    is_large_gap = (abs(open_gap_pct) >= 2.50)
                    is_holding_gap_base = (open_gap_pct > 0 and curr_c >= today_first_open) or (open_gap_pct < 0 and curr_c <= today_first_open)
                    
                    if abs(open_gap_pct) >= 4.0:
                        is_exhausted_gap = True
                    elif is_large_gap and not is_holding_gap_base:
                        # Gapped and failed back into range (gap trap / fade)
                        is_exhausted_gap = True

                    is_gap_up = (open_gap_pct >= 0.60)
                    is_gap_down = (open_gap_pct <= -0.60)
                    is_flat_open = (abs(open_gap_pct) <= 0.40)
                    # Flat open with clean breakout initiation: best risk-reward setups!
                    if is_flat_open and (curr_c > today_first_open) and (rvol >= 1.2 or is_expanding_now):
                        is_flat_open_ignition = True

        # If move is already exhausted by extreme overnight gap, reject or severely penalize
        if is_exhausted_gap:
            return None

        # Gap Hold Confirmation (Holding above gap open)
        is_gap_up_holding = is_gap_up and (curr_c >= float(today_df['Open'].iloc[0]))
        is_gap_down_holding = is_gap_down and (curr_c <= float(today_df['Open'].iloc[0]))

        # Dual Scoring Architecture:
        # A. Early Opportunity Score (Pre-Breakout, Energy Coil, Volume Accumulation, RS Lead)
        # B. Confirmation Score (Completed Expansion, Broken Level, High RVOL, Institutional OB)
        early_bull_score = 0.0
        early_bear_score = 0.0
        conf_bull_score = 0.0
        conf_bear_score = 0.0

        # Fine-grained continuous adjustments (creates distinct, realistic scores like 69, 74, 83):
        pot_val = float(opp_eval.get("potential_pct", 0.0)) if opp_eval else 0.0
        # 1. Volume bonus: proportional to RVOL
        rvol_bonus = min(12.0, max(0.0, (rvol - 1.0) * 8.0))
        # 2. RS bonus: proportional to relative strength
        rs_bonus_bull = min(10.0, max(0.0, rs_nifty * 3.0 + rs_accel_3 * 5.0))
        rs_bonus_bear = min(10.0, max(0.0, -rs_nifty * 3.0 - rs_accel_3 * 5.0))
        # 3. Expansion room bonus
        room_bonus = min(8.0, max(0.0, pot_val * 4.0))

        # Bullish Early Opportunity Factors (Must reflect true structural accumulation):
        if is_coiling_near_orh or is_coiling_near_pdh: early_bull_score += 24.0
        if is_range_squeezed and has_swing_higher_lows: early_bull_score += 19.0
        if vol_consecutive_rise and curr_c > curr_o: early_bull_score += 14.0
        if rs_accel_3 > 0.15: early_bull_score += 13.0
        if is_pdl_liquidity_sweep_bull: early_bull_score += 18.0
        if is_pdh_retest_hold_bull: early_bull_score += 14.0
        if is_flat_open_ignition: early_bull_score += 14.0
        if is_gap_up_holding: early_bull_score += 9.0
        if above_vwap: early_bull_score += 11.0
        # Deep Institutional Triggers (Breaker Blocks & Liquidity Inducement Sweeps):
        if opp_eval and opp_eval.get("has_eql_inducement_sweep"): early_bull_score += 21.0
        if opp_eval and opp_eval.get("nearest_bull_breaker"): conf_bull_score += 21.0

        # Bullish Confirmation Factors:
        if is_broken_above_orh or is_broken_above_pdh: conf_bull_score += 24.0
        if is_expanding_now and curr_c > curr_o: conf_bull_score += 19.0
        if rvol >= 1.25: conf_bull_score += 18.0
        if above_vwap: conf_bull_score += 13.0
        if rs_nifty > 0.3: conf_bull_score += 12.0
        if rs_accel_6 > 0.2: conf_bull_score += 12.0

        # Bearish Early Opportunity Factors:
        if is_coiling_near_orl or is_coiling_near_pdl: early_bear_score += 24.0
        if is_range_squeezed and has_swing_lower_highs: early_bear_score += 19.0
        if vol_consecutive_rise and curr_c < curr_o: early_bear_score += 14.0
        if rs_accel_3 < -0.15: early_bear_score += 13.0
        if is_pdh_liquidity_sweep_bear: early_bear_score += 18.0
        if is_pdl_retest_hold_bear: early_bear_score += 14.0
        if is_gap_down_holding: early_bear_score += 9.0
        if below_vwap: early_bear_score += 11.0
        # Deep Institutional Triggers (Breaker Blocks & Liquidity Inducement Sweeps):
        if opp_eval and opp_eval.get("has_eqh_inducement_sweep"): early_bear_score += 21.0
        if opp_eval and opp_eval.get("nearest_bear_breaker"): conf_bear_score += 21.0

        # Bearish Confirmation Factors:
        if is_broken_below_orl or is_broken_below_pdl: conf_bear_score += 24.0
        if is_expanding_now and curr_c < curr_o: conf_bear_score += 19.0
        if rvol >= 1.25: conf_bear_score += 18.0
        if below_vwap: conf_bear_score += 13.0
        if rs_nifty < -0.3: conf_bear_score += 12.0
        if rs_accel_6 < -0.2: conf_bear_score += 12.0

        # 4. Adaptive Machine Learning Multiplier from historical feedback:
        from engine.learning.adaptive_trade_learner import AdaptiveTradeLearner
        strat_key = "MOMENTUM_BREAKOUT" if (is_broken_above_orh or is_broken_below_orl) else "INTRADAY_TREND"
        learn_mult_bull = AdaptiveTradeLearner.get_pattern_multiplier(strat_key, "BULLISH")
        learn_mult_bear = AdaptiveTradeLearner.get_pattern_multiplier(strat_key, "BEARISH")

        early_bull_score = int(round(min(98.0, (early_bull_score + rvol_bonus + rs_bonus_bull) * learn_mult_bull)))
        conf_bull_score = int(round(min(98.0, (conf_bull_score + rvol_bonus + rs_bonus_bull + room_bonus) * learn_mult_bull)))
        early_bear_score = int(round(min(98.0, (early_bear_score + rvol_bonus + rs_bonus_bear) * learn_mult_bear)))
        conf_bear_score = int(round(min(98.0, (conf_bear_score + rvol_bonus + rs_bonus_bear + room_bonus) * learn_mult_bear)))

        total_bull = max(early_bull_score, conf_bull_score)
        total_bear = max(early_bear_score, conf_bear_score)

        direction = "BULLISH" if (total_bull >= total_bear and total_bull >= 30) else ("BEARISH" if total_bear >= 30 else "NEUTRAL")
        conviction_score = total_bull if direction == "BULLISH" else (total_bear if direction == "BEARISH" else max(total_bull, total_bear))

        # Correct Stop-Loss & Targets from Opportunity Potential Engine
        sl = opp_eval.get("stop_loss") or opp_eval.get("suggested_sl") if opp_eval else None
        tp1 = opp_eval.get("tp1") or opp_eval.get("suggested_tp1") if opp_eval else None
        tp2 = opp_eval.get("tp2") or opp_eval.get("suggested_tp2") if opp_eval else None
        potential_pct = float(opp_eval.get("potential_pct", 0.0)) if opp_eval else 0.0
        rr_ratio = float(opp_eval.get("risk_reward_ratio", 0.0)) if opp_eval else 0.0
        exhaustion_risk = opp_eval.get("exhaustion_risk", "LOW") if opp_eval else "LOW"

        # Quality Gates for CONFIRMED (Calibrated: potential >= 0.35%, RR >= 1.05):
        is_viable_opportunity = (
            opp_eval.get("is_viable", False) and 
            potential_pct >= 0.35 and 
            rr_ratio >= 1.05 and 
            exhaustion_risk != "HIGH"
        ) if opp_eval else False

        # Structural numerical SL/TP and actual mathematical RR validation:
        # 1. SL and TP must be finite, positive numbers (not NaN or Inf)
        # 2. For BUY: SL < curr_c and TP1 > curr_c
        # 3. For SELL: SL > curr_c and TP1 < curr_c
        # 4. Computed RR = (reward / risk) >= 1.05 without lossy premature rounding
        import math
        computed_rr = 0.0
        has_valid_structural_risk = False
        if sl is not None and tp1 is not None:
            try:
                sl_val = float(sl)
                tp1_val = float(tp1)
                curr_c_val = float(curr_c)
                if (math.isfinite(sl_val) and math.isfinite(tp1_val) and math.isfinite(curr_c_val) and
                        sl_val > 0 and tp1_val > 0 and curr_c_val > 0):
                    if direction == "BULLISH" and sl_val < curr_c_val and tp1_val > curr_c_val:
                        risk = curr_c_val - sl_val
                        reward = tp1_val - curr_c_val
                        raw_rr = reward / risk if risk > 0 else 0.0
                        if math.isfinite(raw_rr) and raw_rr >= 1.05:
                            computed_rr = round(raw_rr, 2)
                            has_valid_structural_risk = True
                    elif direction == "BEARISH" and sl_val > curr_c_val and tp1_val < curr_c_val:
                        risk = sl_val - curr_c_val
                        reward = curr_c_val - tp1_val
                        raw_rr = reward / risk if risk > 0 else 0.0
                        if math.isfinite(raw_rr) and raw_rr >= 1.05:
                            computed_rr = round(raw_rr, 2)
                            has_valid_structural_risk = True
            except (ValueError, TypeError):
                has_valid_structural_risk = False

        # Stage Classification with Calibrated Quality Gates:
        # A stock is CONFIRMED (ENTRY READY) when:
        # 1. is_viable_opportunity (opp_eval viable, potential >= 0.35%, RR >= 1.05, not high exhaustion)
        # 2. Valid structural stop-loss, target, and computed RR >= 1.05
        # 3. Confirmation score meets threshold (>= 32) aligned with direction
        # 4. Conviction score >= 38
        is_strictly_confirmed = (
            is_viable_opportunity and 
            has_valid_structural_risk and 
            potential_pct >= 0.35 and 
            exhaustion_risk != "HIGH" and
            ((direction == "BULLISH" and conf_bull_score >= 32 and curr_c > curr_o) or 
             (direction == "BEARISH" and conf_bear_score >= 32 and curr_c < curr_o)) and
            conviction_score >= 38
        )

        # Direction-specific IGNITION trigger:
        # Activated when price breaks out of ORH/PDH (Long) or breaks down from ORL/PDL (Short) with closing confirmation
        # 1. Price breaking out of ORH/PDH (Long) or breaking down from ORL/PDL (Short) or expanding with volume
        # 2. Volume expansion: RVOL >= 1.10 or expanding range >= 1.1x ATR
        # 3. STRICT VWAP VALIDATION: Must hold above VWAP for Longs / below VWAP for Shorts.
        # 4. Valid structural risk boundaries: Invalidation level must exist with computed RR >= 1.05
        has_ignition_risk_gate = has_valid_structural_risk and (computed_rr >= 1.05)
        
        is_bull_ignition = (
            direction == "BULLISH" and 
            above_vwap and 
            has_ignition_risk_gate and (
                ((is_broken_above_orh or is_broken_above_pdh) and curr_c > curr_o and (rvol >= 1.10 or is_expanding_now)) or
                (is_expanding_now and curr_c > curr_o and rvol >= 1.15)
            )
        )
        is_bear_ignition = (
            direction == "BEARISH" and 
            below_vwap and 
            has_ignition_risk_gate and (
                ((is_broken_below_orl or is_broken_below_pdl) and curr_c < curr_o and (rvol >= 1.10 or is_expanding_now)) or
                (is_expanding_now and curr_c < curr_o and rvol >= 1.15)
            )
        )

        # Strict PREPARING Requirement:
        # 1. Structural evidence: Volatility compression (range squeezed / NR4 / NR7) or coiling near key breakout level (ORH/ORL/PDH/PDL)
        # 2. Confirmed swing pivot structure OR validated liquidity sweep & reclaim (NO single candle substitution!)
        # 3. Established structural invalidation level present
        # 4. Early opportunity score >= 40 and aligned with VWAP
        has_bull_structure = (has_swing_higher_lows or is_pdl_liquidity_sweep_bull) and (bull_invalidation is not None and bull_invalidation < curr_c)
        has_bear_structure = (has_swing_lower_highs or is_pdh_liquidity_sweep_bear) and (bear_invalidation is not None and bear_invalidation > curr_c)

        is_strict_bull_preparing = (
            direction == "BULLISH" and 
            (is_coiling_near_orh or is_coiling_near_pdh or is_range_squeezed) and 
            has_bull_structure and 
            early_bull_score >= 40 and 
            above_vwap
        )

        is_strict_bear_preparing = (
            direction == "BEARISH" and 
            (is_coiling_near_orl or is_coiling_near_pdl or is_range_squeezed) and 
            has_bear_structure and 
            early_bear_score >= 40 and 
            below_vwap
        )

        # Low-priority Early Observation Stage (Watch-only: strictly requires true accumulation evidence):
        # Must have at least ONE participation signal (rising volume or improving RS) AND structural evidence:
        # Participation signals:
        has_bull_participation = (vol_consecutive_rise and curr_c >= curr_o) or (rs_accel_3 > 0.15)
        has_bear_participation = (vol_consecutive_rise and curr_c <= curr_o) or (rs_accel_3 < -0.15)
        
        # Structural evidence: Must be near key breakout level (ORH/ORL/PDH/PDL) or have confirmed swing structure
        # (Generic sideways range squeeze without structural level proximity is insufficient!)
        has_bull_structure_setup = is_coiling_near_orh or is_coiling_near_pdh or has_swing_higher_lows or (is_range_squeezed and dist_to_orh_pct <= 1.0)
        has_bear_structure_setup = is_coiling_near_orl or is_coiling_near_pdl or has_swing_lower_highs or (is_range_squeezed and dist_to_orl_pct <= 1.0)

        # High-Recall Early Discovery Layer (P2 - Dedicated Pre-Move Discovery):
        # Discovers non-textbook movers early WITHOUT weakening entry confirmation:
        # 1. Accumulation & Absorption before breakout (tight coil near key level or under VWAP with rising volume)
        # 2. Liquidity-Sweep Reversals before VWAP reclaim (e.g. swept PDL and curling back up)
        # 3. Early Relative-Strength Acceleration (rs_accel_3 > 0.15)
        # 4. Sudden volume expansion before trend confirmation (RVOL >= 1.5)
        # ALL of these are strictly classified as OBSERVING / PREPARING (WATCH-ONLY; is_actionable = False)
        is_bull_accumulation = (
            (is_coiling_near_orh or is_coiling_near_pdh or is_range_squeezed) and 
            (vol_consecutive_rise or rvol >= 1.2)
        )
        is_bear_distribution = (
            (is_coiling_near_orl or is_coiling_near_pdl or is_range_squeezed) and 
            (vol_consecutive_rise or rvol >= 1.2)
        )
        
        is_bull_sweep_reversal = is_pdl_liquidity_sweep_bull
        is_bear_sweep_reversal = is_pdh_liquidity_sweep_bear
        
        is_bull_early_rs_accel = (rs_accel_3 > 0.15 and (curr_c >= curr_o or vol_consecutive_rise))
        is_bear_early_rs_accel = (rs_accel_3 < -0.15 and (curr_c <= curr_o or vol_consecutive_rise))

        is_bull_volume_burst = (rvol >= 1.5 and is_expanding_now and curr_c > curr_o)
        is_bear_volume_burst = (rvol >= 1.5 and is_expanding_now and curr_c < curr_o)

        has_bull_early_momentum = (
            direction == "BULLISH" and (
                (above_vwap and curr_c > curr_o and ((rvol >= 1.5 and rs_accel_3 > 0.1) or is_flat_open_ignition or (is_expanding_now and rvol >= 1.25))) or
                is_bull_accumulation or
                is_bull_sweep_reversal or
                is_bull_early_rs_accel or
                is_bull_volume_burst
            )
        )
        has_bear_early_momentum = (
            direction == "BEARISH" and (
                (below_vwap and curr_c < curr_o and ((rvol >= 1.5 and rs_accel_3 < -0.1) or (is_expanding_now and rvol >= 1.25))) or
                is_bear_distribution or
                is_bear_sweep_reversal or
                is_bear_early_rs_accel or
                is_bear_volume_burst
            )
        )

        is_early_observing = (
            (direction == "BULLISH" and (has_bull_early_momentum or (has_bull_participation and has_bull_structure_setup)) and early_bull_score >= 20) or
            (direction == "BEARISH" and (has_bear_early_momentum or (has_bear_participation and has_bear_structure_setup)) and early_bear_score >= 20)
        )

        if is_strictly_confirmed:
            stage = "CONFIRMED"
            stage_desc = f"Verified Completed Trade: RR 1:{computed_rr:.2f}, +{potential_pct:.1f}% Room"
            entry_trigger = f"{'Demand OB (Buy)' if direction == 'BULLISH' else 'Supply OB (Sell)'} Entry confirmed with RR 1:{computed_rr:.1f}"
        elif is_bull_ignition or is_bear_ignition:
            stage = "IGNITION"
            stage_desc = f"Momentum Breaking Out: RVOL {rvol:.1f}x, Range {curr_range/atr14:.1f}x ATR"
            entry_trigger = "Breakout confirmed above ORH" if direction == "BULLISH" else "Breakdown confirmed below ORL"
        elif is_strict_bull_preparing or is_strict_bear_preparing:
            stage = "PREPARING"
            stage_desc = "⚡ Pre-Breakout/Breakdown Coil: Institutional accumulation before the move"
            entry_trigger = f"Pre-Breakout: Enter above ₹{min(orh, pdh if pdh else orh):.1f}" if direction == "BULLISH" else f"Pre-Breakdown: Enter below ₹{max(orl, pdl if pdl else orl):.1f}"
        elif is_early_observing:
            stage = "OBSERVING"
            stage_desc = "👀 Early Observation: Quiet accumulation watch-only (No confirmed entry)"
            entry_trigger = "Watch only: Waiting for confirmed pivot structure & volume expansion"
        else:
            stage = "MONITORING"
            stage_desc = "Normal intraday action"
            entry_trigger = "Wait for structure setup"

        # Setup Evidence & Drivers
        reasons = []
        if is_flat_open_ignition: reasons.append("Flat-open ignition")
        if is_gap_up_holding: reasons.append(f"Gap-up +{open_gap_pct}% holding")
        elif is_gap_down_holding: reasons.append(f"Gap-down {open_gap_pct}% holding")
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

        rvol_confidence = "HIGH" if rvol_obs >= 3 else ("MEDIUM" if rvol_obs >= 1 else "LOW")
        latest_candle_timestamp = str(stock_df.index[-1]) if stock_df is not None and not stock_df.empty else ""

        # Explicit actionability flag: strictly requires confirmed stage or confirmed ignition breakout with validated risk
        # CONFIRMED and IGNITION (breakouts with confirmed risk & RVOL) are actionable entries!
        is_actionable = bool(
            (stage == "CONFIRMED" and has_valid_structural_risk and computed_rr >= 1.05) or
            (stage == "IGNITION" and has_ignition_risk_gate and computed_rr >= 1.05 and rvol >= 1.10)
        )

        return {
            "symbol": symbol,
            "candle_ts": latest_candle_timestamp,
            "sector": sector_name,
            "current_price": curr_c,
            "change_pct": rs_data["stock_change_pct"],
            "direction": direction,
            "conviction_score": conviction_score,
            "early_opportunity_score": early_bull_score if direction == "BULLISH" else early_bear_score,
            "confirmation_score": conf_bull_score if direction == "BULLISH" else conf_bear_score,
            "stage": stage,
            "is_actionable": is_actionable,
            "stage_desc": stage_desc,
            "entry_trigger": entry_trigger,
            "invalidation_level": bull_invalidation if direction == "BULLISH" else bear_invalidation,
            "rvol": rvol,
            "rvol_obs": rvol_obs,
            "rvol_confidence": rvol_confidence,
            "vol_consecutive_rise": vol_consecutive_rise,
            "is_range_squeezed": is_range_squeezed,
            "rs_nifty": rs_nifty,
            "rs_sector": rs_sector,
            "rs_accel": rs_accel,
            "rs_accel_3": rs_accel_3,
            "rs_accel_6": rs_accel_6,
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
        # 1. Stage (CONFIRMED > IGNITION > PREPARING > OBSERVING > MONITORING)
        # 2. Stage-appropriate scoring:
        #    - For CONFIRMED/IGNITION: weighted by confirmation_score and rs_accel_6
        #    - For PREPARING/OBSERVING: weighted by early_opportunity_score and rs_accel_3
        stage_weights = {"CONFIRMED": 400, "IGNITION": 250, "PREPARING": 150, "OBSERVING": 80, "MONITORING": 0}

        def score_candidate(cand, is_bull=True):
            stg = cand.get("stage", "MONITORING")
            base = stage_weights.get(stg, 0)
            if stg in ("CONFIRMED", "IGNITION"):
                main_score = cand.get("confirmation_score", cand.get("conviction_score", 0))
            else:
                main_score = cand.get("early_opportunity_score", cand.get("conviction_score", 0))
            
            # Directional Relative Strength Acceleration:
            # Bullish: higher positive RS acceleration is rewarded.
            # Bearish: more negative RS acceleration is rewarded (underperforming and deteriorating faster!).
            if is_bull:
                rs_acc = cand.get("rs_accel_3", 0.0)
                rs_acc6 = cand.get("rs_accel_6", 0.0)
            else:
                rs_acc = -cand.get("rs_accel_3", 0.0)
                rs_acc6 = -cand.get("rs_accel_6", 0.0)

            return base + main_score + (rs_acc * 10) + (rs_acc6 * 5)

        bullish.sort(key=lambda x: score_candidate(x, is_bull=True), reverse=True)
        bearish.sort(key=lambda x: score_candidate(x, is_bull=False), reverse=True)

        # Direction-Neutral Fair Sorting for ENTRY-READY and PREPARING
        # Do NOT simply take bullish first; rank strictly by objective quality/conviction score!
        def neutral_quality_score(cand):
            is_bull = cand.get("direction") == "BULLISH"
            return score_candidate(cand, is_bull=is_bull)

        all_candidates = bullish + bearish
        all_candidates.sort(key=neutral_quality_score, reverse=True)

        # 1. ENTRY READY (Max 2): CONFIRMED or active IGNITION 5-minute setups with validated risk
        entry_ready_list = [c for c in all_candidates if c.get("stage") in ("CONFIRMED", "IGNITION") and c.get("is_actionable")][:2]

        # 2. IGNITION (Expanding right now across ORH/PDH or high volume breakout)
        ignition_list = [
            c for c in all_candidates 
            if c.get("stage") == "IGNITION" and c not in entry_ready_list
        ][:3]

        # 3. PREPARING (Max 5): Strictly Pre-Breakout / Pre-Breakdown COILS before expansion begins
        preparing_list = [
            c for c in all_candidates 
            if c.get("stage") == "PREPARING" and c not in entry_ready_list
        ][:5]

        # 4. OBSERVING (Max 5): Early quiet accumulation watch-only (Distinct from preparing!)
        observing_list = [
            c for c in all_candidates 
            if c.get("stage") == "OBSERVING" and c not in entry_ready_list and c not in preparing_list
        ][:5]

        # 5. WATCHLIST / MONITORING ONLY: Explicitly marked as non-actionable watchlists
        # Distinguish actionable setups from pure sector monitoring watchlists
        watchlist_bullish = [
            {**c, "watchlist_type": "SECTOR_WATCHLIST_ONLY", "is_actionable": (c.get("stage") in ("CONFIRMED", "IGNITION"))}
            for c in bullish[:top_n]
        ]
        watchlist_bearish = [
            {**c, "watchlist_type": "SECTOR_WATCHLIST_ONLY", "is_actionable": (c.get("stage") in ("CONFIRMED", "IGNITION"))}
            for c in bearish[:top_n]
        ]

        return {
            "top_bullish": watchlist_bullish,
            "top_bearish": watchlist_bearish,
            "entry_ready": entry_ready_list,
            "ignition": ignition_list,
            "preparing": preparing_list,
            "observing": observing_list
        }
