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
    def evaluate_strategies(df, prev_close=None, sector_bias="NEUTRAL", catalyst_info=None, anomaly_info=None, daily_df=None, market_regime_info=None, as_of_ts=None):
        if df is None or df.empty or len(df) < 5:
            return None

        # Ensure single-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.copy()

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

        # 3. UNIFIED MULTI-DURATION OPENING RANGE (5m, 15m, 30m) & GAP
        today_open = float(today_df['Open'].iloc[0])
        # 5m ORB (first 1 completed candle), 15m ORB (first 3 completed candles), 30m ORB (first 6 completed candles)
        or_5m_h = float(today_df['High'].iloc[0])
        or_5m_l = float(today_df['Low'].iloc[0])
        or_15m_h = float(today_df['High'].iloc[:3].max()) if len(today_df) >= 3 else or_5m_h
        or_15m_l = float(today_df['Low'].iloc[:3].min()) if len(today_df) >= 3 else or_5m_l
        or_30m_h = float(today_df['High'].iloc[:6].max()) if len(today_df) >= 6 else or_15m_h
        or_30m_l = float(today_df['Low'].iloc[:6].min()) if len(today_df) >= 6 else or_15m_l

        # Unified ORB selection: finalize range once respective period is complete
        if len(today_df) >= 6:
            or_high = or_30m_h
            or_low = or_30m_l
            orb_type = "30m ORB"
        elif len(today_df) >= 3:
            or_high = or_15m_h
            or_low = or_15m_l
            orb_type = "15m ORB"
        else:
            or_high = or_5m_h
            or_low = or_5m_l
            orb_type = "5m ORB"

        shared_orb_levels = {
            "orh": or_high,
            "orl": or_low,
            "orb_type": orb_type,
            "or_5m": (or_5m_h, or_5m_l),
            "or_15m": (or_15m_h, or_15m_l),
            "or_30m": (or_30m_h, or_30m_l)
        }
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

        # UNIFIED TIME-OF-DAY RVOL (Across Anomaly and Institutional Engine)
        from engine.anomaly_engine import AnomalyDetector
        rvol = AnomalyDetector.calculate_time_of_day_rvol(df)

        # SMC: Robust Incremental Order Blocks & Confirmed Swing Points (Zero Lookahead)
        # 1. Swing points with explicit confirmation timestamps:
        # A pivot at index i is confirmed only after candles i+1 and i+2 close (confirmation_idx = i+2).
        # We only consider confirmed swings within the last 40 candles so old broken levels do not corrupt structure.
        swing_high_records = []
        swing_low_records = []
        hist_df = df
        d_times = hist_df.index
        n_hist = len(hist_df)
        
        # Only look for pivots confirmed up to the current bar (i + 2 < n_hist)
        for i in range(max(2, n_hist - 45), n_hist - 2):
            confirm_idx = i + 2
            confirm_ts = str(d_times[confirm_idx])
            pivot_ts = str(d_times[i])
            
            if float(h.iloc[i]) == max([float(x) for x in h.iloc[i-2:i+3]]):
                p_high = float(h.iloc[i])
                # Check status at current candle n_hist - 1
                is_swept = any(float(h.iloc[m]) > p_high and float(c.iloc[m]) < p_high for m in range(confirm_idx+1, n_hist))
                is_broken = any(float(c.iloc[m]) > p_high for m in range(confirm_idx+1, n_hist))
                swing_high_records.append({
                    "price": p_high,
                    "pivot_idx": i,
                    "pivot_ts": pivot_ts,
                    "confirm_idx": confirm_idx,
                    "confirm_ts": confirm_ts,
                    "is_swept": is_swept,
                    "is_broken": is_broken
                })
                
            if float(l.iloc[i]) == min([float(x) for x in l.iloc[i-2:i+3]]):
                p_low = float(l.iloc[i])
                is_swept = any(float(l.iloc[m]) < p_low and float(c.iloc[m]) > p_low for m in range(confirm_idx+1, n_hist))
                is_broken = any(float(c.iloc[m]) < p_low for m in range(confirm_idx+1, n_hist))
                swing_low_records.append({
                    "price": p_low,
                    "pivot_idx": i,
                    "pivot_ts": pivot_ts,
                    "confirm_idx": confirm_idx,
                    "confirm_ts": confirm_ts,
                    "is_swept": is_swept,
                    "is_broken": is_broken
                })

        # Most recent active confirmed swing points (unbroken or swept)
        unbroken_highs = [rec["price"] for rec in swing_high_records if not rec["is_broken"]]
        unbroken_lows = [rec["price"] for rec in swing_low_records if not rec["is_broken"]]
        prev_swing_high = unbroken_highs[-1] if unbroken_highs else (swing_high_records[-1]["price"] if swing_high_records else float(h.iloc[:-1].max()))
        prev_swing_low = unbroken_lows[-1] if unbroken_lows else (swing_low_records[-1]["price"] if swing_low_records else float(l.iloc[:-1].min()))

        swing_highs = [rec["price"] for rec in swing_high_records]
        swing_lows = [rec["price"] for rec in swing_low_records]

        # 2. Incremental order block lifecycle state machine:
        # States: CREATED -> ACTIVE -> PARTIALLY_MITIGATED -> FULLY_MITIGATED -> INVALIDATED
        # Requires displacement (expansion candle range >= 1.25x ATR or RVOL >= 1.20)
        active_bull_obs = []
        active_bear_obs = []
        vol_series = v.values
        
        for k in range(1, len(hist_df)):
            c_prev = float(c.iloc[k-1])
            o_prev = float(o.iloc[k-1])
            h_prev = float(h.iloc[k-1])
            l_prev = float(l.iloc[k-1])
            c_curr = float(c.iloc[k])
            o_curr = float(o.iloc[k])
            h_curr = float(h.iloc[k])
            l_curr = float(l.iloc[k])
            v_curr = float(v.iloc[k])
            curr_rng = h_curr - l_curr
            
            # 15-period volume mean up to bar k-1 for local RVOL
            past_vols = vol_series[max(0, k-15):k]
            v_mean = float(np.mean(past_vols)) if len(past_vols) > 0 else (v_curr or 1.0)
            bar_rvol = v_curr / (v_mean or 1.0)
            
            # Displacement check on confirming candle k:
            # Body must expand with significant movement (range >= 1.25x ATR OR bar RVOL >= 1.20)
            has_displacement = (curr_rng >= (atr14 * 1.20)) or (bar_rvol >= 1.20) or (abs(c_curr - o_curr) >= (atr14 * 0.8))
            
            # Bullish Demand OB: Down candle followed by displacement breaking above its high
            if c_prev < o_prev and c_curr > h_prev and has_displacement:
                active_bull_obs.append({
                    "top": h_prev,
                    "bottom": l_prev,
                    "mid": (h_prev + l_prev) / 2.0,
                    "created_at_idx": k,
                    "created_ts": str(d_times[k]),
                    "state": "ACTIVE"  # CREATED -> ACTIVE
                })
            
            # Bearish Supply OB: Up candle followed by displacement breaking below its low
            if c_prev > o_prev and c_curr < l_prev and has_displacement:
                active_bear_obs.append({
                    "top": h_prev,
                    "bottom": l_prev,
                    "mid": (h_prev + l_prev) / 2.0,
                    "created_at_idx": k,
                    "created_ts": str(d_times[k]),
                    "state": "ACTIVE"
                })
            
            # Lifecycle state progression for candles subsequent to creation:
            # - TOUCH 0-50% into zone: PARTIALLY_MITIGATED (wick enters upper/lower half)
            # - PIERCE past mid (50% equilibrium): FULLY_MITIGATED
            # - PIERCE past bottom (bull) or top (bear): INVALIDATED
            for ob in active_bull_obs:
                if ob["created_at_idx"] < k and ob["state"] in ("ACTIVE", "PARTIALLY_MITIGATED", "FULLY_MITIGATED"):
                    if l_curr < ob["bottom"]:
                        ob["state"] = "INVALIDATED"
                    elif l_curr <= ob["mid"]:
                        ob["state"] = "FULLY_MITIGATED"
                    elif l_curr <= ob["top"]:
                        ob["state"] = "PARTIALLY_MITIGATED"
            
            for ob in active_bear_obs:
                if ob["created_at_idx"] < k and ob["state"] in ("ACTIVE", "PARTIALLY_MITIGATED", "FULLY_MITIGATED"):
                    if h_curr > ob["top"]:
                        ob["state"] = "INVALIDATED"
                    elif h_curr >= ob["mid"]:
                        ob["state"] = "FULLY_MITIGATED"
                    elif h_curr >= ob["bottom"]:
                        ob["state"] = "PARTIALLY_MITIGATED"

        # Relevant Order Blocks:
        # Demand OB must be BELOW or touching current price (cannot be above current price for Bullish!)
        # Supply OB must be ABOVE or touching current price (cannot be below current price for Bearish!)
        fresh_bull_obs = [ob for ob in active_bull_obs if ob["state"] in ("ACTIVE", "PARTIALLY_MITIGATED") and ob["top"] <= (curr_c * 1.005)]
        fresh_bear_obs = [ob for ob in active_bear_obs if ob["state"] in ("ACTIVE", "PARTIALLY_MITIGATED") and ob["bottom"] >= (curr_c * 0.995)]
        fully_mitigated_bull_obs = [ob for ob in active_bull_obs if ob["state"] == "FULLY_MITIGATED"]
        fully_mitigated_bear_obs = [ob for ob in active_bear_obs if ob["state"] == "FULLY_MITIGATED"]
        
        near_demand_ob = min(fresh_bull_obs, key=lambda ob: abs(curr_c - ob['top'])) if fresh_bull_obs else None
        near_supply_ob = min(fresh_bear_obs, key=lambda ob: abs(curr_c - ob['bottom'])) if fresh_bear_obs else None

        # Fair Value Gaps (FVG)
        fvg_bull = len(df) >= 3 and float(l.iloc[-1]) > float(h.iloc[-3])
        fvg_bear = len(df) >= 3 and float(h.iloc[-1]) < float(l.iloc[-3])

        # Break of Structure (BOS) & Liquidity Sweeps using verified prior swing levels (excluding current candle)
        prior_h = h.iloc[:-1]
        prior_l = l.iloc[:-1]
        prev_swing_high = float(swing_highs[-1]) if swing_highs else float(prior_h.max())
        prev_swing_low = float(swing_lows[-1]) if swing_lows else float(prior_l.min())

        bos_bullish = curr_c > prev_swing_high and curr_c > vwap
        bos_bearish = curr_c < prev_swing_low and curr_c < vwap

        # Liquidity Sweep & Stop Hunt Reversal: Low/High pierces level but candle closes back within structure
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

        # Strategy E & F: VWAP Reclaim / Rejection (Comparing historical candle to its own session VWAP)
        if len(today_df) >= 2 and len(session_vwap_series) >= 2:
            prev_candle_c = float(today_df['Close'].iloc[-2])
            prev_candle_vwap = float(session_vwap_series.iloc[-2])
            if prev_candle_c < prev_candle_vwap and curr_c > vwap and ema9 >= ema21:
                evaluated_signals.append({"strategy": "VWAP Reclaim", "direction": "BUY", "weight": 22})
            elif prev_candle_c > prev_candle_vwap and curr_c < vwap and ema9 <= ema21:
                evaluated_signals.append({"strategy": "VWAP Rejection", "direction": "SELL", "weight": 22})

        # Strategy G: EMA 9/21 Trend Momentum
        if ema9 > ema21 and curr_c > ema9 and 52 <= rsi <= 76:
            evaluated_signals.append({"strategy": "EMA Momentum", "direction": "BUY", "weight": 18})
        elif ema9 < ema21 and curr_c < ema9 and 24 <= rsi <= 48:
            evaluated_signals.append({"strategy": "EMA Breakdown", "direction": "SELL", "weight": 18})

        # Strategy H: RVOL Institutional Breakout
        if rvol >= 2.0 and abs(curr_c - curr_o) / curr_c >= 0.008:
            evaluated_signals.append({"strategy": "RVOL Anomaly Breakout", "direction": "BUY" if curr_c > curr_o else "SELL", "weight": 25})

        # Strategy I: Flat-Open Sudden Momentum / Anomaly Engine
        if anomaly_info and anomaly_info.get("is_flat_open") and anomaly_info.get("is_sudden_expansion"):
            if anomaly_info.get("direction") == "BULLISH" and curr_c > vwap:
                evaluated_signals.append({"strategy": "Flat-Open Sudden Momentum", "direction": "BUY", "weight": 26})
            elif anomaly_info.get("direction") == "BEARISH" and curr_c < vwap:
                evaluated_signals.append({"strategy": "Flat-Open Sudden Momentum", "direction": "SELL", "weight": 26})
        elif abs(gap_pct) <= 0.65 and rvol >= 1.6:
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

        # Strategy P: Dedicated Momentum Ignition Engine
        from engine.momentum_ignition_engine import MomentumIgnitionEngine
        ignition_info = MomentumIgnitionEngine.detect_ignition(df, vwap=vwap)
        if ignition_info.get("is_ignited"):
            ign_dir = ignition_info.get("direction")
            ign_score = ignition_info.get("ignition_score", 65)
            # Assign proportionate weight based on ignition intensity
            ign_weight = min(28, max(20, int(ign_score * 0.3)))
            evaluated_signals.append({"strategy": "Momentum Ignition Surge", "direction": ign_dir, "weight": ign_weight})

        # Apply Persistent Multi-Dimensional Strategy Learning Multipliers (Calibrated strictly when N >= 15 observations)
        from engine.strategy_learning_engine import StrategyLearningEngine
        reg_type = market_regime_info.get("regime") if market_regime_info else None
        
        # Compute IST time bucket for current bar
        current_time_bucket = None
        try:
            import pytz
            last_ts = df.index[-1]
            if hasattr(last_ts, 'tz') and last_ts.tz is None:
                ist_ts = last_ts.tz_localize('Asia/Kolkata')
            else:
                ist_ts = last_ts
            current_time_bucket = StrategyLearningEngine.get_time_bucket(ist_ts.hour, ist_ts.minute)
        except Exception:
            current_time_bucket = "OPENING_MOMENTUM"

        for sig in evaluated_signals:
            strat_name = sig["strategy"]
            cutoff_ts = as_of_ts if as_of_ts is not None else (df.index[-1] if df is not None and not df.empty else None)
            mult = StrategyLearningEngine.get_combined_multiplier(
                strategy=strat_name,
                market_regime=reg_type,
                time_bucket=current_time_bucket,
                as_of_ts=cutoff_ts
            )
            sig["weight"] = int(sig["weight"] * mult)

        # Macro Market Regime Alignment Boost / Penalty
        macro_penalty_buy = 0
        macro_penalty_sell = 0
        if market_regime_info:
            from engine.market_regime_engine import MarketRegimeEngine
            stock_change_pct = ((curr_c - today_open) / today_open) * 100 if today_open else 0.0
            stock_vwap_diff = curr_c - vwap
            mkt_resp = MarketRegimeEngine.calculate_stock_response_to_market(
                stock_change=stock_change_pct,
                stock_vwap_diff=stock_vwap_diff,
                market_regime_info=market_regime_info,
                sector_bias=sector_bias
            )
            boost = mkt_resp.get("confluence_boost", 0)
            if boost > 0:
                target_dir = "BUY" if stock_change_pct > 0 else "SELL"
                evaluated_signals.append({"strategy": "Market-Wide Macro Confluence", "direction": target_dir, "weight": boost})
            elif boost < 0:
                # Stock is fighting against strong macro regime: apply directional penalty
                if market_regime_info.get("bias") == "BEARISH":
                    # Strong bearish macro: penalize buy setups
                    macro_penalty_buy = abs(boost)
                elif market_regime_info.get("bias") == "BULLISH":
                    # Strong bullish macro: penalize short setups
                    macro_penalty_sell = abs(boost)

        # Rank strategies by weight/evidence
        buy_signals = [s for s in evaluated_signals if s["direction"] == "BUY"]
        sell_signals = [s for s in evaluated_signals if s["direction"] == "SELL"]

        # Apply macro penalty with sensible floor
        raw_buy_score = sum(s["weight"] for s in buy_signals)
        raw_sell_score = sum(s["weight"] for s in sell_signals)
        buy_score = max(0, min(100, raw_buy_score - macro_penalty_buy))
        sell_score = max(0, min(100, raw_sell_score - macro_penalty_sell))

        # Sort strategies by weight descending so strongest strategy is primary
        buy_signals.sort(key=lambda x: x["weight"], reverse=True)
        sell_signals.sort(key=lambda x: x["weight"], reverse=True)

        # Pre-calculate candidate directional structural Stop-Loss
        candidate_dir = "BUY" if (buy_score >= 50 and buy_score > sell_score) else ("SELL" if (sell_score >= 50 and sell_score > buy_score) else "NEUTRAL")
        candidate_sl = None
        if candidate_dir == "BUY":
            raw_sl = vwap * 0.996 if curr_c > vwap else (curr_c - (atr14 * 1.5))
            candidate_sl = round(min(raw_sl, curr_c - (atr14 * 1.2)), 2)
            if candidate_sl >= curr_c:
                candidate_sl = round(curr_c - (atr14 * 1.5), 2)
        elif candidate_dir == "SELL":
            raw_sl = vwap * 1.004 if curr_c < vwap else (curr_c + (atr14 * 1.5))
            candidate_sl = round(max(raw_sl, curr_c + (atr14 * 1.2)), 2)
            if candidate_sl <= curr_c:
                candidate_sl = round(curr_c + (atr14 * 1.5), 2)

        # DEDICATED OPPORTUNITY POTENTIAL & OBSTACLE CLEARANCE EVALUATION WITH ACTUAL STOP LOSS AND DEDICATED DAILY DF
        from engine.opportunity_potential_engine import OpportunityPotentialEngine
        opp_eval = OpportunityPotentialEngine.evaluate_potential(
            df=df,
            current_price=curr_c,
            direction=candidate_dir,
            prev_close=prev_close,
            vwap=vwap,
            supply_obs=fresh_bear_obs,
            demand_obs=fresh_bull_obs,
            stop_loss=candidate_sl,
            daily_df=daily_df,
            orb_levels=shared_orb_levels
        )

        if buy_score >= 50 and buy_score > sell_score and opp_eval["is_viable"]:
            direction = "BUY"
            composite_score = buy_score
            primary_strategy = buy_signals[0]["strategy"]
            triggered = [s["strategy"] for s in buy_signals]
            sl = candidate_sl
            tp1 = opp_eval["tp1"]
            tp2 = opp_eval["tp2"]

        elif sell_score >= 50 and sell_score > buy_score and opp_eval["is_viable"]:
            direction = "SELL"
            composite_score = sell_score
            primary_strategy = sell_signals[0]["strategy"]
            triggered = [s["strategy"] for s in sell_signals]
            sl = candidate_sl
            tp1 = opp_eval["tp1"]
            tp2 = opp_eval["tp2"]

        else:
            direction = "NEUTRAL"
            composite_score = max(buy_score, sell_score)
            if not opp_eval["is_viable"] and (buy_score >= 50 or sell_score >= 50):
                primary_strategy = f"WAIT FOR ENTRY ({opp_eval['reason']})"
            else:
                primary_strategy = "WAIT FOR ENTRY (NO TRADE)"
            triggered = [s["strategy"] for s in (buy_signals + sell_signals)]
            sl = None
            tp1 = None
            tp2 = None

        # Compute Win Probability from statistical performance database
        from core.storage import get_strategy_performance_stats
        cutoff_ts = as_of_ts if as_of_ts is not None else (df.index[-1] if df is not None and not df.empty else None)
        perf_list = get_strategy_performance_stats(as_of_ts=cutoff_ts)
        strat_stat = next((s for s in perf_list if s.get("strategy") == primary_strategy), None)
        if strat_stat and strat_stat.get("total_signals", 0) >= 5:
            win_prob = strat_stat.get("win_rate", 50.0)
        else:
            # Baseline probability modeled as bounded function of composite score
            win_prob = round(min(88.0, max(45.0, 40.0 + (composite_score * 0.48))), 1) if direction != "NEUTRAL" else 50.0

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
            "probability_pct": win_prob,
            "primary_strategy": primary_strategy,
            "triggered_strategies": triggered,
            "suggested_sl": sl,
            "suggested_tp1": tp1,
            "suggested_tp2": tp2,
            "is_viable": opp_eval.get("is_viable", False),
            "opportunity_eval": opp_eval,
            "confidence_penalty": opp_eval.get("confidence_penalty", 0),
            "daily_atr_source": opp_eval.get("daily_atr_source", "NONE"),
            "potential_pct": opp_eval.get("potential_pct", 0.0),
            "potential_tp2_pct": opp_eval.get("potential_tp2_pct", 0.0),
            "nearest_obstacle": opp_eval.get("nearest_obstacle", "None"),
            "exhaustion_risk": opp_eval.get("exhaustion_risk", "LOW"),
            "opportunity_reason": opp_eval.get("reason", ""),
            "nearest_demand_ob": near_demand_ob,
            "nearest_supply_ob": near_supply_ob,
            "active_demand_obs": fresh_bull_obs,
            "active_supply_obs": fresh_bear_obs,
            "fully_mitigated_demand_obs": fully_mitigated_bull_obs,
            "fully_mitigated_supply_obs": fully_mitigated_bear_obs,
            "swing_high_records": swing_high_records,
            "swing_low_records": swing_low_records,
            "or_high": or_high,
            "or_low": or_low,
            "prev_close": prev_close
        }
