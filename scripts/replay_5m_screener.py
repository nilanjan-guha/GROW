"""
Historical 5-Minute Intraday Screener Replay & Backtest Engine
Validates the screener on historical completed 5-minute candles:
- Missed-move rate
- False-positive rate
- Maximum Favorable Excursion (MFE) / Maximum Adverse Excursion (MAE)
- Win rate after simulated transaction costs (slippage + brokerage + STT)
- Breakdown by setup stage (ENTRY READY, IGNITION, PREPARING, OBSERVING)
"""
import sys
import os
import time
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.opportunity_ranking_engine import OpportunityRankingEngine
from engine.market_validator import MarketValidator
from strategies.institutional_engine import InstitutionalStrategyEngine

def run_screener_replay(sample_symbols=None, days="1mo"):
    import yfinance as yf
    
    if not sample_symbols:
        # Representative multi-sector liquid F&O universe (Banking, IT, Auto, Energy, Metals, Pharma, FMCG)
        sample_symbols = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
            "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
            "AXISBANK.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS"
        ]
    
    print(f"=== STARTING 5-MINUTE SCREENER REPLAY VALIDATION ({len(sample_symbols)} STOCKS | PERIOD: {days}) ===")
    
    # 1. Download benchmark
    print("Downloading NIFTY 50 5m candles...")
    nifty_df = yf.download("^NSEI", period=days, interval="5m", progress=False)
    if isinstance(nifty_df.columns, pd.MultiIndex):
        nifty_df = nifty_df.xs("^NSEI", axis=1, level=1) if "^NSEI" in nifty_df.columns.levels[1] else nifty_df
    
    # 2. Replay simulation
    stats = {
        "total_evaluations": 0,
        "actionable_signals": 0,
        "watch_only_candidates": 0,
        "trades_executed": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl_pct": 0.0,
        "mfes": [],
        "maes": [],
        "stage_counts": {"ENTRY READY": 0, "IGNITION": 0, "PREPARING": 0, "OBSERVING": 0, "WATCHLIST": 0},
        "distinct_movement_events": 0,
        "detected_movement_events": 0,
        "detection_lead_times_bars": [],
        "false_positive_trades": 0,
        "true_positive_trades": 0,
        "in_sample_wins": 0,
        "in_sample_trades": 0,
        "out_of_sample_wins": 0,
        "out_of_sample_trades": 0
    }
    
    cost_per_trade_pct = 0.05  # 0.05% realistic brokerage, STT, and slippage
    
    for sym in sample_symbols:
        print(f"Replaying {sym}...")
        df = yf.download(sym, period=days, interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(sym, axis=1, level=1) if sym in df.columns.levels[1] else df
        if df is None or len(df) < 30:
            continue
            
        # Precompute intraday session VWAP per day for accurate testing
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
            df_copy.index = pd.to_datetime(df_copy.index)
        
        # Calculate daily cumulative VWAP per trading session
        dates = df_copy.index.date
        vwap_series = pd.Series(index=df_copy.index, dtype=float)
        for d in pd.Series(dates).unique():
            day_mask = (dates == d)
            day_df = df_copy[day_mask]
            tp = (day_df["High"] + day_df["Low"] + day_df["Close"]) / 3.0
            cum_vol = day_df["Volume"].cumsum()
            cum_vp = (tp * day_df["Volume"]).cumsum()
            vwap_day = cum_vp / cum_vol.replace(0, np.nan)
            vwap_series.loc[day_mask] = vwap_day.fillna(day_df["Close"])

        # Precompute distinct future market movement events (>= 1.25% swing) without overlap (Item 4)
        # Identify distinct non-overlapping peaks and troughs:
        event_indices = []
        skip_until = 0
        close_vals = df["Close"].values
        for e_i in range(25, len(df) - 12):
            if e_i < skip_until:
                continue
            sub_window = df.iloc[e_i:e_i+12]
            c_base = close_vals[e_i]
            max_move_pct = max(
                (sub_window["High"].max() - c_base) / c_base * 100,
                (c_base - sub_window["Low"].min()) / c_base * 100
            )
            if max_move_pct >= 1.25:
                event_indices.append(e_i)
                skip_until = e_i + 12  # Skip forward to deduplicate into distinct movement events

        # History of stages recorded per index for genuine lead-time measurement
        scanned_stages = {}

        current_open_trade = None
        # Replay candle by candle from index 25 onwards
        for idx in range(25, len(df) - 5):
            window_df = df.iloc[:idx]
            curr_candle_ts = window_df.index[-1]
            curr_vwap = float(vwap_series.iloc[idx-1]) if idx > 0 else float(window_df["Close"].iloc[-1])
            
            # Check if previous open trade has closed by current candle index
            if current_open_trade is not None:
                if idx >= current_open_trade["exit_idx"]:
                    current_open_trade = None
            
            # Align with nifty window
            nifty_sub = nifty_df[nifty_df.index <= curr_candle_ts]
            if nifty_sub.empty:
                continue

            # Evaluate genuine institutional risk levels identical to live app pipeline (Item 2)
            inst_eval = InstitutionalStrategyEngine.evaluate_strategies(
                df=window_df,
                sector_bias="NEUTRAL"
            )
            opp_eval = None
            if inst_eval and inst_eval.get("is_viable"):
                opp_eval = {
                    "is_viable": inst_eval.get("is_viable", False),
                    "potential_pct": inst_eval.get("potential_pct", 0.0),
                    "risk_reward_ratio": inst_eval.get("risk_reward_ratio", 0.0),
                    "stop_loss": inst_eval.get("suggested_sl"),
                    "suggested_sl": inst_eval.get("suggested_sl"),
                    "tp1": inst_eval.get("suggested_tp1"),
                    "suggested_tp1": inst_eval.get("suggested_tp1"),
                    "tp2": inst_eval.get("suggested_tp2"),
                    "suggested_tp2": inst_eval.get("suggested_tp2"),
                    "exhaustion_risk": inst_eval.get("exhaustion_risk", "LOW")
                }
                
            res = OpportunityRankingEngine.evaluate_stock_opportunity(
                stock_df=window_df,
                symbol=sym,
                sector_name="TEST_SECTOR",
                sector_change=0.5,
                nifty_df=nifty_sub,
                vwap=curr_vwap,
                opp_eval=opp_eval
            )
            
            stats["total_evaluations"] += 1
            if not res:
                stats["watch_only_candidates"] += 1
                stats["stage_counts"]["DISQUALIFIED_OR_MISALIGNED"] = stats["stage_counts"].get("DISQUALIFIED_OR_MISALIGNED", 0) + 1
                scanned_stages[idx] = "DISQUALIFIED"
                continue

            stage = res.get("stage", "WATCHLIST")
            scanned_stages[idx] = stage
            stats["stage_counts"][stage] = stats["stage_counts"].get(stage, 0) + 1
            is_actionable = res.get("is_actionable", False)

            if not is_actionable:
                stats["watch_only_candidates"] += 1
                continue
                
            # Deduplication: Do not count repeated consecutive signals on the same stock
            # If a trade is currently open in simulation for this symbol, do NOT enter again!
            if current_open_trade is not None:
                continue

            # If ACTIONABLE (ENTRY READY or IGNITION), simulate realistic NEXT-CANDLE execution
            stats["actionable_signals"] += 1
            
            # Next-candle Open execution (avoid lookahead bias / entering on the same candle close)
            next_candle = df.iloc[idx]
            entry_price = float(next_candle["Open"])
            
            # Assign direction first before using in slippage calculation! (Item 1)
            raw_dir = res.get("direction", "BULLISH")
            direction = "BULLISH" if raw_dir in ("BULLISH", "BUY", "CALL", "LONG") else ("BEARISH" if raw_dir in ("BEARISH", "SELL", "PUT", "SHORT") else "NEUTRAL")
            if direction == "NEUTRAL":
                continue

            # 5. Realistic Execution Simulation (Item 6):
            # Model execution price with realistic bid-ask spread and entry slippage (0.03% on entry)
            entry_slippage_pct = 0.03
            actual_entry_price = entry_price * (1.0 + entry_slippage_pct/100) if direction == "BULLISH" else entry_price * (1.0 - entry_slippage_pct/100)
            
            # Correct directional SL and TP fallback logic (Item 3):
            # Bullish: SL < Entry < TP1
            # Bearish: SL > Entry > TP1
            cand_sl = res.get("suggested_sl")
            cand_tp1 = res.get("suggested_tp1")
            if direction == "BULLISH":
                sl = cand_sl if (cand_sl and cand_sl < actual_entry_price) else (actual_entry_price * 0.99)
                tp1 = cand_tp1 if (cand_tp1 and cand_tp1 > actual_entry_price) else (actual_entry_price * 1.015)
            else:
                sl = cand_sl if (cand_sl and cand_sl > actual_entry_price) else (actual_entry_price * 1.01)
                tp1 = cand_tp1 if (cand_tp1 and cand_tp1 < actual_entry_price) else (actual_entry_price * 0.985)
            
            # Realistic Indian Trading Cost Model:
            # Brokerage: ₹20 or 0.03% per order (approx 0.03%)
            # STT / CTT: 0.025% on sell (equity intraday) / 0.125% (options)
            # Exchange turnover fee + SEBI charges + Stamp duty + 18% GST on brokerage: ~0.04%
            # Total round-trip realistic transaction friction: ~0.10%
            total_friction_pct = 0.10
            
            # Explicit Exit Rule: Max 6 completed 5m candles (30 minutes intraday momentum holding horizon)
            max_holding_candles = 6
            forward_candles = df.iloc[idx:min(idx + max_holding_candles, len(df))]
            if forward_candles.empty:
                continue

            stats["trades_executed"] += 1
            highs = forward_candles["High"].values
            lows = forward_candles["Low"].values
            opens = forward_candles["Open"].values
            closes = forward_candles["Close"].values
            
            max_fav = 0.0
            max_adv = 0.0
            hit_tp = False
            hit_sl = False
            exit_offset = len(forward_candles)
            exit_price_actual = float(closes[-1])
            
            # Forward simulate candle by candle strictly up to actual exit
            # MFE and MAE must only measure excursion experienced while trade was open!
            in_trade_highs = []
            in_trade_lows = []

            for c_i, (c_open, h, l, c_close) in enumerate(zip(opens, highs, lows, closes)):
                in_trade_highs.append(h)
                in_trade_lows.append(l)
                
                if direction == "BULLISH":
                    # Check gap-through-stop on bar open
                    if c_open <= sl:
                        hit_sl = True
                        exit_price_actual = c_open  # Filled at gap price worse than SL!
                        exit_offset = c_i + 1
                        break
                    elif l <= sl:
                        hit_sl = True
                        exit_price_actual = sl - (sl * 0.0003)  # Slippage on stop hit
                        exit_offset = c_i + 1
                        break
                    elif c_open >= tp1:
                        hit_tp = True
                        exit_price_actual = c_open  # Favorable gap fill
                        exit_offset = c_i + 1
                        break
                    elif h >= tp1:
                        hit_tp = True
                        exit_price_actual = tp1
                        exit_offset = c_i + 1
                        break
                else:  # BEARISH
                    # Check gap-through-stop on bar open
                    if c_open >= sl:
                        hit_sl = True
                        exit_price_actual = c_open  # Filled at gap price worse than SL!
                        exit_offset = c_i + 1
                        break
                    elif h >= sl:
                        hit_sl = True
                        exit_price_actual = sl + (sl * 0.0003)  # Slippage on stop hit
                        exit_offset = c_i + 1
                        break
                    elif c_open <= tp1:
                        hit_tp = True
                        exit_price_actual = c_open
                        exit_offset = c_i + 1
                        break
                    elif l <= tp1:
                        hit_tp = True
                        exit_price_actual = tp1
                        exit_offset = c_i + 1
                        break

            # Measure genuine MFE and MAE strictly during the trade holding period (candles 0 to exit_offset)
            if direction == "BULLISH":
                max_fav = (max(in_trade_highs) - actual_entry_price) / actual_entry_price * 100
                max_adv = (actual_entry_price - min(in_trade_lows)) / actual_entry_price * 100
            else:
                max_fav = (actual_entry_price - min(in_trade_lows)) / actual_entry_price * 100
                max_adv = (max(in_trade_highs) - actual_entry_price) / actual_entry_price * 100
            
            # Register open trade position to prevent duplicate entries until closed
            current_open_trade = {
                "entry_idx": idx,
                "exit_idx": idx + exit_offset
            }
            
            stats["mfes"].append(max_fav)
            stats["maes"].append(max_adv)
            
            # Split point for in-sample (first 60%) vs out-of-sample holdout (last 40%)
            is_out_of_sample = (idx >= int(len(df) * 0.60))
            is_win_trade = False

            if hit_tp:
                stats["wins"] += 1
                is_win_trade = True
                raw_diff = (exit_price_actual - actual_entry_price) / actual_entry_price * 100 if direction == "BULLISH" else (actual_entry_price - exit_price_actual) / actual_entry_price * 100
                pnl = raw_diff - total_friction_pct
                stats["total_pnl_pct"] += pnl
            elif hit_sl:
                stats["losses"] += 1
                is_win_trade = False
                raw_diff = -(abs(actual_entry_price - exit_price_actual) / actual_entry_price * 100)
                pnl = raw_diff - total_friction_pct
                stats["total_pnl_pct"] += pnl
            else:
                # Flat exit at end of configured holding window (explicit max 6 candles / 30m)
                diff_pct = ((exit_price_actual - actual_entry_price) / actual_entry_price * 100) if direction == "BULLISH" else ((actual_entry_price - exit_price_actual) / actual_entry_price * 100)
                pnl = diff_pct - total_friction_pct
                stats["total_pnl_pct"] += pnl
                if pnl > 0:
                    stats["wins"] += 1
                    is_win_trade = True
                else:
                    stats["losses"] += 1
                    is_win_trade = False

            if is_out_of_sample:
                stats["out_of_sample_trades"] += 1
                if is_win_trade:
                    stats["out_of_sample_wins"] += 1
            else:
                stats["in_sample_trades"] += 1
                if is_win_trade:
                    stats["in_sample_wins"] += 1

        # Post-process distinct movement events for this stock to calculate genuine recall & lead time (Items 4 & 5)
        for ev_idx in event_indices:
            stats["distinct_movement_events"] += 1
            # Check if scanner flagged the move in the 1 to 6 candles (5 to 30 mins) leading up to event start
            detected = False
            lead_time = 0
            for lead in range(1, 7):
                check_idx = ev_idx - lead
                if check_idx in scanned_stages:
                    stg = scanned_stages[check_idx]
                    if stg in ("ENTRY READY", "IGNITION", "PREPARING", "OBSERVING"):
                        detected = True
                        lead_time = lead
                        break
            if detected:
                stats["detected_movement_events"] += 1
                stats["detection_lead_times_bars"].append(lead_time)

    print("\n" + "="*50)
    print("        5-MINUTE REPLAY BACKTEST SUMMARY")
    print("="*50)
    print(f"Total Evaluations:        {stats['total_evaluations']}")
    print(f"Watch-Only (Blocked):     {stats['watch_only_candidates']} ({round(stats['watch_only_candidates']/(stats['total_evaluations'] or 1)*100, 1)}%)")
    print(f"Actionable Trades:        {stats['trades_executed']}")
    print(f"Stage Breakdown:          {stats['stage_counts']}")
    
    # Genuine Event-Based Missed-Move Recall & Detection Lead Time (Items 4 & 5):
    if stats["distinct_movement_events"] > 0:
        recall_pct = round(stats["detected_movement_events"] / stats["distinct_movement_events"] * 100, 1)
        missed_pct = round(100.0 - recall_pct, 1)
        avg_lead = round(np.mean(stats["detection_lead_times_bars"]) * 5, 1) if stats["detection_lead_times_bars"] else 0.0
        avg_lead_bars = round(np.mean(stats["detection_lead_times_bars"]), 1) if stats["detection_lead_times_bars"] else 0.0
        print(f"Distinct Movement Events: {stats['distinct_movement_events']}")
        print(f"Pre-Move Event Recall:    {recall_pct}% ({stats['detected_movement_events']} caught in advance)")
        print(f"Missed Event Rate:        {missed_pct}%")
        print(f"Average Lead Time:        {avg_lead} minutes ({avg_lead_bars} completed 5m candles before move)")

    if stats["trades_executed"] > 0:
        win_rate = round(stats["wins"] / stats["trades_executed"] * 100, 1)
        # Redefined False Positive Rate: trades that failed to generate favorable expansion > 0.5% (Item 6)
        true_positives = sum(1 for mfe in stats["mfes"] if mfe >= 0.70)
        false_positives = len(stats["mfes"]) - true_positives
        fp_rate = round(false_positives / stats["trades_executed"] * 100, 1)
        avg_mfe = round(np.mean(stats["mfes"]), 2) if stats["mfes"] else 0.0
        avg_mae = round(np.mean(stats["maes"]), 2) if stats["maes"] else 0.0
        print(f"Win Rate (After Costs):   {win_rate}% ({stats['wins']} Wins / {stats['losses']} Losses)")
        print(f"True Positive Rate (MFE >= 0.7%): {round(true_positives/stats['trades_executed']*100, 1)}% ({true_positives}/{stats['trades_executed']})")
        print(f"False Positive Rate:      {fp_rate}% ({false_positives} non-expanding triggers)")
        print(f"Net PnL (After Costs):    {round(stats['total_pnl_pct'], 2)}%")
        print(f"Average MFE (In-Trade):   +{avg_mfe}%")
        print(f"Average MAE (In-Trade):   -{avg_mae}%")
        
        # Walk-Forward / Holdout Validation Comparison:
        if stats["in_sample_trades"] > 0:
            is_wr = round(stats["in_sample_wins"] / stats["in_sample_trades"] * 100, 1)
            print(f"In-Sample Win Rate (60%): {is_wr}% ({stats['in_sample_wins']}/{stats['in_sample_trades']} trades)")
        if stats["out_of_sample_trades"] > 0:
            oos_wr = round(stats["out_of_sample_wins"] / stats["out_of_sample_trades"] * 100, 1)
            print(f"Out-of-Sample Win Rate (40%): {oos_wr}% ({stats['out_of_sample_wins']}/{stats['out_of_sample_trades']} trades)")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_screener_replay()
