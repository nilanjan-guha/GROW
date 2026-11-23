"""
Historical 5-Minute Intraday Screener Replay & Backtest Engine
Uses the exact same ScreenerPipeline as the live scanner:
- Zero manufactured risk levels: trades require valid, structural SL/TP and RR >= 1.20 from strategy evaluation
- Next-candle Open execution with entry slippage and realistic Indian transaction costs (0.10%)
- MFE/MAE calculated strictly up to actual exit candle
- Explicit holding horizon: max 6 candles (30 minutes)
- Non-overlapping distinct directional movement events (bullish and bearish separated)
- Direction-matched recall and genuine lead-time measurement
- Statistical Precision, Recall, and False Positive Rate across all market setups
"""
import sys
import os
import time
import math
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.screener_pipeline import ScreenerPipeline

def run_screener_replay(sample_symbols=None, days="1mo"):
    import yfinance as yf
    
    if not sample_symbols:
        # Broad multi-sector liquid F&O universe
        sample_symbols = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
            "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
            "AXISBANK.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS"
        ]
    
    print(f"=== STARTING 5-MINUTE SCREENER REPLAY VALIDATION ({len(sample_symbols)} STOCKS | PERIOD: {days}) ===")
    
    # 1. Download benchmark NIFTY 50 5m candles
    print("Downloading NIFTY 50 5m candles...")
    nifty_df = yf.download("^NSEI", period=days, interval="5m", progress=False)
    if isinstance(nifty_df.columns, pd.MultiIndex):
        nifty_df = nifty_df.xs("^NSEI", axis=1, level=1) if "^NSEI" in nifty_df.columns.levels[1] else nifty_df
    
    stats = {
        "total_evaluations": 0,
        "watch_only_candidates": 0,
        "trades_executed": 0,
        "rejected_no_risk": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl_pct": 0.0,
        "mfes": [],
        "maes": [],
        "stage_counts": {"ENTRY READY": 0, "IGNITION": 0, "PREPARING": 0, "OBSERVING": 0, "MONITORING": 0},
        "bullish_events": 0,
        "bearish_events": 0,
        "detected_bullish_events": 0,
        "detected_bearish_events": 0,
        "bullish_lead_times": [],
        "bearish_lead_times": [],
        "in_sample_wins": 0,
        "in_sample_trades": 0,
        "out_of_sample_wins": 0,
        "out_of_sample_trades": 0
    }
    
    entry_slippage_pct = 0.03
    total_friction_pct = 0.10
    max_holding_candles = 6

    for sym in sample_symbols:
        print(f"Replaying {sym}...")
        df = yf.download(sym, period=days, interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(sym, axis=1, level=1) if sym in df.columns.levels[1] else df
        if df is None or len(df) < 30:
            continue
            
        # 1. Identify distinct, non-overlapping Bullish and Bearish movement events
        # A bullish event: subsequent 12 candles produce high >= +1.25%
        # A bearish event: subsequent 12 candles produce low <= -1.25%
        bullish_event_indices = []
        bearish_event_indices = []
        skip_bull_until = 0
        skip_bear_until = 0
        closes = df["Close"].values
        highs_all = df["High"].values
        lows_all = df["Low"].values

        for e_i in range(25, len(df) - 12):
            c_base = closes[e_i]
            if c_base <= 0:
                continue
            
            # Check Bullish Event
            if e_i >= skip_bull_until:
                max_h = np.max(highs_all[e_i:e_i+12])
                if ((max_h - c_base) / c_base * 100) >= 1.25:
                    bullish_event_indices.append(e_i)
                    skip_bull_until = e_i + 12

            # Check Bearish Event
            if e_i >= skip_bear_until:
                min_l = np.min(lows_all[e_i:e_i+12])
                if ((c_base - min_l) / c_base * 100) >= 1.25:
                    bearish_event_indices.append(e_i)
                    skip_bear_until = e_i + 12

        # Download 1mo Daily OHLC for true Wilder Daily ATR14 calculation point-in-time
        daily_df = yf.download(sym, period="3mo", interval="1d", progress=False)
        if isinstance(daily_df.columns, pd.MultiIndex):
            daily_df = daily_df.xs(sym, axis=1, level=1) if sym in daily_df.columns.levels[1] else daily_df

        scanned_history = {}
        current_open_trade = None

        # Replay candle by candle from index 25 onwards
        for idx in range(25, len(df) - 5):
            window_df = df.iloc[:idx]
            # The signal candle is explicitly the completed last candle of window_df (idx - 1)
            signal_candle_idx = idx - 1
            signal_candle_ts = window_df.index[-1]
            
            if current_open_trade is not None:
                if idx >= current_open_trade["exit_idx"]:
                    current_open_trade = None
            
            nifty_sub = nifty_df[nifty_df.index <= signal_candle_ts]
            if nifty_sub.empty:
                continue

            # Point-in-time Daily context
            daily_sub = daily_df[daily_df.index.date < signal_candle_ts.date()] if daily_df is not None and not daily_df.empty else None

            # Point-in-time Sector relative change against benchmark
            stock_day_open = float(window_df[window_df.index.date == signal_candle_ts.date()]["Open"].iloc[0]) if len(window_df[window_df.index.date == signal_candle_ts.date()]) > 0 else float(window_df["Open"].iloc[0])
            stock_change_pt = round((float(window_df["Close"].iloc[-1]) - stock_day_open) / stock_day_open * 100, 2)
            nifty_day_open = float(nifty_sub[nifty_sub.index.date == signal_candle_ts.date()]["Open"].iloc[0]) if len(nifty_sub[nifty_sub.index.date == signal_candle_ts.date()]) > 0 else float(nifty_sub["Open"].iloc[0])
            nifty_change_pt = round((float(nifty_sub["Close"].iloc[-1]) - nifty_day_open) / nifty_day_open * 100, 2) if nifty_day_open > 0 else 0.0
            
            # Point-in-time Sector Bias
            sector_bias = "BULLISH" if stock_change_pt > nifty_change_pt + 0.3 else ("BEARISH" if stock_change_pt < nifty_change_pt - 0.3 else "NEUTRAL")

            # Complete live-equivalent pipeline with point-in-time historical context
            pipeline_res = ScreenerPipeline.evaluate_candidate_unified(
                stock_df=window_df,
                symbol=sym,
                sector_change=stock_change_pt,
                sector_bias=sector_bias,
                nifty_df=nifty_sub,
                daily_df=daily_sub,
                is_replay=True
            )
            
            stats["total_evaluations"] += 1
            if not pipeline_res:
                stats["watch_only_candidates"] += 1
                scanned_history[signal_candle_idx] = {"stage": "DISQUALIFIED", "direction": "NEUTRAL", "ts": signal_candle_ts}
                continue

            opp_res = pipeline_res.get("opp_res")
            if not opp_res:
                stats["watch_only_candidates"] += 1
                scanned_history[signal_candle_idx] = {"stage": "DISQUALIFIED", "direction": "NEUTRAL", "ts": signal_candle_ts}
                continue

            stage = opp_res.get("stage", "WATCHLIST")
            direction = opp_res.get("direction", "NEUTRAL")
            is_actionable = opp_res.get("is_actionable", False)
            
            stats["stage_counts"][stage] = stats["stage_counts"].get(stage, 0) + 1
            scanned_history[signal_candle_idx] = {"stage": stage, "direction": direction, "ts": signal_candle_ts}

            if not is_actionable or direction == "NEUTRAL":
                stats["watch_only_candidates"] += 1
                continue

            # Deduplication
            if current_open_trade is not None:
                continue

            # ZERO MANUFACTURED RISK: Require genuine structural risk levels from institutional evaluation!
            sl = opp_res.get("suggested_sl")
            tp1 = opp_res.get("suggested_tp1")
            entry_price = float(df["Open"].iloc[idx])
            
            if sl is None or tp1 is None:
                stats["rejected_no_risk"] += 1
                stats["watch_only_candidates"] += 1
                continue

            # Validate structural boundaries
            if direction == "BULLISH":
                if not (sl < entry_price and tp1 > entry_price):
                    stats["rejected_no_risk"] += 1
                    stats["watch_only_candidates"] += 1
                    continue
                risk = entry_price - sl
                reward = tp1 - entry_price
            else:
                if not (sl > entry_price and tp1 < entry_price):
                    stats["rejected_no_risk"] += 1
                    stats["watch_only_candidates"] += 1
                    continue
                risk = sl - entry_price
                reward = entry_price - tp1

            if risk <= 0 or (reward / risk) < 1.20:
                stats["rejected_no_risk"] += 1
                stats["watch_only_candidates"] += 1
                continue

            # Executable trade verified
            stats["trades_executed"] += 1
            actual_entry = entry_price * (1.0 + entry_slippage_pct/100) if direction == "BULLISH" else entry_price * (1.0 - entry_slippage_pct/100)
            
            forward_candles = df.iloc[idx:min(idx + max_holding_candles, len(df))]
            if forward_candles.empty:
                continue

            opens = forward_candles["Open"].values
            highs = forward_candles["High"].values
            lows = forward_candles["Low"].values
            closes_sub = forward_candles["Close"].values
            
            hit_tp = False
            hit_sl = False
            exit_offset = len(forward_candles)
            exit_price_actual = float(closes_sub[-1])
            in_trade_highs = []
            in_trade_lows = []

            for c_i, (c_open, h, l, c_close) in enumerate(zip(opens, highs, lows, closes_sub)):
                in_trade_highs.append(h)
                in_trade_lows.append(l)
                
                if direction == "BULLISH":
                    if c_open <= sl:
                        hit_sl = True
                        exit_price_actual = c_open
                        exit_offset = c_i + 1
                        break
                    elif l <= sl:
                        hit_sl = True
                        exit_price_actual = sl - (sl * 0.0003)
                        exit_offset = c_i + 1
                        break
                    elif c_open >= tp1:
                        hit_tp = True
                        exit_price_actual = c_open
                        exit_offset = c_i + 1
                        break
                    elif h >= tp1:
                        hit_tp = True
                        exit_price_actual = tp1
                        exit_offset = c_i + 1
                        break
                else:  # BEARISH
                    if c_open >= sl:
                        hit_sl = True
                        exit_price_actual = c_open
                        exit_offset = c_i + 1
                        break
                    elif h >= sl:
                        hit_sl = True
                        exit_price_actual = sl + (sl * 0.0003)
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

            # In-trade MFE & MAE strictly up to exit
            if direction == "BULLISH":
                max_fav = (max(in_trade_highs) - actual_entry) / actual_entry * 100
                max_adv = (actual_entry - min(in_trade_lows)) / actual_entry * 100
            else:
                max_fav = (actual_entry - min(in_trade_lows)) / actual_entry * 100
                max_adv = (max(in_trade_highs) - actual_entry) / actual_entry * 100

            stats["mfes"].append(max_fav)
            stats["maes"].append(max_adv)
            
            current_open_trade = {"entry_idx": idx, "exit_idx": idx + exit_offset}
            is_out_of_sample = (idx >= int(len(df) * 0.60))
            is_win = False

            if hit_tp:
                stats["wins"] += 1
                is_win = True
                diff = (exit_price_actual - actual_entry) / actual_entry * 100 if direction == "BULLISH" else (actual_entry - exit_price_actual) / actual_entry * 100
                stats["total_pnl_pct"] += (diff - total_friction_pct)
            elif hit_sl:
                stats["losses"] += 1
                diff = -(abs(actual_entry - exit_price_actual) / actual_entry * 100)
                stats["total_pnl_pct"] += (diff - total_friction_pct)
            else:
                diff = ((exit_price_actual - actual_entry) / actual_entry * 100) if direction == "BULLISH" else ((actual_entry - exit_price_actual) / actual_entry * 100)
                pnl = diff - total_friction_pct
                stats["total_pnl_pct"] += pnl
                if pnl > 0:
                    stats["wins"] += 1
                    is_win = True
                else:
                    stats["losses"] += 1

            if is_out_of_sample:
                stats["out_of_sample_trades"] += 1
                if is_win: stats["out_of_sample_wins"] += 1
            else:
                stats["in_sample_trades"] += 1
                if is_win: stats["in_sample_wins"] += 1

        # Evaluate Direction-Matched Pre-Move Event Recall & Lead Times
        for b_idx in bullish_event_indices:
            stats["bullish_events"] += 1
            for lead in range(1, 7):
                c_idx = b_idx - lead
                entry = scanned_history.get(c_idx, {})
                if entry.get("direction") == "BULLISH" and entry.get("stage") in ("ENTRY READY", "IGNITION", "PREPARING", "OBSERVING"):
                    stats["detected_bullish_events"] += 1
                    stats["bullish_lead_times"].append(lead)
                    break

        for s_idx in bearish_event_indices:
            stats["bearish_events"] += 1
            for lead in range(1, 7):
                c_idx = s_idx - lead
                entry = scanned_history.get(c_idx, {})
                if entry.get("direction") == "BEARISH" and entry.get("stage") in ("ENTRY READY", "IGNITION", "PREPARING", "OBSERVING"):
                    stats["detected_bearish_events"] += 1
                    stats["bearish_lead_times"].append(lead)
                    break

    print("\n" + "="*55)
    print("        5-MINUTE SCREENER REPLAY REPORT")
    print("="*55)
    print(f"Total Candidate Evaluations: {stats['total_evaluations']}")
    print(f"Watch-Only (Quality Gated):  {stats['watch_only_candidates']} ({round(stats['watch_only_candidates']/(stats['total_evaluations'] or 1)*100, 1)}%)")
    print(f"Executed Actionable Trades:  {stats['trades_executed']}")
    print(f"Signals Rejected (No Risk):  {stats['rejected_no_risk']}")
    print(f"Stage Breakdown:             {stats['stage_counts']}")
    
    # Event Recall (Direction-Matched)
    total_events = stats["bullish_events"] + stats["bearish_events"]
    total_detected = stats["detected_bullish_events"] + stats["detected_bearish_events"]
    if total_events > 0:
        recall_pct = round(total_detected / total_events * 100, 1)
        bull_recall = round(stats["detected_bullish_events"] / (stats["bullish_events"] or 1) * 100, 1)
        bear_recall = round(stats["detected_bearish_events"] / (stats["bearish_events"] or 1) * 100, 1)
        avg_lead_min = round(np.mean(stats["bullish_lead_times"] + stats["bearish_lead_times"]) * 5, 1) if (stats["bullish_lead_times"] + stats["bearish_lead_times"]) else 0.0
        
        print("\n--- DIRECTION-MATCHED EVENT RECALL ---")
        print(f"Distinct Market Events:      {total_events} ({stats['bullish_events']} Bullish / {stats['bearish_events']} Bearish)")
        print(f"Pre-Move Recall (Matched):   {recall_pct}% ({total_detected}/{total_events} detected before expansion)")
        print(f"  • Bullish Recall:          {bull_recall}% ({stats['detected_bullish_events']}/{stats['bullish_events']})")
        print(f"  • Bearish Recall:          {bear_recall}% ({stats['detected_bearish_events']}/{stats['bearish_events']})")
        print(f"Average Pre-Move Lead Time:  {avg_lead_min} minutes before start")

    # Trade Quality & Precision
    if stats["trades_executed"] > 0:
        win_rate = round(stats["wins"] / stats["trades_executed"] * 100, 1)
        expansions = sum(1 for mfe in stats["mfes"] if mfe >= 0.70)
        expansion_rate = round(expansions / stats["trades_executed"] * 100, 1)
        non_expansions = len(stats["mfes"]) - expansions
        fpr_expansion = round(non_expansions / stats["trades_executed"] * 100, 1)
        avg_mfe = round(np.mean(stats["mfes"]), 2) if stats["mfes"] else 0.0
        avg_mae = round(np.mean(stats["maes"]), 2) if stats["maes"] else 0.0
        
        print("\n--- TRADE EXECUTION & PERFORMANCE (AFTER 0.10% COSTS) ---")
        print(f"Win Rate:                    {win_rate}% ({stats['wins']} Wins / {stats['losses']} Losses)")
        print(f"Setup Expansion Rate (MFE):  {expansion_rate}% (MFE >= 0.70%)")
        print(f"Non-Expanding Trigger Rate:  {fpr_expansion}%")
        print(f"Average MFE (In-Trade):      +{avg_mfe}%")
        print(f"Average MAE (In-Trade):      -{avg_mae}%")
        print(f"Net Realized PnL:            {round(stats['total_pnl_pct'], 2)}%")
        
        if stats["in_sample_trades"] > 0:
            is_wr = round(stats["in_sample_wins"] / stats["in_sample_trades"] * 100, 1)
            print(f"In-Sample Win Rate (60%):    {is_wr}% ({stats['in_sample_wins']}/{stats['in_sample_trades']})")
        if stats["out_of_sample_trades"] > 0:
            oos_wr = round(stats["out_of_sample_wins"] / stats["out_of_sample_trades"] * 100, 1)
            print(f"Out-of-Sample Win Rate (40%):{oos_wr}% ({stats['out_of_sample_wins']}/{stats['out_of_sample_trades']})")
    print("="*55 + "\n")

if __name__ == "__main__":
    run_screener_replay()
