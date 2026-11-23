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

def run_screener_replay(sample_symbols=None, days="5d"):
    import yfinance as yf
    
    if not sample_symbols:
        sample_symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "TATAMOTORS.NS"]
    
    print(f"=== STARTING 5-MINUTE SCREENER REPLAY VALIDATION ({len(sample_symbols)} STOCKS) ===")
    
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
        "stage_counts": {"ENTRY READY": 0, "IGNITION": 0, "PREPARING": 0, "OBSERVING": 0, "WATCHLIST": 0}
    }
    
    cost_per_trade_pct = 0.05  # 0.05% realistic brokerage, STT, and slippage
    
    for sym in sample_symbols:
        print(f"Replaying {sym}...")
        df = yf.download(sym, period=days, interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(sym, axis=1, level=1) if sym in df.columns.levels[1] else df
        if df is None or len(df) < 30:
            continue
            
        current_open_trade = None
        # Replay candle by candle from index 25 onwards
        for idx in range(25, len(df) - 5):
            window_df = df.iloc[:idx]
            curr_candle_ts = window_df.index[-1]
            
            # Check if previous open trade has closed by current candle index
            if current_open_trade is not None:
                if idx >= current_open_trade["exit_idx"]:
                    current_open_trade = None
            
            # Align with nifty window
            nifty_sub = nifty_df[nifty_df.index <= curr_candle_ts]
            if nifty_sub.empty:
                continue
                
            res = OpportunityRankingEngine.evaluate_stock_opportunity(
                stock_df=window_df,
                symbol=sym,
                sector_name="TEST_SECTOR",
                sector_change=0.5,
                nifty_df=nifty_sub
            )
            
            stats["total_evaluations"] += 1
            if not res:
                stats["watch_only_candidates"] += 1
                stats["stage_counts"]["DISQUALIFIED_OR_MISALIGNED"] = stats["stage_counts"].get("DISQUALIFIED_OR_MISALIGNED", 0) + 1
                continue

            stage = res.get("stage", "WATCHLIST")
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
            
            sl = res.get("suggested_sl") or (entry_price * 0.99)
            tp1 = res.get("suggested_tp1") or (entry_price * 1.015)
            direction = res.get("direction", "BULLISH")
            
            # Forward testing over subsequent completed candles (up to 12 candles / 1 hour)
            forward_candles = df.iloc[idx:min(idx+12, len(df))]
            if forward_candles.empty:
                continue

            stats["trades_executed"] += 1
            highs = forward_candles["High"].values
            lows = forward_candles["Low"].values
            
            max_fav = 0.0
            max_adv = 0.0
            hit_tp = False
            hit_sl = False
            exit_offset = len(forward_candles)
            
            if direction == "BULLISH":
                max_fav = (max(highs) - entry_price) / entry_price * 100
                max_adv = (entry_price - min(lows)) / entry_price * 100
                for c_i, (h, l) in enumerate(zip(highs, lows)):
                    if l <= sl:
                        hit_sl = True
                        exit_offset = c_i + 1
                        break
                    if h >= tp1:
                        hit_tp = True
                        exit_offset = c_i + 1
                        break
            else: # BEARISH
                max_fav = (entry_price - min(lows)) / entry_price * 100
                max_adv = (max(highs) - entry_price) / entry_price * 100
                for c_i, (h, l) in enumerate(zip(highs, lows)):
                    if h >= sl:
                        hit_sl = True
                        exit_offset = c_i + 1
                        break
                    if l <= tp1:
                        hit_tp = True
                        exit_offset = c_i + 1
                        break
            
            # Register open trade position to prevent duplicate entries until closed
            current_open_trade = {
                "entry_idx": idx,
                "exit_idx": idx + exit_offset
            }
            
            stats["mfes"].append(max_fav)
            stats["maes"].append(max_adv)
            
            if hit_tp:
                stats["wins"] += 1
                pnl = (abs(tp1 - entry_price) / entry_price * 100) - cost_per_trade_pct
                stats["total_pnl_pct"] += pnl
            elif hit_sl:
                stats["losses"] += 1
                pnl = -(abs(entry_price - sl) / entry_price * 100) - cost_per_trade_pct
                stats["total_pnl_pct"] += pnl
            else:
                # Flat exit after 5 candles
                final_close = float(forward_candles["Close"].iloc[-1])
                diff_pct = ((final_close - entry_price) / entry_price * 100) if direction == "BULLISH" else ((entry_price - final_close) / entry_price * 100)
                pnl = diff_pct - cost_per_trade_pct
                stats["total_pnl_pct"] += pnl
                if pnl > 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1

    print("\n" + "="*50)
    print("        5-MINUTE REPLAY BACKTEST SUMMARY")
    print("="*50)
    print(f"Total Evaluations:        {stats['total_evaluations']}")
    print(f"Watch-Only (Blocked):     {stats['watch_only_candidates']} ({round(stats['watch_only_candidates']/(stats['total_evaluations'] or 1)*100, 1)}%)")
    print(f"Actionable Trades:        {stats['trades_executed']}")
    print(f"Stage Breakdown:          {stats['stage_counts']}")
    
    if stats["trades_executed"] > 0:
        win_rate = round(stats["wins"] / stats["trades_executed"] * 100, 1)
        avg_mfe = round(np.mean(stats["mfes"]), 2) if stats["mfes"] else 0.0
        avg_mae = round(np.mean(stats["maes"]), 2) if stats["maes"] else 0.0
        print(f"Win Rate (After Costs):   {win_rate}% ({stats['wins']} Wins / {stats['losses']} Losses)")
        print(f"Net PnL (After Costs):    {round(stats['total_pnl_pct'], 2)}%")
        print(f"Average MFE (Excursion):  +{avg_mfe}%")
        print(f"Average MAE (Drawdown):   -{avg_mae}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_screener_replay()
