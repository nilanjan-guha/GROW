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
        "gross_pnl_pct": 0.0,
        "total_pnl_pct": 0.0,
        "trade_net_pnls": [],
        "trade_gross_pnls": [],
        "trade_r_multiples": [],
        "equity_curve": [100.0],
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
    
    # Realistic Indian Intraday Equity/Futures Statutory Friction:
    # Brokerage (₹20/order or 0.03%), STT (0.025% on sell side for equity intraday / 0.0125% for futures),
    # Exchange turnover charge (NSE 0.00325%), GST (18% on brokerage+txn), Stamp duty (0.003% on buy), SEBI fee (0.0001%)
    # Total round-trip statutory friction: ~0.08% to 0.12% depending on side and turnover
    entry_slippage_pct = 0.03
    exit_slippage_pct = 0.03
    base_statutory_friction_pct = 0.08  # STT + Exchange + GST + Stamp Duty
    max_holding_candles = 6
    max_concurrent_portfolio_trades = 4  # Concurrency limit across shared portfolio capital

    # Pre-download and prepare all symbols data
    symbol_data = {}
    from engine.groww_sector_engine import GrowwSectorEngine
    sec_engine = GrowwSectorEngine()
    sec_engine.build_complete_fno_sector_universe_mapping(set(sample_symbols))

    for sym in sample_symbols:
        print(f"Loading data for {sym}...")
        df = yf.download(sym, period=days, interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(sym, axis=1, level=1) if sym in df.columns.levels[1] else df
        if df is None or len(df) < 30:
            continue

        # Timezone normalize
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")

        # 1. Identify distinct, non-overlapping Bullish and Bearish movement events
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
            if e_i >= skip_bull_until:
                max_h = np.max(highs_all[e_i:e_i+12])
                if ((max_h - c_base) / c_base * 100) >= 1.25:
                    bullish_event_indices.append(e_i)
                    skip_bull_until = e_i + 12

            if e_i >= skip_bear_until:
                min_l = np.min(lows_all[e_i:e_i+12])
                if ((c_base - min_l) / c_base * 100) >= 1.25:
                    bearish_event_indices.append(e_i)
                    skip_bear_until = e_i + 12

        # Daily OHLC for ATR14
        daily_df = yf.download(sym, period="3mo", interval="1d", progress=False)
        if isinstance(daily_df.columns, pd.MultiIndex):
            daily_df = daily_df.xs(sym, axis=1, level=1) if sym in daily_df.columns.levels[1] else daily_df

        stock_clean_sym = sym.replace(".NS", "").upper()
        norm_sector = sec_engine.full_universe_stock_map.get(stock_clean_sym, "GENERAL")
        sector_ticker = sec_engine.get_sector_index_ticker(norm_sector)
        sector_df = None
        try:
            sec_raw = yf.download(sector_ticker, period=days, interval="5m", progress=False)
            if not sec_raw.empty:
                sector_df = sec_raw.xs(sector_ticker, axis=1, level=1) if isinstance(sec_raw.columns, pd.MultiIndex) and sector_ticker in sec_raw.columns.levels[1] else sec_raw
                if not pd.api.types.is_datetime64_any_dtype(sector_df.index):
                    sector_df.index = pd.to_datetime(sector_df.index)
                if sector_df.index.tz is None:
                    sector_df.index = sector_df.index.tz_localize("Asia/Kolkata")
                else:
                    sector_df.index = sector_df.index.tz_convert("Asia/Kolkata")
        except Exception:
            sector_df = None

        symbol_data[sym] = {
            "df": df,
            "daily_df": daily_df,
            "sector_df": sector_df,
            "bullish_event_indices": bullish_event_indices,
            "bearish_event_indices": bearish_event_indices,
            "scanned_history": {}
        }

    # Normalize NIFTY timezone
    if not pd.api.types.is_datetime64_any_dtype(nifty_df.index):
        nifty_df.index = pd.to_datetime(nifty_df.index)
    if nifty_df.index.tz is None:
        nifty_df.index = nifty_df.index.tz_localize("Asia/Kolkata")
    else:
        nifty_df.index = nifty_df.index.tz_convert("Asia/Kolkata")

    # Build Master Chronological Timeline of 5-minute timestamps across all symbols
    symbol_index_sets = [set(data["df"].index[25:-5]) for data in symbol_data.values() if data.get("df") is not None and len(data["df"]) >= 30]
    if not symbol_index_sets:
        print("[ERROR] No usable historical symbol data available for timeline replay.")
        return stats
    all_timestamps = sorted(list(set.union(*symbol_index_sets)))
    print(f"Master chronological timeline built: {len(all_timestamps)} unique 5m candles across {len(symbol_data)} stocks.")

    # Global Chronological In-Sample / Out-of-Sample Split (Item 6):
    # Defined globally across the master timeline rather than on per-stock lengths
    oos_split_idx = int(len(all_timestamps) * 0.60)
    global_oos_cutoff_ts = all_timestamps[oos_split_idx] if oos_split_idx < len(all_timestamps) else all_timestamps[-1]
    print(f"Global In-Sample (60%) up to {global_oos_cutoff_ts} | Out-of-Sample (40%) beyond.")

    # Shared Portfolio State:
    # Starting portfolio equity: 100.0 (normalized base)
    # Total capital available, reserved capital, and continuous mark-to-market tracking
    # Max concurrent positions: 4
    # Max allocation per trade: 25% of current equity reserved AT ENTRY
    portfolio_equity = 100.0
    active_portfolio_positions = [] # List of active trades with reserved capital and live mark-to-market

    for current_ts in all_timestamps:
        # 1. Release active positions whose exit_ts has arrived or passed at or before current_ts
        still_active = []
        for pos in active_portfolio_positions:
            if current_ts >= pos["exit_ts"]:
                # Realize position into shared portfolio equity using entry-reserved capital
                pnl_mult = (pos["net_pnl"] / 100.0) * pos["alloc"]
                portfolio_equity = portfolio_equity * (1.0 + pnl_mult)
                stats["trades_executed"] += 1
                stats["trade_net_pnls"].append(pos["net_pnl"])
                stats["trade_gross_pnls"].append(pos["gross_pnl"])
                stats["total_pnl_pct"] += pos["net_pnl"]
                stats["gross_pnl_pct"] += pos["gross_pnl"]
                stats["trade_r_multiples"].append(pos["r_mult"])
                stats["mfes"].append(pos["mfe"])
                stats["maes"].append(pos["mae"])
                if pos["net_pnl"] > 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
                if pos["is_oos"]:
                    stats["out_of_sample_trades"] += 1
                    if pos["net_pnl"] > 0:
                        stats["out_of_sample_wins"] += 1
                else:
                    stats["in_sample_trades"] += 1
                    if pos["net_pnl"] > 0:
                        stats["in_sample_wins"] += 1
            else:
                still_active.append(pos)
        active_portfolio_positions = still_active

        # Compute continuous mark-to-market portfolio equity for this timestamp:
        # Portfolio Equity = Realized Equity + Sum of current unrealized PnL across active positions
        unrealized_total_mult = 0.0
        for pos in active_portfolio_positions:
            pos_sym = pos["symbol"]
            pos_df = symbol_data[pos_sym]["df"]
            if current_ts in pos_df.index:
                curr_bar_c = float(pos_df.loc[current_ts, "Close"])
                pos_dir = pos["direction"]
                entry_p = pos["entry_price"]
                unrealized_pct = ((curr_bar_c - entry_p) / entry_p * 100) if pos_dir == "BULLISH" else ((entry_p - curr_bar_c) / entry_p * 100)
                unrealized_total_mult += (unrealized_pct / 100.0) * pos["alloc"]
        
        current_mtm_equity = portfolio_equity * (1.0 + unrealized_total_mult)
        stats["equity_curve"].append(current_mtm_equity)

        # 2. Check each candidate symbol for this exact completed candle timestamp
        for sym, s_data in symbol_data.items():
            df = s_data["df"]
            if current_ts not in df.index:
                continue
            idx = df.index.get_loc(current_ts)
            if idx < 25 or idx >= len(df) - 5:
                continue

            window_df = df.iloc[:idx]
            signal_candle_idx = idx - 1
            signal_candle_ts = window_df.index[-1]

            nifty_sub = nifty_df[nifty_df.index <= signal_candle_ts]
            if nifty_sub.empty:
                continue

            # Strict Completed Candle Timestamp Equality on NIFTY:
            # Replay requires exact NIFTY completed candle timestamp alignment before calculating relative performance (Item D)
            latest_nifty_ts = nifty_sub.index[-1]
            if latest_nifty_ts != signal_candle_ts:
                continue

            sector_df = s_data["sector_df"]
            sector_sub = sector_df[sector_df.index <= signal_candle_ts] if sector_df is not None else None
            daily_df = s_data["daily_df"]
            daily_sub = daily_df[daily_df.index.date < signal_candle_ts.date()] if daily_df is not None and not daily_df.empty else None

            # Calculate Point-in-time Relative Performance
            stock_day_open = float(window_df[window_df.index.date == signal_candle_ts.date()]["Open"].iloc[0]) if len(window_df[window_df.index.date == signal_candle_ts.date()]) > 0 else float(window_df["Open"].iloc[0])
            stock_change_pt = round((float(window_df["Close"].iloc[-1]) - stock_day_open) / stock_day_open * 100, 2)
            nifty_day_open = float(nifty_sub[nifty_sub.index.date == signal_candle_ts.date()]["Open"].iloc[0]) if len(nifty_sub[nifty_sub.index.date == signal_candle_ts.date()]) > 0 else float(nifty_sub["Open"].iloc[0])
            nifty_change_pt = round((float(nifty_sub["Close"].iloc[-1]) - nifty_day_open) / nifty_day_open * 100, 2) if nifty_day_open > 0 else 0.0

            # Strict 3-Way Completed Candle Timestamp Alignment for Sector Bias (Item 1 & D):
            # If sector completed candle timestamp does not match signal candle timestamp, sector bias MUST be NEUTRAL
            sector_change_pt = 0.0
            sector_is_fresh = False
            if sector_sub is not None and not sector_sub.empty:
                sec_candle_ts = sector_sub.index[-1]
                if sec_candle_ts == signal_candle_ts:
                    sec_day_sub = sector_sub[sector_sub.index.date == signal_candle_ts.date()]
                    if not sec_day_sub.empty:
                        sec_open = float(sec_day_sub["Open"].iloc[0])
                        sec_last = float(sec_day_sub["Close"].iloc[-1])
                        sector_change_pt = round((sec_last - sec_open) / sec_open * 100, 2) if sec_open > 0 else 0.0
                        sector_is_fresh = True

            if sector_is_fresh and (stock_change_pt > sector_change_pt) and (sector_change_pt >= nifty_change_pt - 0.2):
                sector_bias = "BULLISH"
            elif sector_is_fresh and (stock_change_pt < sector_change_pt) and (sector_change_pt <= nifty_change_pt + 0.2):
                sector_bias = "BEARISH"
            else:
                sector_bias = "NEUTRAL"

            pipeline_res = ScreenerPipeline.evaluate_candidate_unified(
                stock_df=window_df,
                symbol=sym,
                sector_change=sector_change_pt,
                sector_bias=sector_bias,
                nifty_df=nifty_sub,
                daily_df=daily_sub,
                is_replay=True,
                sector_df=sector_sub
            )

            stats["total_evaluations"] += 1
            if not pipeline_res:
                stats["watch_only_candidates"] += 1
                s_data["scanned_history"][signal_candle_idx] = {"stage": "DISQUALIFIED", "direction": "NEUTRAL", "ts": signal_candle_ts}
                continue

            opp_res = pipeline_res.get("opp_res")
            if not opp_res:
                stats["watch_only_candidates"] += 1
                s_data["scanned_history"][signal_candle_idx] = {"stage": "DISQUALIFIED", "direction": "NEUTRAL", "ts": signal_candle_ts}
                continue

            stage = opp_res.get("stage", "WATCHLIST")
            direction = opp_res.get("direction", "NEUTRAL")
            is_actionable = opp_res.get("is_actionable", False)

            stats["stage_counts"][stage] = stats["stage_counts"].get(stage, 0) + 1
            s_data["scanned_history"][signal_candle_idx] = {"stage": stage, "direction": direction, "ts": signal_candle_ts}

            if not is_actionable or direction == "NEUTRAL":
                stats["watch_only_candidates"] += 1
                continue

            # Check if this stock is ALREADY in an active trade
            if any(p["symbol"] == sym for p in active_portfolio_positions):
                continue

            # Shared Portfolio Concurrency Gate: Max 4 simultaneous positions
            if len(active_portfolio_positions) >= max_concurrent_portfolio_trades:
                continue

            # Validate risk levels
            sl = opp_res.get("suggested_sl")
            tp1 = opp_res.get("suggested_tp1")
            entry_price = float(df["Open"].iloc[idx])

            if sl is None or tp1 is None:
                stats["rejected_no_risk"] += 1
                stats["watch_only_candidates"] += 1
                continue

            actual_entry = entry_price * (1.0 + entry_slippage_pct/100) if direction == "BULLISH" else entry_price * (1.0 - entry_slippage_pct/100)

            if direction == "BULLISH":
                if not (sl < actual_entry and tp1 > actual_entry):
                    stats["rejected_no_risk"] += 1
                    stats["watch_only_candidates"] += 1
                    continue
                risk = actual_entry - sl
                reward = tp1 - actual_entry
            else:
                if not (sl > actual_entry and tp1 < actual_entry):
                    stats["rejected_no_risk"] += 1
                    stats["watch_only_candidates"] += 1
                    continue
                risk = sl - actual_entry
                reward = actual_entry - tp1

            if risk <= 0 or (reward / risk) < 1.25:
                stats["rejected_no_risk"] += 1
                stats["watch_only_candidates"] += 1
                continue

            # Calculate Trade Outcome Forward
            forward_candles = df.iloc[idx:min(idx + max_holding_candles, len(df))]
            if forward_candles.empty:
                continue

            opens = forward_candles["Open"].values
            highs = forward_candles["High"].values
            lows = forward_candles["Low"].values
            closes_sub = forward_candles["Close"].values

            hit_tp = False
            hit_sl = False
            exit_offset = len(forward_candles) - 1  # 0-indexed offset within forward_candles (0 to 5)
            exit_price_actual = float(closes_sub[-1])
            in_trade_highs = []
            in_trade_lows = []

            for c_i, (c_open, h, l, c_close) in enumerate(zip(opens, highs, lows, closes_sub)):
                in_trade_highs.append(h)
                in_trade_lows.append(l)

                if direction == "BULLISH":
                    touched_sl = (l <= sl) or (c_open <= sl)
                    touched_tp = (h >= tp1) or (c_open >= tp1)
                    if touched_sl and touched_tp:
                        hit_sl = True
                        base_exit = c_open if c_open <= sl else sl
                        exit_price_actual = base_exit * (1.0 - exit_slippage_pct/100)
                        exit_offset = c_i
                        break
                    elif c_open <= sl:
                        hit_sl = True
                        exit_price_actual = c_open * (1.0 - exit_slippage_pct/100)
                        exit_offset = c_i
                        break
                    elif l <= sl:
                        hit_sl = True
                        exit_price_actual = sl * (1.0 - exit_slippage_pct/100)
                        exit_offset = c_i
                        break
                    elif c_open >= tp1:
                        hit_tp = True
                        exit_price_actual = c_open * (1.0 - exit_slippage_pct/100)
                        exit_offset = c_i
                        break
                    elif h >= tp1:
                        hit_tp = True
                        exit_price_actual = tp1 * (1.0 - exit_slippage_pct/100)
                        exit_offset = c_i
                        break
                else:
                    touched_sl = (h >= sl) or (c_open >= sl)
                    touched_tp = (l <= tp1) or (c_open <= tp1)
                    if touched_sl and touched_tp:
                        hit_sl = True
                        base_exit = c_open if c_open >= sl else sl
                        exit_price_actual = base_exit * (1.0 + exit_slippage_pct/100)
                        exit_offset = c_i
                        break
                    elif c_open >= sl:
                        hit_sl = True
                        exit_price_actual = c_open * (1.0 + exit_slippage_pct/100)
                        exit_offset = c_i
                        break
                    elif h >= sl:
                        hit_sl = True
                        exit_price_actual = sl * (1.0 + exit_slippage_pct/100)
                        exit_offset = c_i
                        break
                    elif c_open <= tp1:
                        hit_tp = True
                        exit_price_actual = c_open * (1.0 + exit_slippage_pct/100)
                        exit_offset = c_i
                        break
                    elif l <= tp1:
                        hit_tp = True
                        exit_price_actual = tp1 * (1.0 + exit_slippage_pct/100)
                        exit_offset = c_i
                        break

            # If no target or stop hit, position closes at end-of-horizon bar close with exit slippage (Item 4)
            if not hit_tp and not hit_sl:
                exit_price_actual = exit_price_actual * (1.0 - exit_slippage_pct/100) if direction == "BULLISH" else exit_price_actual * (1.0 + exit_slippage_pct/100)

            # In-trade MFE & MAE strictly up to exit
            if direction == "BULLISH":
                max_fav = (max(in_trade_highs[:exit_offset+1]) - actual_entry) / actual_entry * 100
                max_adv = (actual_entry - min(in_trade_lows[:exit_offset+1])) / actual_entry * 100
            else:
                max_fav = (actual_entry - min(in_trade_lows[:exit_offset+1])) / actual_entry * 100
                max_adv = (max(in_trade_highs[:exit_offset+1]) - actual_entry) / actual_entry * 100

            trade_friction_pct = base_statutory_friction_pct + (0.015 if direction == "BEARISH" else 0.010)

            if hit_tp:
                gross_diff = (exit_price_actual - actual_entry) / actual_entry * 100 if direction == "BULLISH" else (actual_entry - exit_price_actual) / actual_entry * 100
            elif hit_sl:
                gross_diff = -(abs(actual_entry - exit_price_actual) / actual_entry * 100)
            else:
                gross_diff = ((exit_price_actual - actual_entry) / actual_entry * 100) if direction == "BULLISH" else ((actual_entry - exit_price_actual) / actual_entry * 100)

            net_pnl = gross_diff - trade_friction_pct
            risk_pct = (risk / actual_entry) * 100
            r_mult = net_pnl / risk_pct if risk_pct > 0 else (1.0 if net_pnl > 0 else -1.0)
            
            # Correct exit timestamp alignment (Item 3):
            # idx is candle 0 in forward_candles, exit_offset is 0-indexed (e.g. 0 to 5)
            # The candle at idx + exit_offset is the EXACT completed candle where exit occurs!
            exit_idx = idx + exit_offset
            exit_ts = df.index[min(exit_idx, len(df) - 1)]

            # Register into shared active portfolio positions
            active_portfolio_positions.append({
                "symbol": sym,
                "direction": direction,
                "entry_price": actual_entry,
                "entry_ts": current_ts,
                "exit_ts": exit_ts,
                "net_pnl": net_pnl,
                "gross_pnl": gross_diff,
                "alloc": 0.25, # Max 25% allocation per concurrent trade
                "r_mult": r_mult,
                "mfe": max_fav,
                "mae": max_adv,
                "is_oos": (current_ts >= global_oos_cutoff_ts)
            })

    # Flush any remaining positions that completed at the end of the timeline
    for pos in active_portfolio_positions:
        pnl_mult = (pos["net_pnl"] / 100.0) * pos["alloc"]
        prev_eq = stats["equity_curve"][-1]
        new_eq = prev_eq * (1.0 + pnl_mult)
        stats["equity_curve"].append(new_eq)
        stats["trades_executed"] += 1
        stats["trade_net_pnls"].append(pos["net_pnl"])
        stats["trade_gross_pnls"].append(pos["gross_pnl"])
        stats["total_pnl_pct"] += pos["net_pnl"]
        stats["gross_pnl_pct"] += pos["gross_pnl"]
        stats["trade_r_multiples"].append(pos["r_mult"])
        stats["mfes"].append(pos["mfe"])
        stats["maes"].append(pos["mae"])
        if pos["net_pnl"] > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        if pos["is_oos"]:
            stats["out_of_sample_trades"] += 1
            if pos["net_pnl"] > 0:
                stats["out_of_sample_wins"] += 1
        else:
            stats["in_sample_trades"] += 1
            if pos["net_pnl"] > 0:
                stats["in_sample_wins"] += 1

    # Evaluate Direction-Matched Pre-Move Event Recall & Lead Times across all symbols
    for sym, s_data in symbol_data.items():
        scanned_history = s_data["scanned_history"]
        for b_idx in s_data["bullish_event_indices"]:
            stats["bullish_events"] += 1
            for lead in range(1, 7):
                c_idx = b_idx - lead
                entry = scanned_history.get(c_idx, {})
                if entry.get("direction") == "BULLISH" and entry.get("stage") in ("ENTRY READY", "IGNITION", "PREPARING", "OBSERVING"):
                    stats["detected_bullish_events"] += 1
                    stats["bullish_lead_times"].append(lead)
                    break

        for s_idx in s_data["bearish_event_indices"]:
            stats["bearish_events"] += 1
            for lead in range(1, 7):
                c_idx = s_idx - lead
                entry = scanned_history.get(c_idx, {})
                if entry.get("direction") == "BEARISH" and entry.get("stage") in ("ENTRY READY", "IGNITION", "PREPARING", "OBSERVING"):
                    stats["detected_bearish_events"] += 1
                    stats["bearish_lead_times"].append(lead)
                    break
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
        
        # Portfolio Level Calculations: Compounded Return, Profit Factor, Expectancy, Max Drawdown
        winning_trades = [p for p in stats["trade_net_pnls"] if p > 0]
        losing_trades = [p for p in stats["trade_net_pnls"] if p <= 0]
        gross_profits = sum(winning_trades) if winning_trades else 0.0
        gross_losses = abs(sum(losing_trades)) if losing_trades else 0.0001
        profit_factor = round(gross_profits / gross_losses, 2)
        
        # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
        avg_win_pct = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss_pct = abs(np.mean(losing_trades)) if losing_trades else 0.0
        expectancy_pct = round(((stats['wins'] / stats['trades_executed']) * avg_win_pct) - ((stats['losses'] / stats['trades_executed']) * avg_loss_pct), 2)
        avg_r = round(float(np.mean(stats["trade_r_multiples"])), 2) if stats["trade_r_multiples"] else 0.0
        
        # Compounded Portfolio Return & Max Drawdown:
        eq_curve = stats["equity_curve"]
        compounded_return_pct = round(eq_curve[-1] - 100.0, 2)
        # Peak-to-trough max drawdown
        peaks = np.maximum.accumulate(eq_curve)
        drawdowns = (np.array(eq_curve) - peaks) / peaks * 100
        max_drawdown_pct = round(abs(float(np.min(drawdowns))), 2)

        print("\n--- TRADE EXECUTION & PERFORMANCE (AFTER 0.10% COSTS) ---")
        print(f"Win Rate:                    {win_rate}% ({stats['wins']} Wins / {stats['losses']} Losses)")
        print(f"Setup Expansion Rate (MFE):  {expansion_rate}% (MFE >= 0.70%)")
        print(f"Non-Expanding Trigger Rate:  {fpr_expansion}%")
        print(f"Average MFE (In-Trade):      +{avg_mfe}%")
        print(f"Average MAE (In-Trade):      -{avg_mae}%")
        print(f"Gross PnL (Before Costs):    {round(stats['gross_pnl_pct'], 2)}%")
        print(f"Net Realized PnL (Sum):      {round(stats['total_pnl_pct'], 2)}%")
        print(f"Compounded Equity Return:    {compounded_return_pct}% (20% alloc per trade)")
        print(f"Maximum Portfolio Drawdown:  {max_drawdown_pct}%")
        print(f"Profit Factor:               {profit_factor}")
        print(f"Expectancy Per Trade:        +{expectancy_pct}%")
        print(f"Average Realized R:          {avg_r}R")
        
        if stats["in_sample_trades"] > 0:
            is_wr = round(stats["in_sample_wins"] / stats["in_sample_trades"] * 100, 1)
            print(f"In-Sample Win Rate (60%):    {is_wr}% ({stats['in_sample_wins']}/{stats['in_sample_trades']})")
        if stats["out_of_sample_trades"] > 0:
            oos_wr = round(stats["out_of_sample_wins"] / stats["out_of_sample_trades"] * 100, 1)
            print(f"Out-of-Sample Win Rate (40%):{oos_wr}% ({stats['out_of_sample_wins']}/{stats['out_of_sample_trades']})")
    print("="*55 + "\n")

if __name__ == "__main__":
    run_screener_replay()
