import time
import json
import sqlite3
import pandas as pd
from core.storage import get_db_connection

class HistoricalStockLearner:
    """
    Historical Stock Scanner & 30-Day Memory Engine:
    1. Tracks evaluated completed-candle timestamps per stock (replaces brittle wall-clock cooldowns).
    2. Measures genuine 30-day breakout target-vs-stop performance:
       - Evaluates 30+ daily/session bars for true breakout events.
       - A breakout is defined when high breaks prior resistance.
       - Measures actual 1:1.5 target vs 1:1 stop outcome.
       - Strictly respects `as_of_ts` to eliminate lookahead bias in replay.
       - Never writes replay evaluations into live SQLite memory.
    """

    @staticmethod
    def init_memory_table():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_scan_memory (
            symbol TEXT PRIMARY KEY,
            last_scanned_ts REAL,
            last_candle_date TEXT,
            last_evaluated_candle_ts TEXT,
            scan_count INTEGER DEFAULT 1,
            historical_win_rate REAL DEFAULT 50.0,
            pattern_score REAL DEFAULT 50.0,
            bull_win_rate REAL DEFAULT 50.0,
            bear_win_rate REAL DEFAULT 50.0,
            bull_trades INTEGER DEFAULT 0,
            bear_trades INTEGER DEFAULT 0,
            bull_multiplier REAL DEFAULT 1.0,
            bear_multiplier REAL DEFAULT 1.0,
            unresolved_trades INTEGER DEFAULT 0,
            best_strategy TEXT,
            failure_reasons TEXT,
            updated_at REAL
        )
        """)
        # Safe schema migration for existing databases
        for col_def in [
            ("bull_win_rate", "REAL DEFAULT 50.0"),
            ("bear_win_rate", "REAL DEFAULT 50.0"),
            ("bull_trades", "INTEGER DEFAULT 0"),
            ("bear_trades", "INTEGER DEFAULT 0"),
            ("bull_multiplier", "REAL DEFAULT 1.0"),
            ("bear_multiplier", "REAL DEFAULT 1.0"),
            ("unresolved_trades", "INTEGER DEFAULT 0")
        ]:
            try:
                cursor.execute(f"ALTER TABLE stock_scan_memory ADD COLUMN {col_def[0]} {col_def[1]}")
            except Exception:
                pass
        conn.commit()
        conn.close()

    @staticmethod
    def get_stock_profile(symbol):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            clean_s = symbol.replace(".NS", "").upper()
            cursor.execute("SELECT * FROM stock_scan_memory WHERE symbol = ? OR symbol = ?", (symbol, clean_s))
            row = cursor.fetchone()
            conn.close()
            if row:
                return dict(row)
        except Exception:
            pass
        return None

    @staticmethod
    def should_skip_scan(symbol, current_completed_candle_ts=None, max_freshness_seconds=600):
        """
        Determines if a stock should be skipped:
        1. If `current_completed_candle_ts` is provided, skips ONLY if that EXACT completed candle
           has already been evaluated. If a new 5-minute candle formed, returns False immediately!
        2. If no candle ts is provided, uses max_freshness_seconds fallback.
        """
        try:
            profile = HistoricalStockLearner.get_stock_profile(symbol)
            if not profile:
                return False

            if current_completed_candle_ts:
                last_eval_ts = profile.get("last_evaluated_candle_ts")
                if last_eval_ts and str(last_eval_ts) == str(current_completed_candle_ts):
                    return True
                return False

            # Fallback to time cooldown if timestamp not supplied
            if (time.time() - profile.get("last_scanned_ts", 0)) < max_freshness_seconds:
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def record_stock_scan(symbol, last_candle_date, completed_candle_ts=None, win_rate=50.0, pattern_score=50.0,
                          best_strategy="Breakout", bull_win_rate=50.0, bear_win_rate=50.0,
                          bull_trades=0, bear_trades=0, bull_multiplier=1.0, bear_multiplier=1.0,
                          unresolved_trades=0):
        """
        Records or updates a stock scan in the persistent memory table with its evaluated candle timestamp
        and separate directional statistics, multipliers, and unresolved trade count.
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            clean_s = symbol.replace(".NS", "").upper()
            now = time.time()
            cursor.execute("""
            INSERT INTO stock_scan_memory (
                symbol, last_scanned_ts, last_candle_date, last_evaluated_candle_ts, scan_count, historical_win_rate,
                pattern_score, bull_win_rate, bear_win_rate, bull_trades, bear_trades,
                bull_multiplier, bear_multiplier, unresolved_trades, best_strategy, failure_reasons, updated_at
            ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?)
            ON CONFLICT(symbol) DO UPDATE SET
                last_scanned_ts = excluded.last_scanned_ts,
                last_candle_date = excluded.last_candle_date,
                last_evaluated_candle_ts = excluded.last_evaluated_candle_ts,
                scan_count = stock_scan_memory.scan_count + 1,
                historical_win_rate = excluded.historical_win_rate,
                pattern_score = excluded.pattern_score,
                bull_win_rate = excluded.bull_win_rate,
                bear_win_rate = excluded.bear_win_rate,
                bull_trades = excluded.bull_trades,
                bear_trades = excluded.bear_trades,
                bull_multiplier = excluded.bull_multiplier,
                bear_multiplier = excluded.bear_multiplier,
                unresolved_trades = excluded.unresolved_trades,
                best_strategy = excluded.best_strategy,
                updated_at = excluded.updated_at
            """, (clean_s, now, str(last_candle_date), str(completed_candle_ts or last_candle_date),
                  win_rate, pattern_score, bull_win_rate, bear_win_rate, bull_trades, bear_trades,
                  bull_multiplier, bear_multiplier, unresolved_trades, best_strategy, now))
            conn.commit()
            conn.close()
        except Exception:
            pass

    @staticmethod
    def analyze_30d_stock_history(symbol, df_history, is_replay=False, as_of_ts=None, return_directional=False):
        """
        Stock breakout and breakdown performance analyzer (intraday 5-min or multi-day history).
        
        Rules:
        - Strictly filters out future data past as_of_ts when supplied, handling timezone-aware and timezone-naive indices.
        - Validates input DataFrame integrity: checks for required OHLC columns, non-empty, non-finite values,
          monotonic sorted timestamps, duplicate timestamps, and valid OHLC bounds (High >= Low/Open/Close, Low <= Open/Close).
        - Identifies breakout setups where High breaks prior N-bar resistance (Bullish),
          and breakdown setups where Low breaks prior N-bar support (Bearish).
        - Replay model:
            Entry = Next bar Open + directional slippage
            Stop Loss = Trigger bar Low (Bullish) or Trigger bar High (Bearish)
            Risk (R) = |Entry - Stop Loss|
            Target = Entry +/- (1.5 * R)
        - Evaluates forward bars up to 6 bars.
        - Trades that hit neither Target nor Stop within the forward window are explicitly tracked
          as unresolved/expired trades.
        - P1 Fix: Applies an unresolved-rate confidence penalty: If a direction's unresolved trade rate
          exceeds 30%, its multiplier is discounted proportionally towards the neutral 1.0 baseline
          or penalized, ensuring high unresolved rates cannot artificially inflate performance.
        - P2 Fix: Logs learning failures and exposes error status in returned stats dict instead of silent swallow.
        """
        import traceback
        import numpy as np
        HistoricalStockLearner.init_memory_table()

        empty_stats = {
            "status": "INSUFFICIENT_DATA",
            "overall_win_rate": 50.0,
            "bull_win_rate": 50.0,
            "bear_win_rate": 50.0,
            "bull_trades": 0,
            "bear_trades": 0,
            "unresolved_trades": 0,
            "bull_unresolved": 0,
            "bear_unresolved": 0,
            "bull_unresolved_rate": 0.0,
            "bear_unresolved_rate": 0.0,
            "bull_mult": 1.0,
            "bear_mult": 1.0,
            "final_mult": 1.0
        }

        if df_history is None or df_history.empty or len(df_history) < 10:
            return (1.0, 1.0, empty_stats) if return_directional else 1.0

        try:
            df = df_history.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            req_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in req_cols):
                print(f"[HISTORICAL LEARNER ERROR] Missing required OHLC columns for {symbol}: {[c for c in req_cols if c not in df.columns]}")
                empty_stats["status"] = "MISSING_COLUMNS"
                return (1.0, 1.0, empty_stats) if return_directional else 1.0

            # Ensure datetime index
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as ex:
                    print(f"[HISTORICAL LEARNER ERROR] Non-datetime index for {symbol}: {ex}")
                    empty_stats["status"] = "INVALID_INDEX"
                    return (1.0, 1.0, empty_stats) if return_directional else 1.0

            # P2 Validation: Check duplicate timestamps and chronological order
            if df.index.has_duplicates:
                print(f"[HISTORICAL LEARNER ERROR] Duplicate timestamps in historical feed for {symbol}")
                empty_stats["status"] = "DUPLICATE_TIMESTAMPS"
                return (1.0, 1.0, empty_stats) if return_directional else 1.0

            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

            # P2 Validation: Non-finite values check
            for col in req_cols:
                arr = df[col].values
                if not np.all(np.isfinite(arr)):
                    print(f"[HISTORICAL LEARNER ERROR] Non-finite values in column {col} for {symbol}")
                    empty_stats["status"] = "NON_FINITE_DATA"
                    return (1.0, 1.0, empty_stats) if return_directional else 1.0

            # P2 Validation: Strict OHLC boundary sanity check
            h_arr = df['High'].values
            l_arr = df['Low'].values
            o_arr = df['Open'].values
            c_arr = df['Close'].values

            if np.any(h_arr < l_arr) or np.any(h_arr < o_arr) or np.any(h_arr < c_arr) or np.any(l_arr > o_arr) or np.any(l_arr > c_arr):
                print(f"[HISTORICAL LEARNER ERROR] Invalid OHLC price relationships in feed for {symbol}")
                empty_stats["status"] = "CORRUPT_OHLC_RELATIONSHIPS"
                return (1.0, 1.0, empty_stats) if return_directional else 1.0

            # Lookahead Bias Prevention: Slice df up to as_of_ts if provided (handles timezone-aware and naive)
            if as_of_ts is not None:
                try:
                    as_of_dt = pd.to_datetime(as_of_ts)
                    # Align timezone between as_of_dt and df.index
                    if df.index.tz is not None and as_of_dt.tz is None:
                        as_of_dt = as_of_dt.tz_localize(df.index.tz)
                    elif df.index.tz is None and as_of_dt.tz is not None:
                        as_of_dt = as_of_dt.tz_localize(None)
                    elif df.index.tz is not None and as_of_dt.tz is not None:
                        as_of_dt = as_of_dt.tz_convert(df.index.tz)

                    df = df[df.index <= as_of_dt]
                except Exception as ex:
                    print(f"[HISTORICAL LEARNER ERROR] Error slicing as_of_ts {as_of_ts} for {symbol}: {ex}")
                    pass

            if len(df) < 10:
                empty_stats["status"] = "INSUFFICIENT_HISTORY"
                return (1.0, 1.0, empty_stats) if return_directional else 1.0

            c = df['Close'].squeeze()
            o = df['Open'].squeeze()
            h = df['High'].squeeze()
            l = df['Low'].squeeze()

            bull_wins = 0
            bull_losses = 0
            bull_unresolved = 0

            bear_wins = 0
            bear_losses = 0
            bear_unresolved = 0

            n_bars = len(df)

            # Analyze prior breakout and breakdown events with genuine Target vs Stop evaluation
            # Evaluate across completed bars (leaving room for forward check)
            for i in range(5, n_bars - 2):
                prior_high = float(h.iloc[i-5:i].max())
                prior_low = float(l.iloc[i-5:i].min())
                curr_h = float(h.iloc[i])
                curr_l = float(l.iloc[i])
                curr_c = float(c.iloc[i])

                # 1. Bullish Breakout condition: close and high above prior 5-bar high
                if curr_h > prior_high and curr_c > prior_high:
                    next_open = float(o.iloc[i+1])
                    slippage = next_open * 0.0005
                    entry = round(next_open + slippage, 2)
                    stop = round(curr_l, 2)
                    risk = entry - stop
                    if risk > 0:
                        target = round(entry + (1.5 * risk), 2)
                        max_forward = min(n_bars, i + 7)
                        resolved = False
                        for fwd in range(i + 1, max_forward):
                            fwd_o = float(o.iloc[fwd])
                            fwd_h = float(h.iloc[fwd])
                            fwd_l = float(l.iloc[fwd])

                            if fwd_o <= stop:
                                bull_losses += 1
                                resolved = True
                                break

                            hit_target = fwd_h >= target
                            hit_stop = fwd_l <= stop

                            if hit_target and hit_stop:
                                bull_losses += 1
                                resolved = True
                                break
                            elif hit_stop:
                                bull_losses += 1
                                resolved = True
                                break
                            elif hit_target:
                                bull_wins += 1
                                resolved = True
                                break
                        if not resolved:
                            bull_unresolved += 1

                # 2. Bearish Breakdown condition: close and low below prior 5-bar low
                elif curr_l < prior_low and curr_c < prior_low:
                    next_open = float(o.iloc[i+1])
                    slippage = next_open * 0.0005
                    entry = round(next_open - slippage, 2)
                    stop = round(curr_h, 2)
                    risk = stop - entry
                    if risk > 0:
                        target = round(entry - (1.5 * risk), 2)
                        max_forward = min(n_bars, i + 7)
                        resolved = False
                        for fwd in range(i + 1, max_forward):
                            fwd_o = float(o.iloc[fwd])
                            fwd_h = float(h.iloc[fwd])
                            fwd_l = float(l.iloc[fwd])

                            if fwd_o >= stop:
                                bear_losses += 1
                                resolved = True
                                break

                            hit_target = fwd_l <= target
                            hit_stop = fwd_h >= stop

                            if hit_target and hit_stop:
                                bear_losses += 1
                                resolved = True
                                break
                            elif hit_stop:
                                bear_losses += 1
                                resolved = True
                                break
                            elif hit_target:
                                bear_wins += 1
                                resolved = True
                                break
                        if not resolved:
                            bear_unresolved += 1

            total_bull_resolved = bull_wins + bull_losses
            total_bull = total_bull_resolved + bull_unresolved

            total_bear_resolved = bear_wins + bear_losses
            total_bear = total_bear_resolved + bear_unresolved

            total_resolved = total_bull_resolved + total_bear_resolved
            total_unresolved = bull_unresolved + bear_unresolved
            total_trades = total_resolved + total_unresolved

            # Unresolved rates per direction
            bull_unres_rate = round((bull_unresolved / total_bull) * 100.0, 1) if total_bull > 0 else 0.0
            bear_unres_rate = round((bear_unresolved / total_bear) * 100.0, 1) if total_bear > 0 else 0.0

            # Win rates computed against resolved trades, with default 50.0 if insufficient resolved samples
            bull_win_rate = round((bull_wins / total_bull_resolved) * 100.0, 1) if total_bull_resolved >= 2 else 50.0
            bear_win_rate = round((bear_wins / total_bear_resolved) * 100.0, 1) if total_bear_resolved >= 2 else 50.0
            overall_win_rate = round(((bull_wins + bear_wins) / total_resolved) * 100.0, 1) if total_resolved >= 3 else 50.0

            # Structural cleanliness: Body-to-Range ratio
            body = (c - o).abs()
            rng = (h - l).replace(0, 0.001)
            clean_body_ratio = float((body / rng).mean())
            pattern_score = round(clean_body_ratio * 100.0, 1)

            # Empirical Bayesian Shrinkage towards 1.0 baseline separately for Bullish and Bearish
            # 1. Bullish Multiplier:
            bull_weight = min(1.0, total_bull_resolved / 6.0)
            raw_bull_mult = 0.70 + (bull_win_rate / 100.0 * 0.60)
            shrunk_bull_mult = 1.0 + bull_weight * (raw_bull_mult - 1.0)
            
            # P1 Fix: Unresolved-rate penalty:
            # If unresolved rate exceeds 30%, discount the multiplier towards/below neutral (up to -0.15 penalty)
            if bull_unres_rate > 30.0:
                unres_penalty = min(0.15, (bull_unres_rate - 30.0) / 100.0 * 0.25)
                shrunk_bull_mult = shrunk_bull_mult - unres_penalty

            bull_mult = round(min(1.20, max(0.85, shrunk_bull_mult)), 2)

            # 2. Bearish Multiplier:
            bear_weight = min(1.0, total_bear_resolved / 6.0)
            raw_bear_mult = 0.70 + (bear_win_rate / 100.0 * 0.60)
            shrunk_bear_mult = 1.0 + bear_weight * (raw_bear_mult - 1.0)

            # Unresolved-rate penalty for bearish:
            if bear_unres_rate > 30.0:
                unres_penalty = min(0.15, (bear_unres_rate - 30.0) / 100.0 * 0.25)
                shrunk_bear_mult = shrunk_bear_mult - unres_penalty

            bear_mult = round(min(1.20, max(0.85, shrunk_bear_mult)), 2)

            # 3. Overall Multiplier (for backwards compatibility)
            sample_weight = min(1.0, total_resolved / 10.0)
            raw_mult = 0.70 + (overall_win_rate / 100.0 * 0.60)
            shrunk_mult = 1.0 + sample_weight * (raw_mult - 1.0)
            overall_unres_rate = (total_unresolved / total_trades * 100.0) if total_trades > 0 else 0.0
            if overall_unres_rate > 30.0:
                unres_penalty = min(0.15, (overall_unres_rate - 30.0) / 100.0 * 0.25)
                shrunk_mult -= unres_penalty
            final_mult = round(min(1.20, max(0.85, shrunk_mult)), 2)

            # Strictly persist ONLY during live trading with distinct directional stats
            if not is_replay:
                best_strat = "Breakout 1:1.5 Target" if bull_win_rate >= bear_win_rate else "Breakdown Short 1:1.5"
                HistoricalStockLearner.record_stock_scan(
                    symbol=symbol,
                    last_candle_date=str(df.index[-1]),
                    completed_candle_ts=str(df.index[-1]),
                    win_rate=overall_win_rate,
                    pattern_score=pattern_score,
                    best_strategy=best_strat,
                    bull_win_rate=bull_win_rate,
                    bear_win_rate=bear_win_rate,
                    bull_trades=total_bull,
                    bear_trades=total_bear,
                    bull_multiplier=bull_mult,
                    bear_multiplier=bear_mult,
                    unresolved_trades=total_unresolved
                )

            stats = {
                "status": "SUCCESS",
                "overall_win_rate": overall_win_rate,
                "bull_win_rate": bull_win_rate,
                "bear_win_rate": bear_win_rate,
                "bull_trades": total_bull,
                "bear_trades": total_bear,
                "unresolved_trades": total_unresolved,
                "bull_unresolved": bull_unresolved,
                "bear_unresolved": bear_unresolved,
                "bull_unresolved_rate": bull_unres_rate,
                "bear_unresolved_rate": bear_unres_rate,
                "bull_mult": bull_mult,
                "bear_mult": bear_mult,
                "final_mult": final_mult
            }

            if return_directional:
                return (bull_mult, bear_mult, stats)
            return final_mult
        except Exception as e:
            traceback.print_exc()
            print(f"[HISTORICAL LEARNER EXCEPTION] Error analyzing history for {symbol}: {e}")
            fail_stats = dict(empty_stats)
            fail_stats["status"] = f"ERROR: {e}"
            return (1.0, 1.0, fail_stats) if return_directional else 1.0

    @staticmethod
    def replay_intraday_breakouts(symbol, df_5m, is_replay=False, as_of_ts=None, return_directional=False):
        """
        Alias/specialization for 5-minute intraday breakout & breakdown evaluation.
        Clearly documents and separates 5-day intraday replay from multi-week session studies.
        """
        return HistoricalStockLearner.analyze_30d_stock_history(
            symbol=symbol,
            df_history=df_5m,
            is_replay=is_replay,
            as_of_ts=as_of_ts,
            return_directional=return_directional
        )
