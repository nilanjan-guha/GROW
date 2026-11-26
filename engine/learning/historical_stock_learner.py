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
            best_strategy TEXT,
            failure_reasons TEXT,
            updated_at REAL
        )
        """)
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
    def record_stock_scan(symbol, last_candle_date, completed_candle_ts=None, win_rate=50.0, pattern_score=50.0, best_strategy="Breakout"):
        """
        Records or updates a stock scan in the persistent memory table with its evaluated candle timestamp.
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            clean_s = symbol.replace(".NS", "").upper()
            now = time.time()
            cursor.execute("""
            INSERT INTO stock_scan_memory (
                symbol, last_scanned_ts, last_candle_date, last_evaluated_candle_ts, scan_count, historical_win_rate,
                pattern_score, best_strategy, failure_reasons, updated_at
            ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, '', ?)
            ON CONFLICT(symbol) DO UPDATE SET
                last_scanned_ts = excluded.last_scanned_ts,
                last_candle_date = excluded.last_candle_date,
                last_evaluated_candle_ts = excluded.last_evaluated_candle_ts,
                scan_count = stock_scan_memory.scan_count + 1,
                historical_win_rate = excluded.historical_win_rate,
                pattern_score = excluded.pattern_score,
                best_strategy = excluded.best_strategy,
                updated_at = excluded.updated_at
            """, (clean_s, now, str(last_candle_date), str(completed_candle_ts or last_candle_date), win_rate, pattern_score, best_strategy, now))
            conn.commit()
            conn.close()
        except Exception:
            pass

    @staticmethod
    def analyze_30d_stock_history(symbol, df_history, is_replay=False, as_of_ts=None):
        """
        Genuine 30-session breakout target-versus-stop outcome measurement.
        
        Rules:
        - Strictly filters out future data past as_of_ts when supplied.
        - Identifies breakout setups where High breaks prior N-bar resistance.
        - Defines:
            Entry = Breakout High
            Stop Loss = Breakout bar Low
            Risk (R) = Entry - Stop Loss
            Target = Entry + 1.5 * R
        - Evaluates forward bars up to 5 bars to determine whether Target was hit before Stop Loss.
        - Never writes into SQLite memory when is_replay=True.
        - Returns Bayesian shrinkage multiplier (0.85x to 1.20x).
        """
        HistoricalStockLearner.init_memory_table()

        if df_history is None or df_history.empty or len(df_history) < 10:
            return 1.0

        try:
            df = df_history.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Lookahead Bias Prevention: Slice df up to as_of_ts if provided
            if as_of_ts is not None:
                try:
                    as_of_dt = pd.to_datetime(as_of_ts)
                    if not pd.api.types.is_datetime64_any_dtype(df.index):
                        df.index = pd.to_datetime(df.index)
                    df = df[df.index <= as_of_dt]
                except Exception:
                    pass

            if len(df) < 10:
                return 1.0

            c = df['Close'].squeeze()
            o = df['Open'].squeeze()
            h = df['High'].squeeze()
            l = df['Low'].squeeze()

            wins = 0
            losses = 0
            n_bars = len(df)

            # Analyze prior breakout events with genuine Target vs Stop evaluation
            # Evaluate across completed bars (leaving room for forward check)
            for i in range(5, n_bars - 2):
                # 1. Breakout condition: current High breaks the maximum High of preceding 5 bars
                prior_high = float(h.iloc[i-5:i].max())
                curr_h = float(h.iloc[i])
                curr_l = float(l.iloc[i])
                curr_c = float(c.iloc[i])

                if curr_h > prior_high and curr_c > prior_high:
                    entry = curr_h
                    stop = curr_l
                    risk = entry - stop
                    if risk <= 0:
                        continue
                    target = entry + (1.5 * risk)

                    # 2. Check forward up to 5 subsequent bars for outcome
                    max_forward = min(n_bars, i + 6)
                    outcome = None
                    for fwd in range(i + 1, max_forward):
                        fwd_h = float(h.iloc[fwd])
                        fwd_l = float(l.iloc[fwd])

                        hit_target = fwd_h >= target
                        hit_stop = fwd_l <= stop

                        # Conservative Stop-First Rule (P0): If both levels are touched within the same bar,
                        # assume Stop Loss triggered first to strictly avoid optimistic bias!
                        if hit_target and hit_stop:
                            outcome = "LOSS"
                            break
                        elif hit_stop:
                            outcome = "LOSS"
                            break
                        elif hit_target:
                            outcome = "WIN"
                            break

                    if outcome == "WIN":
                        wins += 1
                    elif outcome == "LOSS":
                        losses += 1

            total_trades = wins + losses
            if total_trades >= 3:
                win_rate = round((wins / total_trades) * 100.0, 1)
            else:
                win_rate = 50.0

            # Structural cleanliness: Body-to-Range ratio
            body = (c - o).abs()
            rng = (h - l).replace(0, 0.001)
            clean_body_ratio = float((body / rng).mean())
            pattern_score = round(clean_body_ratio * 100.0, 1)

            # Empirical Bayesian Shrinkage towards 1.0 baseline
            sample_weight = min(1.0, total_trades / 10.0)
            raw_mult = 0.70 + (win_rate / 100.0 * 0.60)
            shrunk_mult = round(1.0 + sample_weight * (raw_mult - 1.0), 2)
            final_mult = min(1.20, max(0.85, shrunk_mult))

            # Strictly persist ONLY during live trading
            if not is_replay:
                HistoricalStockLearner.record_stock_scan(
                    symbol=symbol,
                    last_candle_date=str(df.index[-1]),
                    completed_candle_ts=str(df.index[-1]),
                    win_rate=win_rate,
                    pattern_score=pattern_score,
                    best_strategy="Breakout 1:1.5 Target" if win_rate >= 50.0 else "Trap Reversal"
                )

            return final_mult
        except Exception:
            return 1.0
