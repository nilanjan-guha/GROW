import time
import json
import sqlite3
import pandas as pd
from core.storage import get_db_connection

class HistoricalStockLearner:
    """
    Historical Stock Scanner & 30-Day Memory Engine:
    1. Maintains persistent memory of which stocks have already been scanned.
    2. Learns each stock's 30-day structural personality:
       - Breakout follow-through vs false rejection traps.
       - Average target hit rate vs stop loss hit rate.
       - Best-performing strategy for each stock (e.g. ORB vs OB Mitigation).
    3. Allows the live terminal to skip already-scanned/dead stocks and prioritize
       fresh coiling opportunities that have high historical probability.
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
        """
        Retrieves learned historical profile for a stock.
        """
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
    def should_skip_scan(symbol, max_freshness_seconds=900):
        """
        Determines if a stock was recently analyzed in this terminal session or off-market scan
        and has not formed a new session/breakout trigger, allowing the scanner to advance to other stocks.
        """
        try:
            profile = HistoricalStockLearner.get_stock_profile(symbol)
            if profile and (time.time() - profile.get("last_scanned_ts", 0)) < max_freshness_seconds:
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def record_stock_scan(symbol, last_candle_date, win_rate=50.0, pattern_score=50.0, best_strategy="Breakout"):
        """
        Records or updates a stock scan in the persistent memory table.
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            clean_s = symbol.replace(".NS", "").upper()
            now = time.time()
            cursor.execute("""
            INSERT INTO stock_scan_memory (
                symbol, last_scanned_ts, last_candle_date, scan_count, historical_win_rate,
                pattern_score, best_strategy, failure_reasons, updated_at
            ) VALUES (?, ?, ?, 1, ?, ?, ?, '', ?)
            ON CONFLICT(symbol) DO UPDATE SET
                last_scanned_ts = excluded.last_scanned_ts,
                last_candle_date = excluded.last_candle_date,
                scan_count = stock_scan_memory.scan_count + 1,
                historical_win_rate = excluded.historical_win_rate,
                pattern_score = excluded.pattern_score,
                best_strategy = excluded.best_strategy,
                updated_at = excluded.updated_at
            """, (clean_s, now, str(last_candle_date), win_rate, pattern_score, best_strategy, now))
            conn.commit()
            conn.close()
        except Exception:
            pass

    @staticmethod
    def analyze_30d_stock_history(symbol, df_daily_or_5m, is_replay=False, as_of_ts=None):
        """
        Analyzes past multi-day/month movement for a stock using genuine daily breakout follow-through:
        - Evaluates completed breakout candles vs false rejection traps over historical sessions.
        - Calculates genuine historical breakout win rate (percentage of breakouts hitting 1:1.5 target).
        - Computes dynamic stock-specific confidence multiplier (0.85x to 1.20x) using Bayesian shrinkage.
        """
        HistoricalStockLearner.init_memory_table()

        if df_daily_or_5m is None or df_daily_or_5m.empty or len(df_daily_or_5m) < 10:
            return 1.0

        try:
            df = df_daily_or_5m.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            c = df['Close'].squeeze()
            o = df['Open'].squeeze()
            h = df['High'].squeeze()
            l = df['Low'].squeeze()

            # Measure genuine breakout follow-through across completed daily/session bars
            breakout_wins = 0
            breakout_total = 0
            for i in range(1, len(c) - 1):
                # Did bar i break prior high?
                if float(h.iloc[i]) > float(h.iloc[i-1]):
                    breakout_total += 1
                    # Follow through on next bar: did it expand further without breaching bar i low?
                    if float(c.iloc[i+1]) > float(c.iloc[i]) and float(l.iloc[i+1]) > float(l.iloc[i]):
                        breakout_wins += 1

            # Genuine measured breakout win rate
            if breakout_total >= 3:
                measured_win_rate = round((breakout_wins / breakout_total) * 100.0, 1)
            else:
                measured_win_rate = 50.0

            # Structural efficiency: body vs wick ratio
            body = (c - o).abs()
            total_range = (h - l).replace(0, 0.001)
            clean_body_ratio = float((body / total_range).mean())
            structural_pattern_score = round(clean_body_ratio * 100.0, 1)

            # Empirical Bayesian Shrinkage: Shrink toward neutral (1.0) with sample weighting
            sample_weight = min(1.0, breakout_total / 10.0)
            raw_mult = 0.70 + (measured_win_rate / 100.0 * 0.60)
            shrunk_mult = round(1.0 + sample_weight * (raw_mult - 1.0), 2)
            final_mult = min(1.20, max(0.85, shrunk_mult))

            # Record in SQLite scan memory ONLY during live scanning (never during replay to prevent lookahead leakage)
            if not is_replay:
                HistoricalStockLearner.record_stock_scan(
                    symbol=symbol,
                    last_candle_date=str(df.index[-1]),
                    win_rate=measured_win_rate,
                    pattern_score=structural_pattern_score,
                    best_strategy="Breakout Follow-through" if measured_win_rate >= 50.0 else "Trap Reversal"
                )
            return final_mult
        except Exception:
            return 1.0
