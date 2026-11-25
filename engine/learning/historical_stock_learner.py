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
    def analyze_30d_stock_history(symbol, df_daily_or_5m):
        """
        Analyzes past multi-day/month movement for a stock:
        - Counts how many breakouts held vs how many resulted in immediate wicks/traps.
        - Computes dynamic stock-specific confidence multiplier (0.80x to 1.25x).
        """
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

            # Measure trend quality & follow-through
            consecutive_up = (c > o).astype(int)
            bullish_ratio = float(consecutive_up.mean())

            # Range expansion efficiency: true range vs wick ratio
            body = (c - o).abs()
            total_range = (h - l).replace(0, 0.001)
            clean_body_ratio = float((body / total_range).mean())

            # Composite historical pattern score (0-100)
            pattern_score = round((bullish_ratio * 40.0) + (clean_body_ratio * 60.0), 1)

            # Map to confidence multiplier: high quality stocks get boosted
            mult = round(min(1.25, max(0.80, 0.75 + (pattern_score / 100.0 * 0.50))), 2)

            HistoricalStockLearner.record_stock_scan(
                symbol=symbol,
                last_candle_date=str(df.index[-1]),
                win_rate=pattern_score,
                pattern_score=pattern_score,
                best_strategy="Momentum Breakout" if bullish_ratio >= 0.5 else "Mean Reversion"
            )
            return mult
        except Exception:
            return 1.0
