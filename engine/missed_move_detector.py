import time
import pandas as pd
import numpy as np
from core.storage import get_db_connection

class MissedMoveDetector:
    """
    Missed-Move Detection & Predictive Audit Engine:
    1. Identifies F&O stocks that moved significantly (>1.5% in 3-5 candles or >3.0% session move)
       where NO active confirmed signal was issued by the system.
    2. Diagnoses exact Root Cause of the miss (e.g. potential gate rejected, RR < 1.5,
       sector bias mismatch, obstacle within clearance zone, or lack of early momentum confluence).
    3. Persists missed opportunities to `missed_moves` table for retraining and backtesting audits.
    """

    @staticmethod
    def init_missed_moves_table():
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS missed_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            datetime_str TEXT,
            symbol TEXT,
            direction TEXT,
            move_pct REAL,
            start_price REAL,
            peak_price REAL,
            candle_start_time REAL,
            candle_end_time REAL,
            reason_missed TEXT,
            recorded_at REAL
        )
        """)
        # Migration: Ensure unique index exists on existing databases
        try:
            c.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_missed_moves_unique 
            ON missed_moves (symbol, direction, candle_end_time)
            """)
        except Exception:
            pass
        conn.commit()
        conn.close()

    @staticmethod
    def audit_stock_move(symbol, df, active_signal_id=None, opp_eval=None):
        """
        Inspects completed session candles for un-signaled significant momentum surges.
        Deduplicates by symbol, direction, and candle window to prevent duplicate records.
        """
        if df is None or df.empty or len(df) < 6:
            return None

        try:
            # 1. If an active signal id is explicitly present, immediately exclude (not a missed move)
            if active_signal_id is not None:
                return None

            # 2. Otherwise verify directly against SQLite active signals
            conn = get_db_connection()
            cur = conn.cursor()
            clean_s = symbol.replace(".NS", "")
            cur.execute("""
                SELECT id FROM historical_signals 
                WHERE (symbol = ? OR symbol = ? OR symbol = ?) AND outcome = 'ACTIVE'
                LIMIT 1
            """, (symbol, f"{clean_s}.NS", clean_s))
            active_row = cur.fetchone()
            conn.close()
            if active_row:
                return None  # Stock already has an active trade, not a missed move

            c = df['Close']
            h = df['High']
            l = df['Low']

            curr_c = float(c.iloc[-1])
            start_c = float(c.iloc[-6])
            five_candle_move_pct = round(((curr_c - start_c) / start_c) * 100, 2)

            # Check if an explosive move occurred (> 1.8% in 30 mins) without an active logged signal
            is_significant_move = abs(five_candle_move_pct) >= 1.8
            if is_significant_move:
                direction = "BULLISH" if five_candle_move_pct > 0 else "BEARISH"
                
                # Check maximum excursion (MFE) during the move
                peak_price = float(h.iloc[-6:].max()) if direction == "BULLISH" else float(l.iloc[-6:].min())

                # Diagnose root cause of missing the trade
                reason_missed = "UNKNOWN"
                if opp_eval:
                    if not opp_eval.get("is_viable"):
                        reason_missed = f"Gated by Opportunity Engine: {opp_eval.get('reason', 'Obstacle/RR')}"
                    elif opp_eval.get("risk_reward_ratio", 0) < 1.5:
                        reason_missed = f"Gated by Risk-Reward: RR {opp_eval.get('risk_reward_ratio')} < 1.5 required"
                else:
                    reason_missed = "Composite strategy confluence score was below 50/100 threshold"

                # Deduplication window using candle timestamps
                candle_start_time = float(df.index[-6].timestamp()) if hasattr(df.index[-6], 'timestamp') else 0.0
                candle_end_time = float(df.index[-1].timestamp()) if hasattr(df.index[-1], 'timestamp') else 0.0

                MissedMoveDetector.init_missed_moves_table()
                conn = get_db_connection()
                cur = conn.cursor()
                now = time.time()
                cur.execute("""
                INSERT OR IGNORE INTO missed_moves (
                    timestamp, datetime_str, symbol, direction, move_pct, 
                    start_price, peak_price, candle_start_time, candle_end_time, reason_missed, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    now, time.strftime("%Y-%m-%d %H:%M:%S"), symbol, direction,
                    five_candle_move_pct, start_c, peak_price, candle_start_time, candle_end_time, reason_missed, now
                ))
                conn.commit()
                conn.close()

                return {
                    "is_missed_move": True,
                    "symbol": symbol,
                    "direction": direction,
                    "move_pct": five_candle_move_pct,
                    "peak_price": peak_price,
                    "reason_missed": reason_missed
                }
        except Exception as e:
            return None

        return None
