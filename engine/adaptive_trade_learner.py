import time
import json
import math
from core.storage import get_db_connection

class AdaptiveTradeLearner:
    """
    Intraday Trade Learning & Post-Mortem Feedback Engine:
    1. Diagnoses WHY a trade won or lost (e.g. SL hit due to reversal wick, premature entry, low RVOL, or clean TP hit).
    2. Stores post-mortem analysis for every closed trade.
    3. Calculates real-time strategy & setup multipliers to boost proven patterns and penalize failing setups in future rankings.
    """

    @staticmethod
    def init_learning_tables():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_post_mortems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER,
            symbol TEXT,
            direction TEXT,
            strategy TEXT,
            outcome TEXT,
            net_pnl REAL,
            mfe REAL,
            mae REAL,
            hold_time_minutes REAL,
            primary_reason TEXT,
            learnings_json TEXT,
            created_at REAL,
            FOREIGN KEY (signal_id) REFERENCES historical_signals(id)
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS pattern_learning_weights (
            pattern_key TEXT PRIMARY KEY,
            total_samples INTEGER DEFAULT 0,
            win_count INTEGER DEFAULT 0,
            loss_count INTEGER DEFAULT 0,
            avg_mfe REAL DEFAULT 0.0,
            avg_mae REAL DEFAULT 0.0,
            success_rate REAL DEFAULT 50.0,
            confidence_multiplier REAL DEFAULT 1.0,
            last_updated REAL
        )
        """)
        conn.commit()
        conn.close()

    @staticmethod
    def analyze_and_record_outcome(signal_row, candles_df=None):
        """
        Runs post-mortem root-cause analysis on a completed signal:
        - Why did it hit TP or SL?
        - Was volume high or fading?
        - Did it reach 80% to target before reversing (trailing SL failure)?
        - Was it stopped out immediately (false breakout)?
        """
        try:
            sig_id = signal_row["id"]
            symbol = signal_row["symbol"]
            direction = signal_row["direction"]
            strategy = signal_row.get("strategy") or "MOMENTUM_BREAKOUT"
            outcome = signal_row.get("outcome") or "UNKNOWN"
            entry_p = float(signal_row.get("entry_price") or 0)
            sl = float(signal_row.get("stop_loss") or 0)
            tp1 = float(signal_row.get("target_1") or 0)
            net_pnl = float(signal_row.get("net_pnl") or 0)
            mfe = float(signal_row.get("mfe") or 0)
            mae = float(signal_row.get("mae") or 0)
            t_entry = float(signal_row.get("timestamp") or time.time())
            t_close = float(signal_row.get("closed_at") or time.time())
            hold_minutes = max(0.0, round((t_close - t_entry) / 60.0, 1))

            target_dist = abs(tp1 - entry_p) if (tp1 and entry_p) else 1.0
            sl_dist = abs(entry_p - sl) if (sl and entry_p) else 1.0

            mfe_ratio = round(mfe / target_dist, 2) if target_dist > 0 else 0.0
            mae_ratio = round(abs(mae) / sl_dist, 2) if sl_dist > 0 else 0.0

            is_win = "WIN" in outcome or outcome == "TP1_HIT" or outcome == "TP2_HIT" or net_pnl > 0
            primary_reason = ""
            diagnosis = {}

            if is_win:
                if mfe_ratio >= 1.0:
                    primary_reason = "Clean Target Reached: Strong momentum continuation with minimal drawdown."
                else:
                    primary_reason = "Profitable Exit: Reached partial target before session close."
                diagnosis["success_factor"] = "High directional volume follow-through"
            else:
                if mfe_ratio >= 0.70:
                    primary_reason = f"Reversal Near Target: Move went {int(mfe_ratio*100)}% to TP1 before reversing into SL (Recommend tighter trailing SL)."
                    diagnosis["failure_factor"] = "Trailing stop too loose after strong initial move"
                elif mae_ratio >= 1.0 and hold_minutes <= 15:
                    primary_reason = "Immediate False Breakout: Trap against order block within first 15 mins (Liquidity trap)."
                    diagnosis["failure_factor"] = "Lack of follow-through volume; opposing institutional absorption"
                elif outcome == "CLOSED_EOD_SQUAREOFF":
                    primary_reason = "Time Decay / Sideways Consolidation: Trade stayed in tight range until 3:30 PM."
                    diagnosis["failure_factor"] = "Low intraday volatility expansion"
                else:
                    primary_reason = "Stop Loss Hit: Structural invalidation reached."
                    diagnosis["failure_factor"] = "Market structure reversed below key pivot"

            diagnosis["mfe_ratio"] = mfe_ratio
            diagnosis["mae_ratio"] = mae_ratio
            diagnosis["hold_minutes"] = hold_minutes

            conn = get_db_connection()
            cursor = conn.cursor()

            # Record post-mortem if not already recorded
            cursor.execute("SELECT id FROM trade_post_mortems WHERE signal_id = ?", (sig_id,))
            existing = cursor.fetchone()
            if not existing:
                cursor.execute("""
                INSERT INTO trade_post_mortems (
                    signal_id, symbol, direction, strategy, outcome, net_pnl, mfe, mae,
                    hold_time_minutes, primary_reason, learnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sig_id, symbol, direction, strategy, outcome, net_pnl, mfe, mae,
                    hold_minutes, primary_reason, json.dumps(diagnosis), time.time()
                ))

                # Update pattern learning weight
                pattern_key = f"{strategy}_{direction}"
                cursor.execute("SELECT * FROM pattern_learning_weights WHERE pattern_key = ?", (pattern_key,))
                p_row = cursor.fetchone()
                if p_row:
                    tot = p_row["total_samples"] + 1
                    wins = p_row["win_count"] + (1 if is_win else 0)
                    losses = p_row["loss_count"] + (0 if is_win else 1)
                    win_pct = round((wins / tot) * 100.0, 1)
                    # Multiplier: 0.75 (heavily failing) up to 1.30 (consistently winning)
                    mult = round(min(1.30, max(0.75, 0.70 + (win_pct / 100.0 * 0.60))), 2)
                    cursor.execute("""
                    UPDATE pattern_learning_weights
                    SET total_samples = ?, win_count = ?, loss_count = ?, success_rate = ?, confidence_multiplier = ?, last_updated = ?
                    WHERE pattern_key = ?
                    """, (tot, wins, losses, win_pct, mult, time.time(), pattern_key))
                else:
                    win_pct = 100.0 if is_win else 0.0
                    mult = 1.15 if is_win else 0.85
                    cursor.execute("""
                    INSERT INTO pattern_learning_weights (
                        pattern_key, total_samples, win_count, loss_count, success_rate, confidence_multiplier, last_updated
                    ) VALUES (?, 1, ?, ?, ?, ?, ?)
                    """, (pattern_key, 1 if is_win else 0, 0 if is_win else 1, win_pct, mult, time.time()))

                conn.commit()
            conn.close()
            return {"success": True, "reason": primary_reason, "diagnosis": diagnosis}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def get_pattern_multiplier(strategy, direction):
        """
        Returns dynamic score multiplier (0.75 - 1.30) learned from historical live market feedback.
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            pattern_key = f"{strategy}_{direction}"
            cursor.execute("SELECT confidence_multiplier FROM pattern_learning_weights WHERE pattern_key = ?", (pattern_key,))
            row = cursor.fetchone()
            conn.close()
            if row and row["confidence_multiplier"]:
                return float(row["confidence_multiplier"])
        except Exception:
            pass
        return 1.0

    @staticmethod
    def get_recent_learnings(limit=10):
        """
        Fetches the latest trade post-mortems for UI display.
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM trade_post_mortems 
            ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
            return rows
        except Exception:
            return []
