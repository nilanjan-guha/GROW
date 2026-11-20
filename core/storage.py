import os
import sqlite3
import json
import time

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trading_signals.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS historical_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        datetime_str TEXT,
        symbol TEXT,
        direction TEXT,
        strategy TEXT,
        entry_price REAL,
        stop_loss REAL,
        target_1 REAL,
        target_2 REAL,
        score REAL,
        confidence TEXT,
        catalyst TEXT,
        features_json TEXT,
        market_regime TEXT,
        mfe REAL DEFAULT 0.0,
        mae REAL DEFAULT 0.0,
        outcome TEXT DEFAULT 'ACTIVE',
        final_pnl REAL DEFAULT 0.0,
        closed_at REAL
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS strategy_performance (
        strategy TEXT PRIMARY KEY,
        total_signals INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        win_rate REAL DEFAULT 0.0,
        avg_rr REAL DEFAULT 0.0,
        profit_factor REAL DEFAULT 0.0,
        updated_at REAL
    )
    """)
    # Automatic migration to guarantee columns exist if table was created in an earlier schema
    cursor.execute("PRAGMA table_info(historical_signals)")
    existing_cols = {r[1] for r in cursor.fetchall()}
    for col_name, col_type in [
        ("mfe", "REAL DEFAULT 0.0"),
        ("mae", "REAL DEFAULT 0.0"),
        ("outcome", "TEXT DEFAULT 'ACTIVE'"),
        ("final_pnl", "REAL DEFAULT 0.0"),
        ("closed_at", "REAL")
    ]:
        if col_name not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE historical_signals ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass

    conn.commit()
    conn.close()

def log_signal(signal_dict):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO historical_signals (
            timestamp, datetime_str, symbol, direction, strategy, 
            entry_price, stop_loss, target_1, target_2, score, 
            confidence, catalyst, features_json, market_regime, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            time.strftime("%Y-%m-%d %H:%M:%S"),
            signal_dict.get("symbol"),
            signal_dict.get("direction"),
            signal_dict.get("strategy"),
            signal_dict.get("entry_price"),
            signal_dict.get("stop_loss"),
            signal_dict.get("target_1"),
            signal_dict.get("target_2"),
            signal_dict.get("score", 0.0),
            signal_dict.get("confidence", "MED"),
            signal_dict.get("catalyst", "Price Action"),
            json.dumps(signal_dict.get("features", {})),
            signal_dict.get("market_regime", "TRENDING"),
            "ACTIVE"
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print("[STORAGE] Error logging signal:", e)

def evaluate_open_outcomes(latest_prices):
    """
    Evaluates all ACTIVE signals against latest prices to update:
    MFE (Max Favourable Excursion), MAE (Max Adverse Excursion),
    and marks outcome as WIN (Target reached) or LOSS (Stop Loss hit).
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM historical_signals WHERE outcome = 'ACTIVE'")
        active_signals = cursor.fetchall()
        now = time.time()

        for sig in active_signals:
            sym = sig["symbol"]
            curr_price = latest_prices.get(sym) or latest_prices.get(f"{sym}.NS")
            if not curr_price:
                continue

            entry = float(sig["entry_price"])
            sl = float(sig["stop_loss"])
            tp1 = float(sig["target_1"])
            direction = sig["direction"]
            strategy = sig["strategy"]
            mfe = float(sig["mfe"] or 0.0)
            mae = float(sig["mae"] or 0.0)

            outcome = "ACTIVE"
            final_pnl = 0.0

            if direction == "BUY":
                curr_diff = curr_price - entry
                mfe = max(mfe, curr_diff)
                mae = min(mae, curr_diff)
                if curr_price >= tp1:
                    outcome = "WIN"
                    final_pnl = round(curr_price - entry, 2)
                elif curr_price <= sl:
                    outcome = "LOSS"
                    final_pnl = round(curr_price - entry, 2)
            elif direction == "SELL":
                curr_diff = entry - curr_price
                mfe = max(mfe, curr_diff)
                mae = min(mae, curr_diff)
                if curr_price <= tp1:
                    outcome = "WIN"
                    final_pnl = round(entry - curr_price, 2)
                elif curr_price >= sl:
                    outcome = "LOSS"
                    final_pnl = round(entry - curr_price, 2)

            cursor.execute("""
            UPDATE historical_signals 
            SET mfe = ?, mae = ?, outcome = ?, final_pnl = ?, closed_at = ?
            WHERE id = ?
            """, (mfe, mae, outcome, final_pnl, now if outcome != "ACTIVE" else None, sig["id"]))

            # If trade resolved, update strategy performance statistics
            if outcome in ("WIN", "LOSS"):
                update_strategy_performance(cursor, strategy, outcome)

        conn.commit()
        conn.close()
    except Exception as e:
        print("[STORAGE] Error evaluating outcomes:", e)

def update_strategy_performance(cursor, strategy, outcome):
    cursor.execute("SELECT * FROM strategy_performance WHERE strategy = ?", (strategy,))
    row = cursor.fetchone()
    if row:
        total = row["total_signals"] + 1
        wins = row["wins"] + (1 if outcome == "WIN" else 0)
        losses = row["losses"] + (1 if outcome == "LOSS" else 0)
        win_rate = round((wins / total) * 100, 1)
        profit_factor = round(wins / (losses or 1), 2)
        cursor.execute("""
        UPDATE strategy_performance 
        SET total_signals = ?, wins = ?, losses = ?, win_rate = ?, profit_factor = ?, updated_at = ?
        WHERE strategy = ?
        """, (total, wins, losses, win_rate, profit_factor, time.time(), strategy))
    else:
        wins = 1 if outcome == "WIN" else 0
        losses = 1 if outcome == "LOSS" else 0
        cursor.execute("""
        INSERT INTO strategy_performance (strategy, total_signals, wins, losses, win_rate, profit_factor, updated_at)
        VALUES (?, 1, ?, ?, ?, ?, ?)
        """, (strategy, wins, losses, 100.0 if wins else 0.0, 1.0, time.time()))

def get_strategy_performance_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM strategy_performance ORDER BY total_signals DESC")
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []

def get_recent_signals(limit=50):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM historical_signals ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print("[STORAGE] Error fetching signals:", e)
        return []

init_db()
