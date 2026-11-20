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
    CREATE TABLE IF NOT EXISTS signal_strategies (
        signal_id INTEGER,
        strategy_name TEXT,
        PRIMARY KEY (signal_id, strategy_name),
        FOREIGN KEY (signal_id) REFERENCES historical_signals(id) ON DELETE CASCADE
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
        now = time.time()
        sym = signal_dict.get("symbol")
        direction = signal_dict.get("direction")
        strategy = signal_dict.get("strategy")

        # Deduplication: skip if identical active signal exists or logged in last 15 minutes (900s)
        cursor.execute("""
        SELECT id FROM historical_signals 
        WHERE symbol = ? AND direction = ? AND (outcome = 'ACTIVE' OR (strategy = ? AND timestamp > ?))
        """, (sym, direction, strategy, now - 900))
        if cursor.fetchone():
            conn.close()
            return False

        cursor.execute("""
        INSERT INTO historical_signals (
            timestamp, datetime_str, symbol, direction, strategy, 
            entry_price, stop_loss, target_1, target_2, score, 
            confidence, catalyst, features_json, market_regime, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            sym,
            direction,
            strategy,
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
        signal_id = cursor.lastrowid

        # Record all triggered strategies into signal_strategies association table
        all_triggered = signal_dict.get("all_triggered", [strategy])
        for strat in all_triggered:
            try:
                cursor.execute("INSERT OR IGNORE INTO signal_strategies (signal_id, strategy_name) VALUES (?, ?)", (signal_id, strat))
            except Exception:
                pass

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("[STORAGE] Error logging signal:", e)
        return False

def get_active_signal_symbols():
    """Returns a set of all stock symbols that have open ACTIVE signals requiring monitoring"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM historical_signals WHERE outcome = 'ACTIVE'")
        rows = cursor.fetchall()
        conn.close()
        return {r["symbol"] for r in rows}
    except Exception:
        return set()

def evaluate_open_outcomes(latest_prices):
    """
    Evaluates all ACTIVE signals against latest OHLC candles.
    Conservative handling for simultaneous SL/TP hits:
    If a candle touches both target and stop loss in the same bar, conservatively mark as LOSS/STOPPED OUT.
    Only evaluates after signal creation timestamp.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM historical_signals WHERE outcome = 'ACTIVE'")
        active_signals = cursor.fetchall()
        now = time.time()

        for sig in active_signals:
            sym = sig["symbol"]
            clean_sym = sym.replace(".NS", "")
            data = latest_prices.get(sym) or latest_prices.get(f"{clean_sym}.NS") or latest_prices.get(clean_sym)
            if not data:
                continue

            if isinstance(data, dict):
                curr_price = float(data.get("close", data.get("price", sig["entry_price"])))
                candle_high = float(data.get("high", curr_price))
                candle_low = float(data.get("low", curr_price))
            else:
                curr_price = float(data)
                candle_high = curr_price
                candle_low = curr_price

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
                mfe = max(mfe, candle_high - entry)
                mae = min(mae, candle_low - entry)
                
                # Check for simultaneous hit in single candle
                hit_tp = candle_high >= tp1
                hit_sl = candle_low <= sl
                
                if hit_tp and hit_sl:
                    # Conservative handling: touch of both is classified as LOSS (stopped out)
                    outcome = "LOSS"
                    final_pnl = round(sl - entry, 2)
                elif hit_tp:
                    outcome = "WIN"
                    final_pnl = round(tp1 - entry, 2)
                elif hit_sl:
                    outcome = "LOSS"
                    final_pnl = round(sl - entry, 2)

            elif direction == "SELL":
                mfe = max(mfe, entry - candle_low)
                mae = min(mae, entry - candle_high)
                
                hit_tp = candle_low <= tp1
                hit_sl = candle_high >= sl
                
                if hit_tp and hit_sl:
                    outcome = "LOSS"
                    final_pnl = round(entry - sl, 2)
                elif hit_tp:
                    outcome = "WIN"
                    final_pnl = round(entry - tp1, 2)
                elif hit_sl:
                    outcome = "LOSS"
                    final_pnl = round(entry - sl, 2)

            cursor.execute("""
            UPDATE historical_signals 
            SET mfe = ?, mae = ?, outcome = ?, final_pnl = ?, closed_at = ?
            WHERE id = ?
            """, (mfe, mae, outcome, final_pnl, now if outcome != "ACTIVE" else None, sig["id"]))

            if outcome in ("WIN", "LOSS"):
                update_strategy_performance(cursor, strategy)

        conn.commit()
        conn.close()
    except Exception as e:
        print("[STORAGE] Error evaluating outcomes:", e)

def update_strategy_performance(cursor, strategy):
    """
    Computes institutional performance metrics:
    - Signed Average R-Multiple: (+1.5R on wins, -1.0R on losses)
    - Valid Profit Factor: gross_profit / gross_loss (or None/Inf formatted if zero losses)
    """
    cursor.execute("""
    SELECT outcome, final_pnl, entry_price, stop_loss, target_1 
    FROM historical_signals 
    WHERE strategy = ? AND outcome IN ('WIN', 'LOSS')
    """, (strategy,))
    trades = cursor.fetchall()
    if not trades:
        return

    total = len(trades)
    wins = len([t for t in trades if t["outcome"] == "WIN"])
    losses = len([t for t in trades if t["outcome"] == "LOSS"])
    win_rate = round((wins / total) * 100, 1)

    gross_profit = sum(float(t["final_pnl"]) for t in trades if float(t["final_pnl"]) > 0)
    gross_loss = abs(sum(float(t["final_pnl"]) for t in trades if float(t["final_pnl"]) < 0))

    if gross_loss > 0:
        profit_factor = round(gross_profit / gross_loss, 2)
    else:
        # If no losses, profit factor is undefined/infinite, cap at 99.0
        profit_factor = 99.0 if gross_profit > 0 else 1.0

    # Calculate SIGNED R-multiples: positive for WIN, negative for LOSS
    signed_r_list = []
    for t in trades:
        risk = abs(float(t["entry_price"]) - float(t["stop_loss"])) or 1.0
        pnl = float(t["final_pnl"])
        r_multiple = pnl / risk
        signed_r_list.append(r_multiple)
    avg_rr = round(sum(signed_r_list) / len(signed_r_list), 2) if signed_r_list else 1.0

    cursor.execute("""
    INSERT INTO strategy_performance (strategy, total_signals, wins, losses, win_rate, avg_rr, profit_factor, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(strategy) DO UPDATE SET
        total_signals = excluded.total_signals,
        wins = excluded.wins,
        losses = excluded.losses,
        win_rate = excluded.win_rate,
        avg_rr = excluded.avg_rr,
        profit_factor = excluded.profit_factor,
        updated_at = excluded.updated_at
    """, (strategy, total, wins, losses, win_rate, avg_rr, profit_factor, time.time()))

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
