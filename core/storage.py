import os
import sqlite3
import json
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "trading_signals.db")

# One-time migration: If old root trading_signals.db exists and new data/ db does not, copy it over
OLD_ROOT_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trading_signals.db")
if os.path.exists(OLD_ROOT_DB) and not os.path.exists(DB_PATH):
    try:
        import shutil
        shutil.move(OLD_ROOT_DB, DB_PATH)
        print("[STORAGE MIGRATION] Migrated legacy root trading_signals.db to data/trading_signals.db")
    except Exception as me:
        print("[STORAGE MIGRATION] Error migrating database:", me)

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
        ("closed_at", "REAL"),
        ("last_candle_time", "REAL")
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
        ORDER BY id DESC LIMIT 1
        """, (sym, direction, strategy, now - 900))
        existing = cursor.fetchone()
        if existing:
            existing_id = existing["id"]
            conn.close()
            return existing_id

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
        return signal_id
    except Exception as e:
        print("[STORAGE] Error logging signal:", e)
        return None

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

def evaluate_open_outcomes(data_map):
    """
    Evaluates all ACTIVE signals against chronological OHLC candles formed AFTER signal entry.
    Supports either:
    1. Full DataFrame per symbol (chronologically iterates each subsequent 15m candle)
    2. Dict with candle timestamps
    
    Guarantees:
    - Never evaluates candles timestamped before or equal to signal generation.
    - Tracks last_candle_time per signal to process all intermediate candles in chronological order.
    - If a single candle touches both Target and SL, conservatively marks as LOSS/STOPPED OUT.
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
            data = None
            for k in (sym, f"{clean_sym}.NS", clean_sym):
                if k in data_map:
                    data = data_map[k]
                    break
            if data is None:
                continue

            entry = float(sig["entry_price"])
            sl = float(sig["stop_loss"])
            tp1 = float(sig["target_1"])
            direction = sig["direction"]
            strategy = sig["strategy"]
            sig_time = float(sig["timestamp"])
            last_candle_time = float(sig["last_candle_time"] or sig_time)
            mfe = float(sig["mfe"] or 0.0)
            mae = float(sig["mae"] or 0.0)

            outcome = "ACTIVE"
            final_pnl = 0.0
            newest_candle_time = last_candle_time

            # Case A: Full DataFrame provided
            import pandas as pd
            if isinstance(data, pd.DataFrame) and not data.empty:
                df = data.copy()
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index)
                
                # Iterate each completed candle formed strictly after the signal timestamp
                now_ts = time.time()
                for idx, (c_dt, row) in enumerate(df.iterrows()):
                    c_ts = c_dt.timestamp() if hasattr(c_dt, 'timestamp') else float(c_dt)
                    # Ignore candles at or before signal creation, or already processed
                    if c_ts <= last_candle_time or c_ts <= sig_time:
                        continue

                    # If this is the very last candle in df during live market hours and candle hasn't closed yet (300s for 5m candle), do not process it
                    is_last_candle = (idx == len(df) - 1)
                    if is_last_candle and (c_ts + 300 > now_ts):
                        continue

                    newest_candle_time = max(newest_candle_time, c_ts)
                    c_high = float(row.get('High', entry))
                    c_low = float(row.get('Low', entry))

                    if direction == "BUY":
                        mfe = max(mfe, c_high - entry)
                        mae = min(mae, c_low - entry)
                        hit_tp = c_high >= tp1
                        hit_sl = c_low <= sl
                        if hit_tp and hit_sl:
                            outcome = "LOSS"
                            final_pnl = round(sl - entry, 2)
                            break
                        elif hit_tp:
                            outcome = "WIN"
                            final_pnl = round(tp1 - entry, 2)
                            break
                        elif hit_sl:
                            outcome = "LOSS"
                            final_pnl = round(sl - entry, 2)
                            break

                    elif direction == "SELL":
                        mfe = max(mfe, entry - c_low)
                        mae = min(mae, entry - c_high)
                        hit_tp = c_low <= tp1
                        hit_sl = c_high >= sl
                        if hit_tp and hit_sl:
                            outcome = "LOSS"
                            final_pnl = round(entry - sl, 2)
                            break
                        elif hit_tp:
                            outcome = "WIN"
                            final_pnl = round(entry - tp1, 2)
                            break
                        elif hit_sl:
                            outcome = "LOSS"
                            final_pnl = round(entry - sl, 2)
                            break

            # Case B: Single candle dictionary
            elif isinstance(data, dict):
                c_ts = float(data.get("timestamp", now))
                # Enforce completed-candle rule: candle must be finished (c_ts + 300 <= now)
                is_completed = data.get("is_completed", (c_ts + 300 <= now))
                if is_completed and c_ts > last_candle_time and c_ts > sig_time:
                    newest_candle_time = c_ts
                    c_high = float(data.get("high", entry))
                    c_low = float(data.get("low", entry))

                    if direction == "BUY":
                        mfe = max(mfe, c_high - entry)
                        mae = min(mae, c_low - entry)
                        hit_tp = c_high >= tp1
                        hit_sl = c_low <= sl
                        if hit_tp and hit_sl:
                            outcome = "LOSS"
                            final_pnl = round(sl - entry, 2)
                        elif hit_tp:
                            outcome = "WIN"
                            final_pnl = round(tp1 - entry, 2)
                        elif hit_sl:
                            outcome = "LOSS"
                            final_pnl = round(sl - entry, 2)
                    elif direction == "SELL":
                        mfe = max(mfe, entry - c_low)
                        mae = min(mae, entry - c_high)
                        hit_tp = c_low <= tp1
                        hit_sl = c_high >= sl
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
            SET mfe = ?, mae = ?, outcome = ?, final_pnl = ?, closed_at = ?, last_candle_time = ?
            WHERE id = ?
            """, (mfe, mae, outcome, final_pnl, now if outcome != "ACTIVE" else None, newest_candle_time, sig["id"]))

            if outcome in ("WIN", "LOSS"):
                update_strategy_performance(cursor, strategy, sig["id"])

        conn.commit()
        conn.close()
    except Exception as e:
        print("[STORAGE] Error evaluating outcomes:", e)

def update_strategy_performance(cursor, strategy, signal_id=None):
    """
    Computes institutional performance metrics across:
    1. Primary strategy
    2. All confluence strategies associated with the signal via signal_strategies
    - Signed Average R-Multiple: (+1.5R on wins, -1.0R on losses)
    - Valid Profit Factor: gross_profit / gross_loss (or None/Inf formatted if zero losses)
    """
    # Fetch all strategies to update (primary strategy + any co-triggered strategies)
    strategies_to_update = {strategy}
    if signal_id:
        try:
            cursor.execute("SELECT strategy_name FROM signal_strategies WHERE signal_id = ?", (signal_id,))
            for r in cursor.fetchall():
                strategies_to_update.add(r["strategy_name"])
        except Exception:
            pass

    for strat in strategies_to_update:
        # Check trades matching either primary strategy OR co-triggered in signal_strategies
        cursor.execute("""
        SELECT hs.outcome, hs.final_pnl, hs.entry_price, hs.stop_loss, hs.target_1 
        FROM historical_signals hs
        LEFT JOIN signal_strategies ss ON hs.id = ss.signal_id
        WHERE (hs.strategy = ? OR ss.strategy_name = ?) AND hs.outcome IN ('WIN', 'LOSS')
        GROUP BY hs.id
        """, (strat, strat))
        trades = cursor.fetchall()
        if not trades:
            continue

        total = len(trades)
        wins = len([t for t in trades if t["outcome"] == "WIN"])
        losses = len([t for t in trades if t["outcome"] == "LOSS"])
        win_rate = round((wins / total) * 100, 1)

        gross_profit = sum(float(t["final_pnl"]) for t in trades if float(t["final_pnl"]) > 0)
        gross_loss = abs(sum(float(t["final_pnl"]) for t in trades if float(t["final_pnl"]) < 0))

        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        else:
            profit_factor = 99.0 if gross_profit > 0 else 1.0

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
        """, (strat, total, wins, losses, win_rate, avg_rr, profit_factor, time.time()))

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
