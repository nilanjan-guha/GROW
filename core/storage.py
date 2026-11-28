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
        net_profit_factor REAL DEFAULT 0.0,
        total_gross_pnl REAL DEFAULT 0.0,
        total_net_pnl REAL DEFAULT 0.0,
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
        ("last_candle_time", "REAL"),
        ("current_stop_loss", "REAL"),
        ("tp1_hit", "INTEGER DEFAULT 0"),
        ("tp1_hit_time", "REAL"),
        ("tp2_hit", "INTEGER DEFAULT 0"),
        ("tp2_hit_time", "REAL"),
        ("net_pnl", "REAL DEFAULT 0.0"),
        ("brokerage_costs", "REAL DEFAULT 0.0"),
        ("signal_price", "REAL"),
        ("expected_entry_price", "REAL"),
        ("actual_entry_price", "REAL"),
        ("entry_timestamp", "REAL"),
        ("execution_status", "TEXT DEFAULT 'SIGNAL_GENERATED'"),
        ("signal_candle_start", "REAL"),
        ("signal_candle_close_time", "REAL"),
        ("signal_generated_at", "REAL"),
        ("quantity", "INTEGER DEFAULT 100"),
        ("instrument_type", "TEXT DEFAULT 'EQUITY_INTRADAY'"),
        ("instrument_id", "TEXT"),
        ("contract_expiry", "TEXT"),
        ("contract_source", "TEXT")
    ]:
        if col_name not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE historical_signals ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass

    cursor.execute("PRAGMA table_info(strategy_performance)")
    existing_perf_cols = {r[1] for r in cursor.fetchall()}
    for col_name, col_type in [
        ("net_profit_factor", "REAL DEFAULT 0.0"),
        ("total_gross_pnl", "REAL DEFAULT 0.0"),
        ("total_net_pnl", "REAL DEFAULT 0.0"),
        ("full_wins", "INTEGER DEFAULT 0"),
        ("partial_wins", "INTEGER DEFAULT 0"),
        ("net_positive_trades", "INTEGER DEFAULT 0"),
        ("broker_trades", "INTEGER DEFAULT 0"),
        ("paper_trades", "INTEGER DEFAULT 0")
    ]:
        if col_name not in existing_perf_cols:
            try:
                cursor.execute(f"ALTER TABLE strategy_performance ADD COLUMN {col_name} {col_type}")
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
            entry_price, signal_price, expected_entry_price, actual_entry_price, entry_timestamp, execution_status,
            signal_candle_start, signal_candle_close_time, signal_generated_at, quantity, instrument_type,
            instrument_id, contract_expiry, contract_source,
            stop_loss, current_stop_loss, target_1, target_2, score, 
            confidence, catalyst, features_json, market_regime, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            sym,
            direction,
            strategy,
            signal_dict.get("entry_price"),
            signal_dict.get("signal_price", signal_dict.get("entry_price")),
            signal_dict.get("expected_entry_price", signal_dict.get("entry_price")),
            signal_dict.get("actual_entry_price"),
            signal_dict.get("entry_timestamp"),
            signal_dict.get("execution_status", "SIGNAL_GENERATED"),
            signal_dict.get("signal_candle_start"),
            signal_dict.get("signal_candle_close_time"),
            signal_dict.get("signal_generated_at", now),
            int(signal_dict.get("quantity") or 100),
            signal_dict.get("instrument_type", "EQUITY_INTRADAY"),
            signal_dict.get("instrument_id"),
            str(signal_dict.get("contract_expiry")) if signal_dict.get("contract_expiry") is not None else None,
            signal_dict.get("contract_source"),
            signal_dict.get("stop_loss"),
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

def evaluate_candle_transition(direction, entry, orig_sl, sl, tp1, tp2, tp1_hit_prior, c_open, c_high, c_low, c_ts, est_costs, cost_calc_fn=None):
    """
    Consolidated pure function for 5-minute single-candle outcome evaluation.
    Used identically across both DataFrame and Dictionary evaluation paths.
    Enforces strict chronological sequencing, conservative same-candle stop-first ambiguity rules,
    and dynamically recalculates transaction friction at actual exit prices when cost_calc_fn is provided.
    """
    # Normalize direction to BUY or SELL
    norm_dir = {"BUY": "BUY", "BULLISH": "BUY", "SELL": "SELL", "BEARISH": "SELL"}.get(str(direction).upper())
    if not norm_dir:
        # Unrecognized direction: fail closed, do not execute
        return {
            "outcome": "ACTIVE", "final_pnl": 0.0, "net_pnl": 0.0, "brokerage_costs": est_costs,
            "tp1_hit": 1 if tp1_hit_prior else 0, "tp1_hit_time": None, "tp2_hit": 0, "tp2_hit_time": None,
            "sl": sl, "should_break": False
        }

    # Reject missing, non-finite, non-positive, or structurally inconsistent candle values (corrupted feed protection)
    import math
    try:
        f_entry = float(entry)
        f_orig_sl = float(orig_sl)
        f_sl = float(sl)
        f_tp1 = float(tp1)
        f_open = float(c_open)
        f_high = float(c_high)
        f_low = float(c_low)
    except (ValueError, TypeError):
        return {
            "outcome": "ACTIVE", "final_pnl": 0.0, "net_pnl": 0.0, "brokerage_costs": est_costs,
            "tp1_hit": 1 if tp1_hit_prior else 0, "tp1_hit_time": None, "tp2_hit": 0, "tp2_hit_time": None,
            "sl": sl, "should_break": False
        }

    for val in (f_entry, f_orig_sl, f_sl, f_tp1, f_open, f_high, f_low):
        if not math.isfinite(val) or val <= 0:
            return {
                "outcome": "ACTIVE", "final_pnl": 0.0, "net_pnl": 0.0, "brokerage_costs": est_costs,
                "tp1_hit": 1 if tp1_hit_prior else 0, "tp1_hit_time": None, "tp2_hit": 0, "tp2_hit_time": None,
                "sl": sl, "should_break": False
            }

    # Structural consistency: High must be >= max(Open, Low), Low must be <= min(Open, High)
    # Strictly enforces: f_low <= f_open <= f_high and f_high >= f_low
    if not (f_low <= f_open <= f_high) or (f_high < f_low):
        return {
            "outcome": "ACTIVE", "final_pnl": 0.0, "net_pnl": 0.0, "brokerage_costs": est_costs,
            "tp1_hit": 1 if tp1_hit_prior else 0, "tp1_hit_time": None, "tp2_hit": 0, "tp2_hit_time": None,
            "sl": sl, "should_break": False
        }

    entry, orig_sl, sl, tp1, c_open, c_high, c_low = f_entry, f_orig_sl, f_sl, f_tp1, f_open, f_high, f_low

    if tp2 is not None:
        try:
            f_tp2 = float(tp2)
            tp2 = f_tp2 if (math.isfinite(f_tp2) and f_tp2 > 0) else None
        except (ValueError, TypeError):
            tp2 = None

    direction = norm_dir
    outcome = "ACTIVE"
    final_pnl = 0.0
    brokerage_costs = est_costs
    net_pnl = 0.0
    tp1_hit = 1 if tp1_hit_prior else 0
    tp1_hit_time = None
    tp2_hit = 0
    tp2_hit_time = None
    updated_sl = sl
    should_break = False

    def compute_net(pnl_val, exit_p1, exit_p2=None, is_partial=False):
        if cost_calc_fn:
            try:
                if is_partial and exit_p2 is not None:
                    # Accurately compute multi-exit costs with 1 entry leg and 2 half-quantity exit legs
                    tot_cost = cost_calc_fn([(exit_p1, 0.5), (exit_p2, 0.5)])
                else:
                    tot_cost = cost_calc_fn([(exit_p1, 1.0)])
                return round(tot_cost, 2), round(pnl_val - tot_cost, 2)
            except Exception:
                pass
        return est_costs, round(pnl_val - est_costs, 2)

    if direction == "BUY":
        # 1. Position entered this candle with TP1 already confirmed in a previous candle:
        if tp1_hit_prior:
            if c_open < sl:
                outcome = "WIN_PARTIAL"
                final_pnl = round(0.5 * (tp1 - entry) + 0.5 * (c_open - entry), 2)
                brokerage_costs, net_pnl = compute_net(final_pnl, tp1, c_open, is_partial=True)
                should_break = True
            else:
                hit_tp2_now = tp2 and (c_high >= tp2)
                hit_be_sl_now = c_low <= sl
                if hit_tp2_now and hit_be_sl_now:
                    outcome = "WIN_PARTIAL"
                    final_pnl = round(0.5 * (tp1 - entry) + 0.5 * (sl - entry), 2)
                    brokerage_costs, net_pnl = compute_net(final_pnl, tp1, sl, is_partial=True)
                    should_break = True
                elif hit_tp2_now:
                    tp2_hit = 1
                    tp2_hit_time = c_ts
                    outcome = "WIN_FULL"
                    final_pnl = round(0.5 * (tp1 - entry) + 0.5 * (tp2 - entry), 2)
                    brokerage_costs, net_pnl = compute_net(final_pnl, tp1, tp2, is_partial=True)
                    should_break = True
                elif hit_be_sl_now:
                    outcome = "WIN_PARTIAL"
                    final_pnl = round(0.5 * (tp1 - entry) + 0.5 * (sl - entry), 2)
                    brokerage_costs, net_pnl = compute_net(final_pnl, tp1, sl, is_partial=True)
                    should_break = True

        # 2. Position entered this candle with TP1 not yet hit:
        else:
            if c_open <= orig_sl:
                outcome = "LOSS"
                final_pnl = round(c_open - entry, 2)
                brokerage_costs, net_pnl = compute_net(final_pnl, c_open, is_partial=False)
                should_break = True
            else:
                hit_tp1_now = c_high >= tp1
                hit_orig_sl_now = c_low <= orig_sl
                if hit_tp1_now and hit_orig_sl_now:
                    outcome = "LOSS"
                    final_pnl = round(orig_sl - entry, 2)
                    brokerage_costs, net_pnl = compute_net(final_pnl, orig_sl, is_partial=False)
                    should_break = True
                elif hit_orig_sl_now:
                    outcome = "LOSS"
                    final_pnl = round(orig_sl - entry, 2)
                    brokerage_costs, net_pnl = compute_net(final_pnl, orig_sl, is_partial=False)
                    should_break = True
                elif hit_tp1_now:
                    tp1_hit = 1
                    tp1_hit_time = c_ts
                    updated_sl = max(sl, entry)
                    hit_tp2_same = bool(tp2 and c_high >= tp2)
                    hit_be_same = bool(c_low <= updated_sl)

                    if hit_tp2_same and hit_be_same:
                        # Ambiguous same-candle sequence: conservative stop-first
                        outcome = "WIN_PARTIAL"
                        final_pnl = round(0.5 * (tp1 - entry) + 0.5 * (updated_sl - entry), 2)
                        brokerage_costs, net_pnl = compute_net(final_pnl, tp1, updated_sl, is_partial=True)
                        should_break = True
                    elif hit_be_same:
                        outcome = "WIN_PARTIAL"
                        final_pnl = round(0.5 * (tp1 - entry) + 0.5 * (updated_sl - entry), 2)
                        brokerage_costs, net_pnl = compute_net(final_pnl, tp1, updated_sl, is_partial=True)
                        should_break = True
                    elif hit_tp2_same:
                        tp2_hit = 1
                        tp2_hit_time = c_ts
                        outcome = "WIN_FULL"
                        final_pnl = round(0.5 * (tp1 - entry) + 0.5 * (tp2 - entry), 2)
                        brokerage_costs, net_pnl = compute_net(final_pnl, tp1, tp2, is_partial=True)
                        should_break = True

    elif direction == "SELL":
        # 1. Position entered this candle with TP1 already confirmed in a previous candle:
        if tp1_hit_prior:
            if c_open > sl:
                outcome = "WIN_PARTIAL"
                final_pnl = round(0.5 * (entry - tp1) + 0.5 * (entry - c_open), 2)
                brokerage_costs, net_pnl = compute_net(final_pnl, tp1, c_open, is_partial=True)
                should_break = True
            else:
                hit_tp2_now = tp2 and (c_low <= tp2)
                hit_be_sl_now = c_high >= sl
                if hit_tp2_now and hit_be_sl_now:
                    outcome = "WIN_PARTIAL"
                    final_pnl = round(0.5 * (entry - tp1) + 0.5 * (entry - sl), 2)
                    brokerage_costs, net_pnl = compute_net(final_pnl, tp1, sl, is_partial=True)
                    should_break = True
                elif hit_tp2_now:
                    tp2_hit = 1
                    tp2_hit_time = c_ts
                    outcome = "WIN_FULL"
                    final_pnl = round(0.5 * (entry - tp1) + 0.5 * (entry - tp2), 2)
                    brokerage_costs, net_pnl = compute_net(final_pnl, tp1, tp2, is_partial=True)
                    should_break = True
                elif hit_be_sl_now:
                    outcome = "WIN_PARTIAL"
                    final_pnl = round(0.5 * (entry - tp1) + 0.5 * (entry - sl), 2)
                    brokerage_costs, net_pnl = compute_net(final_pnl, tp1, sl, is_partial=True)
                    should_break = True

        # 2. Position entered this candle with TP1 not yet hit:
        else:
            if c_open >= orig_sl:
                outcome = "LOSS"
                final_pnl = round(entry - c_open, 2)
                brokerage_costs, net_pnl = compute_net(final_pnl, c_open, is_partial=False)
                should_break = True
            else:
                hit_tp1_now = c_low <= tp1
                hit_orig_sl_now = c_high >= orig_sl
                if hit_tp1_now and hit_orig_sl_now:
                    outcome = "LOSS"
                    final_pnl = round(entry - orig_sl, 2)
                    brokerage_costs, net_pnl = compute_net(final_pnl, orig_sl, is_partial=False)
                    should_break = True
                elif hit_orig_sl_now:
                    outcome = "LOSS"
                    final_pnl = round(entry - orig_sl, 2)
                    brokerage_costs, net_pnl = compute_net(final_pnl, orig_sl, is_partial=False)
                    should_break = True
                elif hit_tp1_now:
                    tp1_hit = 1
                    tp1_hit_time = c_ts
                    updated_sl = min(sl, entry)
                    hit_tp2_same = bool(tp2 and c_low <= tp2)
                    hit_be_same = bool(c_high >= updated_sl)

                    if hit_tp2_same and hit_be_same:
                        outcome = "WIN_PARTIAL"
                        final_pnl = round(0.5 * (entry - tp1) + 0.5 * (entry - updated_sl), 2)
                        brokerage_costs, net_pnl = compute_net(final_pnl, tp1, updated_sl, is_partial=True)
                        should_break = True
                    elif hit_be_same:
                        outcome = "WIN_PARTIAL"
                        final_pnl = round(0.5 * (entry - tp1) + 0.5 * (entry - updated_sl), 2)
                        brokerage_costs, net_pnl = compute_net(final_pnl, tp1, updated_sl, is_partial=True)
                        should_break = True
                    elif hit_tp2_same:
                        tp2_hit = 1
                        tp2_hit_time = c_ts
                        outcome = "WIN_FULL"
                        final_pnl = round(0.5 * (entry - tp1) + 0.5 * (entry - tp2), 2)
                        brokerage_costs, net_pnl = compute_net(final_pnl, tp1, tp2, is_partial=True)
                        should_break = True

    return {
        "outcome": outcome,
        "final_pnl": final_pnl,
        "net_pnl": net_pnl,
        "brokerage_costs": brokerage_costs,
        "tp1_hit": tp1_hit,
        "tp1_hit_time": tp1_hit_time,
        "tp2_hit": tp2_hit,
        "tp2_hit_time": tp2_hit_time,
        "sl": updated_sl,
        "should_break": should_break
    }

def validate_candle_record(c_ts, raw_open, raw_high, raw_low, now_ts, is_completed_flag=None, enforce_grid=True):
    """
    Unified authoritative validator for single 5-minute candles across both DataFrame and Dictionary paths.
    
    Strict guarantees:
    1. Timestamp must be a positive, finite float.
    2. 5-Minute Grid: candle timestamp must align with 5-minute multiples (300 seconds), allowing 1.5s clock jitter.
    3. Physical Clock Completion: candle cannot be in the future (c_ts + 300 <= now_ts).
    4. Completion Flag:
       - If omitted (None), candle is considered complete because clock has passed (c_ts + 300 <= now_ts).
       - If provided, it must be strictly a boolean True (or truthy int 1); False, 0, or strings like "false" are rejected.
    5. Structural OHLC integrity:
       - Open, High, Low must be positive, finite numbers.
       - Strictly Low <= Open <= High and High >= Low.
       
    Returns (True, c_ts, c_open, c_high, c_low) if valid, otherwise (False, None, None, None, None).
    """
    import math
    if c_ts is None:
        return False, None, None, None, None
    try:
        ts = float(c_ts)
        if not math.isfinite(ts) or ts <= 0:
            return False, None, None, None, None
    except (ValueError, TypeError):
        return False, None, None, None, None

    # 1. Mandatory 5-Minute exchange grid alignment (multiples of 300 seconds)
    # When enforce_grid is True (default), strictly reject timestamps off the 300s grid.
    if enforce_grid:
        grid_rem = round(ts) % 300
        if 2 <= grid_rem <= 298:
            return False, None, None, None, None

    # 2. Clock completion check: candle close (ts + 300) must be <= authoritative now
    if ts + 300 > now_ts:
        return False, None, None, None, None

    # 3. Explicit completion flag validation: reject non-boolean / string false values
    if is_completed_flag is not None:
        if isinstance(is_completed_flag, str) and is_completed_flag.lower().strip() in ("false", "0", "no"):
            return False, None, None, None, None
        if is_completed_flag is False or is_completed_flag == 0:
            return False, None, None, None, None
        # Must be boolean True or integer 1 (reject arbitrary numbers like 1.5, 2)
        if type(is_completed_flag) is bool:
            if not is_completed_flag:
                return False, None, None, None, None
        elif isinstance(is_completed_flag, int):
            if is_completed_flag != 1:
                return False, None, None, None, None
        else:
            return False, None, None, None, None

    # 4. OHLC structural integrity
    if raw_open is None or raw_high is None or raw_low is None:
        return False, None, None, None, None
    try:
        o = float(raw_open)
        h = float(raw_high)
        l = float(raw_low)
    except (ValueError, TypeError):
        return False, None, None, None, None

    if not (math.isfinite(o) and math.isfinite(h) and math.isfinite(l)):
        return False, None, None, None, None
    if o <= 0 or h <= 0 or l <= 0:
        return False, None, None, None, None
    if not (l <= o <= h) or (h < l):
        return False, None, None, None, None

    return True, ts, o, h, l

def evaluate_open_outcomes(data_map, force_eod=False, evaluation_time=None):
    """
    Evaluates all ACTIVE signals against chronological 5-minute OHLC candles formed strictly AFTER signal candle close.
    Supports either:
    1. Full DataFrame per symbol (chronologically iterates each subsequent 5-minute candle)
    2. Dict with candle timestamps
    
    Guarantees:
    - Never evaluates candles timestamped before or equal to the completed signal candle close.
    - If candle timestamp reflects candle START (standard exchange grid), evaluates candles where c_ts >= signal_candle_close_time.
    - Tracks last_candle_time per signal to process all intermediate candles in chronological order.
    - If a single candle touches both Target and SL, conservatively marks as LOSS/STOPPED OUT.
    """
    try:
        import datetime
        import pytz
        import pandas as pd
        import math
        outcomes_changed = False
        from engine.cost_engine import IndianMarketCosts

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM historical_signals WHERE outcome = 'ACTIVE'")
        active_signals = cursor.fetchall()

        # STRICT SEPARATION OF LIVE CLOCK vs REPLAY CLOCK:
        # 1. In LIVE mode (evaluation_time is None), time.time() is the authoritative clock.
        #    Candles timestamped in the future can NEVER advance 'now'. Future-dated candles FAIL-CLOSED.
        # 2. In REPLAY mode, the caller MUST provide an explicit evaluation_time.
        #    We NEVER infer or advance 'now' from incoming feed candle timestamps.
        if evaluation_time is not None:
            now = float(evaluation_time)
        else:
            now = time.time()

        # Auto-Squareoff at Market Close: if market is closed (past 15:30 IST or non-trading day),
        # any intraday trade from a previous session or after 15:30 IST is automatically squared off at the latest closing price!
        from engine.market_validator import MarketValidator
        ist_now = MarketValidator.get_ist_now()
        is_mkt_open = MarketValidator.is_market_open(ist_now)
        current_minute = ist_now.hour * 60 + ist_now.minute

        for sig in active_signals:
            sym = sig["symbol"]
            clean_sym = sym.replace(".NS", "")
            data = None
            for k in (sym, f"{clean_sym}.NS", clean_sym):
                if k in data_map:
                    data = data_map[k]
                    break

            entry = float(sig["entry_price"])
            orig_sl = float(sig["stop_loss"])
            curr_sl = float(sig["current_stop_loss"]) if sig["current_stop_loss"] is not None else orig_sl
            sl = curr_sl
            tp1 = float(sig["target_1"])
            tp2 = float(sig["target_2"]) if sig["target_2"] is not None else None
            direction = sig["direction"]
            strategy = sig["strategy"]
            sig_time = float(sig["timestamp"])
            last_candle_time = float(sig["last_candle_time"] or 0.0)
            sig_candle_close = float(sig["signal_candle_close_time"] or 0.0)
            # Threshold: candle must be formed after signal candle close time if available, or signal generation time
            # For 5-minute candles where index represents candle START (e.g. 09:20 start completes at 09:25),
            # if sig_candle_close is recorded as 09:20:00 (epoch), the first executable candle starting at 09:20:00 is evaluated!
            # If not recorded, fallback: c_ts > last_candle_time and (c_ts >= sig_time - 300 if c_ts is candle start)
            # Threshold: candle must be formed strictly after signal candle close time.
            # If signal_candle_close_time is missing, fail-closed for that signal (do not guess sig_time - 300)
            if sig_candle_close > 0:
                eval_min_time = sig_candle_close
            elif sig_time > 0:
                eval_min_time = sig_time
            else:
                continue

            mfe = float(sig["mfe"] or 0.0)
            mae = float(sig["mae"] or 0.0)

            outcome = sig["outcome"] or "ACTIVE"
            final_pnl = float(sig["final_pnl"] or 0.0)
            newest_candle_time = last_candle_time
            tp1_hit = int(sig["tp1_hit"] or 0)
            tp1_hit_time = sig["tp1_hit_time"]
            tp2_hit = int(sig["tp2_hit"] or 0)
            tp2_hit_time = sig["tp2_hit_time"]
            net_pnl = float(sig["net_pnl"] or 0.0)
            brokerage_costs = float(sig["brokerage_costs"] or 0.0)

            # Realistic Indian F&O equity / quantity-based cost model
            qty = int(sig["quantity"] or 100)
            inst_type = sig["instrument_type"] or "EQUITY_INTRADAY"
            default_costs = IndianMarketCosts.calculate_intraday_costs(entry, entry, quantity=qty, instrument_type=inst_type)
            est_costs = default_costs["cost_per_share"]

            def make_cost_calc(entry_p, quantity, inst):
                def _calc(exit_tuples):
                    # exit_tuples: list of (exit_price, fraction)
                    raw_exits = []
                    allocated_qty = 0
                    for idx, (ep, frac) in enumerate(exit_tuples):
                        is_last = (idx == len(exit_tuples) - 1)
                        if is_last:
                            sub_q = max(0, quantity - allocated_qty)
                        else:
                            sub_q = int(round(quantity * frac))
                            allocated_qty += sub_q
                        if sub_q > 0:
                            raw_exits.append((ep, sub_q))

                    if not raw_exits:
                        return est_costs

                    res_cost = IndianMarketCosts.calculate_multi_exit_costs(
                        entry_price=entry_p, exits=raw_exits, total_quantity=quantity, instrument_type=inst
                    )
                    return res_cost["cost_per_share"]
                return _calc

            cost_calc_fn = make_cost_calc(entry, qty, inst_type)

            # 0. Intraday Session Close Auto-Squareoff:
            # Applies when:
            # - Explicitly forced (force_eod=True)
            # - Signal is from a prior trading day (sig_dt.date() < current_eval_dt.date())
            # Note: During off-market hours or testing (evaluation_time is None and market closed), if data is passed
            # in data_map to evaluate intraday candles or test setups, we do NOT prematurely truncate them with EOD square-off
            # unless force_eod is True or the signal is from a prior day without fresh intra-session candles.
            try:
                sig_dt = datetime.datetime.fromtimestamp(sig_time, pytz.timezone('Asia/Kolkata'))
                current_eval_dt = datetime.datetime.fromtimestamp(now, pytz.timezone('Asia/Kolkata'))
                is_prior_day = sig_dt.date() < current_eval_dt.date()
                
                # EOD squareoff should apply if explicitly forced, OR if signal is from a prior calendar day
                # (and either no new candle feed is provided, or force_eod is enabled)
                if force_eod:
                    should_eod_squareoff = True
                elif is_prior_day:
                    should_eod_squareoff = True
                else:
                    should_eod_squareoff = False
            except Exception:
                should_eod_squareoff = False

            if should_eod_squareoff:
                exit_price = None
                if isinstance(data, pd.DataFrame) and not data.empty:
                    try:
                        raw_c = data['Close'].iloc[-1]
                        val_c = float(raw_c)
                        if math.isfinite(val_c) and val_c > 0:
                            exit_price = val_c
                    except Exception:
                        exit_price = None
                elif isinstance(data, dict):
                    try:
                        raw_ep = data.get("close", data.get("current_price"))
                        if raw_ep is not None:
                            val_ep = float(raw_ep)
                            if math.isfinite(val_ep) and val_ep > 0:
                                exit_price = val_ep
                    except Exception:
                        exit_price = None

                # For stale sessions from previous days where data wasn't passed in data_map:
                # Prefer the last verified completed 5-minute close; if unavailable, keep outcome unverified (do not invent an exit with signal_price/entry_price)
                if exit_price is None:
                    # Keep signal active and do not fabricate a fake breakeven exit or use old signal price
                    continue

                pnl = round(exit_price - entry if direction == "BUY" else entry - exit_price, 2)
                outcome = "CLOSED_EOD_SQUAREOFF" if not is_prior_day else "CLOSED_STALE_SESSION"
                final_pnl = pnl

                # Compute actual friction at actual exit_price across the position
                try:
                    if cost_calc_fn:
                        brokerage_costs = cost_calc_fn([(exit_price, 1.0)])
                    else:
                        actual_costs = IndianMarketCosts.calculate_intraday_costs(entry, exit_price, quantity=qty, instrument_type=inst_type)
                        brokerage_costs = actual_costs["cost_per_share"]
                except Exception:
                    brokerage_costs = est_costs

                net_pnl = round(final_pnl - brokerage_costs, 2)

                cursor.execute("""
                UPDATE historical_signals 
                SET outcome = ?, final_pnl = ?, closed_at = ?, net_pnl = ?, brokerage_costs = ?
                WHERE id = ?
                """, (outcome, final_pnl, now, net_pnl, brokerage_costs, sig["id"]))
                outcomes_changed = True

                # Run Trade Learning Post-Mortem Feedback
                try:
                    from engine.adaptive_trade_learner import AdaptiveTradeLearner
                    sig_updated = dict(sig)
                    sig_updated.update({"outcome": outcome, "final_pnl": final_pnl, "closed_at": now, "net_pnl": net_pnl})
                    AdaptiveTradeLearner.analyze_and_record_outcome(sig_updated)
                except Exception:
                    pass
                continue

            if data is None:
                continue

            # Case A: Full DataFrame provided
            import pandas as pd
            if isinstance(data, pd.DataFrame) and not data.empty:
                df = data.copy()
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    try:
                        df.index = pd.to_datetime(df.index)
                    except Exception:
                        continue
                
                # Check for duplicate timestamps and sort monotonically
                if df.index.has_duplicates:
                    df = df[~df.index.duplicated(keep='last')]
                if not df.index.is_monotonic_increasing:
                    df = df.sort_index()

                # Normalize to Asia/Kolkata timezone awareness to avoid local server timezone skew
                import pytz
                ist = pytz.timezone('Asia/Kolkata')
                if df.index.tz is None:
                    df.index = df.index.tz_localize(ist)
                elif str(df.index.tz) != 'Asia/Kolkata':
                    df.index = df.index.tz_convert(ist)

                # Iterate each completed candle formed strictly after the completed signal candle
                now_ts = time.time()
                for idx, (c_dt, row) in enumerate(df.iterrows()):
                    try:
                        raw_c_ts = c_dt.timestamp() if hasattr(c_dt, 'timestamp') else float(c_dt)
                    except Exception:
                        continue

                    # Safe extraction of OHLC
                    raw_open = row.get('Open')
                    raw_high = row.get('High')
                    raw_low = row.get('Low')

                    # Unified candle validation: enforces positive finite ts, 5-min grid, completion (ts + 300 <= now), and OHLC integrity
                    is_valid, c_ts, c_open, c_high, c_low = validate_candle_record(raw_c_ts, raw_open, raw_high, raw_low, now)
                    if not is_valid:
                        continue

                    if (c_ts - last_candle_time) <= 1e-3 or (c_ts - eval_min_time) < -1e-3:
                        continue

                    newest_candle_time = max(newest_candle_time, c_ts)

                    tp1_hit_prior = bool(tp1_hit)
                    if direction == "BUY":
                        mfe = max(mfe, c_high - entry)
                        mae = min(mae, c_low - entry)
                    else:
                        mfe = max(mfe, entry - c_low)
                        mae = min(mae, entry - c_high)

                    res = evaluate_candle_transition(
                        direction=direction, entry=entry, orig_sl=orig_sl, sl=sl,
                        tp1=tp1, tp2=tp2, tp1_hit_prior=tp1_hit_prior,
                        c_open=c_open, c_high=c_high, c_low=c_low, c_ts=c_ts,
                        est_costs=est_costs, cost_calc_fn=cost_calc_fn
                    )

                    outcome = res["outcome"]
                    final_pnl = res["final_pnl"]
                    net_pnl = res["net_pnl"]
                    brokerage_costs = res["brokerage_costs"]
                    if res["tp1_hit"]:
                        tp1_hit = 1
                        tp1_hit_time = res["tp1_hit_time"] or tp1_hit_time
                    if res["tp2_hit"]:
                        tp2_hit = 1
                        tp2_hit_time = res["tp2_hit_time"] or tp2_hit_time
                    sl = res["sl"]

                    if res["should_break"]:
                        break

            # Case B: Single candle dictionary
            elif isinstance(data, dict):
                raw_ts = data.get("timestamp")
                raw_open = data.get("open")
                raw_high = data.get("high")
                raw_low = data.get("low")
                raw_is_comp = data.get("is_completed")

                # Unified authoritative candle validation: enforces positive finite ts, mandatory 5-min grid, completion (ts + 300 <= now), completion flag, and OHLC integrity
                is_valid, c_ts, c_open, c_high, c_low = validate_candle_record(raw_ts, raw_open, raw_high, raw_low, now, is_completed_flag=raw_is_comp, enforce_grid=True)
                if not is_valid:
                    continue

                if c_ts > last_candle_time and c_ts >= eval_min_time:
                    newest_candle_time = c_ts

                    tp1_hit_prior = bool(tp1_hit)
                    if direction == "BUY":
                        mfe = max(mfe, c_high - entry)
                        mae = min(mae, c_low - entry)
                    else:
                        mfe = max(mfe, entry - c_low)
                        mae = min(mae, entry - c_high)

                    res = evaluate_candle_transition(
                        direction=direction, entry=entry, orig_sl=orig_sl, sl=sl,
                        tp1=tp1, tp2=tp2, tp1_hit_prior=tp1_hit_prior,
                        c_open=c_open, c_high=c_high, c_low=c_low, c_ts=c_ts,
                        est_costs=est_costs, cost_calc_fn=cost_calc_fn
                    )

                    outcome = res["outcome"]
                    final_pnl = res["final_pnl"]
                    net_pnl = res["net_pnl"]
                    brokerage_costs = res["brokerage_costs"]
                    if res["tp1_hit"]:
                        tp1_hit = 1
                        tp1_hit_time = res["tp1_hit_time"] or tp1_hit_time
                    if res["tp2_hit"]:
                        tp2_hit = 1
                        tp2_hit_time = res["tp2_hit_time"] or tp2_hit_time
                    sl = res["sl"]

            cursor.execute("""
            UPDATE historical_signals 
            SET mfe = ?, mae = ?, outcome = ?, final_pnl = ?, closed_at = ?, last_candle_time = ?,
                current_stop_loss = ?, tp1_hit = ?, tp1_hit_time = ?, tp2_hit = ?, tp2_hit_time = ?, net_pnl = ?, brokerage_costs = ?
            WHERE id = ?
            """, (
                mfe, mae, outcome, final_pnl, now if outcome != "ACTIVE" else None, newest_candle_time,
                sl, tp1_hit, tp1_hit_time, tp2_hit, tp2_hit_time, net_pnl, brokerage_costs, sig["id"]
            ))

            if outcome in ("WIN", "WIN_FULL", "WIN_PARTIAL", "LOSS"):
                update_strategy_performance(cursor, strategy, sig["id"])
                outcomes_changed = True

            # If trade has concluded, trigger Adaptive Post-Mortem Analysis
            if outcome != "ACTIVE":
                try:
                    from engine.adaptive_trade_learner import AdaptiveTradeLearner
                    sig_completed = dict(sig)
                    sig_completed.update({
                        "outcome": outcome, "final_pnl": final_pnl, "closed_at": now,
                        "mfe": mfe, "mae": mae, "net_pnl": net_pnl
                    })
                    AdaptiveTradeLearner.analyze_and_record_outcome(sig_completed)
                except Exception:
                    pass

        conn.commit()
        conn.close()

        # Trigger strategy learning dynamic calibration strictly after commit & connection close
        if outcomes_changed:
            try:
                from engine.strategy_learning_engine import StrategyLearningEngine
                StrategyLearningEngine.calibrate_weights()
            except Exception as e:
                print("[STORAGE] Error in post-outcome calibration:", e)
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
        SELECT hs.outcome, hs.final_pnl, hs.net_pnl, hs.entry_price, hs.stop_loss, hs.target_1, hs.target_2 
        FROM historical_signals hs
        LEFT JOIN signal_strategies ss ON hs.id = ss.signal_id
        WHERE (hs.strategy = ? OR ss.strategy_name = ?) AND hs.outcome IN ('WIN', 'WIN_FULL', 'WIN_PARTIAL', 'LOSS')
        GROUP BY hs.id
        """, (strat, strat))
        trades = cursor.fetchall()
        if not trades:
            continue

        total = len(trades)
        full_wins = len([t for t in trades if t["outcome"] in ("WIN", "WIN_FULL")])
        partial_wins = len([t for t in trades if t["outcome"] == "WIN_PARTIAL"])
        wins = full_wins + partial_wins
        losses = len([t for t in trades if t["outcome"] == "LOSS"])
        net_positive_trades = len([t for t in trades if (float(t["net_pnl"]) if t["net_pnl"] is not None else float(t["final_pnl"])) > 0])
        win_rate = round((wins / total) * 100, 1)

        gross_profit = sum(float(t["final_pnl"]) for t in trades if float(t["final_pnl"]) > 0)
        gross_loss = abs(sum(float(t["final_pnl"]) for t in trades if float(t["final_pnl"]) < 0))

        net_profit = sum(
            float(t["net_pnl"]) if t["net_pnl"] is not None else float(t["final_pnl"])
            for t in trades
            if (float(t["net_pnl"]) if t["net_pnl"] is not None else float(t["final_pnl"])) > 0
        )
        net_loss = abs(sum(
            float(t["net_pnl"]) if t["net_pnl"] is not None else float(t["final_pnl"])
            for t in trades
            if (float(t["net_pnl"]) if t["net_pnl"] is not None else float(t["final_pnl"])) < 0
        ))

        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        else:
            profit_factor = 99.0 if gross_profit > 0 else 1.0

        if net_loss > 0:
            net_profit_factor = round(net_profit / net_loss, 2)
        else:
            net_profit_factor = 99.0 if net_profit > 0 else 1.0

        total_gross_pnl = round(sum(float(t["final_pnl"]) for t in trades), 2)
        total_net_pnl = round(
            sum(float(t["net_pnl"]) if t["net_pnl"] is not None else float(t["final_pnl"]) for t in trades),
            2
        )

        signed_r_list = []
        for t in trades:
            risk = abs(float(t["entry_price"]) - float(t["stop_loss"])) or 1.0
            pnl = float(t["net_pnl"]) if t["net_pnl"] is not None else float(t["final_pnl"])
            r_multiple = pnl / risk
            signed_r_list.append(r_multiple)
        avg_rr = round(sum(signed_r_list) / len(signed_r_list), 2) if signed_r_list else 1.0

        cursor.execute("""
        INSERT INTO strategy_performance (
            strategy, total_signals, wins, losses, win_rate, avg_rr, 
            profit_factor, net_profit_factor, total_gross_pnl, total_net_pnl,
            full_wins, partial_wins, net_positive_trades, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(strategy) DO UPDATE SET
            total_signals = excluded.total_signals,
            wins = excluded.wins,
            losses = excluded.losses,
            win_rate = excluded.win_rate,
            avg_rr = excluded.avg_rr,
            profit_factor = excluded.profit_factor,
            net_profit_factor = excluded.net_profit_factor,
            total_gross_pnl = excluded.total_gross_pnl,
            total_net_pnl = excluded.total_net_pnl,
            full_wins = excluded.full_wins,
            partial_wins = excluded.partial_wins,
            net_positive_trades = excluded.net_positive_trades,
            updated_at = excluded.updated_at
        """, (
            strat, total, wins, losses, win_rate, avg_rr,
            profit_factor, net_profit_factor, total_gross_pnl, total_net_pnl,
            full_wins, partial_wins, net_positive_trades, time.time()
        ))

def get_strategy_performance_stats(as_of_ts=None):
    """
    Returns strategy performance statistics.
    If as_of_ts is provided (point-in-time backtesting/replay), strictly excludes
    any trades or learning signals that closed or occurred AFTER as_of_ts to prevent lookahead leakage.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        if as_of_ts is not None:
            # Convert as_of_ts to epoch float if needed
            if hasattr(as_of_ts, 'timestamp'):
                as_of_num = as_of_ts.timestamp()
            else:
                try:
                    import pandas as pd
                    as_of_num = pd.to_datetime(as_of_ts).timestamp()
                except Exception:
                    as_of_num = float(as_of_ts)

            # STRICT POINT-IN-TIME LEARNING:
            # Only include trades where outcome completion time (closed_at) is known and <= as_of_num.
            # Never fall back to entry timestamp, which leaks trades that completed after as_of_num!
            cursor.execute("""
                SELECT 
                    strategy,
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN outcome LIKE '%WIN%' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    AVG(net_pnl) as avg_pnl
                FROM historical_signals 
                WHERE outcome IS NOT NULL 
                  AND outcome != 'ACTIVE' 
                  AND closed_at IS NOT NULL 
                  AND closed_at <= ?
                GROUP BY strategy
            """, (as_of_num,))
            rows = []
            for r in cursor.fetchall():
                d = dict(r)
                tot = d.get("total_signals", 0)
                w = d.get("wins", 0)
                d["win_rate"] = round((w / tot * 100), 1) if tot > 0 else 50.0
                rows.append(d)
        else:
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
