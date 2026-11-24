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
        ("instrument_type", "TEXT DEFAULT 'EQUITY_INTRADAY'")
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
            stop_loss, current_stop_loss, target_1, target_2, score, 
            confidence, catalyst, features_json, market_regime, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    # Reject missing, non-finite, or non-positive candle values (corrupted feed protection)
    import math
    for val in (entry, orig_sl, sl, tp1, c_open, c_high, c_low):
        if val is None or not math.isfinite(float(val)) or float(val) <= 0:
            return {
                "outcome": "ACTIVE", "final_pnl": 0.0, "net_pnl": 0.0, "brokerage_costs": est_costs,
                "tp1_hit": 1 if tp1_hit_prior else 0, "tp1_hit_time": None, "tp2_hit": 0, "tp2_hit_time": None,
                "sl": sl, "should_break": False
            }

    if tp2 is not None and (not math.isfinite(float(tp2)) or float(tp2) <= 0):
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
                    c1 = cost_calc_fn(exit_p1, fraction=0.5)
                    c2 = cost_calc_fn(exit_p2, fraction=0.5)
                    tot_cost = c1 + c2
                else:
                    tot_cost = cost_calc_fn(exit_p1, fraction=1.0)
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

def evaluate_open_outcomes(data_map, force_eod=False):
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
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM historical_signals WHERE outcome = 'ACTIVE'")
        active_signals = cursor.fetchall()
        now = time.time()
        import datetime
        import pytz
        import pandas as pd
        outcomes_changed = False
        from engine.cost_engine import IndianMarketCosts

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
                def _calc(exit_p, fraction=1.0):
                    sub_qty = max(1, int(round(quantity * fraction)))
                    c = IndianMarketCosts.calculate_intraday_costs(entry_p, exit_p, quantity=sub_qty, instrument_type=inst)
                    return c["cost_per_share"] * fraction
                return _calc

            cost_calc_fn = make_cost_calc(entry, qty, inst_type)

            # 0. Intraday Session Close Auto-Squareoff:
            # Applies when signal is from an older calendar day or when called with force_eod=True
            try:
                sig_dt = datetime.datetime.fromtimestamp(sig_time, pytz.timezone('Asia/Kolkata'))
                is_stale_day = (sig_dt.date() < ist_now.date())
                should_eod_squareoff = is_stale_day or (force_eod and not is_mkt_open)
            except Exception:
                should_eod_squareoff = False

            if should_eod_squareoff:
                exit_price = entry
                if isinstance(data, pd.DataFrame) and not data.empty:
                    exit_price = float(data['Close'].iloc[-1])
                elif isinstance(data, dict):
                    exit_price = float(data.get("close", data.get("current_price", entry)))

                pnl = round(exit_price - entry if direction == "BUY" else entry - exit_price, 2)
                outcome = "CLOSED_EOD_SQUAREOFF"
                final_pnl = pnl
                brokerage_costs = est_costs
                net_pnl = round(final_pnl - brokerage_costs, 2)

                cursor.execute("""
                UPDATE historical_signals 
                SET outcome = ?, final_pnl = ?, closed_at = ?, net_pnl = ?, brokerage_costs = ?
                WHERE id = ?
                """, (outcome, final_pnl, now, net_pnl, brokerage_costs, sig["id"]))
                outcomes_changed = True
                continue

            if data is None:
                continue

            # Case A: Full DataFrame provided
            import pandas as pd
            if isinstance(data, pd.DataFrame) and not data.empty:
                df = data.copy()
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index)
                
                # Iterate each completed candle formed strictly after the completed signal candle
                now_ts = time.time()
                for idx, (c_dt, row) in enumerate(df.iterrows()):
                    c_ts = c_dt.timestamp() if hasattr(c_dt, 'timestamp') else float(c_dt)
                    if (c_ts - last_candle_time) <= 1e-3 or (c_ts - eval_min_time) < -1e-3:
                        continue

                    is_last_candle = (idx == len(df) - 1)
                    if is_last_candle and (c_ts + 300 > now_ts):
                        continue

                    newest_candle_time = max(newest_candle_time, c_ts)
                    c_open = float(row.get('Open', entry))
                    c_high = float(row.get('High', entry))
                    c_low = float(row.get('Low', entry))

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
                c_ts = float(data.get("timestamp", now))
                is_completed = data.get("is_completed", (c_ts + 300 <= now))
                if is_completed and c_ts > last_candle_time and c_ts >= eval_min_time:
                    newest_candle_time = c_ts
                    c_open = float(data.get("open", entry))
                    c_high = float(data.get("high", entry))
                    c_low = float(data.get("low", entry))

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
