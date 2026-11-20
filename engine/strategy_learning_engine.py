import time
import json
from core.storage import get_db_connection

class StrategyLearningEngine:
    """
    Persistent Strategy Learning & Dynamic Calibration Engine:
    1. Tracks multi-dimensional trade statistics:
       - By Strategy
       - By Market Regime (STRONG_BULLISH, STRONG_BEARISH, RANGEBOUND)
       - By Time of Day (Opening 09:15-10:30, Midday 10:30-13:30, Afternoon 13:30-15:30)
       - By Sector & Score Tier
    2. Minimum Sample Size Gate (N >= 15 verified closed trades) before adjusting strategy weights.
    3. Dynamically computes weight multiplier (0.50x to 1.50x) based on rolling net win rate & profit factor.
    """

    MIN_OBSERVATIONS = 15

    @staticmethod
    def init_learning_tables():
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS strategy_calibrations (
            dimension_key TEXT PRIMARY KEY,
            dimension_type TEXT,
            sample_size INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0.0,
            net_profit_factor REAL DEFAULT 0.0,
            weight_multiplier REAL DEFAULT 1.0,
            updated_at REAL
        )
        """)
        conn.commit()
        conn.close()

    @staticmethod
    def get_time_bucket(hour, minute):
        total_mins = hour * 60 + minute
        if total_mins <= 630:    # Up to 10:30
            return "OPENING_MOMENTUM"
        elif total_mins <= 810:  # 10:30 to 13:30
            return "MIDDAY_CHOP"
        else:
            return "AFTERNOON_EXPANSION"

    @staticmethod
    def calibrate_weights():
        """
        Inspects closed trades in `historical_signals` and calculates weight adjustments.
        Only modifies multipliers if sample size >= MIN_OBSERVATIONS (15).
        """
        StrategyLearningEngine.init_learning_tables()
        conn = get_db_connection()
        c = conn.cursor()

        # Query all closed trades
        c.execute("""
        SELECT strategy, market_regime, outcome, net_pnl, final_pnl, timestamp
        FROM historical_signals
        WHERE outcome IN ('WIN', 'WIN_FULL', 'WIN_PARTIAL', 'LOSS')
        """)
        trades = c.fetchall()

        if not trades:
            conn.close()
            return {}

        # Aggregate by Strategy
        strategy_stats = {}
        for t in trades:
            strat = t["strategy"]
            if strat not in strategy_stats:
                strategy_stats[strat] = {"wins": 0, "losses": 0, "net_profit": 0.0, "net_loss": 0.0}
            
            pnl = float(t["net_pnl"]) if t["net_pnl"] is not None else float(t["final_pnl"])
            is_win = t["outcome"] in ("WIN", "WIN_FULL", "WIN_PARTIAL")
            if is_win:
                strategy_stats[strat]["wins"] += 1
                strategy_stats[strat]["net_profit"] += max(0.0, pnl)
            else:
                strategy_stats[strat]["losses"] += 1
                strategy_stats[strat]["net_loss"] += abs(min(0.0, pnl))

        calibrations = {}
        now = time.time()
        for strat, stats in strategy_stats.items():
            total = stats["wins"] + stats["losses"]
            win_rate = (stats["wins"] / total * 100) if total > 0 else 0.0
            npf = (stats["net_profit"] / stats["net_loss"]) if stats["net_loss"] > 0 else (2.5 if stats["net_profit"] > 0 else 1.0)

            # Rule: Only adjust weight if total observations >= MIN_OBSERVATIONS
            if total >= StrategyLearningEngine.MIN_OBSERVATIONS:
                # Calibrate multiplier: 0.50x (poor) to 1.50x (exceptional)
                if win_rate >= 65.0 and npf >= 1.5:
                    multiplier = min(1.50, round(1.0 + (win_rate - 50.0)/100.0, 2))
                elif win_rate <= 40.0 or npf < 0.9:
                    multiplier = max(0.50, round(0.50 + (win_rate / 100.0), 2))
                else:
                    multiplier = 1.0
            else:
                multiplier = 1.0  # Baseline unchanged until sufficient statistical sample size

            calibrations[strat] = {
                "sample_size": total,
                "win_rate": round(win_rate, 1),
                "net_profit_factor": round(npf, 2),
                "weight_multiplier": multiplier
            }

            c.execute("""
            INSERT INTO strategy_calibrations (dimension_key, dimension_type, sample_size, win_rate, net_profit_factor, weight_multiplier, updated_at)
            VALUES (?, 'STRATEGY', ?, ?, ?, ?, ?)
            ON CONFLICT(dimension_key) DO UPDATE SET
                sample_size = excluded.sample_size,
                win_rate = excluded.win_rate,
                net_profit_factor = excluded.net_profit_factor,
                weight_multiplier = excluded.weight_multiplier,
                updated_at = excluded.updated_at
            """, (strat, total, round(win_rate, 1), round(npf, 2), multiplier, now))

        # Multi-dimensional Learning: Aggregate and learn by Market Regime and Time-Of-Day
        regime_stats = {}
        time_stats = {}
        for t in trades:
            regime = t["market_regime"] or "TRENDING"
            if regime not in regime_stats:
                regime_stats[regime] = {"wins": 0, "losses": 0, "net_profit": 0.0, "net_loss": 0.0}
            
            pnl = float(t["net_pnl"]) if t["net_pnl"] is not None else float(t["final_pnl"])
            is_win = t["outcome"] in ("WIN", "WIN_FULL", "WIN_PARTIAL")
            if is_win:
                regime_stats[regime]["wins"] += 1
                regime_stats[regime]["net_profit"] += max(0.0, pnl)
            else:
                regime_stats[regime]["losses"] += 1
                regime_stats[regime]["net_loss"] += abs(min(0.0, pnl))

            # Time of day dimension
            trade_ts = float(t["timestamp"] or now)
            import datetime
            dt = datetime.datetime.fromtimestamp(trade_ts)
            t_bucket = StrategyLearningEngine.get_time_bucket(dt.hour, dt.minute)
            if t_bucket not in time_stats:
                time_stats[t_bucket] = {"wins": 0, "losses": 0, "net_profit": 0.0, "net_loss": 0.0}
            if is_win:
                time_stats[t_bucket]["wins"] += 1
                time_stats[t_bucket]["net_profit"] += max(0.0, pnl)
            else:
                time_stats[t_bucket]["losses"] += 1
                time_stats[t_bucket]["net_loss"] += abs(min(0.0, pnl))

        # Record regime calibrations
        for reg, stats in regime_stats.items():
            tot = stats["wins"] + stats["losses"]
            wr = (stats["wins"] / tot * 100) if tot > 0 else 0.0
            npf_val = (stats["net_profit"] / stats["net_loss"]) if stats["net_loss"] > 0 else (2.0 if stats["net_profit"] > 0 else 1.0)
            mult = 1.0
            if tot >= StrategyLearningEngine.MIN_OBSERVATIONS:
                mult = min(1.30, max(0.70, round(1.0 + (wr - 50.0)/100.0, 2)))
            c.execute("""
            INSERT INTO strategy_calibrations (dimension_key, dimension_type, sample_size, win_rate, net_profit_factor, weight_multiplier, updated_at)
            VALUES (?, 'MARKET_REGIME', ?, ?, ?, ?, ?)
            ON CONFLICT(dimension_key) DO UPDATE SET
                sample_size = excluded.sample_size,
                win_rate = excluded.win_rate,
                net_profit_factor = excluded.net_profit_factor,
                weight_multiplier = excluded.weight_multiplier,
                updated_at = excluded.updated_at
            """, (f"REGIME_{reg}", tot, round(wr, 1), round(npf_val, 2), mult, now))

        # Record time-of-day calibrations
        for tb, stats in time_stats.items():
            tot = stats["wins"] + stats["losses"]
            wr = (stats["wins"] / tot * 100) if tot > 0 else 0.0
            npf_val = (stats["net_profit"] / stats["net_loss"]) if stats["net_loss"] > 0 else (2.0 if stats["net_profit"] > 0 else 1.0)
            mult = 1.0
            if tot >= StrategyLearningEngine.MIN_OBSERVATIONS:
                mult = min(1.30, max(0.70, round(1.0 + (wr - 50.0)/100.0, 2)))
            c.execute("""
            INSERT INTO strategy_calibrations (dimension_key, dimension_type, sample_size, win_rate, net_profit_factor, weight_multiplier, updated_at)
            VALUES (?, 'TIME_OF_DAY', ?, ?, ?, ?, ?)
            ON CONFLICT(dimension_key) DO UPDATE SET
                sample_size = excluded.sample_size,
                win_rate = excluded.win_rate,
                net_profit_factor = excluded.net_profit_factor,
                weight_multiplier = excluded.weight_multiplier,
                updated_at = excluded.updated_at
            """, (f"TIME_{tb}", tot, round(wr, 1), round(npf_val, 2), mult, now))

        conn.commit()
        conn.close()
        return calibrations

    @staticmethod
    def get_strategy_multiplier(strategy_name):
        """Fetches learned weight multiplier (default 1.0)"""
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT weight_multiplier FROM strategy_calibrations WHERE dimension_key = ?", (strategy_name,))
            row = c.fetchone()
            conn.close()
            if row:
                return float(row["weight_multiplier"])
        except Exception:
            pass
        return 1.0
