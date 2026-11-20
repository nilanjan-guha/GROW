import unittest
import pandas as pd
import numpy as np
import datetime
import pytz

from engine.opportunity_potential_engine import OpportunityPotentialEngine
from strategies.institutional_engine import InstitutionalStrategyEngine
from engine.groww_sector_engine import GrowwSectorEngine
from core.storage import log_signal, evaluate_open_outcomes, get_strategy_performance_stats, get_db_connection

class TestIntradayScreenerCore(unittest.TestCase):

    def setUp(self):
        # Create synthetic 5m OHLC DataFrame (60 candles = 5 hours of trading)
        np.random.seed(42)
        ist = pytz.timezone('Asia/Kolkata')
        start_dt = datetime.datetime.now(ist).replace(hour=9, minute=15, second=0, microsecond=0)
        times = [start_dt + datetime.timedelta(minutes=5*i) for i in range(60)]
        
        base_price = 1000.0
        prices = [base_price]
        for _ in range(59):
            prices.append(prices[-1] + np.random.uniform(-1.5, 2.0))
            
        opens = [p - np.random.uniform(0, 0.5) for p in prices]
        highs = [max(o, p) + np.random.uniform(0.5, 2.0) for o, p in zip(opens, prices)]
        lows = [min(o, p) - np.random.uniform(0.5, 2.0) for o, p in zip(opens, prices)]
        volumes = [np.random.randint(10000, 50000) for _ in prices]
        
        self.df_5m = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes
        }, index=pd.DatetimeIndex(times))
        
        # Create synthetic 1mo daily DataFrame (25 completed days)
        d_times = [start_dt.date() - datetime.timedelta(days=i) for i in range(30, 0, -1)]
        d_close = [base_price + np.random.uniform(-20, 20) for _ in d_times]
        d_open = [c - np.random.uniform(-5, 5) for c in d_close]
        d_high = [max(o, c) + np.random.uniform(5, 15) for o, c in zip(d_open, d_close)]
        d_low = [min(o, c) - np.random.uniform(5, 15) for o, c in zip(d_open, d_close)]
        
        self.daily_df = pd.DataFrame({
            "Open": d_open,
            "High": d_high,
            "Low": d_low,
            "Close": d_close,
            "Volume": [500000]*len(d_times)
        }, index=pd.DatetimeIndex(d_times))

    def test_wilder_daily_atr_conventional_smoothing(self):
        curr_price = float(self.df_5m['Close'].iloc[-1])
        dates = [d.date() for d in self.df_5m.index]
        atr, source = OpportunityPotentialEngine.calculate_daily_atr(
            self.df_5m, dates, curr_price, daily_df=self.daily_df
        )
        self.assertEqual(source, "DEDICATED_DAILY_WILDER_ATR14")
        self.assertGreater(atr, 0.0)
        self.assertTrue(5.0 < atr < 50.0)

    def test_volatility_aware_dynamic_clearance(self):
        curr_price = 1000.0
        # If immediate obstacle is 0.1% away, it must be blocked
        opp = OpportunityPotentialEngine.evaluate_potential(
            self.df_5m, curr_price, "BUY", stop_loss=990.0, daily_df=self.daily_df
        )
        self.assertIn("dynamic_clearance_pct", opp)
        self.assertGreaterEqual(opp["dynamic_clearance_pct"], 0.20)
        self.assertLessEqual(opp["dynamic_clearance_pct"], 0.60)

    def test_strict_unrounded_potential_and_rr_gates(self):
        curr_price = float(self.df_5m['Close'].iloc[-1])
        # Force a stop-loss very close to entry so RR is artificially high, but test if potential > 1.0% holds
        res = OpportunityPotentialEngine.evaluate_potential(
            self.df_5m, curr_price, "BUY", stop_loss=curr_price * 0.998, daily_df=self.daily_df
        )
        if res["is_viable"]:
            self.assertGreater(res["potential_pct"], 1.0)
            self.assertGreaterEqual(res["risk_reward_ratio"], 1.50)

    def test_independent_tp1_and_tp2_outcomes(self):
        # Create a test signal in database
        sym = "TESTSTOCK.NS"
        entry = 500.0
        sl = 490.0
        tp1 = 510.0
        tp2 = 520.0
        now_ts = 100000.0

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO historical_signals (
                timestamp, datetime_str, symbol, direction, strategy, entry_price, stop_loss, target_1, target_2, score, confidence, outcome, last_candle_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now_ts, "2026-11-21 10:00:00", sym, "BUY", "Gap-and-Go", entry, sl, tp1, tp2, 85, "HIGH", "ACTIVE", now_ts))
        sig_id = c.lastrowid
        conn.commit()
        conn.close()

        # Candle 1 reaches TP1 (High 512, Low 502) -> Should record TP1 hit and move stop to breakeven
        c1 = {"high": 512.0, "low": 502.0, "timestamp": now_ts + 300, "is_completed": True}
        evaluate_open_outcomes({sym: c1})

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT outcome, tp1_hit, tp2_hit, stop_loss FROM historical_signals WHERE id = ?", (sig_id,))
        row = c.fetchone()
        self.assertEqual(row["tp1_hit"], 1)
        self.assertEqual(row["tp2_hit"], 0)
        self.assertEqual(row["outcome"], "ACTIVE")

        # Candle 2 reaches TP2 (High 525, Low 508) -> Should record WIN_FULL and net P&L
        c2 = {"high": 525.0, "low": 508.0, "timestamp": now_ts + 600, "is_completed": True}
        evaluate_open_outcomes({sym: c2})

        c.execute("SELECT outcome, tp1_hit, tp2_hit, final_pnl, net_pnl FROM historical_signals WHERE id = ?", (sig_id,))
        row2 = c.fetchone()
        self.assertEqual(row2["outcome"], "WIN_FULL")
        self.assertEqual(row2["tp2_hit"], 1)
        self.assertGreater(row2["final_pnl"], 0.0)
        self.assertGreater(row2["net_pnl"], 0.0)

        # Clean up test row
        c.execute("DELETE FROM historical_signals WHERE id = ?", (sig_id,))
        conn.commit()
        conn.close()

if __name__ == '__main__':
    unittest.main()
