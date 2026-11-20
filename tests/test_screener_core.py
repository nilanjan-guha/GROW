import unittest
import os
import tempfile
import sqlite3
import pandas as pd
import numpy as np
import datetime
import pytz

from engine.opportunity_potential_engine import OpportunityPotentialEngine
from strategies.institutional_engine import InstitutionalStrategyEngine
import core.storage as storage

class TestIntradayScreenerCore(unittest.TestCase):

    def setUp(self):
        # Create an isolated temporary SQLite database for test execution
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        self.orig_db_path = storage.DB_PATH
        storage.DB_PATH = self.temp_db_path
        storage.init_db()

        # Create synthetic 5m OHLC DataFrame (60 candles)
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

    def tearDown(self):
        storage.DB_PATH = self.orig_db_path
        os.close(self.temp_db_fd)
        try:
            os.remove(self.temp_db_path)
        except Exception:
            pass

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
        opp = OpportunityPotentialEngine.evaluate_potential(
            self.df_5m, curr_price, "BUY", stop_loss=990.0, daily_df=self.daily_df
        )
        self.assertIn("dynamic_clearance_pct", opp)
        self.assertGreaterEqual(opp["dynamic_clearance_pct"], 0.20)
        self.assertLessEqual(opp["dynamic_clearance_pct"], 0.60)

    def test_strict_unrounded_potential_and_rr_gates(self):
        curr_price = float(self.df_5m['Close'].iloc[-1])
        res = OpportunityPotentialEngine.evaluate_potential(
            self.df_5m, curr_price, "BUY", stop_loss=curr_price * 0.998, daily_df=self.daily_df
        )
        if res["is_viable"]:
            self.assertGreater(res["potential_pct"], 1.0)
            self.assertGreaterEqual(res["risk_reward_ratio"], 1.50)

    def test_buy_independent_tp1_and_tp2_lifecycle(self):
        sym = "TEST_BUY.NS"
        entry, sl, tp1, tp2 = 500.0, 490.0, 510.0, 520.0

        sig_id = storage.log_signal({
            "symbol": sym, "direction": "BUY", "strategy": "Gap-and-Go",
            "entry_price": entry, "stop_loss": sl, "target_1": tp1, "target_2": tp2,
            "score": 85, "confidence": "HIGH"
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        c.execute("SELECT timestamp FROM historical_signals WHERE id = ?", (sig_id,))
        now_ts = float(c.fetchone()["timestamp"])

        # Candle 1 reaches TP1 (High 512, Low 502) -> TP1 hit, stop moved to breakeven (500)
        c1 = {"high": 512.0, "low": 502.0, "timestamp": now_ts + 300, "is_completed": True}
        storage.evaluate_open_outcomes({sym: c1})

        c.execute("SELECT outcome, tp1_hit, tp2_hit, current_stop_loss FROM historical_signals WHERE id = ?", (sig_id,))
        row = c.fetchone()
        self.assertEqual(row["tp1_hit"], 1)
        self.assertEqual(row["tp2_hit"], 0)
        self.assertEqual(row["outcome"], "ACTIVE")
        self.assertEqual(row["current_stop_loss"], entry)

        # Candle 2 reaches TP2 (High 525, Low 508) -> WIN_FULL
        c2 = {"high": 525.0, "low": 508.0, "timestamp": now_ts + 600, "is_completed": True}
        storage.evaluate_open_outcomes({sym: c2})

        c.execute("SELECT outcome, tp1_hit, tp2_hit, final_pnl, net_pnl FROM historical_signals WHERE id = ?", (sig_id,))
        row2 = c.fetchone()
        self.assertEqual(row2["outcome"], "WIN_FULL")
        self.assertEqual(row2["tp2_hit"], 1)
        self.assertGreater(row2["final_pnl"], 0.0)
        self.assertGreater(row2["net_pnl"], 0.0)
        conn.close()

    def test_sell_independent_tp1_and_tp2_lifecycle(self):
        sym = "TEST_SELL.NS"
        entry, sl, tp1, tp2 = 500.0, 510.0, 490.0, 480.0

        sig_id = storage.log_signal({
            "symbol": sym, "direction": "SELL", "strategy": "Opening Range Breakdown",
            "entry_price": entry, "stop_loss": sl, "target_1": tp1, "target_2": tp2,
            "score": 80, "confidence": "HIGH"
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        c.execute("SELECT timestamp FROM historical_signals WHERE id = ?", (sig_id,))
        now_ts = float(c.fetchone()["timestamp"])

        # Candle 1 reaches TP1 for SELL (High 498, Low 488) -> TP1 hit, stop moved to breakeven (500)
        c1 = {"high": 498.0, "low": 488.0, "timestamp": now_ts + 300, "is_completed": True}
        storage.evaluate_open_outcomes({sym: c1})

        c.execute("SELECT outcome, tp1_hit, current_stop_loss FROM historical_signals WHERE id = ?", (sig_id,))
        row = c.fetchone()
        self.assertEqual(row["tp1_hit"], 1)
        self.assertEqual(row["current_stop_loss"], entry)

        # Candle 2 reaches TP2 for SELL (High 492, Low 475) -> WIN_FULL
        c2 = {"high": 492.0, "low": 475.0, "timestamp": now_ts + 600, "is_completed": True}
        storage.evaluate_open_outcomes({sym: c2})

        c.execute("SELECT outcome, tp2_hit, final_pnl, net_pnl FROM historical_signals WHERE id = ?", (sig_id,))
        row2 = c.fetchone()
        self.assertEqual(row2["outcome"], "WIN_FULL")
        self.assertEqual(row2["tp2_hit"], 1)
        self.assertGreater(row2["final_pnl"], 0.0)
        self.assertGreater(row2["net_pnl"], 0.0)
        conn.close()

    def test_ambiguous_same_candle_tp1_and_sl_conservative_accounting(self):
        sym = "TEST_AMBIGUOUS.NS"
        entry, sl, tp1 = 500.0, 490.0, 515.0

        sig_id = storage.log_signal({
            "symbol": sym, "direction": "BUY", "strategy": "EMA Momentum",
            "entry_price": entry, "stop_loss": sl, "target_1": tp1, "target_2": 530.0,
            "score": 75, "confidence": "HIGH"
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        c.execute("SELECT timestamp FROM historical_signals WHERE id = ?", (sig_id,))
        now_ts = float(c.fetchone()["timestamp"])

        # Candle touches BOTH TP1 (High 518 >= 515) and original SL (Low 488 <= 490) in the same bar
        c_ambiguous = {"high": 518.0, "low": 488.0, "timestamp": now_ts + 300, "is_completed": True}
        storage.evaluate_open_outcomes({sym: c_ambiguous})

        c.execute("SELECT outcome, tp1_hit, final_pnl, net_pnl FROM historical_signals WHERE id = ?", (sig_id,))
        row = c.fetchone()
        # Conservative accounting must treat same-bar cross as LOSS
        self.assertEqual(row["outcome"], "LOSS")
        self.assertEqual(row["tp1_hit"], 0)
        self.assertLess(row["final_pnl"], 0.0)
        self.assertLess(row["net_pnl"], 0.0)
        conn.close()

    def test_tp1_followed_by_breakeven_sl(self):
        sym = "TEST_BE.NS"
        entry, sl, tp1, tp2 = 500.0, 490.0, 510.0, 530.0

        sig_id = storage.log_signal({
            "symbol": sym, "direction": "BUY", "strategy": "VWAP Reclaim",
            "entry_price": entry, "stop_loss": sl, "target_1": tp1, "target_2": tp2,
            "score": 78, "confidence": "HIGH"
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        c.execute("SELECT timestamp FROM historical_signals WHERE id = ?", (sig_id,))
        now_ts = float(c.fetchone()["timestamp"])

        # Candle 1 hits TP1
        c1 = {"high": 512.0, "low": 502.0, "timestamp": now_ts + 300, "is_completed": True}
        storage.evaluate_open_outcomes({sym: c1})

        # Candle 2 reverses back to entry price (breakeven stop triggered)
        c2 = {"high": 508.0, "low": 499.0, "timestamp": now_ts + 600, "is_completed": True}
        storage.evaluate_open_outcomes({sym: c2})

        c.execute("SELECT outcome, tp1_hit, tp2_hit, final_pnl FROM historical_signals WHERE id = ?", (sig_id,))
        row = c.fetchone()
        self.assertEqual(row["outcome"], "WIN_PARTIAL")
        self.assertEqual(row["tp1_hit"], 1)
        self.assertEqual(row["tp2_hit"], 0)
        self.assertEqual(row["final_pnl"], 5.0)
        conn.close()

if __name__ == '__main__':
    unittest.main()
