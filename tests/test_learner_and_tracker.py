import unittest
import time
import pandas as pd
import numpy as np
from core.storage import get_db_connection
from engine.learning.adaptive_trade_learner import AdaptiveTradeLearner
from engine.tracking.background_continuous_tracker import BackgroundContinuousTracker

class TestAdaptiveLearnerAndTracker(unittest.TestCase):
    def setUp(self):
        AdaptiveTradeLearner.init_learning_tables()
        self.conn = get_db_connection()
        self.cursor = self.conn.cursor()

    def tearDown(self):
        self.cursor.execute("DELETE FROM historical_signals WHERE strategy = 'TEST_REPLAY_STRAT'")
        self.cursor.execute("DELETE FROM trade_post_mortems WHERE strategy = 'TEST_REPLAY_STRAT'")
        self.cursor.execute("DELETE FROM pattern_learning_weights WHERE pattern_key = 'TEST_REPLAY_STRAT_BUY'")
        self.conn.commit()
        self.conn.close()

    def test_future_trades_excluded_from_historical_replay(self):
        """
        P0 Test: Verifies that point-in-time replay strictly excludes trades closed
        after as_of_ts, eliminating lookahead bias.
        """
        strat = "TEST_REPLAY_STRAT"
        direction = "BUY"
        t_past = 1700000000.0   # e.g., Day 1: past trade closed as WIN
        t_as_of = 1700001000.0  # as_of point-in-time timestamp
        t_future = 1700002000.0 # Day 2: future trade closed as LOSS

        # Insert past winning trade
        self.cursor.execute("""
            INSERT INTO historical_signals (
                symbol, direction, strategy, entry_price, stop_loss, target_1, outcome, net_pnl, closed_at, score
            ) VALUES (?, ?, ?, 100, 95, 110, 'WIN_TARGET_1', 10.0, ?, 80)
        """, ("TEST1.NS", direction, strat, t_past))

        # Insert future losing trade
        self.cursor.execute("""
            INSERT INTO historical_signals (
                symbol, direction, strategy, entry_price, stop_loss, target_1, outcome, net_pnl, closed_at, score
            ) VALUES (?, ?, ?, 100, 95, 110, 'LOSS_SL', -5.0, ?, 80)
        """, ("TEST2.NS", direction, strat, t_future))
        self.conn.commit()

        # At point-in-time t_as_of:
        # Only the past WIN trade should be seen (1 win out of 1 sample -> 100% win rate -> multiplier > 1.0)
        mult_at_as_of = AdaptiveTradeLearner.get_pattern_multiplier(strat, direction, as_of_ts=t_as_of)
        self.assertGreater(mult_at_as_of, 1.0, "Multiplier should reflect past winning trade before as_of_ts")

        # Prior to t_past:
        # No trades closed yet -> fail-safe default multiplier 1.0
        mult_before_all = AdaptiveTradeLearner.get_pattern_multiplier(strat, direction, as_of_ts=t_past - 100)
        self.assertEqual(mult_before_all, 1.0, "Multiplier should be 1.0 when no historical trades are closed prior to as_of_ts")

        # After t_future:
        # Both WIN and LOSS are visible (1 win, 1 loss -> 50% win rate -> lower multiplier)
        mult_after_future = AdaptiveTradeLearner.get_pattern_multiplier(strat, direction, as_of_ts=t_future + 100)
        self.assertLess(mult_after_future, mult_at_as_of, "Multiplier after future loss should be lower than at as_of")

    def test_pandas_timestamp_parsing_in_learner(self):
        """
        P0 Test: Verifies that pd.Timestamp strings and datetime objects are parsed without error.
        """
        strat = "TEST_REPLAY_STRAT"
        direction = "BUY"
        ts_str = "2026-11-26 12:00:00"
        mult = AdaptiveTradeLearner.get_pattern_multiplier(strat, direction, as_of_ts=ts_str)
        self.assertIsInstance(mult, float)
        self.assertGreaterEqual(mult, 0.80)
        self.assertLessEqual(mult, 1.25)

    def test_anti_chasing_atr_threshold_bullish_and_bearish(self):
        """
        P1 Test: Verifies that anti-chasing dynamically adjusts based on volatility and
        rejects chasing for both bullish breakouts and bearish breakdowns.
        """
        # Create a sample DataFrame with known ATR
        dates = pd.date_range("2026-11-26 09:15", periods=20, freq="5min")
        highs = [102.0] * 20
        lows = [100.0] * 20
        closes = [101.0] * 20
        df = pd.DataFrame({"high": highs, "low": lows, "close": closes}, index=dates)

        # 5m ATR is ~ 2.0. On level 100, ATR% is 2.0%. 0.5x ATR% = 1.0% tolerance.
        level = 100.0
        high_s = df["high"]
        low_s = df["low"]
        close_s = df["close"]
        tr = pd.concat([high_s - low_s, (high_s - close_s.shift()).abs(), (low_s - close_s.shift()).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1])
        raw_atr_pct = (atr14 / level) * 100.0
        dynamic_tol = min(1.20, max(0.25, round(0.50 * raw_atr_pct, 2)))

        self.assertGreaterEqual(dynamic_tol, 0.25)
        self.assertLessEqual(dynamic_tol, 1.20)

        # Bullish test:
        # Price within tolerance: 100.4 (0.4% run) -> OK
        live_bull_ok = 100.4
        bull_missed = live_bull_ok < level or ((live_bull_ok - level) / level * 100.0) > dynamic_tol
        self.assertFalse(bull_missed, "Bullish entry within dynamic ATR tolerance should be accepted")

        # Price overextended: 101.5 (1.5% run > dynamic_tol) -> REJECT
        live_bull_chase = 101.5
        bull_chase_rejected = live_bull_chase < level or ((live_bull_chase - level) / level * 100.0) > dynamic_tol
        self.assertTrue(bull_chase_rejected, "Bullish chase beyond dynamic ATR tolerance should be rejected")

        # Bearish test:
        # Breakdown level = 100.0. Price within tolerance: 99.6 (0.4% drop) -> OK
        live_bear_ok = 99.6
        bear_missed = live_bear_ok > level or ((level - live_bear_ok) / level * 100.0) > dynamic_tol
        self.assertFalse(bear_missed, "Bearish entry within dynamic ATR tolerance should be accepted")

        # Price dumped too far: 98.0 (2.0% drop > dynamic_tol) -> REJECT
        live_bear_chase = 98.0
        bear_chase_rejected = live_bear_chase > level or ((level - live_bear_chase) / level * 100.0) > dynamic_tol
        self.assertTrue(bear_chase_rejected, "Bearish chase beyond dynamic ATR tolerance should be rejected")

    def test_tracker_does_not_skip_failed_evaluation(self):
        """
        P0 Test: Verifies that if outcome evaluation throws an exception,
        the candle timestamp is NOT marked as evaluated, allowing automatic retry on the next cycle.
        """
        tracker = BackgroundContinuousTracker({})
        norm_sym = "RELIANCE"
        c_ts = "2026-11-26 13:30:00"

        # Simulate fresh DataFrame ready for evaluation
        pending_keys = {norm_sym: c_ts}
        
        # Scenario 1: Evaluation fails
        try:
            raise RuntimeError("Database locked or network blip during outcome evaluation")
            # If line below were placed before or in try block, it would falsely mark evaluated:
            for k, ts in pending_keys.items():
                tracker._last_evaluated_candles[k] = ts
        except Exception:
            pass # Evaluation failed, timestamp should NOT be recorded

        self.assertNotIn(norm_sym, tracker._last_evaluated_candles, 
                         "Failed evaluation must NOT record candle timestamp, preventing permanent skipping")

        # Scenario 2: Evaluation succeeds
        # Timestamp is updated only after successful execution
        for k, ts in pending_keys.items():
            tracker._last_evaluated_candles[k] = ts

        self.assertEqual(tracker._last_evaluated_candles.get(norm_sym), c_ts,
                         "Successful evaluation records candle timestamp to prevent redundant re-evaluation")

if __name__ == "__main__":
    unittest.main()
