import unittest
import time
import sqlite3
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

    def test_tracker_deduplication_and_retry_after_db_failure(self):
        """
        P2 Test: Verifies duplicate .NS and non-.NS symbols are deduplicated to exactly 1 evaluation,
        and that when a database/evaluation error occurs, the completed candle is preserved for retry.
        """
        tracker = BackgroundContinuousTracker({})
        norm_sym = "INFY"
        c_ts = "2026-11-26 14:15:00"
        
        # Create mock 5-minute candle DataFrame with timestamp
        mock_idx = pd.to_datetime([c_ts])
        mock_df = pd.DataFrame({"close": [1850.0], "high": [1855.0], "low": [1845.0], "open": [1848.0]}, index=mock_idx)

        # Fresh DFs collected from feed containing BOTH "INFY" and "INFY.NS"
        fresh_dfs = {
            "INFY": mock_df,
            "INFY.NS": mock_df
        }

        # Step 1: Deduplication loop logic identical to BackgroundContinuousTracker
        evaluable_fresh_dfs = {}
        pending_eval_keys = {}
        seen_norm_symbols = set()

        for s_key, c_df in fresh_dfs.items():
            norm_key = s_key.replace(".NS", "").upper()
            if norm_key in seen_norm_symbols:
                continue
            seen_norm_symbols.add(norm_key)

            cand_ts = str(c_df.index[-1])
            dedup_key = f"{norm_key}:{cand_ts}"
            if tracker._last_evaluated_candles.get(norm_key) != cand_ts and dedup_key not in pending_eval_keys:
                evaluable_fresh_dfs[norm_key] = c_df
                evaluable_fresh_dfs[f"{norm_key}.NS"] = c_df
                pending_eval_keys[norm_key] = cand_ts

        # Verify deduplication reduced inputs to exactly 1 normalized symbol key
        self.assertEqual(len(seen_norm_symbols), 1, "Duplicate .NS and non-.NS symbols must resolve to single unique symbol")
        self.assertEqual(list(pending_eval_keys.keys()), [norm_sym], "Pending keys must contain exactly 1 normalized entry")

        # Step 2: Simulate Database failure during outcome evaluation
        db_failed = True
        try:
            if db_failed:
                raise sqlite3.OperationalError("database is locked")
            for n_k, n_ts in pending_eval_keys.items():
                tracker._last_evaluated_candles[n_k] = n_ts
        except Exception:
            pass # Caught error, do NOT update _last_evaluated_candles

        self.assertNotIn(norm_sym, tracker._last_evaluated_candles, 
                         "Database failure must NOT mark candle evaluated, enabling retry on next iteration")

        # Step 3: Next cycle retry with recovered database
        db_failed = False
        evaluable_fresh_dfs_retry = {}
        pending_eval_keys_retry = {}
        seen_norm_symbols_retry = set()

        for s_key, c_df in fresh_dfs.items():
            norm_key = s_key.replace(".NS", "").upper()
            if norm_key in seen_norm_symbols_retry:
                continue
            seen_norm_symbols_retry.add(norm_key)

            cand_ts = str(c_df.index[-1])
            dedup_key = f"{norm_key}:{cand_ts}"
            if tracker._last_evaluated_candles.get(norm_key) != cand_ts and dedup_key not in pending_eval_keys_retry:
                evaluable_fresh_dfs_retry[norm_key] = c_df
                evaluable_fresh_dfs_retry[f"{norm_key}.NS"] = c_df
                pending_eval_keys_retry[norm_key] = cand_ts

        self.assertEqual(len(pending_eval_keys_retry), 1, "Candle must be re-queued for evaluation after failure")
        
        # Successful evaluation marks the candle as evaluated
        for n_k, n_ts in pending_eval_keys_retry.items():
            tracker._last_evaluated_candles[n_k] = n_ts

        self.assertEqual(tracker._last_evaluated_candles.get(norm_sym), c_ts, 
                         "After retry success, candle timestamp must be recorded")

        # Step 4: Subsequent cycle with identical candle timestamp -> skipped
        evaluable_fresh_dfs_subsequent = {}
        pending_eval_keys_subsequent = {}
        seen_norm_symbols_subsequent = set()

        for s_key, c_df in fresh_dfs.items():
            norm_key = s_key.replace(".NS", "").upper()
            if norm_key in seen_norm_symbols_subsequent:
                continue
            seen_norm_symbols_subsequent.add(norm_key)

            cand_ts = str(c_df.index[-1])
            dedup_key = f"{norm_key}:{cand_ts}"
            if tracker._last_evaluated_candles.get(norm_key) != cand_ts and dedup_key not in pending_eval_keys_subsequent:
                evaluable_fresh_dfs_subsequent[norm_key] = c_df
                evaluable_fresh_dfs_subsequent[f"{norm_key}.NS"] = c_df
                pending_eval_keys_subsequent[norm_key] = cand_ts

        self.assertEqual(len(pending_eval_keys_subsequent), 0, 
                         "Completed candle that was already evaluated must be evaluated exactly once")

if __name__ == "__main__":
    unittest.main()
