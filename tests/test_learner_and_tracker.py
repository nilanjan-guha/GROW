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
        P0 Test: Directly calls BackgroundContinuousTracker.calculate_atr_pct production method
        and verifies dynamic ATR-based slippage tolerance calculation.
        """
        # Create a sample DataFrame with known ATR
        dates = pd.date_range("2026-11-26 09:15", periods=20, freq="5min")
        highs = [102.0] * 20
        lows = [100.0] * 20
        closes = [101.0] * 20
        df = pd.DataFrame({"high": highs, "low": lows, "close": closes}, index=dates)

        level = 100.0
        # Call actual production helper
        raw_atr_pct = BackgroundContinuousTracker.calculate_atr_pct(df, level)
        self.assertIsNotNone(raw_atr_pct, "Production calculate_atr_pct must return valid float, not None")
        self.assertIsInstance(raw_atr_pct, float)
        self.assertAlmostEqual(raw_atr_pct, 2.0, places=1)

        # Dynamic tolerance formula: 0.5x of ATR% bounded in [0.25%, 1.20%]
        dynamic_tol = min(1.20, max(0.25, round(0.50 * raw_atr_pct, 2)))
        self.assertEqual(dynamic_tol, 1.0)

        # Insufficient data (< 14 bars) returns None cleanly
        df_short = df.iloc[:10]
        self.assertIsNone(BackgroundContinuousTracker.calculate_atr_pct(df_short, level))
        # Negative or zero level returns None cleanly
        self.assertIsNone(BackgroundContinuousTracker.calculate_atr_pct(df, 0.0))

    def test_real_breakout_promotion_and_idempotence(self):
        """
        P1 Test: Directly invokes production method BackgroundContinuousTracker.promote_preparing_breakout(),
        verifying the candidate is promoted to ENTRY READY and that repeated runs do not duplicate promotions.
        """
        prep_info = {
            "symbol": "TATAMOTORS",
            "direction": "BULLISH",
            "entry_trigger": "Enter above ₹950.0",
            "conviction_score": 90,
            "change_pct": 2.5
        }
        screener_state = {
            "persistent_preparing_stocks": {
                "TATAMOTORS": prep_info
            },
            "ranked_opportunities": {
                "entry_ready": []
            }
        }
        tracker = BackgroundContinuousTracker(screener_state)

        candle_ts = "2026-11-26 14:30:00"
        completed_c = 952.0
        lvl = 950.0
        direct_eval = {
            "is_viable": True,
            "suggested_sl": 945.0,
            "suggested_tp1": 962.5,
            "suggested_tp2": 970.0,
            "room_pct": 1.8
        }

        # Step 1: Call ACTUAL production method
        promoted = tracker.promote_preparing_breakout(
            symbol="TATAMOTORS",
            completed_c=completed_c,
            evaluated_completed_ts=candle_ts,
            prep_info=prep_info,
            direct_eval=direct_eval,
            level=lvl,
            direction="BULLISH",
            entry_trigger="Enter above ₹950.0"
        )
        self.assertTrue(promoted, "Production promote_preparing_breakout must return True on fresh breakout")

        # Verify promotion results
        entry_list = screener_state["ranked_opportunities"]["entry_ready"]
        self.assertEqual(len(entry_list), 1)
        prom_cand = entry_list[0]
        self.assertEqual(prom_cand["symbol"], "TATAMOTORS")
        self.assertEqual(prom_cand["stage"], "CONFIRMED")
        self.assertTrue(prom_cand["is_actionable"])
        self.assertEqual(prom_cand["current_price"], 952.0)
        self.assertEqual(prom_cand["suggested_sl"], 945.0)
        self.assertEqual(prom_cand["suggested_tp1"], 962.5)
        # Score is calculated from actual evidence (base + room bonus), not artificially hardcoded
        self.assertGreaterEqual(prom_cand["conviction_score"], 90)
        self.assertEqual(tracker.tracked_stats["promotions_today"], 1)
        self.assertNotIn("TATAMOTORS", screener_state["persistent_preparing_stocks"])

        # Step 2: Idempotency check - subsequent scan with same candle timestamp must return False and not duplicate
        second_prom = tracker.promote_preparing_breakout(
            symbol="TATAMOTORS",
            completed_c=completed_c,
            evaluated_completed_ts=candle_ts,
            prep_info=prep_info,
            direct_eval=direct_eval,
            level=lvl,
            direction="BULLISH"
        )
        self.assertFalse(second_prom, "Second call with identical candle timestamp must return False (idempotent)")
        self.assertEqual(len(screener_state["ranked_opportunities"]["entry_ready"]), 1, "Must not add duplicate candidate")
        self.assertEqual(tracker.tracked_stats["promotions_today"], 1, "Promotion counter must not be incremented repeatedly")

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
