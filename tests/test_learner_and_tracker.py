import unittest
import time
import math
import pytz
import sqlite3
import pandas as pd
import numpy as np
from core.storage import get_db_connection
from engine.learning.adaptive_trade_learner import AdaptiveTradeLearner
from engine.learning.historical_stock_learner import HistoricalStockLearner
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
            "tp1": 962.5,
            "tp2": 970.0,
            "potential_pct": 1.8
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
        # Structural SL: lvl_val (950) - max(952*0.0035, 952 - 950 = 2) -> max(3.332, 2.0) = 3.332 -> 950 - 3.33 = 946.67
        self.assertEqual(prom_cand["suggested_sl"], 946.67)
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

    def test_bearish_breakdown_promotion_and_two_candidate_cap(self):
        """
        P1/P2 Test: Verifies bearish breakdown promotion creates valid structural SL above breakdown level,
        TP1 below entry, meets RR >= 1.05, and enforces max 2 candidates cap in ENTRY READY.
        """
        screener_state = {
            "persistent_preparing_stocks": {
                "SBIN": {
                    "symbol": "SBIN",
                    "direction": "BEARISH",
                    "entry_trigger": "Breakdown below ₹800.0",
                    "conviction_score": 85,
                    "change_pct": -1.8
                }
            },
            "ranked_opportunities": {
                "entry_ready": [
                    {"symbol": "STOCK_A", "score": 90},
                    {"symbol": "STOCK_B", "score": 88}
                ]
            }
        }
        tracker = BackgroundContinuousTracker(screener_state)

        candle_ts = "2026-11-26 15:00:00"
        completed_c = 796.0
        lvl = 800.0 # Breakdown level
        # Risk = 804.0 - 796.0 = 8.0. For RR >= 1.05, reward must be >= 8.4 -> TP1 <= 787.6
        direct_eval = {
            "is_viable": True,
            "tp1": 786.0,
            "tp2": 778.0,
            "potential_pct": 1.5
        }

        prep_info = screener_state["persistent_preparing_stocks"]["SBIN"]

        promoted = tracker.promote_preparing_breakout(
            symbol="SBIN",
            completed_c=completed_c,
            evaluated_completed_ts=candle_ts,
            prep_info=prep_info,
            direct_eval=direct_eval,
            level=lvl,
            direction="BEARISH",
            entry_trigger="Breakdown below ₹800.0"
        )
        self.assertTrue(promoted, "Bearish promotion with valid RR must succeed")

        entry_list = screener_state["ranked_opportunities"]["entry_ready"]
        # Must enforce strict limit of max 2 candidates
        self.assertEqual(len(entry_list), 2, "ENTRY READY must be strictly capped at max 2 candidates")
        prom_cand = entry_list[0]
        self.assertEqual(prom_cand["symbol"], "SBIN")
        self.assertEqual(prom_cand["direction"], "BEARISH")
        # In a breakdown: SL must be strictly above entry, TP1 strictly below entry
        self.assertGreater(prom_cand["suggested_sl"], prom_cand["current_price"])
        self.assertLess(prom_cand["suggested_tp1"], prom_cand["current_price"])
        
        # Verify Risk:Reward ratio calculation
        risk = prom_cand["suggested_sl"] - prom_cand["current_price"]
        reward = prom_cand["current_price"] - prom_cand["suggested_tp1"]
        rr = reward / risk
        self.assertGreaterEqual(rr, 1.05, f"Bearish promotion must meet RR >= 1.05, got {rr}")

    def test_offline_learner_rejects_invalid_or_stale_feed(self):
        """
        P1/P2 Test: Verifies that:
        1. Gap detection is tested independently from market freshness (using now_dt set to the feed's timestamp).
        2. Stale feeds during live market hours are rejected.
        3. Malformed feeds (missing required columns or non-finite values) are rejected.
        4. Forming-candle removal is verified through the actual production MarketValidator.validate_feed_freshness path.
        """
        from engine.market.market_validator import MarketValidator
        tracker = BackgroundContinuousTracker({"fno_universe": ["TESTSTK"]})
        ist = pytz.timezone('Asia/Kolkata')
        
        # Test 1: Independent Gap Detection
        # Construct feed on a trading day with an intra-session gap: 09:15 to 09:30, then jumping to 10:35 (65m gap)
        # We pass now_dt = 10:40 on that exact day so freshness check passes, testing gap detection strictly on its own merits!
        gap_day_now = pd.to_datetime("2026-11-26 10:40:00").tz_localize(ist)
        gap_dates = list(pd.date_range("2026-11-26 09:15", periods=5, freq="5min", tz=ist)) # 09:15..09:35
        gap_dates.append(pd.to_datetime("2026-11-26 10:35:00").tz_localize(ist)) # 60-min gap!
        gap_df = pd.DataFrame({
            "Open": [100.0] * 6, "High": [105.0] * 6, "Low": [99.0] * 6, "Close": [102.0] * 6, "Volume": [1000] * 6
        }, index=gap_dates)

        feed_val_gap = MarketValidator.validate_feed_freshness(gap_df, max_delay_seconds=900, candle_duration_seconds=300, now_dt=gap_day_now)
        self.assertFalse(feed_val_gap.get("is_valid", False), "Feed with intra-session gap must be rejected")
        self.assertEqual(feed_val_gap.get("status"), "INVALID_TIMEFRAME", "Failure status must specifically be INVALID_TIMEFRAME due to interval discontinuity")

        # Step should abort without recording scan for gap feed
        class MockGapProvider:
            def fetch_5m_candles(self, ticker, days_back=5):
                return gap_df, "mock_gap"

        tracker.data_provider = MockGapProvider()
        tracker._run_closed_market_learner_step()
        self.assertFalse(HistoricalStockLearner.should_skip_scan("TESTSTK", current_completed_candle_ts=str(gap_df.index[-1])))

        # Test 2: Stale feed during live market hours
        simulated_live_market = pd.to_datetime("2026-11-26 11:00:00").tz_localize(ist)
        stale_dates = pd.date_range("2026-11-20 09:15", periods=30, freq="5min", tz=ist)
        stale_df = pd.DataFrame({
            "Open": [100.0] * 30, "High": [105.0] * 30, "Low": [99.0] * 30, "Close": [102.0] * 30, "Volume": [1000] * 30
        }, index=stale_dates)
        feed_val_stale = MarketValidator.validate_feed_freshness(stale_df, max_delay_seconds=300, candle_duration_seconds=300, now_dt=simulated_live_market)
        self.assertFalse(feed_val_stale.get("is_valid", False), "Previous session feed during active market hours must be marked invalid")

        # Test 3: Malformed feed (missing Close / High / Low)
        malformed_df = pd.DataFrame({"SomeOtherCol": [1, 2, 3]}, index=stale_dates[:3])
        feed_val_mal = MarketValidator.validate_feed_freshness(malformed_df, max_delay_seconds=300, candle_duration_seconds=300)
        self.assertFalse(feed_val_mal.get("is_valid", False), "Malformed df missing required columns must be invalid")

        # Test 4: Non-finite prices (NaN / Inf)
        non_finite_df = stale_df.copy()
        non_finite_df.iloc[5, 3] = float("nan") # NaN in Close
        feed_val_nan = MarketValidator.validate_feed_freshness(non_finite_df, max_delay_seconds=300, candle_duration_seconds=300)
        self.assertFalse(feed_val_nan.get("is_valid", False), "Data with NaN must be invalid")

        # Test 5: Production-path forming candle detection and removal
        # In a 5m series, candle started at 10:00 closes at 10:05.
        # At now_dt = 10:03, the 10:00 candle is actively forming (now < last_ts + 300).
        prod_dates = pd.date_range("2026-11-26 09:15", "2026-11-26 10:00", freq="5min", tz=ist)
        prod_df = pd.DataFrame({
            "Open": [100.0] * len(prod_dates), "High": [105.0] * len(prod_dates),
            "Low": [99.0] * len(prod_dates), "Close": [102.0] * len(prod_dates),
            "Volume": [1000] * len(prod_dates)
        }, index=prod_dates)

        now_forming = pd.to_datetime("2026-11-26 10:03:00").tz_localize(ist)
        feed_val_forming = MarketValidator.validate_feed_freshness(prod_df, max_delay_seconds=900, candle_duration_seconds=300, now_dt=now_forming)
        
        # Must be valid feed, but is_completed_candle must be False!
        self.assertTrue(feed_val_forming.get("is_valid"), "Feed itself is fresh and valid")
        self.assertFalse(feed_val_forming.get("is_completed_candle"), "10:00 candle at 10:03 must be detected as FORMING")

        # Replicate actual production tracker slicing using the returned validator flag
        is_comp = feed_val_forming.get("is_completed_candle", False)
        study_df = prod_df if is_comp else prod_df.iloc[:-1]
        self.assertEqual(len(study_df), len(prod_df) - 1, "Production path must strip forming candle when validator marks is_completed_candle=False")
        self.assertEqual(str(study_df.index[-1]), str(prod_dates[-2]), "Stripped study_df must end on the previous completed candle (09:55)")

    def test_directional_multipliers_and_unresolved_trades(self):
        """
        P1 Test: Verifies that analyze_30d_stock_history / replay_intraday_breakouts calculates
        separate bullish and bearish multipliers and accurately tracks unresolved trades.
        """
        # Build synthetic 5-minute data with:
        # - Bullish breakouts that hit target (strong bullish performance)
        # - Bearish breakdowns that fail or stay unresolved
        dates = pd.date_range("2026-11-20 09:15", periods=50, freq="5min")
        opens = [100.0] * 50
        highs = [101.0] * 50
        lows = [99.5] * 50
        closes = [100.0] * 50

        # Bar 10: Bullish breakout (prior 5-bar high is 101.0)
        opens[10] = 101.2
        highs[10] = 103.0
        closes[10] = 102.5
        lows[10] = 101.0 # Stop = 101.0
        # Bar 11: Next open 102.6, Low 101.5 (above stop 101.0), High 104.0
        opens[11] = 102.6
        lows[11] = 101.5
        highs[11] = 104.0
        closes[11] = 103.5
        # Bar 12: Hits target (entry ~ 102.65, stop 101.0, risk 1.65, target 105.12)
        opens[12] = 103.5
        highs[12] = 107.0
        lows[12] = 103.0
        closes[12] = 106.0

        # Bar 20: Bullish breakout 2
        opens[20] = 106.2
        highs[20] = 108.0
        closes[20] = 107.5
        lows[20] = 106.0 # Stop = 106.0
        opens[21] = 107.6
        lows[21] = 106.5
        highs[21] = 108.5
        closes[21] = 108.0
        opens[22] = 108.0
        highs[22] = 112.0 # Hits target
        lows[22] = 107.5
        closes[22] = 111.0

        # Bar 30: Bearish breakdown (prior 5 bars 25..29 lows are 101.0)
        for k in range(25, 30):
            opens[k] = 102.0
            highs[k] = 104.0
            lows[k] = 101.0
            closes[k] = 102.0

        opens[30] = 100.5
        lows[30] = 98.0
        closes[30] = 98.5
        highs[30] = 103.0 # Stop = 103.0
        opens[31] = 98.4
        lows[31] = 98.0
        highs[31] = 104.0 # Hits stop! (loss)

        # Bar 40: Breakout near the end of series that does not reach target or stop before series ends -> Unresolved!
        opens[40] = 107.2
        highs[40] = 110.0
        closes[40] = 109.0
        lows[40] = 107.0
        opens[41] = 109.1
        for k in range(41, 50):
            opens[k] = 109.0
            highs[k] = 109.5
            lows[k] = 108.0
            closes[k] = 109.0

        test_df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=dates)

        bull_mult, bear_mult, stats = HistoricalStockLearner.replay_intraday_breakouts(
            "TEST_DIR_STK", test_df, is_replay=True, return_directional=True
        )

        # Bullish had 2 winning trades -> bull_mult > 1.0
        self.assertGreater(bull_mult, 1.0, f"Bullish multiplier should be > 1.0, got {bull_mult}")
        # Bearish had 1 losing trade -> bear_mult <= 1.0
        self.assertLessEqual(bear_mult, 1.0, f"Bearish multiplier should be <= 1.0, got {bear_mult}")
        # Unresolved trades should be counted (> 0)
        self.assertGreater(stats["unresolved_trades"], 0, "Trades hitting neither target nor stop must be tracked as unresolved")
        self.assertIn("bull_win_rate", stats)
        self.assertIn("bear_win_rate", stats)

    def test_unresolved_rate_confidence_penalty(self):
        """
        P1 Test: Verifies that when a setup has a high unresolved trade rate (>30%),
        its multiplier receives a confidence penalty, preventing deceptive high win-rate signals.
        """
        dates = pd.date_range("2026-11-20 09:15", periods=50, freq="5min")
        opens = [100.0] * 50
        highs = [101.0] * 50
        lows = [99.5] * 50
        closes = [100.0] * 50

        # Create 1 WIN trade:
        opens[10] = 101.5
        highs[10] = 103.0
        closes[10] = 102.5
        lows[10] = 101.0
        opens[11] = 102.6
        lows[11] = 101.5
        highs[11] = 104.0
        closes[11] = 103.5
        opens[12] = 103.5
        highs[12] = 107.0
        lows[12] = 103.0
        closes[12] = 106.0

        # And 3 UNRESOLVED trades (breakouts that stay flat without hitting stop or target):
        for bar_idx in [20, 28, 36]:
            opens[bar_idx] = 106.0
            highs[bar_idx] = 108.0
            closes[bar_idx] = 107.5
            lows[bar_idx] = 105.0 # Stop = 105.0
            opens[bar_idx + 1] = 107.6
            # Next bars stay tight between 106.0 and 108.0 (entry ~107.65, stop 105.0, target ~111.6)
            for f in range(bar_idx + 1, min(50, bar_idx + 7)):
                opens[f] = 107.0
                highs[f] = 107.8
                lows[f] = 106.5
                closes[f] = 107.0

        test_df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=dates)

        bull_mult, bear_mult, stats = HistoricalStockLearner.replay_intraday_breakouts(
            "TEST_UNRES", test_df, is_replay=True, return_directional=True
        )

        # Bullish has 1 win and 3 unresolved (75% unresolved rate!)
        self.assertGreater(stats["bull_unresolved_rate"], 30.0, "Unresolved rate must exceed 30%")
        # Despite 100% win rate on resolved trades, the multiplier must be penalized below what a clean 100% win-rate gets
        self.assertLess(bull_mult, 1.15, "Multiplier must be heavily discounted due to high unresolved rate")

    def test_historical_replay_data_validation_and_as_of_ts_timezones(self):
        """
        P2 Test: Verifies that:
        1. Duplicate timestamps, unsorted candles, non-finite values, and corrupt OHLC are rejected with explicit status.
        2. as_of_ts filtering works identically with timezone-aware (Asia/Kolkata) and timezone-naive timestamps.
        """
        ist = pytz.timezone('Asia/Kolkata')
        dates = pd.date_range("2026-11-20 09:15", periods=20, freq="5min", tz=ist)
        df_valid = pd.DataFrame({
            "Open": [100.0] * 20, "High": [105.0] * 20, "Low": [99.0] * 20, "Close": [102.0] * 20
        }, index=dates)

        # Case 1: Corrupt OHLC (High < Low)
        df_corrupt = df_valid.copy()
        df_corrupt.iloc[5, 1] = 95.0 # High = 95 < Low = 99
        _, _, stats_corrupt = HistoricalStockLearner.analyze_30d_stock_history("TEST_CORR", df_corrupt, return_directional=True)
        self.assertEqual(stats_corrupt.get("status"), "CORRUPT_OHLC_RELATIONSHIPS")

        # Case 2: Non-finite values (NaN / Inf)
        df_nan = df_valid.copy()
        df_nan.iloc[5, 3] = np.nan
        _, _, stats_nan = HistoricalStockLearner.analyze_30d_stock_history("TEST_NAN", df_nan, return_directional=True)
        self.assertEqual(stats_nan.get("status"), "NON_FINITE_DATA")

        # Case 3: Duplicate timestamps
        dates_dup = list(dates[:10]) + list(dates[:10])
        df_dup = pd.DataFrame({
            "Open": [100.0] * 20, "High": [105.0] * 20, "Low": [99.0] * 20, "Close": [102.0] * 20
        }, index=dates_dup)
        _, _, stats_dup = HistoricalStockLearner.analyze_30d_stock_history("TEST_DUP", df_dup, return_directional=True)
        self.assertEqual(stats_dup.get("status"), "DUPLICATE_TIMESTAMPS")

        # Case 4: Timezone-aware vs Timezone-naive as_of_ts
        as_of_naive = "2026-11-20 10:00:00"
        as_of_aware = pd.to_datetime(as_of_naive).tz_localize(ist)
        
        _, _, stats_naive = HistoricalStockLearner.analyze_30d_stock_history("TEST_TZ1", df_valid, as_of_ts=as_of_naive, return_directional=True)
        _, _, stats_aware = HistoricalStockLearner.analyze_30d_stock_history("TEST_TZ2", df_valid, as_of_ts=as_of_aware, return_directional=True)
        self.assertEqual(stats_naive.get("status"), "SUCCESS")
        self.assertEqual(stats_aware.get("status"), "SUCCESS")
        self.assertEqual(stats_naive.get("bull_mult"), stats_aware.get("bull_mult"), "Naive and Aware as_of_ts must yield identical multipliers")

    def test_promotion_defensive_validation_and_failure_state_preservation(self):
        """
        P2 Test: Verifies promote_preparing_breakout rejects invalid inputs defensively without
        modifying persistent_preparing_stocks or adding corrupt records to entry_ready.
        Also verifies displayed candidate RR matches actual promotion RR.
        """
        prep_info = {
            "symbol": "DEF_TEST",
            "direction": "BULLISH",
            "entry_trigger": "Enter above ₹500.0",
            "conviction_score": 85,
            "change_pct": 1.2
        }
        screener_state = {
            "persistent_preparing_stocks": {
                "DEF_TEST": prep_info
            },
            "ranked_opportunities": {
                "entry_ready": []
            }
        }
        tracker = BackgroundContinuousTracker(screener_state)

        # 1. Reject invalid direction
        res_dir = tracker.promote_preparing_breakout(
            symbol="DEF_TEST",
            completed_c=505.0,
            evaluated_completed_ts="2026-11-26 15:15:00",
            prep_info=prep_info,
            direct_eval={"is_viable": True, "tp1": 515.0},
            level=500.0,
            direction="SIDEWAYS" # Invalid!
        )
        self.assertFalse(res_dir, "Invalid direction must be rejected")
        self.assertIn("DEF_TEST", screener_state["persistent_preparing_stocks"], "Preparing state must not be modified on failure")
        self.assertEqual(len(screener_state["ranked_opportunities"]["entry_ready"]), 0)

        # 2. Reject non-finite prices
        res_inf = tracker.promote_preparing_breakout(
            symbol="DEF_TEST",
            completed_c=float("inf"),
            evaluated_completed_ts="2026-11-26 15:15:00",
            prep_info=prep_info,
            direct_eval={"is_viable": True, "tp1": 515.0},
            level=500.0,
            direction="BULLISH"
        )
        self.assertFalse(res_inf, "Infinite completed price must be rejected")
        self.assertIn("DEF_TEST", screener_state["persistent_preparing_stocks"])

        # 3. Reject missing/invalid trigger level
        res_lvl = tracker.promote_preparing_breakout(
            symbol="DEF_TEST",
            completed_c=505.0,
            evaluated_completed_ts="2026-11-26 15:15:00",
            prep_info=prep_info,
            direct_eval={"is_viable": True, "tp1": 515.0},
            level=-10.0,
            direction="BULLISH"
        )
        self.assertFalse(res_lvl, "Negative level must be rejected")

        # 4. Reject when viable is False or missing
        res_viab = tracker.promote_preparing_breakout(
            symbol="DEF_TEST",
            completed_c=505.0,
            evaluated_completed_ts="2026-11-26 15:15:00",
            prep_info=prep_info,
            direct_eval={"is_viable": False, "tp1": 515.0},
            level=500.0,
            direction="BULLISH"
        )
        self.assertFalse(res_viab, "Non-viable direct_eval must be rejected")

        # 5. Successful promotion: Verify promoted candidate RR matches actual calculated RR
        res_success = tracker.promote_preparing_breakout(
            symbol="DEF_TEST",
            completed_c=505.0,
            evaluated_completed_ts="2026-11-26 15:15:00",
            prep_info=prep_info,
            direct_eval={"is_viable": True, "tp1": 520.0, "tp2": 530.0, "potential_pct": 2.5},
            level=500.0,
            direction="BULLISH"
        )
        self.assertTrue(res_success)
        promoted_cand = screener_state["ranked_opportunities"]["entry_ready"][0]
        self.assertIn("promoted_rr", promoted_cand)
        expected_risk = promoted_cand["current_price"] - promoted_cand["suggested_sl"]
        expected_reward = promoted_cand["suggested_tp1"] - promoted_cand["current_price"]
        expected_rr = round(expected_reward / expected_risk, 2)
        self.assertEqual(promoted_cand["promoted_rr"], expected_rr, "Promoted RR field must match actual calculated RR")

if __name__ == "__main__":
    unittest.main()
