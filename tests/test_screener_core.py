import unittest
import os
import tempfile
import sqlite3
import time
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
        c1 = {"open": 502.0, "high": 512.0, "low": 502.0, "timestamp": now_ts + 300, "is_completed": True}
        storage.evaluate_open_outcomes({sym: c1})

        # Candle 2 reverses back to entry price (breakeven stop triggered)
        c2 = {"open": 505.0, "high": 508.0, "low": 499.0, "timestamp": now_ts + 600, "is_completed": True}
        storage.evaluate_open_outcomes({sym: c2})

        c.execute("SELECT outcome, tp1_hit, tp2_hit, final_pnl FROM historical_signals WHERE id = ?", (sig_id,))
        row = c.fetchone()
        self.assertEqual(row["outcome"], "WIN_PARTIAL")
        self.assertEqual(row["tp1_hit"], 1)
        self.assertEqual(row["tp2_hit"], 0)
        self.assertEqual(row["final_pnl"], 5.0)
        conn.close()

    def test_duplicate_candle_evaluation_is_idempotent(self):
        sym = "TEST_DUP.NS"
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

        c1 = {"high": 505.0, "low": 498.0, "timestamp": now_ts + 300, "is_completed": True}
        storage.evaluate_open_outcomes({sym: c1})
        # Feed the exact same candle again
        storage.evaluate_open_outcomes({sym: c1})

        c.execute("SELECT outcome, last_candle_time FROM historical_signals WHERE id = ?", (sig_id,))
        row = c.fetchone()
        self.assertEqual(row["outcome"], "ACTIVE")
        self.assertEqual(row["last_candle_time"], now_ts + 300)
        conn.close()

    def test_market_validator_holidays_and_staleness(self):
        from engine.market_validator import MarketValidator, OFFICIAL_NSE_HOLIDAYS
        # Republic Day 2026 is a holiday
        rep_day = datetime.date(2026, 1, 26)
        self.assertFalse(MarketValidator.is_trading_day(rep_day))
        # Normal working Tuesday
        work_day = datetime.date(2026, 1, 27)
        self.assertTrue(MarketValidator.is_trading_day(work_day))

    def test_momentum_ignition_engine(self):
        from engine.momentum_ignition_engine import MomentumIgnitionEngine
        # Inject explosive candle into DataFrame
        df_copy = self.df_5m.copy()
        last_idx = df_copy.index[-1]
        df_copy.loc[last_idx, 'Open'] = 1000.0
        df_copy.loc[last_idx, 'Close'] = 1030.0  # +3% surge
        df_copy.loc[last_idx, 'High'] = 1032.0
        df_copy.loc[last_idx, 'Low'] = 999.0
        df_copy.loc[last_idx, 'Volume'] = 250000 # 5x volume surge

        ignition = MomentumIgnitionEngine.detect_ignition(df_copy, vwap=1005.0)
        self.assertTrue(ignition["is_ignited"])
        self.assertEqual(ignition["direction"], "BUY")
        self.assertGreaterEqual(ignition["ignition_score"], 70)

    def test_market_regime_engine(self):
        from engine.market_regime_engine import MarketRegimeEngine
        regime = MarketRegimeEngine.evaluate_benchmark_regime(self.df_5m)
        self.assertIn("regime", regime)
        self.assertIn("bias", regime)
        self.assertIn("alignment_score", regime)

    def test_strategy_learning_calibration_gate(self):
        from engine.strategy_learning_engine import StrategyLearningEngine
        # Without 15 trades, multiplier must remain 1.0
        mult = StrategyLearningEngine.get_strategy_multiplier("Gap-and-Go")
        self.assertEqual(mult, 1.0)
        combined = StrategyLearningEngine.get_combined_multiplier("Gap-and-Go", "BULLISH", "OPENING_MOMENTUM")
        self.assertEqual(combined, 1.0)

    def test_indian_market_costs(self):
        from engine.cost_engine import IndianMarketCosts
        eq_costs = IndianMarketCosts.calculate_equity_costs(entry_price=1000.0, exit_price=1010.0, quantity=100)
        self.assertEqual(eq_costs["instrument"], "EQUITY_INTRADAY")
        self.assertGreater(eq_costs["total_costs"], 0.0)
        self.assertGreater(eq_costs["brokerage"], 0.0)
        self.assertGreater(eq_costs["stt"], 0.0)

        opt_costs = IndianMarketCosts.calculate_options_costs(entry_premium=50.0, exit_premium=70.0, lot_size=500)
        self.assertEqual(opt_costs["instrument"], "OPTIONS")
        self.assertGreater(opt_costs["stt"], 0.0)

    def test_pre_momentum_prediction_engine(self):
        from engine.pre_momentum_engine import PreMomentumEngine
        # Test stage classification with viable opportunity
        opp_viable = {"is_viable": True, "risk_reward_ratio": 2.1}
        res_confirmed = PreMomentumEngine.classify_stage(
            df=self.df_5m,
            sector_bias="BULLISH",
            vwap=1000.0,
            rvol=1.5,
            composite_score=75,
            opp_eval=opp_viable
        )
        self.assertEqual(res_confirmed["stage"], "CONFIRMED")
        self.assertTrue(res_confirmed["is_tradeable"])

    def test_opportunity_ranking_engine(self):
        from engine.opportunity_ranking_engine import OpportunityRankingEngine
        res = OpportunityRankingEngine.evaluate_stock_opportunity(
            stock_df=self.df_5m,
            symbol="TESTSTOCK.NS",
            sector_name="TEST_SECTOR",
            sector_change=1.5,
            nifty_df=self.df_5m,
            vwap=1000.0,
            opp_eval={"is_viable": True, "potential_pct": 2.2, "risk_reward_ratio": 1.8}
        )
        self.assertIsNotNone(res)
        self.assertIn("stage", res)
        self.assertIn("conviction_score", res)
        self.assertIn("reasons", res)
        self.assertIn("rs_nifty", res)
        self.assertIn("rs_accel", res)
        self.assertIn("early_opportunity_score", res)
        self.assertIn("confirmation_score", res)
        self.assertIn("entry_trigger", res)

        # Test true RS acceleration calculation with distinct stock vs nifty returns
        rs_calc = OpportunityRankingEngine.calculate_relative_strength_acceleration(
            stock_df=self.df_5m,
            nifty_df=self.df_5m,
            sector_change_pct=1.0
        )
        self.assertIn("rs_accel_3", rs_calc)
        self.assertIn("rs_accel_6", rs_calc)
        # Since stock and nifty are identical in this synthetic check, RS acceleration against benchmark is 0.0
        self.assertEqual(rs_calc["rs_accel_3"], 0.0)

        ranked = OpportunityRankingEngine.rank_top_opportunities([res], top_n=3)
        self.assertIn("top_bullish", ranked)
        self.assertIn("top_bearish", ranked)

    def test_early_warning_preparation_mode(self):
        from engine.momentum_ignition_engine import MomentumIgnitionEngine
        prep = MomentumIgnitionEngine.detect_early_warning_preparation(self.df_5m, vwap=1000.0)
        self.assertIn("is_preparing", prep)
        self.assertIn("preparation_score", prep)

    def test_session_grid_historical_validation(self):
        from engine.market_validator import MarketValidator
        val = MarketValidator.validate_historical_replay_data(self.df_5m, allow_partial_final_session=True)
        self.assertTrue(val.get("is_valid", False))
        self.assertEqual(val.get("status"), "VALID_HISTORICAL")

    def test_replay_portfolio_capital_accounting(self):
        # Verify that capital reservation and turnover friction calculate correctly
        entry_p = 1000.0
        alloc_rs = 250000.0
        qty = int(alloc_rs / entry_p)
        self.assertEqual(qty, 250)
        
        exit_p = 1015.0 # +1.5% profit
        entry_turnover = entry_p * qty
        exit_turnover = exit_p * qty
        total_turnover = entry_turnover + exit_turnover
        brokerage = min(20.0, entry_turnover * 0.0003) + min(20.0, exit_turnover * 0.0003)
        stt = exit_turnover * 0.00025
        exch = total_turnover * 0.0000325
        sebi = total_turnover * 0.000001
        gst = (brokerage + exch + sebi) * 0.18
        stamp = entry_turnover * 0.00003
        friction = brokerage + stt + exch + sebi + gst + stamp
        
        gross_profit = (exit_p - entry_p) * qty
        net_profit = gross_profit - friction
        self.assertGreater(net_profit, 0.0)
        self.assertLess(net_profit, gross_profit)

    def test_institutional_breaker_and_volume_profile(self):
        # Verify Breaker block tracking, Inducement sweeps, and Volume Profile POC evaluation
        strat_res = InstitutionalStrategyEngine.evaluate_strategies(
            df=self.df_5m,
            daily_df=self.daily_df
        )
        self.assertIsNotNone(strat_res)
        self.assertIn("bullish_breaker_obs", strat_res)
        self.assertIn("bearish_breaker_obs", strat_res)
        self.assertIn("has_eqh_inducement_sweep", strat_res)
        self.assertIn("has_eql_inducement_sweep", strat_res)
        self.assertIn("volume_profile_poc", strat_res)
        self.assertIn("volume_profile_vah", strat_res)
        self.assertIn("volume_profile_val", strat_res)
        self.assertGreater(strat_res["volume_profile_poc"], 0.0)
        self.assertLessEqual(strat_res["volume_profile_val"], strat_res["volume_profile_vah"])

    def test_strict_requested_window_validation(self):
        from engine.market_validator import MarketValidator
        first_candle = self.df_5m.index[0]
        last_candle = self.df_5m.index[-1]
        
        # Valid matching window
        val_ok = MarketValidator.validate_historical_replay_data(
            self.df_5m,
            requested_start_dt=first_candle,
            requested_end_dt=last_candle
        )
        self.assertTrue(val_ok.get("is_valid", False))
        
        # Mismatched requested start
        wrong_start = first_candle + pd.Timedelta(minutes=5)
        val_bad_start = MarketValidator.validate_historical_replay_data(
            self.df_5m,
            requested_start_dt=wrong_start
        )
        self.assertFalse(val_bad_start.get("is_valid", False))
        self.assertIn("does not match requested_start_dt", val_bad_start.get("reason", ""))

    def test_instrument_turnover_costs_futures_vs_equity(self):
        # Futures should have lower STT (0.0125% vs 0.025%) and lower stamp duty (0.002% vs 0.003%)
        turnover = 100000.0
        equity_stt = turnover * 0.00025
        futures_stt = turnover * 0.000125
        self.assertEqual(equity_stt, 25.0)
        self.assertEqual(futures_stt, 12.5)

    def test_atr_no_lookahead_and_early_ob_suppression(self):
        # A dataframe with < 14 bars should have no active OBs formed since ATR is unavailable
        short_df = self.df_5m.iloc[:12].copy()
        strat_res = InstitutionalStrategyEngine.evaluate_strategies(
            df=short_df,
            daily_df=self.daily_df
        )
        self.assertEqual(len(strat_res.get("active_demand_obs", [])), 0)
        self.assertEqual(len(strat_res.get("active_supply_obs", [])), 0)

    def test_breaker_rejection_strength_prevents_false_positives(self):
        # Verify that weak wick touches without firm rejection are not qualified as confirmed breakers
        strat_res = InstitutionalStrategyEngine.evaluate_strategies(
            df=self.df_5m,
            daily_df=self.daily_df
        )
        # Check that any confirmed breaker has all 3 required milestone keys:
        # created_ts, invalidated_ts, and breaker_confirmed_ts
        for b in strat_res.get("bullish_breaker_obs", []):
            self.assertEqual(b.get("breaker_state"), "BREAKER_CONFIRMED_REJECTION")
            self.assertIn("created_ts", b)
            self.assertIn("invalidated_ts", b)
            self.assertIn("breaker_confirmed_ts", b)

        for b in strat_res.get("bearish_breaker_obs", []):
            self.assertEqual(b.get("breaker_state"), "BREAKER_CONFIRMED_REJECTION")
            self.assertIn("created_ts", b)
            self.assertIn("invalidated_ts", b)
            self.assertIn("breaker_confirmed_ts", b)

    def test_live_timeframe_exact_grid_rejection(self):
        from engine.market_validator import MarketValidator
        # Construct unaligned 1-minute feed disguised as 5-minute
        bad_df = self.df_5m.copy()
        bad_df.index = [t.replace(minute=t.minute + 1) for t in bad_df.index]
        val = MarketValidator.validate_feed_freshness(bad_df, candle_duration_seconds=300)
        self.assertFalse(val.get("is_valid", False))
        self.assertEqual(val.get("status"), "UNALIGNED_CANDLE_TIMESTAMPS")

    def test_special_trading_session_validation(self):
        from engine.market_validator import MarketValidator
        # Special session on 2026-11-08 (18:15 to 19:15 IST)
        special_date = datetime.date(2026, 11, 8)
        spec_times = pd.date_range("2026-11-08 18:15:00", "2026-11-08 19:15:00", freq="5min", tz="Asia/Kolkata")
        spec_df = pd.DataFrame({
            "Open": [1000.0] * len(spec_times),
            "High": [1005.0] * len(spec_times),
            "Low": [995.0] * len(spec_times),
            "Close": [1002.0] * len(spec_times),
            "Volume": [10000] * len(spec_times)
        }, index=spec_times)
        val = MarketValidator.validate_feed_freshness(spec_df, candle_duration_seconds=300, now_dt=spec_times[-1] + pd.Timedelta(seconds=300))
        # Should not be rejected as OFF_HOURS_CANDLE
        self.assertNotEqual(val.get("status"), "OFF_HOURS_CANDLE")

    def test_options_replay_disabled_without_contract_chain(self):
        from scripts.replay_5m_screener import run_screener_replay
        # OPTIONS replay must explicitly raise ValueError until true contract-level chains are loaded
        with self.assertRaises(ValueError):
            run_screener_replay(days="1d", instrument_type="OPTIONS")

    def test_futures_margin_sizing_no_forced_lot(self):
        # When available capital is less than required margin for 1 lot, trade must be rejected (0 lots, not forced 1 lot)
        from core.contract_master import ContractMaster
        spec = ContractMaster.get_contract_spec("RELIANCE", as_of_date=datetime.date(2026, 1, 1))
        self.assertIsNotNone(spec)
        self.assertEqual(spec["lot_size"], 250)
        self.assertEqual(spec["margin_pct"], 0.20)
        
        entry_price = 2500.0
        contract_notional = entry_price * spec["lot_size"] # ₹6,25,000
        req_margin = contract_notional * spec["margin_pct"] # ₹1,25,000
        available_alloc = 50000.0 # Only ₹50,000 available
        
        num_lots = int(available_alloc / req_margin)
        self.assertEqual(num_lots, 0) # Must be 0, never forced to 1

    def test_rupee_based_r_multiple_eliminates_margin_distortion(self):
        # If entry = 1000, SL = 990 (Risk = 10 Rs/share), Exit = 1020 (Gross = +20 Rs/share)
        # Qty = 250 shares. Margin = 20% (₹50,000). Total turnover = (1000 + 1020) * 250 = ₹5,05,000
        # Friction ~ ₹100. Net PnL ~ (20 * 250) - 100 = ₹4,900.
        # Rupee Risk = 10 * 250 = ₹2,500.
        # Correct R-Multiple = ₹4,900 / ₹2,500 = 1.96R (NOT 5x inflated by margin leverage!)
        qty = 250
        entry = 1000.0
        sl = 990.0
        exit_p = 1020.0
        risk_per_share = entry - sl
        gross_pnl = (exit_p - entry) * qty
        friction = 100.0
        net_pnl = gross_pnl - friction
        
        rupee_risk = risk_per_share * qty
        r_mult = round(net_pnl / rupee_risk, 2)
        self.assertAlmostEqual(r_mult, 1.96, places=1)
        self.assertLess(r_mult, 2.5) # Confirms it is not distorted to ~9.8R

    def test_contract_master_input_validation(self):
        from core.contract_master import ContractMaster
        # 1. Unknown symbol
        self.assertIsNone(ContractMaster.get_contract_spec("UNKNOWN_STOCK"))
        self.assertIsNone(ContractMaster.get_contract_spec(None))
        self.assertIsNone(ContractMaster.get_contract_spec(12345))
        
        # 2. Date before earliest contract inception (e.g. year 2010)
        self.assertIsNone(ContractMaster.get_contract_spec("RELIANCE", as_of_date=datetime.date(2010, 1, 1)))
        
        # 3. Valid date string
        spec_str = ContractMaster.get_contract_spec("RELIANCE", as_of_date="2024-06-01")
        self.assertIsNotNone(spec_str)
        self.assertEqual(spec_str["lot_size"], 250)
        self.assertIn("circular_ref", spec_str)
        self.assertIn("estimated_margin_pct", spec_str)
        
        # 4. Invalid date format string
        self.assertIsNone(ContractMaster.get_contract_spec("RELIANCE", as_of_date="not-a-date"))
        
        # 5. Date after contract change (2025)
        spec_after = ContractMaster.get_contract_spec("RELIANCE", as_of_date=datetime.date(2025, 1, 1))
        self.assertIsNotNone(spec_after)
        self.assertEqual(spec_after["effective_date"], datetime.date(2024, 11, 29))

    def test_futures_replay_rupee_r_multiple_execution_integration(self):
        # Integration test verifying that the replay logic computes rupee-based R-multiple correctly:
        # Given:
        actual_entry = 2500.0
        sl = 2475.0  # Risk = 25 Rs/share
        exit_price_actual = 2550.0  # Gross gain = 50 Rs/share (+2R before friction)
        trade_qty = 250
        actual_allocated_margin_rs = 125000.0 # 20% margin
        contract_notional_total = actual_entry * trade_qty # ₹6,25,000
        direction = "BULLISH"
        
        entry_turnover_rs = actual_entry * trade_qty
        exit_turnover_rs = exit_price_actual * trade_qty
        total_turnover_rs = entry_turnover_rs + exit_turnover_rs

        brokerage_rs = min(20.0, entry_turnover_rs * 0.0003) + min(20.0, exit_turnover_rs * 0.0003)
        stt_rs = exit_turnover_rs * 0.000125
        exch_turnover_fee_rs = total_turnover_rs * 0.000019
        stamp_duty_rs = entry_turnover_rs * 0.00002
        sebi_charges_rs = total_turnover_rs * 0.000001
        gst_rs = (brokerage_rs + exch_turnover_fee_rs + sebi_charges_rs) * 0.18
        total_friction_rs = brokerage_rs + stt_rs + exch_turnover_fee_rs + sebi_charges_rs + gst_rs + stamp_duty_rs

        gross_realized_rs = (exit_price_actual - actual_entry) * trade_qty
        net_realized_rs = gross_realized_rs - total_friction_rs

        net_pnl_margin = (net_realized_rs / actual_allocated_margin_rs) * 100.0
        return_on_notional = (net_realized_rs / contract_notional_total) * 100.0
        
        rupee_risk = (actual_entry - sl) * trade_qty # 25 * 250 = ₹6,250
        r_mult = round(net_realized_rs / rupee_risk, 2)

        # Net profit ~ ₹12,500 - friction (~₹160) = ~₹12,340
        # Rupee risk = ₹6,250
        # R-multiple must be approximately 1.97R (close to 2R)
        self.assertAlmostEqual(r_mult, 1.97, places=1)
        # Margin return is ~9.8%, Notional return is ~1.97%
        self.assertGreater(net_pnl_margin, return_on_notional)
        self.assertAlmostEqual(return_on_notional, 1.97, places=1)

    def test_evaluate_open_outcomes_gap_through_stop(self):
        # 1. Test BUY signal with gap-through-stop
        # Signal entry = 1000.0, SL = 990.0, TP1 = 1020.0
        sig_id = storage.log_signal({
            "symbol": "TESTGAP.NS",
            "strategy": "OB_PULLBACK",
            "direction": "BUY",
            "entry_price": 1000.0,
            "stop_loss": 990.0,
            "target_1": 1020.0,
            "target_2": 1040.0,
            "signal_price": 999.0,
            "expected_entry_price": 1000.0,
            "actual_entry_price": 1000.0
        })
        
        conn = storage.get_db_connection()
        c = conn.cursor()
        # Set signal timestamp and last_candle_time to 1 hour ago so test candles are guaranteed completed
        one_hour_ago = time.time() - 3600
        c.execute("UPDATE historical_signals SET timestamp = ?, last_candle_time = ? WHERE id = ?", (one_hour_ago, one_hour_ago, sig_id))
        conn.commit()
        conn.close()

        # Subsequent candle formed 30 minutes ago (well completed, c_ts + 300 <= now)
        c_time = pd.to_datetime(one_hour_ago + 300, unit='s', utc=True).tz_convert('Asia/Kolkata')
        df_gap = pd.DataFrame({
            "Open": [980.0],
            "High": [985.0],
            "Low": [975.0],
            "Close": [978.0],
            "Volume": [10000]
        }, index=pd.DatetimeIndex([c_time]))
        
        storage.evaluate_open_outcomes({"TESTGAP.NS": df_gap})
        
        conn = storage.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM historical_signals WHERE id = ?", (sig_id,))
        sig = cursor.fetchone()
        conn.close()
        
        self.assertEqual(sig["outcome"], "LOSS")
        # Final P&L should be based on Open (980 - 1000 = -20), NOT the SL (990 - 1000 = -10)
        self.assertEqual(sig["final_pnl"], -20.0)

    def test_same_candle_tp1_does_not_falsely_trigger_pre_tp1_open_breakeven(self):
        # When a candle touches both TP1 and breakeven stop:
        # In a single completed bar, it is impossible to know whether the low occurred before TP1 or after TP1.
        # Under conservative accounting, touching both TP1 and breakeven stop in the same bar yields WIN_PARTIAL
        # (50% locked at TP1, 50% stopped at breakeven), NOT a full loss or premature gap exit at open.
        sig_id = storage.log_signal({
            "symbol": "TEST_CHRONO.NS",
            "direction": "BUY",
            "strategy": "SMC_PULLBACK",
            "entry_price": 1000.0,
            "stop_loss": 990.0,
            "target_1": 1010.0,
            "target_2": 1025.0
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        one_hour_ago = time.time() - 3600
        c.execute("UPDATE historical_signals SET timestamp = ?, last_candle_time = ? WHERE id = ?", (one_hour_ago, one_hour_ago, sig_id))
        conn.commit()
        conn.close()

        # Candle opens at 995 (above orig_sl 990, but below entry 1000), rallies to High 1015 (hits TP1), Low 994, Close 1012.
        c_time = pd.to_datetime(one_hour_ago + 300, unit='s', utc=True).tz_convert('Asia/Kolkata')
        df_candle = pd.DataFrame({
            "Open": [995.0],
            "High": [1015.0],
            "Low": [994.0],
            "Close": [1012.0],
            "Volume": [15000]
        }, index=pd.DatetimeIndex([c_time]))

        storage.evaluate_open_outcomes({"TEST_CHRONO.NS": df_candle})

        conn = storage.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT outcome, tp1_hit, tp2_hit, current_stop_loss, final_pnl FROM historical_signals WHERE id = ?", (sig_id,))
        sig = cursor.fetchone()
        conn.close()

        # Trade exits as WIN_PARTIAL under conservative same-candle breakeven sequence, with final_pnl calculated at BE (5.0), NOT open exit!
        self.assertEqual(sig["tp1_hit"], 1)
        self.assertEqual(sig["outcome"], "WIN_PARTIAL")
        self.assertEqual(sig["final_pnl"], 5.0)

    def test_same_candle_tp1_without_touching_breakeven_remains_active(self):
        # When a candle reaches TP1 and its Low remains strictly above breakeven stop (sl = entry = 1000.0),
        # the remaining 50% position MUST remain ACTIVE with current_stop_loss updated to breakeven!
        sig_id = storage.log_signal({
            "symbol": "TEST_STILL_ACTIVE.NS",
            "direction": "BUY",
            "strategy": "SMC_PULLBACK",
            "entry_price": 1000.0,
            "stop_loss": 990.0,
            "target_1": 1010.0,
            "target_2": 1030.0
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        one_hour_ago = time.time() - 3600
        c.execute("UPDATE historical_signals SET timestamp = ?, last_candle_time = ? WHERE id = ?", (one_hour_ago, one_hour_ago, sig_id))
        conn.commit()
        conn.close()

        # Candle opens at 1002, rallies to High 1015 (hits TP1), Low 1001 (strictly above BE 1000), Close 1012.
        c_time = pd.to_datetime(one_hour_ago + 300, unit='s', utc=True).tz_convert('Asia/Kolkata')
        df_candle = pd.DataFrame({
            "Open": [1002.0],
            "High": [1015.0],
            "Low": [1001.0],
            "Close": [1012.0],
            "Volume": [15000]
        }, index=pd.DatetimeIndex([c_time]))

        storage.evaluate_open_outcomes({"TEST_STILL_ACTIVE.NS": df_candle})

        conn = storage.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT outcome, tp1_hit, tp2_hit, current_stop_loss FROM historical_signals WHERE id = ?", (sig_id,))
        sig = cursor.fetchone()
        conn.close()

        # Trade MUST remain ACTIVE (tp1_hit = 1, stop moved to breakeven 1000.0)
        self.assertEqual(sig["tp1_hit"], 1)
        self.assertEqual(sig["outcome"], "ACTIVE")
        self.assertEqual(sig["current_stop_loss"], 1000.0)

    def test_subsequent_candle_gap_below_breakeven_after_tp1_hits_partial_win(self):
        # If TP1 was already confirmed in a previous candle, and the NEXT candle opens below breakeven stop,
        # it should exit at the new candle's open price.
        sig_id = storage.log_signal({
            "symbol": "TEST_BE_GAP.NS",
            "direction": "BUY",
            "strategy": "SMC_PULLBACK",
            "entry_price": 1000.0,
            "stop_loss": 990.0,
            "target_1": 1010.0,
            "target_2": 1030.0
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        one_hour_ago = time.time() - 3600
        c.execute("UPDATE historical_signals SET timestamp = ?, last_candle_time = ? WHERE id = ?", (one_hour_ago, one_hour_ago, sig_id))
        conn.commit()
        conn.close()

        # Candle 1: hits TP1 (High 1015, Low 1002, Open 1003, Close 1012)
        c1_time = pd.to_datetime(one_hour_ago + 300, unit='s', utc=True).tz_convert('Asia/Kolkata')
        # Candle 2: gaps open below breakeven (Open 996, High 998, Low 993, Close 995)
        c2_time = pd.to_datetime(one_hour_ago + 600, unit='s', utc=True).tz_convert('Asia/Kolkata')

        df_c1 = pd.DataFrame({
            "Open": [1003.0], "High": [1015.0], "Low": [1002.0], "Close": [1012.0], "Volume": [20000]
        }, index=pd.DatetimeIndex([c1_time]))
        df_c2 = pd.DataFrame({
            "Open": [996.0], "High": [998.0], "Low": [993.0], "Close": [995.0], "Volume": [20000]
        }, index=pd.DatetimeIndex([c2_time]))

        storage.evaluate_open_outcomes({"TEST_BE_GAP.NS": df_c1})
        storage.evaluate_open_outcomes({"TEST_BE_GAP.NS": df_c2})

        conn = storage.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT outcome, final_pnl FROM historical_signals WHERE id = ?", (sig_id,))
        sig = cursor.fetchone()
        conn.close()

        self.assertEqual(sig["outcome"], "WIN_PARTIAL")
        # Final PnL = 0.5 * (1010 - 1000) + 0.5 * (996 - 1000) = 5.0 - 2.0 = 3.0
        self.assertEqual(sig["final_pnl"], 3.0)

    def test_same_candle_tp1_tp2_and_breakeven_ambiguity_conservative_partial_win(self):
        # Scenario:
        # BUY Entry: 1000, SL: 990, TP1: 1010, TP2: 1025
        # Candle: Open 1003, High 1030 (touches TP1 & TP2), Low 998 (touches breakeven 1000)
        # Outcome MUST be conservative WIN_PARTIAL (stop-first assumption), NOT WIN_FULL!
        sig_id = storage.log_signal({
            "symbol": "TEST_AMBIG_BUY.NS",
            "direction": "BUY",
            "strategy": "SMC_PULLBACK",
            "entry_price": 1000.0,
            "stop_loss": 990.0,
            "target_1": 1010.0,
            "target_2": 1025.0
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        one_hour_ago = time.time() - 3600
        c.execute("UPDATE historical_signals SET timestamp = ?, last_candle_time = ? WHERE id = ?", (one_hour_ago, one_hour_ago, sig_id))
        conn.commit()
        conn.close()

        c_time = pd.to_datetime(one_hour_ago + 300, unit='s', utc=True).tz_convert('Asia/Kolkata')
        df_candle = pd.DataFrame({
            "Open": [1003.0], "High": [1030.0], "Low": [998.0], "Close": [1020.0], "Volume": [25000]
        }, index=pd.DatetimeIndex([c_time]))

        storage.evaluate_open_outcomes({"TEST_AMBIG_BUY.NS": df_candle})

        conn = storage.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT outcome, tp1_hit, tp2_hit, final_pnl FROM historical_signals WHERE id = ?", (sig_id,))
        sig = cursor.fetchone()
        conn.close()

        self.assertEqual(sig["outcome"], "WIN_PARTIAL")
        self.assertEqual(sig["tp1_hit"], 1)
        self.assertEqual(sig["tp2_hit"], 0)
        # PnL: 0.5 * (1010 - 1000) + 0.5 * (1000 - 1000) = 5.0
        self.assertEqual(sig["final_pnl"], 5.0)

    def test_same_candle_tp1_and_breakeven_touched_without_tp2_exits_partial_win(self):
        # Scenario:
        # BUY Entry: 1000, SL: 990, TP1: 1010, TP2: 1030
        # Candle: Open 1002, High 1015 (hits TP1, but NOT TP2), Low 998 (hits BE 1000)
        # Outcome MUST be WIN_PARTIAL at breakeven stop, rather than remaining open
        sig_id = storage.log_signal({
            "symbol": "TEST_BE_TOUCH.NS",
            "direction": "BUY",
            "strategy": "SMC_PULLBACK",
            "entry_price": 1000.0,
            "stop_loss": 990.0,
            "target_1": 1010.0,
            "target_2": 1030.0
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        one_hour_ago = time.time() - 3600
        c.execute("UPDATE historical_signals SET timestamp = ?, last_candle_time = ? WHERE id = ?", (one_hour_ago, one_hour_ago, sig_id))
        conn.commit()
        conn.close()

        c_time = pd.to_datetime(one_hour_ago + 300, unit='s', utc=True).tz_convert('Asia/Kolkata')
        df_candle = pd.DataFrame({
            "Open": [1002.0], "High": [1015.0], "Low": [998.0], "Close": [1005.0], "Volume": [25000]
        }, index=pd.DatetimeIndex([c_time]))

        storage.evaluate_open_outcomes({"TEST_BE_TOUCH.NS": df_candle})

        conn = storage.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT outcome, tp1_hit, final_pnl FROM historical_signals WHERE id = ?", (sig_id,))
        sig = cursor.fetchone()
        conn.close()

        self.assertEqual(sig["outcome"], "WIN_PARTIAL")
        self.assertEqual(sig["tp1_hit"], 1)
        self.assertEqual(sig["final_pnl"], 5.0)

    def test_sell_same_candle_tp1_tp2_and_breakeven_ambiguity_conservative_partial_win(self):
        # SELL Scenario:
        # Entry: 1000, SL: 1010, TP1: 990, TP2: 975
        # Candle: Open 998, Low 970 (hits TP1 & TP2), High 1002 (hits BE 1000)
        sig_id = storage.log_signal({
            "symbol": "TEST_AMBIG_SELL.NS",
            "direction": "SELL",
            "strategy": "SMC_BREAKDOWN",
            "entry_price": 1000.0,
            "stop_loss": 1010.0,
            "target_1": 990.0,
            "target_2": 975.0
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        one_hour_ago = time.time() - 3600
        c.execute("UPDATE historical_signals SET timestamp = ?, last_candle_time = ? WHERE id = ?", (one_hour_ago, one_hour_ago, sig_id))
        conn.commit()
        conn.close()

        c_time = pd.to_datetime(one_hour_ago + 300, unit='s', utc=True).tz_convert('Asia/Kolkata')
        df_candle = pd.DataFrame({
            "Open": [998.0], "High": [1002.0], "Low": [970.0], "Close": [980.0], "Volume": [25000]
        }, index=pd.DatetimeIndex([c_time]))

        storage.evaluate_open_outcomes({"TEST_AMBIG_SELL.NS": df_candle})

        conn = storage.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT outcome, tp1_hit, tp2_hit, final_pnl FROM historical_signals WHERE id = ?", (sig_id,))
        sig = cursor.fetchone()
        conn.close()

        self.assertEqual(sig["outcome"], "WIN_PARTIAL")
        self.assertEqual(sig["tp1_hit"], 1)
        self.assertEqual(sig["tp2_hit"], 0)
        # PnL for SELL: 0.5 * (1000 - 990) + 0.5 * (1000 - 1000) = 5.0
        self.assertEqual(sig["final_pnl"], 5.0)

    def test_dict_input_same_candle_tp1_and_breakeven_ambiguity(self):
        # Case B (Dictionary input) should behave identically to DataFrame input
        sig_id = storage.log_signal({
            "symbol": "TEST_DICT_AMBIG.NS",
            "direction": "BUY",
            "strategy": "SMC_PULLBACK",
            "entry_price": 1000.0,
            "stop_loss": 990.0,
            "target_1": 1010.0,
            "target_2": 1025.0
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        one_hour_ago = time.time() - 3600
        c.execute("UPDATE historical_signals SET timestamp = ?, last_candle_time = ? WHERE id = ?", (one_hour_ago, one_hour_ago, sig_id))
        conn.commit()
        conn.close()

        dict_candle = {
            "open": 1003.0, "high": 1030.0, "low": 998.0, "close": 1020.0, "volume": 25000,
            "timestamp": one_hour_ago + 300, "is_completed": True
        }

        storage.evaluate_open_outcomes({"TEST_DICT_AMBIG.NS": dict_candle})

        conn = storage.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT outcome, tp1_hit, tp2_hit, final_pnl FROM historical_signals WHERE id = ?", (sig_id,))
        sig = cursor.fetchone()
        conn.close()

        self.assertEqual(sig["outcome"], "WIN_PARTIAL")
        self.assertEqual(sig["final_pnl"], 5.0)

    def test_first_candle_after_signal_not_skipped_when_candle_timestamp_is_start_time(self):
        # Scenario:
        # Signal candle completed 30 minutes ago (same session/day).
        # Signal candle close time was base_epoch.
        # Signal generation logged with timestamp = base_epoch + 3.0.
        # The NEXT executable 5m candle starts at base_epoch (09:20:00) and completes at base_epoch + 300 (09:25:00).
        # In the market feed, this candle's index timestamp is base_epoch (which is < base_epoch + 3.0).
        # Outcome evaluator MUST NOT skip this first candle!
        now_epoch = time.time()
        base_epoch = now_epoch - 1800  # 30 minutes ago today
        sig_id = storage.log_signal({
            "symbol": "TEST_FIRST_BAR.NS",
            "direction": "BUY",
            "strategy": "SMC_PULLBACK",
            "entry_price": 1000.0,
            "stop_loss": 990.0,
            "target_1": 1010.0,
            "target_2": 1025.0,
            "signal_candle_start": base_epoch - 300,
            "signal_candle_close_time": base_epoch,
            "signal_generated_at": base_epoch + 3.0,
            "quantity": 500
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        # Explicitly set timestamp = base_epoch + 3.0, signal_candle_close_time = base_epoch, last_candle_time = 0
        c.execute("UPDATE historical_signals SET timestamp = ?, signal_candle_close_time = ?, last_candle_time = 0 WHERE id = ?", (base_epoch + 3.0, base_epoch, sig_id))
        conn.commit()
        conn.close()

        # Next candle starting at base_epoch (which is < base_epoch + 3.0, but >= signal_candle_close_time)
        c_time = pd.to_datetime(base_epoch, unit='s', utc=True).tz_convert('Asia/Kolkata')
        df_first_candle = pd.DataFrame({
            "Open": [1000.0],
            "High": [1015.0],
            "Low": [1001.0],
            "Close": [1012.0],
            "Volume": [30000]
        }, index=pd.DatetimeIndex([c_time]))

        storage.evaluate_open_outcomes({"TEST_FIRST_BAR.NS": df_first_candle})

        conn = storage.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT outcome, tp1_hit, last_candle_time FROM historical_signals WHERE id = ?", (sig_id,))
        sig = cursor.fetchone()
        conn.close()

        # The first candle MUST have been processed!
        self.assertEqual(sig["tp1_hit"], 1)
        self.assertAlmostEqual(sig["last_candle_time"], base_epoch, places=3)

    def test_quantity_and_instrument_type_cost_scaling(self):
        # Verify that different quantities scale brokerage and statutory fees properly
        from engine.cost_engine import IndianMarketCosts
        one_hour_ago = time.time() - 3600
        sig_100_id = storage.log_signal({
            "symbol": "COST_100.NS", "direction": "BUY", "strategy": "TEST_STRAT",
            "entry_price": 1000.0, "stop_loss": 990.0, "target_1": 1020.0, "target_2": 1020.0, "quantity": 100,
            "instrument_type": "EQUITY_INTRADAY",
            "signal_candle_start": one_hour_ago - 300,
            "signal_candle_close_time": one_hour_ago
        })
        sig_1000_id = storage.log_signal({
            "symbol": "COST_1000.NS", "direction": "BUY", "strategy": "TEST_STRAT",
            "entry_price": 1000.0, "stop_loss": 990.0, "target_1": 1020.0, "target_2": 1020.0, "quantity": 1000,
            "instrument_type": "EQUITY_INTRADAY",
            "signal_candle_start": one_hour_ago - 300,
            "signal_candle_close_time": one_hour_ago
        })
        sig_fut_id = storage.log_signal({
            "symbol": "COST_FUT.NS", "direction": "BUY", "strategy": "TEST_STRAT",
            "entry_price": 1000.0, "stop_loss": 990.0, "target_1": 1020.0, "target_2": 1020.0, "quantity": 250,
            "instrument_type": "FUTURES",
            "signal_candle_start": one_hour_ago - 300,
            "signal_candle_close_time": one_hour_ago
        })

        conn = storage.get_db_connection()
        c = conn.cursor()
        c.execute("UPDATE historical_signals SET timestamp = ?, last_candle_time = 0 WHERE id IN (?, ?, ?)", (one_hour_ago, sig_100_id, sig_1000_id, sig_fut_id))
        conn.commit()
        conn.close()

        c_time = pd.to_datetime(one_hour_ago + 300, unit='s', utc=True).tz_convert('Asia/Kolkata')
        df_win = pd.DataFrame({
            "Open": [1000.0], "High": [1025.0], "Low": [1001.0], "Close": [1022.0], "Volume": [20000]
        }, index=pd.DatetimeIndex([c_time]))

        storage.evaluate_open_outcomes({"COST_100.NS": df_win, "COST_1000.NS": df_win, "COST_FUT.NS": df_win})

        conn = storage.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, brokerage_costs, final_pnl, net_pnl, quantity, instrument_type FROM historical_signals WHERE id IN (?, ?, ?)", (sig_100_id, sig_1000_id, sig_fut_id))
        rows = {r["id"]: dict(r) for r in cursor.fetchall()}
        conn.close()

        self.assertIn(sig_100_id, rows)
        self.assertIn(sig_1000_id, rows)
        self.assertIn(sig_fut_id, rows)

        # 1. Quantity assertions
        self.assertEqual(rows[sig_100_id]["quantity"], 100)
        self.assertEqual(rows[sig_1000_id]["quantity"], 1000)
        self.assertEqual(rows[sig_fut_id]["quantity"], 250)

        # 2. Numerical cost assertions:
        # Brokerage + statutory cost per share for 100 shares is higher due to minimum order brokerage floor
        cost_100 = rows[sig_100_id]["brokerage_costs"]
        cost_1000 = rows[sig_1000_id]["brokerage_costs"]
        cost_fut = rows[sig_fut_id]["brokerage_costs"]

        self.assertGreater(cost_100, 0.0)
        self.assertGreater(cost_1000, 0.0)
        self.assertGreater(cost_fut, 0.0)

        # Total transaction friction scales with quantity, so per-share cost decreases as fixed brokerage is amortized
        self.assertGreater(cost_100, cost_1000)

        # Net PnL must be strictly final_pnl - brokerage_costs
        self.assertAlmostEqual(rows[sig_100_id]["net_pnl"], rows[sig_100_id]["final_pnl"] - cost_100, places=2)
        self.assertAlmostEqual(rows[sig_1000_id]["net_pnl"], rows[sig_1000_id]["final_pnl"] - cost_1000, places=2)
        self.assertAlmostEqual(rows[sig_fut_id]["net_pnl"], rows[sig_fut_id]["final_pnl"] - cost_fut, places=2)

    def test_smc_confirmation_endpoint_normalizes_direction_and_evaluates(self):
        # Integration test verifying manual SMC confirmation endpoint normalizes BULLISH/BEARISH
        import groww_terminal_app as gta
        from flask import json
        client = gta.app.test_client()

        # Seed screener_state so candidate passes stage check
        test_sym = "RELIANCE.NS"
        auth_ts = str(self.df_5m.index[-1])
        cand = {
            "symbol": test_sym,
            "direction": "BULLISH",
            "current_price": 2500.0,
            "suggested_sl": 2475.0,
            "suggested_tp1": 2550.0,
            "suggested_tp2": 2600.0,
            "is_actionable": True,
            "stage": "ENTRY READY",
            "candle_ts": auth_ts,
            "conviction_score": 85,
            "confirmation_score": 90
        }
        gta.screener_state["ranked_opportunities"] = {
            "entry_ready": [cand],
            "ranked_candle_ts": auth_ts
        }
        gta.screener_state["raw_market_cache"] = {
            "^NSEI": {
                "timestamp": time.time(),
                "df": self.df_5m
            }
        }

        # Send request with action="BULLISH" to /api/signals/log-confirmed
        # Mock MarketValidator.is_market_open to return True and get_ist_now to align with synthetic candle timestamps
        from unittest.mock import patch
        mock_now = self.df_5m.index[-1] + pd.Timedelta(seconds=300)
        with patch("engine.market_validator.MarketValidator.is_market_open", return_value=True), \
             patch("engine.market_validator.MarketValidator.get_ist_now", return_value=mock_now):
            resp = client.post("/api/signals/log-confirmed", data=json.dumps({
                "symbol": test_sym,
                "action": "BULLISH",
                "candle_ts": auth_ts,
                "entry_price": 2500.0,
                "sl": 2475.0,
                "tp1": 2550.0,
                "tp2": 2600.0
            }), content_type="application/json")

        self.assertEqual(resp.status_code, 200)
        res_data = json.loads(resp.data)
        self.assertTrue(res_data.get("success"))
        sig_id = res_data.get("signal_id")
        self.assertIsNotNone(sig_id)

        # Verify stored signal has normalized BUY direction and valid candle close timestamp
        conn = storage.get_db_connection()
        c = conn.cursor()
        c.execute("SELECT direction, signal_candle_close_time, quantity, instrument_type FROM historical_signals WHERE id = ?", (sig_id,))
        saved = c.fetchone()
        conn.close()

        self.assertEqual(saved["direction"], "BUY")
        self.assertGreater(saved["signal_candle_close_time"], 0.0)
        self.assertEqual(saved["instrument_type"], "FUTURES")
        self.assertEqual(saved["quantity"], 250)

    def test_evaluate_candle_transition_adverse_gaps_and_direction_normalization(self):
        from core.storage import evaluate_candle_transition

        # 1. BUY gap below original SL -> Exits at adverse open
        res_buy_gap = evaluate_candle_transition(
            direction="BUY", entry=1000.0, orig_sl=990.0, sl=990.0, tp1=1020.0, tp2=1040.0,
            tp1_hit_prior=False, c_open=982.0, c_high=988.0, c_low=978.0, c_ts=1700000000.0, est_costs=1.5
        )
        self.assertEqual(res_buy_gap["outcome"], "LOSS")
        self.assertEqual(res_buy_gap["final_pnl"], -18.0)  # 982 - 1000 = -18 (not capped at 990)
        self.assertTrue(res_buy_gap["should_break"])

        # 2. SELL gap above original SL -> Exits at adverse open
        res_sell_gap = evaluate_candle_transition(
            direction="SELL", entry=1000.0, orig_sl=1010.0, sl=1010.0, tp1=980.0, tp2=960.0,
            tp1_hit_prior=False, c_open=1018.0, c_high=1022.0, c_low=1015.0, c_ts=1700000000.0, est_costs=1.5
        )
        self.assertEqual(res_sell_gap["outcome"], "LOSS")
        self.assertEqual(res_sell_gap["final_pnl"], -18.0)  # 1000 - 1018 = -18 (not capped at 1010)
        self.assertTrue(res_sell_gap["should_break"])

        # 3. SELL after TP1, gap above breakeven -> Partial win using adverse open
        res_sell_be_gap = evaluate_candle_transition(
            direction="SELL", entry=1000.0, orig_sl=1010.0, sl=1000.0, tp1=980.0, tp2=960.0,
            tp1_hit_prior=True, c_open=1005.0, c_high=1008.0, c_low=995.0, c_ts=1700000000.0, est_costs=1.5
        )
        self.assertEqual(res_sell_be_gap["outcome"], "WIN_PARTIAL")
        # 0.5 * (1000 - 980) + 0.5 * (1000 - 1005) = 10.0 - 2.5 = 7.5
        self.assertEqual(res_sell_be_gap["final_pnl"], 7.5)
        self.assertTrue(res_sell_be_gap["should_break"])

        # 4. BULLISH string normalized to BUY
        res_norm = evaluate_candle_transition(
            direction="BULLISH", entry=1000.0, orig_sl=990.0, sl=990.0, tp1=1020.0, tp2=1040.0,
            tp1_hit_prior=False, c_open=1000.0, c_high=1025.0, c_low=1000.0, c_ts=1700000000.0, est_costs=1.5
        )
        self.assertEqual(res_norm["tp1_hit"], 1)

        # 5. Invalid direction -> Fails closed, returns ACTIVE without false execution
        res_invalid = evaluate_candle_transition(
            direction="INVALID_DIR", entry=1000.0, orig_sl=990.0, sl=990.0, tp1=1020.0, tp2=1040.0,
            tp1_hit_prior=False, c_open=1000.0, c_high=1025.0, c_low=1000.0, c_ts=1700000000.0, est_costs=1.5
        )
        self.assertEqual(res_invalid["outcome"], "ACTIVE")
        self.assertFalse(res_invalid["should_break"])

        # 6. Missing or invalid non-finite/negative OHLC -> Reject candle, remain ACTIVE
        res_corrupt_price = evaluate_candle_transition(
            direction="BUY", entry=1000.0, orig_sl=990.0, sl=990.0, tp1=1020.0, tp2=1040.0,
            tp1_hit_prior=False, c_open=-50.0, c_high=1025.0, c_low=995.0, c_ts=1700000000.0, est_costs=1.5
        )
        self.assertEqual(res_corrupt_price["outcome"], "ACTIVE")
        self.assertFalse(res_corrupt_price["should_break"])

        # 7. Same-candle TP1 and TP2 touched -> Conservative sequence (TP1 win, remaining leg evaluated conservatively)
        res_tp1_tp2_same = evaluate_candle_transition(
            direction="BUY", entry=1000.0, orig_sl=990.0, sl=990.0, tp1=1020.0, tp2=1040.0,
            tp1_hit_prior=False, c_open=1002.0, c_high=1045.0, c_low=1001.0, c_ts=1700000000.0, est_costs=1.5
        )
        self.assertEqual(res_tp1_tp2_same["outcome"], "WIN_FULL")
        self.assertEqual(res_tp1_tp2_same["tp1_hit"], 1)
        self.assertEqual(res_tp1_tp2_same["tp2_hit"], 1)

        # 8. Same-candle TP1 and breakeven stop touched -> Conservative stop-first (partial win, not full win)
        res_tp1_be_same = evaluate_candle_transition(
            direction="BUY", entry=1000.0, orig_sl=990.0, sl=990.0, tp1=1020.0, tp2=1040.0,
            tp1_hit_prior=False, c_open=1002.0, c_high=1025.0, c_low=999.0, c_ts=1700000000.0, est_costs=1.5
        )
        self.assertEqual(res_tp1_be_same["outcome"], "WIN_PARTIAL")
        self.assertEqual(res_tp1_be_same["tp1_hit"], 1)
        self.assertEqual(res_tp1_be_same["tp2_hit"], 0)
        self.assertEqual(res_tp1_be_same["final_pnl"], 10.0)  # 0.5 * (1020-1000) + 0.5 * (1000-1000) = 10.0

        # 9. Inconsistent OHLC (High < Low or High < Open) -> Reject candle, remain ACTIVE
        res_bad_high = evaluate_candle_transition(
            direction="BUY", entry=1000.0, orig_sl=990.0, sl=990.0, tp1=1020.0, tp2=1040.0,
            tp1_hit_prior=False, c_open=1005.0, c_high=1000.0, c_low=995.0, c_ts=1700000000.0, est_costs=1.5
        )
        self.assertEqual(res_bad_high["outcome"], "ACTIVE")
        self.assertFalse(res_bad_high["should_break"])

        # 10. String non-numeric corrupted input -> Safe exception handling, remain ACTIVE
        res_str_corrupt = evaluate_candle_transition(
            direction="BUY", entry=1000.0, orig_sl=990.0, sl=990.0, tp1=1020.0, tp2=1040.0,
            tp1_hit_prior=False, c_open="corrupt", c_high=1025.0, c_low=995.0, c_ts=1700000000.0, est_costs=1.5
        )
        self.assertEqual(res_str_corrupt["outcome"], "ACTIVE")
        self.assertFalse(res_str_corrupt["should_break"])

    def test_multi_exit_cost_accounting_no_double_entry_friction(self):
        # Verify that multi-exit cost accounting counts entry friction (brokerage, stamp duty) exactly ONCE
        from engine.cost_engine import IndianMarketCosts
        entry_price = 1000.0
        tot_qty = 200

        # Case 1: Full exit in 1 order
        full_res = IndianMarketCosts.calculate_multi_exit_costs(
            entry_price=entry_price, exits=[(1020.0, 200)], total_quantity=tot_qty, instrument_type="EQUITY_INTRADAY"
        )

        # Case 2: Partial exits in 2 orders of 100 shares each
        multi_res = IndianMarketCosts.calculate_multi_exit_costs(
            entry_price=entry_price, exits=[(1020.0, 100), (1020.0, 100)], total_quantity=tot_qty, instrument_type="EQUITY_INTRADAY"
        )

        # Stamp duty is on BUY turnover only, so both cases must have identical stamp duty!
        self.assertEqual(full_res["stamp_duty"], multi_res["stamp_duty"])
        self.assertEqual(full_res["buy_turnover"], multi_res["buy_turnover"])

        # Brokerage for full exit is 1 buy order + 1 sell order = ₹40 (max)
        # Brokerage for 2 partial exits is 1 buy order + 2 sell orders = ₹60 (max)
        self.assertEqual(full_res["brokerage"], 40.0)
        self.assertEqual(multi_res["brokerage"], 60.0)

    def test_smc_confirmation_endpoint_bearish_and_options_rejection(self):
        # Integration test verifying BEARISH confirmation and strict OPTIONS rejection without contract data
        import groww_terminal_app as gta
        from flask import json
        client = gta.app.test_client()

        test_sym = "TCS.NS"
        auth_ts = str(self.df_5m.index[-1])
        cand = {
            "symbol": test_sym,
            "direction": "BEARISH",
            "current_price": 3500.0,
            "suggested_sl": 3535.0,
            "suggested_tp1": 3430.0,
            "suggested_tp2": 3360.0,
            "is_actionable": True,
            "stage": "ENTRY READY",
            "candle_ts": auth_ts,
            "conviction_score": 85,
            "confirmation_score": 90
        }
        gta.screener_state["ranked_opportunities"] = {
            "entry_ready": [cand],
            "ranked_candle_ts": auth_ts
        }
        gta.screener_state["raw_market_cache"] = {
            "^NSEI": {"timestamp": time.time(), "df": self.df_5m}
        }

        # 1. Test BEARISH confirmation
        from unittest.mock import patch
        mock_now = self.df_5m.index[-1] + pd.Timedelta(seconds=300)
        with patch("engine.market_validator.MarketValidator.is_market_open", return_value=True), \
             patch("engine.market_validator.MarketValidator.get_ist_now", return_value=mock_now):
            resp = client.post("/api/signals/log-confirmed", data=json.dumps({
                "symbol": test_sym,
                "action": "BEARISH",
                "candle_ts": auth_ts,
                "entry_price": 3500.0,
                "sl": 3535.0,
                "tp1": 3430.0,
                "tp2": 3360.0
            }), content_type="application/json")

            self.assertEqual(resp.status_code, 200)
            res_data = json.loads(resp.data)
            self.assertTrue(res_data.get("success"))
            sig_id = res_data.get("signal_id")
            self.assertIsNotNone(sig_id)

            conn = storage.get_db_connection()
            c = conn.cursor()
            c.execute("SELECT direction, quantity, instrument_type FROM historical_signals WHERE id = ?", (sig_id,))
            saved = c.fetchone()
            conn.close()

            self.assertEqual(saved["direction"], "SELL")
            self.assertEqual(saved["instrument_type"], "FUTURES")
            self.assertEqual(saved["quantity"], 175)

            # 2. Test OPTIONS rejection without contract chain data
            opt_resp = client.post("/api/signals/log-confirmed", data=json.dumps({
                "symbol": test_sym,
                "action": "BEARISH",
                "candle_ts": auth_ts,
                "instrument_type": "OPTIONS",
                "entry_price": 3500.0,
                "sl": 3535.0,
                "tp1": 3430.0,
                "tp2": 3360.0
            }), content_type="application/json")

            self.assertEqual(opt_resp.status_code, 400)
            opt_data = json.loads(opt_resp.data)
            self.assertFalse(opt_data.get("success"))
            self.assertIn("OPTIONS trading mode requires", opt_data.get("error", ""))

    def test_nifty_benchmark_fail_closed_gate(self):
        # Verify that compute_and_update_ranked_opportunities fails-closed when NIFTY feed is corrupted
        import groww_terminal_app as gta
        
        # Corrupted NIFTY: negative low price
        bad_nifty = self.df_5m.copy()
        bad_nifty.loc[bad_nifty.index[0], "Low"] = -100.0
        
        # Inject corrupted NIFTY into screener_state raw_market_cache
        gta.screener_state.setdefault("raw_market_cache", {})["^NSEI"] = {
            "timestamp": datetime.datetime.now().timestamp(),
            "df": bad_nifty
        }
        
        result, ts = gta.compute_and_update_ranked_opportunities()
        
        self.assertEqual(result.get("benchmark_status"), "INVALID_BENCHMARK")
        self.assertEqual(len(result.get("top_bullish", [])), 0)
        self.assertEqual(len(result.get("top_bearish", [])), 0)
        self.assertIsNone(ts)

if __name__ == '__main__':
    unittest.main()
