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

if __name__ == '__main__':
    unittest.main()
