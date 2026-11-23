"""
Unified Stock Evaluation Pipeline
Single source of truth used identically by:
1. Live Screener Worker & smc_scan API (groww_terminal_app.py)
2. Historical Screener Replay & Backtesting (scripts/replay_5m_screener.py)

Eliminates divergence in market regime, anomaly detection, sector bias,
institutional confluence scoring, and opportunity ranking.
"""
import pandas as pd
import numpy as np
from strategies.institutional_engine import InstitutionalStrategyEngine
from engine.anomaly_engine import AnomalyDetector
from engine.market_validator import MarketValidator
from engine.opportunity_ranking_engine import OpportunityRankingEngine
from engine.market_regime_engine import MarketRegimeEngine

class ScreenerPipeline:
    @staticmethod
    def evaluate_candidate_unified(
        stock_df,
        symbol,
        sector_name="GENERAL",
        sector_change=0.0,
        sector_bias="NEUTRAL",
        nifty_df=None,
        daily_df=None,
        news_cat=None,
        prev_close=None,
        time_slot_rvol=None,
        rvol_obs=0,
        is_replay=False
    ):
        """
        Executes complete institutional + opportunity ranking analysis on a single stock's 5m history.
        Returns a dictionary with complete signals, risk levels, and ranking evaluation.
        """
        if stock_df is None or len(stock_df) < 15:
            return None

        # Clean columns
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.get_level_values(0)

        # 1. Determine completed candle validation (live feed freshness vs historical integrity)
        if is_replay:
            feed_val = MarketValidator.validate_historical_replay_data(stock_df, candle_duration_seconds=300)
            if not feed_val.get("is_valid", False):
                return None
            is_completed = True
            eval_df = stock_df
        else:
            feed_val = MarketValidator.validate_feed_freshness(stock_df, max_delay_seconds=300, candle_duration_seconds=300)
            if not feed_val.get("is_valid", False):
                return None
            is_completed = feed_val.get("is_completed_candle", False)
            eval_df = stock_df if is_completed else stock_df.iloc[:-1]
        
        if len(eval_df) < 15:
            return None

        # Calculate prev_close if not passed
        if prev_close is None:
            df_c = eval_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_c.index):
                df_c.index = pd.to_datetime(df_c.index)
            dates = sorted(list(set(df_c.index.date)))
            if len(dates) >= 2:
                prev_day_df = df_c[df_c.index.date == dates[-2]]
                prev_close = float(prev_day_df['Close'].iloc[-1])
            else:
                prev_close = float(df_c['Open'].iloc[0])

        # 2. Benchmark evaluation & Macro Regime Context
        nifty_regime = {"regime": "NEUTRAL", "bias": "NEUTRAL", "alignment_score": 50}
        eval_nifty = None
        if nifty_df is not None and not nifty_df.empty:
            if isinstance(nifty_df.columns, pd.MultiIndex):
                nifty_df.columns = nifty_df.columns.get_level_values(0)
            n_val = MarketValidator.validate_feed_freshness(nifty_df, max_delay_seconds=300, candle_duration_seconds=300)
            is_n_comp = n_val.get("is_completed_candle", False)
            eval_nifty = nifty_df if is_n_comp else nifty_df.iloc[:-1]
            if not eval_nifty.empty:
                nifty_regime = MarketRegimeEngine.evaluate_benchmark_regime(eval_nifty)

        # 3. Anomaly detection (Flat open, gaps)
        anomaly_info = AnomalyDetector.detect_flat_open_breakout(df=eval_df, prev_close=prev_close)

        # 4. Institutional 15-Strategy Multi-Layer Engine
        inst_res = InstitutionalStrategyEngine.evaluate_strategies(
            df=eval_df,
            prev_close=prev_close,
            sector_bias=sector_bias,
            catalyst_info=news_cat,
            anomaly_info=anomaly_info,
            daily_df=daily_df,
            market_regime_info=nifty_regime
        )
        if not inst_res:
            return None

        # 5. Extract structural risk boundaries from institutional analysis
        opp_eval = {
            "is_viable": inst_res.get("is_viable", False),
            "potential_pct": inst_res.get("potential_pct", 0.0),
            "risk_reward_ratio": inst_res.get("risk_reward_ratio", 0.0),
            "stop_loss": inst_res.get("suggested_sl"),
            "suggested_sl": inst_res.get("suggested_sl"),
            "tp1": inst_res.get("suggested_tp1"),
            "suggested_tp1": inst_res.get("suggested_tp1"),
            "tp2": inst_res.get("suggested_tp2"),
            "suggested_tp2": inst_res.get("suggested_tp2"),
            "exhaustion_risk": inst_res.get("exhaustion_risk", "LOW")
        }

        # 6. Opportunity Ranking & Predictive Stage Engine
        opp_res = OpportunityRankingEngine.evaluate_stock_opportunity(
            stock_df=eval_df,
            symbol=symbol,
            sector_name=sector_name,
            sector_change=sector_change,
            nifty_df=eval_nifty,
            vwap=inst_res.get("vwap"),
            opp_eval=opp_eval,
            time_slot_rvol=time_slot_rvol,
            rvol_obs=rvol_obs,
            is_liquid=True
        )

        return {
            "inst_res": inst_res,
            "opp_res": opp_res,
            "is_completed_candle": is_completed,
            "eval_df": eval_df,
            "opp_eval": opp_eval
        }
