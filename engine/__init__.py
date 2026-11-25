"""
Modular Engine Package Architecture:
Provides clean, structured sub-modules while preserving seamless root-level imports:
- engine.tracking (Continuous Background Tracker & Diagnostics)
- engine.learning (Adaptive Trade Learning & Post-Mortem Feedback)
- engine.ranking (Opportunity Ranking & Potential Engines)
- engine.market (Market Validator, Regime, & FNO Filters)
- engine.execution (Cost & Slippage Calculation)
"""

from engine.adaptive_trade_learner import AdaptiveTradeLearner
from engine.background_continuous_tracker import BackgroundContinuousTracker
from engine.opportunity_ranking_engine import OpportunityRankingEngine
from engine.opportunity_potential_engine import OpportunityPotentialEngine
from engine.fno_momentum_filter import FNOMomentumFilter
from engine.market_validator import MarketValidator
from engine.cost_engine import IndianMarketCosts
from engine.groww_sector_engine import GrowwSectorEngine

__all__ = [
    "AdaptiveTradeLearner",
    "BackgroundContinuousTracker",
    "OpportunityRankingEngine",
    "OpportunityPotentialEngine",
    "FNOMomentumFilter",
    "MarketValidator",
    "IndianMarketCosts",
    "GrowwSectorEngine"
]
