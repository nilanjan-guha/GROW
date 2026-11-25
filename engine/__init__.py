"""
Modular Engine Architecture:
Exposes all components cleanly from both domain subpackages AND root engine namespace
for maximum backward compatibility and high-speed execution:

Domain Folders:
- engine.tracking      (Continuous Background Tracker & Real-Time Setup Watcher)
- engine.learning      (Adaptive Trade Learner & Post-Mortem Feedback)
- engine.ranking       (Opportunity Ranking, Potential Engine, Ignition, Pre-Momentum)
- engine.market        (FNO Momentum Filter, Sector Engine, Market Validator, Regime, NSE Feed)
- engine.execution     (Transaction & Slippage Cost Engine)
- engine.intelligence  (Anomaly Detector, News Catalyst, Screener Pipeline, Missed Moves)
"""

# 1. Tracking
from engine.tracking.background_continuous_tracker import BackgroundContinuousTracker

# 2. Learning
from engine.learning.adaptive_trade_learner import AdaptiveTradeLearner
from engine.learning.strategy_learning_engine import StrategyLearningEngine

# 3. Ranking
from engine.ranking.opportunity_ranking_engine import OpportunityRankingEngine
from engine.ranking.opportunity_potential_engine import OpportunityPotentialEngine
from engine.ranking.momentum_ignition_engine import MomentumIgnitionEngine
from engine.ranking.pre_momentum_engine import PreMomentumEngine

# 4. Market
from engine.market.fno_momentum_filter import FNOMomentumFilter
from engine.market.groww_sector_engine import GrowwSectorEngine
from engine.market.market_validator import MarketValidator
from engine.market.market_regime_engine import MarketRegimeEngine
from engine.market.nse_data_engine import NSEDataEngine

# 5. Execution
from engine.execution.cost_engine import IndianMarketCosts

# 6. Intelligence
from engine.intelligence.anomaly_engine import AnomalyDetector
from engine.intelligence.news_engine import NewsIntelligenceEngine
from engine.intelligence.screener_pipeline import ScreenerPipeline
from engine.intelligence.missed_move_detector import MissedMoveDetector

__all__ = [
    # Tracking
    "BackgroundContinuousTracker",
    # Learning
    "AdaptiveTradeLearner",
    "StrategyLearningEngine",
    # Ranking
    "OpportunityRankingEngine",
    "OpportunityPotentialEngine",
    "MomentumIgnitionEngine",
    "PreMomentumEngine",
    # Market
    "FNOMomentumFilter",
    "GrowwSectorEngine",
    "MarketValidator",
    "MarketRegimeEngine",
    "NSEDataEngine",
    # Execution
    "IndianMarketCosts",
    # Intelligence
    "AnomalyDetector",
    "NewsIntelligenceEngine",
    "ScreenerPipeline",
    "MissedMoveDetector"
]
