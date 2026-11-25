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

import sys

# 1. Tracking
from engine.tracking.background_continuous_tracker import BackgroundContinuousTracker

# 2. Learning
from engine.learning.adaptive_trade_learner import AdaptiveTradeLearner
from engine.learning.strategy_learning_engine import StrategyLearningEngine
from engine.learning.historical_stock_learner import HistoricalStockLearner

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

# Dynamically map sub-modules to root sys.modules so existing imports
# like `import engine.market_validator` resolve seamlessly without redundant files!
_SUBMODULE_MAP = {
    "engine.background_continuous_tracker": "engine.tracking.background_continuous_tracker",
    "engine.adaptive_trade_learner": "engine.learning.adaptive_trade_learner",
    "engine.strategy_learning_engine": "engine.learning.strategy_learning_engine",
    "engine.opportunity_ranking_engine": "engine.ranking.opportunity_ranking_engine",
    "engine.opportunity_potential_engine": "engine.ranking.opportunity_potential_engine",
    "engine.momentum_ignition_engine": "engine.ranking.momentum_ignition_engine",
    "engine.pre_momentum_engine": "engine.ranking.pre_momentum_engine",
    "engine.fno_momentum_filter": "engine.market.fno_momentum_filter",
    "engine.groww_sector_engine": "engine.market.groww_sector_engine",
    "engine.market_validator": "engine.market.market_validator",
    "engine.market_regime_engine": "engine.market.market_regime_engine",
    "engine.nse_data_engine": "engine.market.nse_data_engine",
    "engine.cost_engine": "engine.execution.cost_engine",
    "engine.anomaly_engine": "engine.intelligence.anomaly_engine",
    "engine.news_engine": "engine.intelligence.news_engine",
    "engine.screener_pipeline": "engine.intelligence.screener_pipeline",
    "engine.missed_move_detector": "engine.intelligence.missed_move_detector",
    "engine.groq_client": "engine.intelligence.groq_client",
}

for legacy_name, target_name in _SUBMODULE_MAP.items():
    if legacy_name not in sys.modules:
        try:
            __import__(target_name)
            sys.modules[legacy_name] = sys.modules[target_name]
        except Exception:
            pass

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
