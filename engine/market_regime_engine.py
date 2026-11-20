import time
import pandas as pd
import numpy as np

class MarketRegimeEngine:
    """
    Market-Wide Momentum & Benchmark Alignment Engine:
    1. Tracks NIFTY 50 and BANKNIFTY live momentum, VWAP, and breakout impulses.
    2. Measures sector surge/dump intensity.
    3. Identifies F&O stock responses (Beta alignment, Relative Strength surge).
    """

    @staticmethod
    def evaluate_benchmark_regime(nifty_df, banknifty_df=None):
        """
        Evaluates current macro market regime:
        - STRONG_BULLISH: NIFTY > VWAP, EMA9 > EMA21, positive 1D change
        - STRONG_BEARISH: NIFTY < VWAP, EMA9 < EMA21, negative 1D change
        - RANGEBOUND / ROTATION: Choppy or conflicting signals
        """
        if nifty_df is None or nifty_df.empty or len(nifty_df) < 5:
            return {"regime": "NEUTRAL", "nifty_momentum": 0.0, "bias": "NEUTRAL", "alignment_score": 50}

        try:
            c = nifty_df['Close']
            h = nifty_df['High']
            l = nifty_df['Low']
            curr_c = float(c.iloc[-1])
            curr_o = float(nifty_df['Open'].iloc[-1])
            
            # Intraday VWAP for NIFTY
            tp = (h + l + c) / 3
            v = nifty_df['Volume'].replace(0, 1) if 'Volume' in nifty_df.columns else pd.Series(1, index=nifty_df.index)
            cum_vol = v.cumsum()
            vwap = float(((tp * v).cumsum() / cum_vol).iloc[-1])

            ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])

            # Session 1D change
            session_open = float(nifty_df['Open'].iloc[0])
            change_pct = ((curr_c - session_open) / session_open) * 100

            # Momentum impulse in last 3 candles
            candle_3_return = ((curr_c - float(c.iloc[-4])) / float(c.iloc[-4])) * 100 if len(c) >= 4 else 0.0

            is_bull = curr_c > vwap and ema9 >= ema21 and candle_3_return >= 0.05
            is_bear = curr_c < vwap and ema9 <= ema21 and candle_3_return <= -0.05

            if is_bull and change_pct > 0.2:
                regime = "STRONG_BULLISH"
                bias = "BULLISH"
                score = 85
            elif is_bull:
                regime = "MILD_BULLISH"
                bias = "BULLISH"
                score = 70
            elif is_bear and change_pct < -0.2:
                regime = "STRONG_BEARISH"
                bias = "BEARISH"
                score = 85
            elif is_bear:
                regime = "MILD_BEARISH"
                bias = "BEARISH"
                score = 70
            else:
                regime = "RANGEBOUND_CHOP"
                bias = "NEUTRAL"
                score = 50

            return {
                "regime": regime,
                "bias": bias,
                "change_pct": round(change_pct, 2),
                "last_3_candles_return_pct": round(candle_3_return, 2),
                "nifty_vwap": round(vwap, 2),
                "alignment_score": score
            }
        except Exception as e:
            return {"regime": "NEUTRAL", "bias": "NEUTRAL", "error": str(e), "alignment_score": 50}

    @staticmethod
    def calculate_stock_response_to_market(stock_change, stock_vwap_diff, market_regime_info, sector_bias="NEUTRAL"):
        """
        Scores how well an individual F&O stock is responding to broad market and sector tailwinds:
        - Returns positive boost if moving with strong NIFTY + Sector confluence.
        - Penalizes fighting against a strong trending macro regime.
        """
        regime = market_regime_info.get("regime", "NEUTRAL")
        mkt_bias = market_regime_info.get("bias", "NEUTRAL")

        confluence_boost = 0
        if mkt_bias == "BULLISH" and sector_bias == "BULLISH" and stock_change > 0 and stock_vwap_diff > 0:
            confluence_boost = 15  # Triple confluence: Market + Sector + Stock
        elif mkt_bias == "BEARISH" and sector_bias == "BEARISH" and stock_change < 0 and stock_vwap_diff < 0:
            confluence_boost = 15  # Triple confluence downside
        elif mkt_bias == "BULLISH" and stock_change < -0.5:
            confluence_boost = -10 # Fighting strong market gainer
        elif mkt_bias == "BEARISH" and stock_change > 0.5:
            confluence_boost = -10 # Fighting strong market breakdown

        return {
            "confluence_boost": confluence_boost,
            "is_aligned": confluence_boost > 0,
            "macro_regime": regime
        }
