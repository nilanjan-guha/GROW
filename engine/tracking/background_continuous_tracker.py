import time
import datetime
import pytz
import traceback
import pandas as pd
import yfinance as yf
from core.storage import evaluate_open_outcomes, get_active_signal_symbols, get_db_connection
from engine.market.market_validator import MarketValidator
from engine.ranking.opportunity_potential_engine import OpportunityPotentialEngine
from engine.learning.adaptive_trade_learner import AdaptiveTradeLearner
from core.groww_data_provider import GrowwDataProvider

# Shared Data Provider Instance
groww_data_provider = GrowwDataProvider()

class BackgroundContinuousTracker:
    """
    Dedicated Independent Background Tracking & Learning Daemon:
    - Runs in its own separate thread, completely decoupled from the 5-minute Universe Screener.
    - Focuses strictly on:
      1. Polling and updating actively coiling PREPARING stocks every 15-30 seconds.
      2. Tracking active OPEN signals, evaluating MFE/MAE and recording TP/SL executions.
      3. Running automated post-mortem root-cause analysis on closed trades.
      4. Promoting triggered PREPARING coils into ENTRY READY without waiting for the full 219 F&O scan.
    """

    def __init__(self, screener_state_ref, compute_ranking_fn=None, data_provider=None):
        self.screener_state = screener_state_ref
        self.compute_ranking_fn = compute_ranking_fn
        self.data_provider = data_provider or groww_data_provider
        self.is_running = False
        self.last_track_cycle = 0
        self.last_outcome_check = 0
        self.tracked_stats = {
            "active_prep_count": 0,
            "open_signals_count": 0,
            "last_cycle_duration_ms": 0,
            "last_evaluated_ts": None,
            "promotions_today": 0
        }

    def start(self):
        import threading
        self.is_running = True
        thread = threading.Thread(target=self._tracker_loop, daemon=True, name="BackgroundContinuousTrackerThread")
        thread.start()
        print("[CONTINUOUS TRACKER DAEMON] Dedicated background tracking & learning daemon started.")
        return thread

    def _tracker_loop(self):
        ist = pytz.timezone('Asia/Kolkata')
        while self.is_running:
            try:
                now = time.time()
                t0 = time.perf_counter()
                today_str = datetime.datetime.now(ist).strftime('%Y-%m-%d')

                # 1. Clean up obsolete day's preparing tracking
                prep_dict = self.screener_state.setdefault("persistent_preparing_stocks", {})
                for p_sym in list(prep_dict.keys()):
                    if prep_dict[p_sym].get("first_detected_date") != today_str:
                        del prep_dict[p_sym]

                active_open_symbols = list(get_active_signal_symbols())
                preparing_symbols = list(prep_dict.keys())
                self.tracked_stats["active_prep_count"] = len(preparing_symbols)
                self.tracked_stats["open_signals_count"] = len(active_open_symbols)

                # Prioritized watchlist: Active open trades + PREPARING coils
                focus_symbols = list(dict.fromkeys(active_open_symbols + preparing_symbols))

                fresh_dfs = {}
                promoted_any = False

                if focus_symbols:
                    # Download compact 5-minute data rapidly for focused symbols
                    for sym in focus_symbols:
                        try:
                            clean_s = sym if sym.endswith(".NS") else f"{sym}.NS"
                            clean_ticker = sym.replace(".NS", "").upper()

                            # Use unified GrowwDataProvider for consistent 5m candle retrieval
                            df, data_src = self.data_provider.fetch_5m_candles(clean_ticker, days_back=2)
                            if df is None or df.empty or len(df) < 5:
                                continue

                            # Establish completed candle validity before any tracker evaluation or promotion (P0)
                            feed_val = MarketValidator.validate_feed_freshness(df, max_delay_seconds=300, candle_duration_seconds=300)
                            if not feed_val.get("is_valid", False):
                                # Reject invalid feed immediately! (P0)
                                continue

                            is_completed = feed_val.get("is_completed_candle", False)
                            eval_completed_df = df if is_completed else df.iloc[:-1]

                            if eval_completed_df.empty or len(eval_completed_df) < 5:
                                continue

                            evaluated_completed_ts = str(eval_completed_df.index[-1])
                            live_forming_price = float(df['Close'].squeeze().iloc[-1])
                            completed_c = float(eval_completed_df['Close'].squeeze().iloc[-1])

                            # Pass ONLY completed candles into trade outcome evaluation (P0)
                            fresh_dfs[sym] = eval_completed_df
                            fresh_dfs[clean_s] = eval_completed_df

                            # Update shared raw market cache with strictly validated completed candle timestamp
                            self.screener_state.setdefault("raw_market_cache", {})[clean_s] = {
                                "timestamp": now,
                                "raw_latest_candle_ts": str(df.index[-1]),
                                "evaluated_completed_candle_ts": evaluated_completed_ts,
                                "live_forming_price": live_forming_price,
                                "source": data_src,
                                "df": df
                            }

                            # Inspect if this PREPARING stock has broken out on a completed 5-minute candle
                            if sym in prep_dict:
                                prep_info = prep_dict[sym]
                                entry_trig = str(prep_info.get("entry_trigger") or "")
                                direction = prep_info.get("direction", "BULLISH")
                                is_breakout = False

                                # Mandatory P0 Rule: Signal promotion MUST evaluate completed candle close, not intrabar ticks!
                                if "Enter above" in entry_trig:
                                    try:
                                        lvl = float(entry_trig.split("₹")[-1].strip())
                                        if completed_c >= lvl:
                                            is_breakout = True
                                    except Exception:
                                        pass
                                elif "Enter below" in entry_trig:
                                    try:
                                        lvl = float(entry_trig.split("₹")[-1].strip())
                                        if completed_c <= lvl:
                                            is_breakout = True
                                    except Exception:
                                        pass

                                if is_breakout:
                                    # Evaluate expansion room strictly on completed candle dataframe
                                    direct_eval = OpportunityPotentialEngine.evaluate_potential(
                                        df=eval_completed_df,
                                        current_price=completed_c,
                                        direction="BUY" if direction == "BULLISH" else "SELL"
                                    )
                                    if direct_eval.get("is_viable", True):
                                        promoted_any = True
                                        self.tracked_stats["promotions_today"] += 1
                                        print(f"[PRE-BREAKOUT TRIGGERED] {sym} confirmed 5m close above/below level @ ₹{completed_c} ({evaluated_completed_ts})!")
                        except Exception:
                            pass
                        time.sleep(0.08)

                # 2. Evaluate open outcomes and trigger adaptive learning post-mortem
                if (now - self.last_outcome_check > 30) and fresh_dfs:
                    try:
                        evaluate_open_outcomes(fresh_dfs)
                        self.last_outcome_check = now
                    except Exception as oe:
                        print("[CONTINUOUS TRACKER] Outcome check error:", oe)

                # 3. If a breakout occurred in a tracked PREPARING stock, recompute rankings instantly
                if promoted_any and self.compute_ranking_fn:
                    try:
                        self.compute_ranking_fn()
                    except Exception:
                        pass

                t1 = time.perf_counter()
                self.tracked_stats["last_cycle_duration_ms"] = round((t1 - t0) * 1000.0, 1)
                self.tracked_stats["last_evaluated_ts"] = datetime.datetime.now(ist).strftime("%H:%M:%S")

                # Track cycle interval: sleep 15 seconds so tracking is continuous and doesn't load CPU
                time.sleep(15)

            except Exception as e:
                traceback.print_exc()
                print("[CONTINUOUS TRACKER EXCEPTION]:", e)
                time.sleep(10)
