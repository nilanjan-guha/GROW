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
        self.data_provider = data_provider or GrowwDataProvider()
        self.is_running = False
        self.last_track_cycle = 0
        self.last_outcome_check = 0
        self._last_evaluated_candles = {} # {symbol: last_completed_candle_ts} to prevent duplicate evaluations (P1)
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
                                    # Anti-Chasing & Viability Safeguard (P1):
                                    # Use dynamic volatility/ATR-based threshold (0.4 * 5m ATR or 0.25% - 1.25% adaptive envelope)
                                    # instead of a static arbitrary percentage.
                                    raw_atr = None
                                    if eval_completed_df is not None and len(eval_completed_df) >= 14:
                                        try:
                                            high_s = eval_completed_df["high"] if "high" in eval_completed_df.columns else eval_completed_df["High"]
                                            low_s = eval_completed_df["low"] if "low" in eval_completed_df.columns else eval_completed_df["Low"]
                                            close_s = eval_completed_df["close"] if "close" in eval_completed_df.columns else eval_completed_df["Close"]
                                            tr1 = high_s - low_s
                                            tr2 = (high_s - close_s.shift()).abs()
                                            tr3 = (low_s - close_s.shift()).abs()
                                            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                                            atr14 = float(tr.rolling(14).mean().iloc[-1])
                                            if not math.isnan(atr14) and atr14 > 0 and lvl > 0:
                                                raw_atr = (atr14 / lvl) * 100.0
                                        except Exception:
                                            raw_atr = None
                                    
                                    # Adaptive slippage tolerance: 0.5x of 5m ATR%, bounded between [0.25%, 1.20%]
                                    if raw_atr is not None:
                                        max_slippage_dist_pct = min(1.20, max(0.25, round(0.50 * raw_atr, 2)))
                                    else:
                                        max_slippage_dist_pct = 0.50

                                    is_missed_or_extended = False
                                    if "Enter above" in entry_trig:
                                        # Bullish breakout: reject if price fell back below level or stretched too far above
                                        if live_forming_price < lvl or ((live_forming_price - lvl) / lvl * 100.0) > max_slippage_dist_pct:
                                            is_missed_or_extended = True
                                    elif "Enter below" in entry_trig:
                                        # Bearish breakdown: reject if price popped back above level or dumped too far below
                                        if live_forming_price > lvl or ((lvl - live_forming_price) / lvl * 100.0) > max_slippage_dist_pct:
                                            is_missed_or_extended = True

                                    # Evaluate expansion room strictly on completed candle dataframe and verified live level
                                    direct_eval = OpportunityPotentialEngine.evaluate_potential(
                                        df=eval_completed_df,
                                        current_price=live_forming_price,
                                        direction="BUY" if direction == "BULLISH" else "SELL"
                                    )
                                    if not is_missed_or_extended and direct_eval.get("is_viable", True):
                                        promoted_any = True
                                        self.tracked_stats["promotions_today"] += 1
                                        print(f"[PRE-BREAKOUT TRIGGERED] {sym} confirmed 5m close @ ₹{completed_c} (Live: ₹{live_forming_price})! Level ₹{lvl} valid, actionable trade entry.")
                                    elif is_missed_or_extended:
                                        print(f"[PRE-BREAKOUT CHASE PREVENTED] {sym} crossed level ₹{lvl} but live price ₹{live_forming_price} is overextended (Tolerance: {max_slippage_dist_pct}%). Scan rejected stale chase.")
                        except Exception:
                            pass
                        time.sleep(0.08)

                # 2. Evaluate open outcomes and trigger adaptive learning post-mortem
                # Safeguard (P0 Fix):
                # - Deduplicate symbols using normalized ticker key (clean symbol)
                # - Only advance/mark self._last_evaluated_candles AFTER evaluate_open_outcomes succeeds!
                # If evaluation fails/throws an exception, the candles are retried next cycle rather than permanently skipped.
                evaluable_fresh_dfs = {}
                pending_eval_keys = {}
                for s_key, c_df in fresh_dfs.items():
                    norm_key = s_key.replace(".NS", "").upper()
                    c_ts = str(c_df.index[-1])
                    if self._last_evaluated_candles.get(norm_key) != c_ts:
                        evaluable_fresh_dfs[norm_key] = c_df
                        evaluable_fresh_dfs[f"{norm_key}.NS"] = c_df
                        pending_eval_keys[norm_key] = c_ts

                if (now - self.last_outcome_check > 15) and evaluable_fresh_dfs:
                    try:
                        evaluate_open_outcomes(evaluable_fresh_dfs)
                        # Mark evaluated ONLY after successful execution without exception
                        for n_k, n_ts in pending_eval_keys.items():
                            self._last_evaluated_candles[n_k] = n_ts
                        self.last_outcome_check = now
                    except Exception as oe:
                        print("[CONTINUOUS TRACKER] Outcome check error:", oe)

                # 3. If a breakout occurred in a tracked PREPARING stock, recompute rankings instantly
                if promoted_any:
                    # Invalidate screener cache immediately so frontend gets fresh ranked breakouts on next poll
                    self.screener_state["ranked_opportunities_ts"] = 0
                    if self.compute_ranking_fn:
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
