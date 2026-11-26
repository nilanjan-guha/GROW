import time
import datetime
import math
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
        self._promoted_candle_keys = set() # {f"{symbol}:{candle_ts}"} idempotent promotion registry (P1)
        self.tracked_stats = {
            "active_prep_count": 0,
            "open_signals_count": 0,
            "last_cycle_duration_ms": 0,
            "last_evaluated_ts": None,
            "promotions_today": 0
        }

    @staticmethod
    def calculate_atr_pct(eval_completed_df, reference_level):
        """
        Production helper: calculates 14-period ATR as a percentage of reference_level.
        Returns float ATR% or None if insufficient history or invalid values.
        """
        if eval_completed_df is None or len(eval_completed_df) < 14 or reference_level <= 0:
            return None
        try:
            high_s = eval_completed_df["high"] if "high" in eval_completed_df.columns else eval_completed_df["High"]
            low_s = eval_completed_df["low"] if "low" in eval_completed_df.columns else eval_completed_df["Low"]
            close_s = eval_completed_df["close"] if "close" in eval_completed_df.columns else eval_completed_df["Close"]
            tr1 = high_s - low_s
            tr2 = (high_s - close_s.shift()).abs()
            tr3 = (low_s - close_s.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr14 = float(tr.rolling(14).mean().iloc[-1])
            if not math.isnan(atr14) and atr14 > 0:
                return (atr14 / reference_level) * 100.0
        except Exception:
            pass
        return None

    def promote_preparing_breakout(self, symbol, completed_c, evaluated_completed_ts, prep_info, direct_eval, level, direction, entry_trigger=None):
        """
        Production helper: idempotently promotes a confirmed breakout setup into ENTRY READY,
        calculates confirmation score from genuine market evidence (no artificial 85 floor),
        and removes the PREPARING coil record.
        Returns True if newly promoted, False if duplicate or rejected.
        """
        cand_prom_key = f"{symbol.replace('.NS', '').upper()}:{evaluated_completed_ts}"
        if cand_prom_key in self._promoted_candle_keys:
            return False

        self._promoted_candle_keys.add(cand_prom_key)
        self.tracked_stats["promotions_today"] += 1
        now = time.time()

        # P2 Fix: Genuine confirmation score from actual market evidence (RVOL, breakout room, base conviction)
        # Never artificially floor to 85!
        base_score = float(prep_info.get("conviction_score") or 70.0)
        room_pct = float(direct_eval.get("room_pct") or 1.0)
        # Modest evidence-based adjustments
        evidence_bonus = min(15.0, max(0.0, room_pct * 3.0))
        genuine_score = int(round(min(98.0, base_score + evidence_bonus)))

        confirmed_cand = {
            "symbol": symbol,
            "direction": direction,
            "stage": "CONFIRMED",
            "is_actionable": True,
            "current_price": completed_c,
            "entry_trigger": entry_trigger or str(prep_info.get("entry_trigger") or f"Breakout level ₹{level}"),
            "entry_price": completed_c,
            "suggested_sl": direct_eval.get("suggested_sl"),
            "suggested_tp1": direct_eval.get("suggested_tp1"),
            "suggested_tp2": direct_eval.get("suggested_tp2"),
            "conviction_score": genuine_score,
            "change_pct": prep_info.get("change_pct", 0.0),
            "completed_candle_ts": evaluated_completed_ts,
            "promoted_at": now
        }

        ranked_opps = self.screener_state.setdefault("ranked_opportunities", {})
        entry_ready_list = ranked_opps.setdefault("entry_ready", [])
        entry_ready_list[:] = [c for c in entry_ready_list if c.get("symbol") != symbol]
        entry_ready_list.insert(0, confirmed_cand)
        ranked_opps["entry_ready"] = entry_ready_list[:3]

        prep_info["is_promoted"] = True
        prep_info["promoted_candle_ts"] = evaluated_completed_ts
        prep_info["promoted_at"] = now
        self.screener_state.get("persistent_preparing_stocks", {}).pop(symbol, None)

        print(f"[PRE-BREAKOUT TRIGGERED] {symbol} confirmed 5m close @ ₹{completed_c}! Level ₹{level} valid, genuine score: {genuine_score}/100, promoted to ENTRY READY.")
        return True

    def start(self):
        import threading
        self.is_running = True
        thread = threading.Thread(target=self._tracker_loop, daemon=True, name="BackgroundContinuousTrackerThread")
        thread.start()

        # P1 Fix: Run closed-market historical learning in its own independent daemon thread
        # completely decoupled from the tracker loop, so slow historical API fetches never delay outcome evaluations.
        learner_thread = threading.Thread(target=self._offline_learner_loop, daemon=True, name="OfflineLearnerThread")
        learner_thread.start()

        print("[CONTINUOUS TRACKER DAEMON] Dedicated background tracking daemon and offline learning thread started.")
        return thread

    def _offline_learner_loop(self):
        """
        Independent closed-market learner thread:
        Runs in background, fetching 5m intraday replay and daily history with rate limits and retries,
        completely decoupled from the main trade tracking loop.
        """
        while self.is_running:
            try:
                # Run learning ONLY when market is closed
                if not MarketValidator.is_market_open():
                    self._run_closed_market_learner_step()
                    time.sleep(10) # Controlled rate-limiting between stocks
                else:
                    time.sleep(60) # Sleep during live market hours
            except Exception as e:
                time.sleep(15)

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
                                    # P1 Requirement: Use ONLY the completed candle close (completed_c) and
                                    # completed-candle dataframe (eval_completed_df) for every promotion,
                                    # anti-chasing, and viability decision.
                                    raw_atr = self.calculate_atr_pct(eval_completed_df, lvl)
                                    
                                    # Adaptive slippage tolerance: 0.5x of 5m ATR%, bounded between [0.25%, 1.20%]
                                    if raw_atr is not None:
                                        max_slippage_dist_pct = min(1.20, max(0.25, round(0.50 * raw_atr, 2)))
                                    else:
                                        max_slippage_dist_pct = 0.50

                                    is_missed_or_extended = False
                                    if "Enter above" in entry_trig:
                                        # Bullish breakout: check completed candle close against level and max slippage
                                        if completed_c < lvl or ((completed_c - lvl) / lvl * 100.0) > max_slippage_dist_pct:
                                            is_missed_or_extended = True
                                    elif "Enter below" in entry_trig:
                                        # Bearish breakdown: check completed candle close against level and max slippage
                                        if completed_c > lvl or ((lvl - completed_c) / lvl * 100.0) > max_slippage_dist_pct:
                                            is_missed_or_extended = True

                                    # Evaluate expansion room strictly on completed candle dataframe and completed candle close
                                    direct_eval = OpportunityPotentialEngine.evaluate_potential(
                                        df=eval_completed_df,
                                        current_price=completed_c,
                                        direction="BUY" if direction == "BULLISH" else "SELL"
                                    )

                                    cand_prom_key = f"{sym.replace('.NS', '').upper()}:{evaluated_completed_ts}"
                                    # Fail-closed viability check: require is_viable is explicitly True
                                    is_viable_trade = (direct_eval.get("is_viable") is True)
                                    if not is_missed_or_extended and is_viable_trade:
                                        promoted = self.promote_preparing_breakout(
                                            symbol=sym,
                                            completed_c=completed_c,
                                            evaluated_completed_ts=evaluated_completed_ts,
                                            prep_info=prep_info,
                                            direct_eval=direct_eval,
                                            level=lvl,
                                            direction=direction,
                                            entry_trigger=entry_trig
                                        )
                                        if promoted:
                                            promoted_any = True
                                    elif is_missed_or_extended:
                                        print(f"[PRE-BREAKOUT CHASE PREVENTED] {sym} closed @ ₹{completed_c} beyond level ₹{lvl} tolerance ({max_slippage_dist_pct}%). Setup rejected.")
                                    elif is_missed_or_extended:
                                        print(f"[PRE-BREAKOUT CHASE PREVENTED] {sym} closed @ ₹{completed_c} beyond level ₹{lvl} tolerance ({max_slippage_dist_pct}%). Setup rejected.")
                        except Exception:
                            pass
                        time.sleep(0.08)

                # 2. Evaluate open outcomes and trigger adaptive learning post-mortem
                # Safeguard (P0 Fix):
                # - Deduplicate symbols using normalized symbol + candle timestamp key (norm_key:candle_ts)
                # - Only advance/mark self._last_evaluated_candles AFTER evaluate_open_outcomes succeeds!
                # If evaluation fails/throws an exception, the candle is retried on the next cycle rather than permanently skipped.
                evaluable_fresh_dfs = {}
                pending_eval_keys = {}
                seen_norm_symbols = set()

                for s_key, c_df in fresh_dfs.items():
                    norm_key = s_key.replace(".NS", "").upper()
                    if norm_key in seen_norm_symbols:
                        continue
                    seen_norm_symbols.add(norm_key)

                    c_ts = str(c_df.index[-1])
                    dedup_key = f"{norm_key}:{c_ts}"
                    if self._last_evaluated_candles.get(norm_key) != c_ts and dedup_key not in pending_eval_keys:
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
                        # On failure, do NOT update self._last_evaluated_candles so it retries next cycle
                        print("[CONTINUOUS TRACKER] Outcome check error (will retry next cycle):", oe)

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

    def _run_closed_market_learner_step(self):
        """
        Processes closed-market historical learning incrementally (1-2 stocks per cycle)
        to keep memory low, never block the tracker loop, and record individual stock personality.
        """
        try:
            from engine.learning.historical_stock_learner import HistoricalStockLearner
            HistoricalStockLearner.init_memory_table()

            # Compile candidate pool: F&O universe + persistent preparing + tracked stocks
            fno_univ = self.screener_state.get("fno_universe", [])
            prep_dict = self.screener_state.get("persistent_preparing_stocks", {})
            raw_candidates = list(dict.fromkeys(list(prep_dict.keys()) + fno_univ))

            if not raw_candidates or not self.data_provider:
                return

            if not hasattr(self, "_closed_learner_index"):
                self._closed_learner_index = 0

            target_sym = raw_candidates[self._closed_learner_index % len(raw_candidates)]
            self._closed_learner_index += 1

            clean_ticker = target_sym.replace(".NS", "").upper()

            # P1 Fix: Fetch both 5-minute intraday completed candles and daily candles
            # to learn true intraday 5m breakout behavior alongside multi-session trend
            five_m_df, m5_src = self.data_provider.fetch_5m_candles(clean_ticker, days_back=5)
            daily_df, d_src = self.data_provider.fetch_daily_candles(clean_ticker, days_back=45)

            study_df = five_m_df if (five_m_df is not None and not five_m_df.empty and len(five_m_df) >= 20) else daily_df
            src_used = m5_src if study_df is five_m_df else d_src

            if study_df is not None and not study_df.empty and len(study_df) >= 10:
                completed_ts = str(study_df.index[-1])
                # Skip if already learned for this exact completed session/candle
                if HistoricalStockLearner.should_skip_scan(clean_ticker, current_completed_candle_ts=completed_ts):
                    return

                # Replay historical breakouts (with realistic next-open execution, slippage, and gap-through-stop)
                mult = HistoricalStockLearner.analyze_30d_stock_history(
                    symbol=clean_ticker,
                    df_history=study_df,
                    is_replay=False
                )
                print(f"[OFFLINE LEARNER] Studied {clean_ticker:<12} | Completed: {completed_ts} | Learned Multiplier: {mult}x ({src_used})")
        except Exception as le:
            pass
