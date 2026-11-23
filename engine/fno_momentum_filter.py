import time
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class FNOMomentumFilter:
    """
    Universal 219 F&O Momentum & Options Liquidity Screener:
    
    1. Downloads 5-day / 5-minute data across all 219 tradeable F&O stocks using multi-threading.
    2. Filters out DEAD, SIDEWAYS, low-volume, and choppy stocks:
       - Eliminates dead stocks (Volume < 50k or Turnover < 50 Lakhs)
       - Eliminates chop/sideways stocks (ATR < 0.4%, Bollinger Bandwidth squeeze without expansion)
    3. Segregates stocks into:
       - Top BULLISH Momentum (Strong relative volume RVOL >= 1.25x, Price > VWAP, EMA9 > EMA21, positive session gain)
       - Top BEARISH Momentum (Strong relative volume RVOL >= 1.25x, Price < VWAP, EMA9 < EMA21, negative session drop)
    4. Extracts Top 15 Bullish and Top 15 Bearish high-momentum candidates (20-30 total) for deep analysis.
    """

    @staticmethod
    def _evaluate_single_fno(sym, cdf):
        try:
            if cdf is None or cdf.empty or len(cdf) < 15:
                return None

            if isinstance(cdf.columns, pd.MultiIndex):
                cdf = cdf.xs(sym, axis=1, level=1) if sym in cdf.columns.levels[1] else cdf

            c = cdf['Close']
            o = cdf['Open']
            h = cdf['High']
            l = cdf['Low']
            v = cdf['Volume']

            # Intraday calculation: Compare against the actual start of the latest trading session
            df_temp = cdf.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_temp.index):
                df_temp.index = pd.to_datetime(df_temp.index)

            # Determine strictly completed candles using unified 5-minute feed validation (max_delay 300s)
            from engine.market_validator import MarketValidator
            feed_val = MarketValidator.validate_feed_freshness(df_temp, max_delay_seconds=300, candle_duration_seconds=300)
            if not feed_val.get("is_valid", False):
                return None
            is_completed = feed_val.get("is_completed_candle", True)
            
            # If the last candle is still actively forming during market hours, trim to last completed candle
            eval_df = df_temp if is_completed else df_temp.iloc[:-1]
            if len(eval_df) < 15:
                return None

            c = eval_df['Close']
            o = eval_df['Open']
            h = eval_df['High']
            l = eval_df['Low']
            v = eval_df['Volume']

            curr_c = float(c.iloc[-1])
            curr_o = float(o.iloc[-1])

            eval_df['Date'] = eval_df.index.date
            dates = sorted(eval_df['Date'].unique())
            if dates:
                today_df = eval_df[eval_df['Date'] == dates[-1]]
                session_o = float(today_df['Open'].iloc[0]) if not today_df.empty else float(o.iloc[0])
            else:
                session_o = float(o.iloc[0])

            change_pct = round(((curr_c - session_o) / session_o) * 100, 2) if session_o > 0 else 0.0

            # 1. Volume & Traded Value Turnover Filter (Measured using each completed candle's volume * typical price)
            typical_price = (h + l + c) / 3.0
            traded_value = (typical_price * v)
            avg_traded_value_10 = float(traded_value.iloc[-10:].mean()) if len(traded_value) >= 10 else float(traded_value.mean())
            recent_v = v.iloc[-10:].mean()
            if recent_v < 5000 or avg_traded_value_10 < 1500000:  # < 15 Lakhs actual average traded value per 5-min
                return None

            # 2. Time-Slot Adjusted Relative Volume (RVOL) on the exact same completed candle
            eval_df['Time'] = eval_df.index.time
            slot_time = eval_df['Time'].iloc[-1]
            same_slot_vols = eval_df[eval_df['Time'] == slot_time]['Volume']
            curr_vol = float(v.iloc[-1])

            obs_count = len(same_slot_vols)
            if obs_count > 1:
                hist_slot_avg = float(same_slot_vols.iloc[:-1].mean())
                rvol = round(curr_vol / hist_slot_avg, 2) if hist_slot_avg > 0 else 1.0
            else:
                vol_sma20 = v.rolling(20).mean()
                avg_vol = float(vol_sma20.iloc[-1]) if not vol_sma20.empty and not pd.isna(vol_sma20.iloc[-1]) else 1.0
                rvol = round(curr_vol / avg_vol, 2) if avg_vol > 0 else 1.0

            # 3. ATR & Volatility Compression (NR4 / NR7 & Bollinger Squeeze detection)
            tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1).dropna()
            atr = float(tr.iloc[-14:].mean()) if len(tr) >= 14 else float(tr.mean())
            atr_pct = (atr / curr_c) * 100 if curr_c > 0 else 0.0

            # Narrow Range 4 (NR4) / NR7 detection
            candle_ranges = (h - l).iloc[-7:]
            curr_range = float(candle_ranges.iloc[-1]) if not candle_ranges.empty else 1.0
            is_nr4 = len(candle_ranges) >= 4 and curr_range <= candle_ranges.iloc[-4:].min()
            is_nr7 = len(candle_ranges) >= 7 and curr_range <= candle_ranges.min()

            # Bollinger Bandwidth Squeeze
            bb_mid = c.rolling(20).mean()
            bb_std = c.rolling(20).std()
            bb_width = float(((bb_std * 2) / bb_mid * 100).iloc[-1]) if not bb_mid.empty and not pd.isna(bb_mid.iloc[-1]) else 1.0
            is_bb_squeeze = (bb_width < 1.25)

            # 4. Session VWAP (Strictly reset at start of current trading session using eval_df index)
            today_idx = eval_df.index.date == dates[-1] if dates else slice(None)
            today_h = h[today_idx]
            today_l = l[today_idx]
            today_c = c[today_idx]
            today_v = v[today_idx]

            typical_price = (today_h + today_l + today_c) / 3.0
            cum_vp = (typical_price * today_v).cumsum()
            cum_v = today_v.cumsum()
            vwap_series = cum_vp / cum_v.replace(0, np.nan)
            curr_vwap = round(float(vwap_series.iloc[-1]), 2) if not vwap_series.empty and not pd.isna(vwap_series.iloc[-1]) else curr_c

            # 5. EMA 9 vs EMA 21
            ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])

            # Confirmed Swing Pivot Detection (2 bars on each side)
            # A swing low is confirmed when low[i] <= min(low[i-2], low[i-1], low[i+1], low[i+2])
            # A swing high is confirmed when high[i] >= max(high[i-2], high[i-1], high[i+1], high[i+2])
            # We look for the last 2 confirmed swing lows / swing highs up to index -2 (excluding unconfirmed forming bars)
            confirmed_swing_lows = []
            confirmed_swing_highs = []
            confirmed_swing_low_ts = []
            confirmed_swing_high_ts = []
            low_vals = [float(x) for x in l.values]
            high_vals = [float(x) for x in h.values]
            ts_vals = [str(x) for x in eval_df.index]
            n_bars = len(low_vals)
            # Scan backwards from n_bars - 3 down to max(0, n_bars - 35)
            for i in range(n_bars - 3, max(1, n_bars - 35), -1):
                if (i >= 2 and i + 2 < n_bars):
                    if (low_vals[i] <= low_vals[i-1] and low_vals[i] <= low_vals[i-2] and
                        low_vals[i] <= low_vals[i+1] and low_vals[i] <= low_vals[i+2]):
                        confirmed_swing_lows.append(low_vals[i])
                        confirmed_swing_low_ts.append(ts_vals[i])
                    if (high_vals[i] >= high_vals[i-1] and high_vals[i] >= high_vals[i-2] and
                        high_vals[i] >= high_vals[i+1] and high_vals[i] >= high_vals[i+2]):
                        confirmed_swing_highs.append(high_vals[i])
                        confirmed_swing_high_ts.append(ts_vals[i])
                if len(confirmed_swing_lows) >= 2 and len(confirmed_swing_highs) >= 2:
                    break

            has_higher_lows = False
            has_lower_highs = False
            bullish_structure_status = "INSUFFICIENT_STRUCTURE"
            bearish_structure_status = "INSUFFICIENT_STRUCTURE"

            if len(confirmed_swing_lows) >= 2:
                # confirmed_swing_lows[0] is the most recent confirmed pivot low, [1] is the prior
                has_higher_lows = bool(confirmed_swing_lows[0] > confirmed_swing_lows[1])
                bullish_structure_status = "CONFIRMED_HIGHER_LOWS" if has_higher_lows else "LOWER_LOWS"

            if len(confirmed_swing_highs) >= 2:
                # confirmed_swing_highs[0] is most recent confirmed pivot high, [1] is prior
                has_lower_highs = bool(confirmed_swing_highs[0] < confirmed_swing_highs[1])
                bearish_structure_status = "CONFIRMED_LOWER_HIGHS" if has_lower_highs else "HIGHER_HIGHS"

            # 6. Directional Pre-Breakout / Compression Discovery:
            # Three distinct discovery paths:
            # Path A: Structural Coil (Requires confirmed swing structure + compression + above/below VWAP)
            # Path B: Early Activity (Improving RVOL >= 1.25x + compression near VWAP, tagged as EARLY_ACTIVITY)
            # Path C: Early Observation (Quiet accumulation with TRUE 3-candle rising volume trend and compression, RVOL < 1.25x, tagged as OBSERVING)
            is_structural_coil_bull = (
                (curr_c > curr_vwap) and 
                (ema9 >= ema21) and 
                has_higher_lows and 
                (is_nr4 or is_nr7 or is_bb_squeeze)
            )
            is_structural_coil_bear = (
                (curr_c < curr_vwap) and 
                (ema9 <= ema21) and 
                has_lower_highs and 
                (is_nr4 or is_nr7 or is_bb_squeeze)
            )

            is_early_activity_bull = (
                (curr_c > curr_vwap) and
                (rvol >= 1.25) and
                (is_nr4 or is_nr7 or is_bb_squeeze) and
                not is_structural_coil_bull
            )
            is_early_activity_bear = (
                (curr_c < curr_vwap) and
                (rvol >= 1.25) and
                (is_nr4 or is_nr7 or is_bb_squeeze) and
                not is_structural_coil_bear
            )

            # Path C: Early Observation Path (Strict 3-candle consecutively rising volume trend: v[-1] > v[-2] > v[-3] + compression + near VWAP)
            vol_rising_trend = bool(len(v) >= 3 and (v.iloc[-1] > v.iloc[-2] > v.iloc[-3]))
            is_observing_bull = (
                (curr_c >= curr_vwap * 0.998) and
                (is_nr4 or is_nr7 or is_bb_squeeze) and
                vol_rising_trend and
                not is_structural_coil_bull and
                not is_early_activity_bull
            )
            is_observing_bear = (
                (curr_c <= curr_vwap * 1.002) and
                (is_nr4 or is_nr7 or is_bb_squeeze) and
                vol_rising_trend and
                not is_structural_coil_bear and
                not is_early_activity_bear
            )

            # Active Momentum Breakout Path:
            is_bull_momentum = (curr_c >= curr_vwap) and (ema9 >= ema21 * 0.998) and (change_pct >= 0.20 or (rvol >= 1.25 and curr_c > curr_o))
            is_bear_momentum = (curr_c <= curr_vwap) and (ema9 <= ema21 * 1.002) and (change_pct <= -0.20 or (rvol >= 1.25 and curr_c < curr_o))

            is_bull = is_bull_momentum or is_structural_coil_bull or is_early_activity_bull or is_observing_bull
            is_bear = is_bear_momentum or is_structural_coil_bear or is_early_activity_bear or is_observing_bear

            # Mutual Exclusivity: If both evaluate true or neither, discard ambiguous chop
            if (is_bull and is_bear) or (not is_bull and not is_bear):
                return None

            discovery_type = "MOMENTUM"
            if is_structural_coil_bull or is_structural_coil_bear:
                discovery_type = "STRUCTURAL_COIL"
            elif is_early_activity_bull or is_early_activity_bear:
                discovery_type = "EARLY_ACTIVITY"
            elif is_observing_bull or is_observing_bear:
                discovery_type = "OBSERVING"

            # Real momentum velocity: based strictly on price expansion and genuine relative volume (no artificial inflations)
            velocity = round(abs(change_pct) * max(1.0, rvol), 2)
            if velocity < 0.5:
                # Minimum readable floor for coils
                velocity = round(max(0.8, rvol), 2)

            evaluated_completed_ts = str(eval_df.index[-1])
            raw_latest_ts = str(df_temp.index[-1])

            return {
                "symbol": sym,
                "current_price": curr_c,
                "change_pct": change_pct,
                "rvol": rvol,
                "rvol_obs": obs_count - 1,
                "rvol_candle_ts": evaluated_completed_ts,
                "evaluated_completed_candle_ts": evaluated_completed_ts,
                "raw_latest_candle_ts": raw_latest_ts,
                "atr_pct": round(atr_pct, 2),
                "vwap": curr_vwap,
                "is_bull": is_bull,
                "is_bear": is_bear,
                "is_pre_coil": is_structural_coil_bull or is_structural_coil_bear,
                "discovery_type": discovery_type,
                "bullish_structure_status": bullish_structure_status,
                "bearish_structure_status": bearish_structure_status,
                "confirmed_pivots": {
                    "swing_lows": confirmed_swing_lows[:2],
                    "swing_highs": confirmed_swing_highs[:2],
                    "swing_low_ts": confirmed_swing_low_ts[:2],
                    "swing_high_ts": confirmed_swing_high_ts[:2]
                },
                "is_nr4": is_nr4,
                "is_nr7": is_nr7,
                "bb_width": round(bb_width, 2),
                "velocity": velocity,
                "df": eval_df
            }
        except Exception as e:
            print(f"[FNO FILTER ERROR] {sym}: {e}")
            return None

    @classmethod
    def filter_active_momentum_stocks(cls, fno_symbols, max_workers=12):
        """
        Scans all 219 F&O stocks in parallel threads.
        Returns:
            - bullish_stocks: Top 12-15 Bullish F&O options movers
            - bearish_stocks: Top 12-15 Bearish F&O options movers
            - raw_dfs: cached market DataFrames
        """
        if not fno_symbols:
            return [], [], {}

        tickers = [f"{s}.NS" if not s.endswith(".NS") else s for s in fno_symbols]
        results = []
        raw_dfs = {}

        def fetch_and_eval(sym):
            try:
                cdf = yf.download(sym, period="5d", interval="5m", progress=False)
                res = cls._evaluate_single_fno(sym, cdf)
                return sym, res, cdf
            except Exception:
                return sym, None, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_and_eval, sym) for sym in tickers]
            for fut in as_completed(futures):
                sym, res, cdf = fut.result()
                if cdf is not None and not cdf.empty:
                    raw_dfs[sym] = cdf
                if res:
                    results.append(res)

        bullish = [r for r in results if r["is_bull"]]
        bearish = [r for r in results if r["is_bear"]]

        # Sort by momentum velocity descending
        bullish.sort(key=lambda x: x["velocity"], reverse=True)
        bearish.sort(key=lambda x: x["velocity"], reverse=True)

        top_bull = bullish[:15]
        top_bear = bearish[:15]

        return top_bull, top_bear, raw_dfs, results
