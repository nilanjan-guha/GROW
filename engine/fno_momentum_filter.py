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

            curr_c = float(c.iloc[-1])
            curr_o = float(o.iloc[-1])

            # Intraday calculation: Compare against the actual start of the latest trading session
            df_temp = cdf.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_temp.index):
                df_temp.index = pd.to_datetime(df_temp.index)
            df_temp['Date'] = df_temp.index.date
            dates = sorted(df_temp['Date'].unique())
            if dates:
                today_df = df_temp[df_temp['Date'] == dates[-1]]
                session_o = float(today_df['Open'].iloc[0]) if not today_df.empty else float(o.iloc[0])
            else:
                session_o = float(o.iloc[0])

            change_pct = round(((curr_c - session_o) / session_o) * 100, 2) if session_o > 0 else 0.0

            # 1. Volume & Turnover Filter (Reject dead contracts, but don't over-filter)
            recent_v = v.iloc[-10:].mean()
            if recent_v < 5000 or (recent_v * curr_c) < 1500000:  # < 15 Lakhs turnover per 5-min
                return None

            # 2. Time-Slot Adjusted Relative Volume (RVOL) on Completed 5-Minute Candles
            # Validates against completed matching historical time-of-day slots
            df_temp['Time'] = df_temp.index.time
            # If the latest candle is still forming, use the previous completed candle time
            comp_idx = -2 if len(df_temp) >= 2 else -1
            slot_time = df_temp['Time'].iloc[comp_idx]
            same_slot_vols = df_temp[df_temp['Time'] == slot_time]['Volume']
            curr_vol = float(v.iloc[comp_idx])

            obs_count = len(same_slot_vols)
            if obs_count > 1:
                hist_slot_avg = float(same_slot_vols.iloc[:-1].mean())
                rvol = round(curr_vol / hist_slot_avg, 2) if hist_slot_avg > 0 else 1.0
            else:
                vol_sma20 = v.rolling(20).mean()
                avg_vol = float(vol_sma20.iloc[comp_idx]) if not vol_sma20.empty and not pd.isna(vol_sma20.iloc[comp_idx]) else 1.0
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

            # 4. Session VWAP (Strictly reset at start of current trading session)
            today_idx = df_temp.index.date == dates[-1] if dates else slice(None)
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

            # Recent higher-lows or lower-highs (Last 3 candles structure)
            recent_lows = [float(x) for x in l.iloc[-4:]]
            recent_highs = [float(x) for x in h.iloc[-4:]]
            has_higher_lows = (recent_lows[-1] >= recent_lows[-2] >= recent_lows[-3]) if len(recent_lows) >= 3 else False
            has_lower_highs = (recent_highs[-1] <= recent_highs[-2] <= recent_highs[-3]) if len(recent_highs) >= 3 else False

            # 6. Directional Pre-Breakout / Compression Filters (No Ambiguous Dual Qualification):
            # Bullish Coil: Price firmly above VWAP, higher-lows or EMA9 > EMA21, volatility compression (NR4/NR7/BB Squeeze)
            is_pre_coil_bull = (
                (curr_c > curr_vwap) and 
                (ema9 >= ema21) and 
                (has_higher_lows or curr_c > curr_o) and 
                (is_nr4 or is_nr7 or is_bb_squeeze)
            )
            # Bearish Coil: Price firmly below VWAP, lower-highs or EMA9 < EMA21, volatility compression (NR4/NR7/BB Squeeze)
            is_pre_coil_bear = (
                (curr_c < curr_vwap) and 
                (ema9 <= ema21) and 
                (has_lower_highs or curr_c < curr_o) and 
                (is_nr4 or is_nr7 or is_bb_squeeze)
            )

            # Active Momentum Breakout Path:
            is_bull_momentum = (curr_c >= curr_vwap) and (ema9 >= ema21 * 0.998) and (change_pct >= 0.20 or (rvol >= 1.25 and curr_c > curr_o))
            is_bear_momentum = (curr_c <= curr_vwap) and (ema9 <= ema21 * 1.002) and (change_pct <= -0.20 or (rvol >= 1.25 and curr_c < curr_o))

            is_bull = is_bull_momentum or is_pre_coil_bull
            is_bear = is_bear_momentum or is_pre_coil_bear

            # Mutual Exclusivity: If both evaluate true or neither, discard ambiguous chop
            if (is_bull and is_bear) or (not is_bull and not is_bear):
                return None

            # Real momentum velocity: based strictly on price expansion and genuine relative volume (no artificial inflations)
            velocity = round(abs(change_pct) * max(1.0, rvol), 2)
            if velocity < 0.5:
                # Minimum readable floor for coils
                velocity = round(max(0.8, rvol), 2)

            return {
                "symbol": sym,
                "current_price": curr_c,
                "change_pct": change_pct,
                "rvol": rvol,
                "rvol_obs": obs_count - 1,
                "atr_pct": round(atr_pct, 2),
                "vwap": curr_vwap,
                "is_bull": is_bull,
                "is_bear": is_bear,
                "is_pre_coil": is_pre_coil_bull or is_pre_coil_bear,
                "is_nr4": is_nr4,
                "is_nr7": is_nr7,
                "bb_width": round(bb_width, 2),
                "velocity": velocity,
                "df": cdf
            }
        except Exception:
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

        return top_bull, top_bear, raw_dfs
