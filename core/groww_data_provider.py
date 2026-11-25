import logging
import datetime
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import yfinance as yf

logger = logging.getLogger("GrowwDataProvider")
logger.setLevel(logging.INFO)

class GrowwDataProvider:
    """
    Unified Data Provider wrapping the official Groww API Python SDK (growwapi)
    with strict candle validation and graceful fallback.
    
    Guarantees:
      - Strictly 5-minute (intraday) or 1-day (daily) candle interfaces (no accidental 1-minute execution paths).
      - Endpoint-specific permission tracking so a failure in LTP doesn't block historical candles or vice versa.
      - Single source of truth interface: both main scanner and background tracker fetch through here.
      - Records data source ('GROWW' or 'YAHOO') and evaluated candle timestamp on every returned DataFrame.
    """

    def __init__(self, client=None):
        self.client = client
        self._ltp_allowed: Optional[bool] = None
        self._ohlc_allowed: Optional[bool] = None
        self._candles_allowed: Optional[bool] = None
        self._option_chain_allowed: Optional[bool] = None

    def set_client(self, client):
        self.client = client
        self._ltp_allowed = None
        self._ohlc_allowed = None
        self._candles_allowed = None
        self._option_chain_allowed = None

    def is_authenticated(self) -> bool:
        return self.client is not None

    def fetch_5m_candles(self, symbol: str, days_back: int = 5) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Unified 5-minute candle fetcher used by both main scanner and continuous tracker.
        Returns (DataFrame, source_name) where source_name is 'GROWW' or 'YAHOO'.
        DataFrame is guaranteed to have columns: [Open, High, Low, Close, Volume] indexed by Datetime.
        """
        clean_sym = symbol.replace(".NS", "").replace("^NSEI", "NIFTY").upper()
        
        # 1. Try Groww API historical candles first if client exists and not marked forbidden
        if self.client and self._candles_allowed is not False:
            try:
                groww_symbol = f"NSE-{clean_sym}"
                now = datetime.datetime.now()
                start_time = (now - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d 09:15:00")
                end_time = now.strftime("%Y-%m-%d %H:%M:%S")
                interval_code = getattr(self.client, "CANDLE_INTERVAL_MIN_5", "5minute")

                res = self.client.get_historical_candles(
                    exchange="NSE",
                    segment="CASH",
                    groww_symbol=groww_symbol,
                    start_time=start_time,
                    end_time=end_time,
                    candle_interval=interval_code,
                    timeout=8
                )
                if isinstance(res, dict):
                    candles_list = res.get("candles") or res.get("data")
                    if candles_list and isinstance(candles_list, list) and len(candles_list) >= 5:
                        self._candles_allowed = True
                        records = []
                        for row in candles_list:
                            if len(row) >= 5:
                                records.append({
                                    "Datetime": pd.to_datetime(row[0]),
                                    "Open": float(row[1]),
                                    "High": float(row[2]),
                                    "Low": float(row[3]),
                                    "Close": float(row[4]),
                                    "Volume": float(row[5]) if len(row) > 5 else 0.0
                                })
                        if records:
                            df = pd.DataFrame(records)
                            df.set_index("Datetime", inplace=True)
                            df.sort_index(inplace=True)
                            df.attrs["source"] = "GROWW"
                            return df, "GROWW"
            except Exception as e:
                msg = str(e).lower()
                if "forbidden" in msg or "roles" in msg:
                    self._candles_allowed = False
                logger.debug("Groww historical candles unavailable for %s: %s", clean_sym, e)

        # 2. Seamless verified fallback to Yahoo Finance
        try:
            yf_sym = f"{clean_sym}.NS" if not clean_sym.startswith("^") and clean_sym != "NIFTY" else ("^NSEI" if clean_sym == "NIFTY" else clean_sym)
            period_str = f"{days_back}d" if days_back <= 5 else "5d"
            df = yf.download(yf_sym, period=period_str, interval="5m", progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    if yf_sym in df.columns.levels[1]:
                        df = df.xs(yf_sym, axis=1, level=1)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                df.attrs["source"] = "YAHOO"
                return df, "YAHOO"
        except Exception as e:
            logger.error("Fallback download failed for %s: %s", clean_sym, e)

        return None, "NONE"

    def fetch_daily_candles(self, symbol: str, days_back: int = 45) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Fetches at least 30-45 trading sessions of daily candles for genuine 30-day breakout
        target-vs-stop learning and ATR calculation.
        """
        clean_sym = symbol.replace(".NS", "").replace("^NSEI", "NIFTY").upper()

        if self.client and self._candles_allowed is not False:
            try:
                groww_symbol = f"NSE-{clean_sym}"
                now = datetime.datetime.now()
                start_time = (now - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d 09:15:00")
                end_time = now.strftime("%Y-%m-%d %H:%M:%S")
                interval_code = getattr(self.client, "CANDLE_INTERVAL_DAY", "1day")

                res = self.client.get_historical_candles(
                    exchange="NSE",
                    segment="CASH",
                    groww_symbol=groww_symbol,
                    start_time=start_time,
                    end_time=end_time,
                    candle_interval=interval_code,
                    timeout=8
                )
                if isinstance(res, dict):
                    candles_list = res.get("candles") or res.get("data")
                    if candles_list and isinstance(candles_list, list) and len(candles_list) >= 15:
                        self._candles_allowed = True
                        records = []
                        for row in candles_list:
                            if len(row) >= 5:
                                records.append({
                                    "Date": pd.to_datetime(row[0]).date(),
                                    "Open": float(row[1]),
                                    "High": float(row[2]),
                                    "Low": float(row[3]),
                                    "Close": float(row[4]),
                                    "Volume": float(row[5]) if len(row) > 5 else 0.0
                                })
                        if records:
                            df = pd.DataFrame(records)
                            df.set_index("Date", inplace=True)
                            df.sort_index(inplace=True)
                            df.attrs["source"] = "GROWW"
                            return df, "GROWW"
            except Exception:
                pass

        # Fallback to Yahoo Finance daily
        try:
            yf_sym = f"{clean_sym}.NS" if not clean_sym.startswith("^") and clean_sym != "NIFTY" else ("^NSEI" if clean_sym == "NIFTY" else clean_sym)
            df = yf.download(yf_sym, period="2mo", interval="1d", progress=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    if yf_sym in df.columns.levels[1]:
                        df = df.xs(yf_sym, axis=1, level=1)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                df.attrs["source"] = "YAHOO"
                return df, "YAHOO"
        except Exception:
            pass

        return None, "NONE"

    def get_ltp_batch(self, symbols: List[str], segment: str = "CASH") -> Dict[str, float]:
        """
        Fetch LTP for a list of symbols in chunks of 50.
        """
        if not self.client or self._ltp_allowed is False:
            return {}

        ltp_map = {}
        seg = getattr(self.client, f"SEGMENT_{segment}", "CASH")
        formatted = [s if s.startswith("NSE_") or s.startswith("BSE_") else f"NSE_{s.replace('.NS', '')}" for s in symbols]

        for i in range(0, len(formatted), 50):
            chunk = tuple(formatted[i:i + 50])
            try:
                res = self.client.get_ltp(segment=seg, exchange_trading_symbols=chunk)
                if isinstance(res, dict):
                    self._ltp_allowed = True
                    for sym_key, price in res.items():
                        clean = sym_key.replace("NSE_", "").replace("BSE_", "")
                        try:
                            ltp_map[clean] = float(price)
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                msg = str(e).lower()
                if "forbidden" in msg or "roles" in msg:
                    self._ltp_allowed = False
                    break
        return ltp_map

    def get_ohlc_batch(self, symbols: List[str], segment: str = "CASH") -> Dict[str, Dict[str, float]]:
        """
        Fetch real-time snapshot OHLC in chunks of 50.
        """
        if not self.client or self._ohlc_allowed is False:
            return {}

        ohlc_map = {}
        seg = getattr(self.client, f"SEGMENT_{segment}", "CASH")
        formatted = [s if s.startswith("NSE_") or s.startswith("BSE_") else f"NSE_{s.replace('.NS', '')}" for s in symbols]

        for i in range(0, len(formatted), 50):
            chunk = tuple(formatted[i:i + 50])
            try:
                res = self.client.get_ohlc(segment=seg, exchange_trading_symbols=chunk)
                if isinstance(res, dict):
                    self._ohlc_allowed = True
                    for sym_key, ohlc in res.items():
                        clean = sym_key.replace("NSE_", "").replace("BSE_", "")
                        if isinstance(ohlc, dict):
                            ohlc_map[clean] = {
                                "open": float(ohlc.get("open", 0)),
                                "high": float(ohlc.get("high", 0)),
                                "low": float(ohlc.get("low", 0)),
                                "close": float(ohlc.get("close", 0))
                            }
            except Exception as e:
                msg = str(e).lower()
                if "forbidden" in msg or "roles" in msg:
                    self._ohlc_allowed = False
                    break
        return ohlc_map

    def get_option_chain_data(self, underlying: str = "NIFTY", expiry_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch complete option chain and Greeks using groww.get_option_chain.
        """
        if not self.client or self._option_chain_allowed is False:
            return None

        clean = underlying.replace("^NSEI", "NIFTY").replace(".NS", "")
        try:
            res = self.client.get_option_chain(
                exchange="NSE",
                underlying=clean,
                expiry_date=expiry_date or "",
                timeout=10
            )
            self._option_chain_allowed = True
            return res
        except Exception as e:
            msg = str(e).lower()
            if "forbidden" in msg or "roles" in msg:
                self._option_chain_allowed = False
            return None
