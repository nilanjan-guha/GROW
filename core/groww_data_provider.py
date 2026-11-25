import logging
import datetime
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

logger = logging.getLogger("GrowwDataProvider")
logger.setLevel(logging.INFO)

class GrowwDataProvider:
    """
    Unified Data Provider wrapping the official Groww API Python SDK (growwapi).
    Provides methods for:
      - Live LTP (Last Traded Price) batch queries (up to 50 instruments)
      - Live OHLC real-time snapshots
      - Historical candle retrieval (5m, 15m, 1d) with automatic DataFrame transformation
      - Option chain and Greek analytics
    Features fail-closed fallback mechanisms so the application remains 100% operational
    whether authenticated with full market-data roles or standard order-basic roles.
    """

    def __init__(self, client=None):
        self.client = client
        self._market_data_allowed: Optional[bool] = None

    def set_client(self, client):
        self.client = client
        self._market_data_allowed = None

    def is_authenticated(self) -> bool:
        return self.client is not None

    def get_ltp_batch(self, symbols: List[str], segment: str = "CASH") -> Dict[str, float]:
        """
        Fetch LTP for a list of trading symbols using growwapi.get_ltp.
        Symbols should be passed like ['RELIANCE', 'TATASTEEL'] or ['NSE_RELIANCE'].
        Batched in chunks of 50 per official documentation.
        """
        if not self.client or self._market_data_allowed is False:
            return {}

        ltp_map = {}
        seg = getattr(self.client, f"SEGMENT_{segment}", "CASH")

        # Format symbols as required: "NSE_<SYMBOL>"
        formatted_symbols = [
            s if s.startswith("NSE_") or s.startswith("BSE_") else f"NSE_{s}"
            for s in symbols
        ]

        # Chunk in batches of 50
        chunk_size = 50
        for i in range(0, len(formatted_symbols), chunk_size):
            chunk = tuple(formatted_symbols[i:i + chunk_size])
            try:
                res = self.client.get_ltp(segment=seg, exchange_trading_symbols=chunk)
                if isinstance(res, dict):
                    self._market_data_allowed = True
                    for sym_key, price in res.items():
                        clean = sym_key.replace("NSE_", "").replace("BSE_", "")
                        try:
                            ltp_map[clean] = float(price)
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                msg = str(e).lower()
                if "forbidden" in msg or "roles" in msg:
                    logger.warning("Groww live-data/ltp API requires market-data subscription: %s", e)
                    self._market_data_allowed = False
                    break
                logger.error("Error fetching LTP batch from Groww: %s", e)
        return ltp_map

    def get_ohlc_batch(self, symbols: List[str], segment: str = "CASH") -> Dict[str, Dict[str, float]]:
        """
        Fetch real-time snapshot OHLC for a list of instruments (up to 50 per call).
        Returns dict: { 'RELIANCE': {'open': ..., 'high': ..., 'low': ..., 'close': ...} }
        """
        if not self.client or self._market_data_allowed is False:
            return {}

        ohlc_map = {}
        seg = getattr(self.client, f"SEGMENT_{segment}", "CASH")
        formatted_symbols = [
            s if s.startswith("NSE_") or s.startswith("BSE_") else f"NSE_{s}"
            for s in symbols
        ]

        chunk_size = 50
        for i in range(0, len(formatted_symbols), chunk_size):
            chunk = tuple(formatted_symbols[i:i + chunk_size])
            try:
                res = self.client.get_ohlc(segment=seg, exchange_trading_symbols=chunk)
                if isinstance(res, dict):
                    self._market_data_allowed = True
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
                    self._market_data_allowed = False
                    break
                logger.error("Error fetching OHLC batch from Groww: %s", e)
        return ohlc_map

    def get_historical_candles_df(
        self,
        symbol: str,
        interval: str = "5minute",
        days_back: int = 5,
        exchange: str = "NSE",
        segment: str = "CASH"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical candle data using client.get_historical_candles
        Transforms response into standard pandas DataFrame with columns:
        [Open, High, Low, Close, Volume] indexed by Datetime.
        """
        if not self.client or self._market_data_allowed is False:
            return None

        clean_sym = symbol.replace(".NS", "").replace("^NSEI", "NIFTY")
        groww_symbol = f"{exchange}-{clean_sym}"
        now = datetime.datetime.now()
        start_time = (now - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d 09:15:00")
        end_time = now.strftime("%Y-%m-%d %H:%M:%S")

        # Map interval string
        interval_code = getattr(self.client, f"CANDLE_INTERVAL_MIN_5", "5minute")
        if interval == "1minute":
            interval_code = getattr(self.client, "CANDLE_INTERVAL_MIN_1", "1minute")
        elif interval == "15minute":
            interval_code = getattr(self.client, "CANDLE_INTERVAL_MIN_15", "15minute")
        elif interval == "1day":
            interval_code = getattr(self.client, "CANDLE_INTERVAL_DAY", "1day")

        try:
            res = self.client.get_historical_candles(
                exchange=exchange,
                segment=segment,
                groww_symbol=groww_symbol,
                start_time=start_time,
                end_time=end_time,
                candle_interval=interval_code,
                timeout=10
            )
            if not res or not isinstance(res, dict):
                return None

            candles_list = res.get("candles") or res.get("data")
            if not candles_list or not isinstance(candles_list, list):
                return None

            # Standard format: [timestamp, open, high, low, close, volume, oi]
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
            if not records:
                return None

            df = pd.DataFrame(records)
            df.set_index("Datetime", inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            msg = str(e).lower()
            if "forbidden" in msg or "roles" in msg:
                self._market_data_allowed = False
            else:
                logger.debug("Groww historical candles fetch failed for %s: %s", symbol, e)
            return None

    def get_option_chain_data(self, underlying: str = "NIFTY", expiry_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch complete option chain with Greeks using groww.get_option_chain.
        Returns parsed strike data, Greeks, and underlying price.
        """
        if not self.client or self._market_data_allowed is False:
            return None

        clean_underlying = underlying.replace("^NSEI", "NIFTY")
        try:
            res = self.client.get_option_chain(
                exchange="NSE",
                underlying=clean_underlying,
                expiry_date=expiry_date or "",
                timeout=10
            )
            return res
        except Exception as e:
            logger.debug("Groww option chain fetch failed: %s", e)
            return None
