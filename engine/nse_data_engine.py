import os
import time
import requests
import json
import threading

class NSEDataEngine:
    """
    NSE India Live Data Engine:
    Fetches real-time structured data from official NSE India APIs:
    - FII / DII Institutional Net Flow
    - Bulk & Block Deals (Institutional Large Deals)
    - Most Active by Volume & Value
    - Short Selling Insights
    Provides caching (TTL 3-5 min) to prevent IP rate-limiting.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NSEDataEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": os.getenv("NSE_MCP_UA", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "*/*"
        })
        self._cache = {
            "fiidii": {"data": None, "ts": 0},
            "largedeals": {"data": None, "ts": 0},
            "most_active": {"data": None, "ts": 0}
        }
        self.TTL = 300  # 5 minutes cache

    def get_fii_dii_activity(self):
        """Returns the latest FII / DII institutional net flow figures"""
        now = time.time()
        if self._cache["fiidii"]["data"] and (now - self._cache["fiidii"]["ts"] < self.TTL):
            return self._cache["fiidii"]["data"]

        try:
            url = "https://www.nseindia.com/api/fiidiiTradeReact"
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                rows = resp.json()
                # Group by date
                grouped = {}
                for row in rows:
                    dt = row.get("date", "").strip()
                    if not dt:
                        continue
                    if dt not in grouped:
                        grouped[dt] = {"date": dt, "fii": {}, "dii": {}}
                    cat = row.get("category", "").strip().upper()
                    if "FII" in cat:
                        grouped[dt]["fii"] = {
                            "buy": float(row.get("buyValue") or 0),
                            "sell": float(row.get("sellValue") or 0),
                            "net": float(row.get("netValue") or 0)
                        }
                    elif "DII" in cat:
                        grouped[dt]["dii"] = {
                            "buy": float(row.get("buyValue") or 0),
                            "sell": float(row.get("sellValue") or 0),
                            "net": float(row.get("netValue") or 0)
                        }

                res_list = list(grouped.values())
                self._cache["fiidii"] = {"data": res_list, "ts": now}
                return res_list
        except Exception as e:
            print("[NSE DATA ENGINE] Error fetching FII/DII:", e)
        return self._cache["fiidii"]["data"] or []

    def get_large_deals(self, symbol=None):
        """Returns recent Bulk and Block deals, optionally filtered by symbol"""
        now = time.time()
        if not self._cache["largedeals"]["data"] or (now - self._cache["largedeals"]["ts"] >= self.TTL):
            try:
                url = "https://www.nseindia.com/api/snapshot-capital-market-largedeal"
                resp = self.session.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    bulk = data.get("BULK_DEALS_DATA", []) or []
                    block = data.get("BLOCK_DEALS_DATA", []) or []
                    parsed = []
                    for b in bulk:
                        parsed.append({
                            "type": "BULK",
                            "symbol": (b.get("symbol") or "").strip().upper(),
                            "name": b.get("name", ""),
                            "client": b.get("clientName", ""),
                            "action": "BUY" if (b.get("buySell") or "").upper().startswith("B") else "SELL",
                            "qty": int(float(b.get("qty") or 0)),
                            "price": float(b.get("watp") or 0),
                            "date": b.get("date", "")
                        })
                    for bl in block:
                        parsed.append({
                            "type": "BLOCK",
                            "symbol": (bl.get("symbol") or "").strip().upper(),
                            "name": bl.get("name", ""),
                            "client": bl.get("clientName", ""),
                            "action": "BUY" if (bl.get("buySell") or "").upper().startswith("B") else "SELL",
                            "qty": int(float(bl.get("qty") or 0)),
                            "price": float(bl.get("watp") or 0),
                            "date": bl.get("date", "")
                        })
                    self._cache["largedeals"] = {"data": parsed, "ts": now}
            except Exception as e:
                print("[NSE DATA ENGINE] Error fetching large deals:", e)

        deals = self._cache["largedeals"]["data"] or []
        if symbol:
            clean = symbol.replace(".NS", "").upper()
            return [d for d in deals if d["symbol"] == clean]
        return deals

    def get_institutional_bias(self, symbol=None):
        """
        Calculates institutional bias score for an individual stock or overall market:
        Returns:
            - fii_net: latest net FII value
            - dii_net: latest net DII value
            - market_inst_bias: BULLISH | BEARISH | NEUTRAL
            - stock_deals: list of large deals for symbol
            - inst_score_modifier: +15, -15, or 0 to add to composite score
        """
        fii_dii = self.get_fii_dii_activity()
        latest = fii_dii[0] if fii_dii else {}
        fii_net = latest.get("fii", {}).get("net", 0.0)
        dii_net = latest.get("dii", {}).get("net", 0.0)
        total_inst_net = fii_net + dii_net

        if total_inst_net > 500:
            market_bias = "BULLISH"
        elif total_inst_net < -500:
            market_bias = "BEARISH"
        else:
            market_bias = "NEUTRAL"

        deals = []
        score_mod = 0
        if symbol:
            deals = self.get_large_deals(symbol)
            buys = sum(d["qty"] * d["price"] for d in deals if d["action"] == "BUY")
            sells = sum(d["qty"] * d["price"] for d in deals if d["action"] == "SELL")
            if buys > sells and buys > 0:
                score_mod += 15
            elif sells > buys and sells > 0:
                score_mod -= 15

        return {
            "date": latest.get("date", ""),
            "fii_net": fii_net,
            "dii_net": dii_net,
            "total_inst_net": total_inst_net,
            "market_inst_bias": market_bias,
            "stock_deals": deals,
            "score_modifier": score_mod
        }
