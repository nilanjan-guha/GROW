"""
Groww Trending Sectors Engine
Fetches live sectors directly from Groww Trending Sectors (https://groww.in/stocks/sectors-trending)
Extracts official gain/loss percentages, matches against 219 F&O stocks, and filters:
- Top 3 Bullish Sectors with Top 5 F&O Gainers
- Top 3 Bearish Sectors with Top 5 F&O Losers (Skips non-F&O sectors)
"""
import requests
import json
import time
import urllib.parse
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

class GrowwSectorEngine:
    def __init__(self):
        self.cached_sectors = {}
        self.stock_to_sector_map = {}  # Symbol -> {"sector": name, "bias": "BULLISH"/"BEARISH", "relative_strength": float}
        self.last_fetch_time = 0
        self.cache_ttl = 300  # 5 minutes TTL

    def fetch_groww_trending_sectors(self, fno_universe_set):
        """
        Fetches all 60 Groww trending sectors directly from Groww.
        Enforces a 300-second cache TTL.
        Identifies top 3 bullish and top 3 bearish sectors that have active F&O stocks.
        If a sector has no moving F&O stocks, it checks the next sector (e.g. 4th, 5th, 6th).
        Maintains a comprehensive stock_to_sector_map across all scanned constituents.
        """
        now = time.time()
        if self.cached_sectors and (now - self.last_fetch_time < self.cache_ttl):
            return self.cached_sectors

        url = 'https://groww.in/stocks/sectors-trending'
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            if r.status_code != 200:
                print(f"[GROWW SECTORS] Error HTTP {r.status_code}")
                return self.cached_sectors
            
            soup = BeautifulSoup(r.text, 'html.parser')
            script = soup.find('script', id='__NEXT_DATA__')
            if not script:
                print("[GROWW SECTORS] __NEXT_DATA__ script tag not found")
                return self.cached_sectors
                
            data = json.loads(script.string)
            sectors_raw = data.get('props', {}).get('pageProps', {}).get('sectors', [])
            if not sectors_raw:
                return self.cached_sectors
        except Exception as e:
            print("[GROWW SECTORS] Network exception:", e)
            return self.cached_sectors

        # Normalize F&O set symbols
        norm_fno = {s.upper().replace(".NS", "") for s in fno_universe_set}

        # Sort all sectors strictly by Groww's official 1D price change percent
        sorted_bullish = sorted([s for s in sectors_raw if s.get('dayChangePercent', 0) > 0], key=lambda x: x.get('dayChangePercent', 0), reverse=True)
        sorted_bearish = sorted([s for s in sectors_raw if s.get('dayChangePercent', 0) < 0], key=lambda x: x.get('dayChangePercent', 0))

        sector_results = {}
        # Build local dictionary first to ensure atomic replacement on completion
        new_stock_to_sector_map = {}

        # 1. Evaluate Bullish Sectors
        bull_count = 0
        for s in sorted_bullish:
            s_name = s['sectorName']
            s_code = s['industryCode']
            change_pct = round(s.get('dayChangePercent', 0.0), 2)
            
            constituents = self._fetch_sector_constituents(s_name, s_code, norm_fno, is_bullish=True, sec_pct=change_pct)
            # Only qualify if it has at least 1 moving positive F&O stock
            if constituents:
                bull_count += 1
                sector_results[s_name] = {
                    "ticker": s_code,
                    "groww_code": s_code,
                    "change": change_pct,
                    "is_bullish": True,
                    "gainers": s.get('positiveStocks', len(constituents)),
                    "losers": s.get('negativeStocks', 0),
                    "top_5": constituents[:5]
                }
                for stk in constituents:
                    new_stock_to_sector_map[stk["symbol"]] = {
                        "sector": s_name,
                        "bias": "BULLISH",
                        "relative_strength": stk.get("relative_strength", 0.0)
                    }
                if bull_count >= 3:
                    break

        # 2. Evaluate Bearish Sectors (Skip sectors with 0 moving F&O losers)
        bear_count = 0
        for s in sorted_bearish:
            s_name = s['sectorName']
            s_code = s['industryCode']
            change_pct = round(s.get('dayChangePercent', 0.0), 2)
            
            constituents = self._fetch_sector_constituents(s_name, s_code, norm_fno, is_bullish=False, sec_pct=change_pct)
            # Skip if no negative moving F&O stock
            if constituents:
                bear_count += 1
                sector_results[s_name] = {
                    "ticker": s_code,
                    "groww_code": s_code,
                    "change": change_pct,
                    "is_bullish": False,
                    "gainers": s.get('positiveStocks', 0),
                    "losers": s.get('negativeStocks', len(constituents)),
                    "top_5": constituents[:5]
                }
                for stk in constituents:
                    new_stock_to_sector_map[stk["symbol"]] = {
                        "sector": s_name,
                        "bias": "BEARISH",
                        "relative_strength": stk.get("relative_strength", 0.0)
                    }
                if bear_count >= 3:
                    break

        # Atomic replacement: assign once rebuild is completely finished
        self.stock_to_sector_map = new_stock_to_sector_map
        self.cached_sectors = sector_results
        self.last_fetch_time = now
        return sector_results

    def _fetch_sector_constituents(self, sector_name, industry_code, norm_fno_set, is_bullish, sec_pct):
        """Fetches individual stocks from Groww sector page and filters only moving F&O constituents"""
        # Safely URL-encode the sector name
        safe_sector_name = urllib.parse.quote(sector_name)
        url = f'https://groww.in/stocks/sectors-trending/{safe_sector_name}?industryCodes={industry_code}'
        try:
            r = requests.get(url, headers=HEADERS, timeout=6)
            if r.status_code != 200:
                return []
            soup = BeautifulSoup(r.text, 'html.parser')
            script = soup.find('script', id='__NEXT_DATA__')
            if not script:
                return []
            data = json.loads(script.string)
            stocks_raw = data.get('props', {}).get('pageProps', {}).get('data', [])
        except Exception as e:
            print(f"[GROWW SECTOR PARSE ERROR] {sector_name}: {e}")
            return []

        qualified_fno = []
        for item in stocks_raw:
            code = str(item.get('nseScriptCode', '')).strip().upper()
            if not code or code not in norm_fno_set:
                continue
            
            try:
                ltp = float(item.get('ltp', 0.0) or 0.0)
                close = float(item.get('close', ltp) or ltp)
                pct = round(((ltp - close) / close * 100), 2) if close > 0 else 0.0
            except (ValueError, TypeError):
                continue
            
            # Directional alignment:
            # Bullish sectors only take stocks with pct > 0
            # Bearish sectors only take stocks with pct < 0
            if is_bullish and pct > 0:
                qualified_fno.append({
                    "symbol": f"{code}.NS",
                    "price": ltp,
                    "change": pct,
                    "relative_strength": round(pct - sec_pct, 2)
                })
            elif not is_bullish and pct < 0:
                qualified_fno.append({
                    "symbol": f"{code}.NS",
                    "price": ltp,
                    "change": pct,
                    "relative_strength": round(pct - sec_pct, 2)
                })

        # Multi-factor ranking:
        # Instead of simple stock_pct - sec_pct, we compute multi-factor momentum & relative strength acceleration
        for stk in qualified_fno:
            pct = stk["change"]
            # Outperformance component over sector baseline
            alpha_over_sector = pct - sec_pct
            # Relative strength momentum weight
            stk["momentum_score"] = round(abs(pct) * 1.5 + abs(alpha_over_sector) * 1.2, 2)

        if is_bullish:
            qualified_fno.sort(key=lambda x: (x.get("momentum_score", 0.0), x["change"]), reverse=True)
        else:
            qualified_fno.sort(key=lambda x: (x.get("momentum_score", 0.0), -x["change"]), reverse=True)

        return qualified_fno
