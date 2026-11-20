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
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

class GrowwSectorEngine:
    def __init__(self):
        self.cached_sectors = {}
        self.sector_metadata = {}
        self.last_fetch_time = 0

    def fetch_groww_trending_sectors(self, fno_universe_set):
        """
        Fetches all 60 Groww trending sectors directly from Groww.
        Identifies top 3 bullish and top 3 bearish sectors that have active F&O stocks.
        If a sector has no moving F&O stocks, it checks the next sector (e.g. 4th, 5th, 6th).
        """
        url = 'https://groww.in/stocks/sectors-trending'
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            if r.status_code != 200:
                print(f"[GROWW SECTORS] Error HTTP {r.status_code}")
                return self.cached_sectors
            
            soup = BeautifulSoup(r.text, 'html.parser')
            script = soup.find('script', id='__NEXT_DATA__')
            if not script:
                return self.cached_sectors
                
            data = json.loads(script.string)
            sectors_raw = data.get('props', {}).get('pageProps', {}).get('sectors', [])
            if not sectors_raw:
                return self.cached_sectors
        except Exception as e:
            print("[GROWW SECTORS] Network exception:", e)
            return self.cached_sectors

        # Sort all sectors strictly by Groww's official 1D price change percent
        sorted_bullish = sorted([s for s in sectors_raw if s.get('dayChangePercent', 0) > 0], key=lambda x: x.get('dayChangePercent', 0), reverse=True)
        sorted_bearish = sorted([s for s in sectors_raw if s.get('dayChangePercent', 0) < 0], key=lambda x: x.get('dayChangePercent', 0))

        sector_results = {}

        # 1. Evaluate Bullish Sectors
        bull_count = 0
        for s in sorted_bullish:
            s_name = s['sectorName']
            s_code = s['industryCode']
            change_pct = round(s.get('dayChangePercent', 0.0), 2)
            
            constituents = self._fetch_sector_constituents(s_name, s_code, fno_universe_set, is_bullish=True, sec_pct=change_pct)
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
                if bull_count >= 3:
                    break

        # 2. Evaluate Bearish Sectors (Skip sectors with 0 moving F&O losers)
        bear_count = 0
        for s in sorted_bearish:
            s_name = s['sectorName']
            s_code = s['industryCode']
            change_pct = round(s.get('dayChangePercent', 0.0), 2)
            
            constituents = self._fetch_sector_constituents(s_name, s_code, fno_universe_set, is_bullish=False, sec_pct=change_pct)
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
                if bear_count >= 3:
                    break

        self.cached_sectors = sector_results
        self.last_fetch_time = time.time()
        return sector_results

    def _fetch_sector_constituents(self, sector_name, industry_code, fno_universe_set, is_bullish, sec_pct):
        """Fetches individual stocks from Groww sector page and filters only moving F&O constituents"""
        url = f'https://groww.in/stocks/sectors-trending/{sector_name}?industryCodes={industry_code}'
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
        except Exception:
            return []

        qualified_fno = []
        for item in stocks_raw:
            code = item.get('nseScriptCode')
            if not code or code not in fno_universe_set:
                continue
            
            ltp = float(item.get('ltp', 0.0))
            close = float(item.get('close', ltp))
            pct = round(((ltp - close) / close * 100), 2) if close > 0 else 0.0
            
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

        # Rank: Bullish by highest gainer; Bearish by biggest loser (most negative)
        if is_bullish:
            qualified_fno.sort(key=lambda x: x["change"], reverse=True)
        else:
            qualified_fno.sort(key=lambda x: x["change"])

        return qualified_fno
