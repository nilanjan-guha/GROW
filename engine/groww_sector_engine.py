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
        self.stock_to_sector_map = {}  # Symbol -> {"sector": name, "bias": "BULLISH"/"BEARISH"/"NEUTRAL", "relative_strength": float}
        self.last_fetch_time = 0
        self.cache_ttl = 300  # 5 minutes TTL
        self.is_stale = False
        self.is_usable = False
        self.last_error = None
        self.expected_fno_coverage = 219
        self.actual_fno_coverage = 0

    def fetch_groww_trending_sectors(self, fno_universe_set):
        """
        Fetches all 60 Groww trending sectors directly from Groww.
        Enforces a 300-second cache TTL.
        Identifies top 3 bullish, top 3 bearish, and neutral sectors that have active F&O stocks.
        If a sector has no moving F&O stocks, it checks the next sector.
        Tracks explicit stale status and actual F&O coverage count.
        """
        now = time.time()
        # Stale check: if cache exists but is older than cache_ttl (300s), mark stale and reject cache!
        if self.cached_sectors:
            if now - self.last_fetch_time > self.cache_ttl:
                self.is_stale = True
            else:
                self.is_stale = False
                return self.cached_sectors

        url = 'https://groww.in/stocks/sectors-trending'
        sectors_raw = []
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            if r.status_code != 200:
                print(f"[GROWW SECTORS] Error HTTP {r.status_code}")
                self.is_stale = True
                self.last_error = f"HTTP_{r.status_code}"
                # Reject stale cache on failure; return empty dict so ranking knows sector data is unavailable/stale
                return {}
            
            soup = BeautifulSoup(r.text, 'html.parser')
            script = soup.find('script', id='__NEXT_DATA__')
            if not script:
                print("[GROWW SECTORS] __NEXT_DATA__ script tag not found")
                self.is_stale = True
                self.last_error = "MISSING_NEXT_DATA"
                return {}
                
            data = json.loads(script.string)
            sectors_raw = data.get('props', {}).get('pageProps', {}).get('sectors', [])
            if not sectors_raw:
                self.is_stale = True
                self.last_error = "EMPTY_SECTOR_DATA"
                return {}
        except Exception as e:
            print("[GROWW SECTORS] Network exception:", e)
            self.is_stale = True
            self.last_error = str(e)
            return {}

        # Normalize F&O set symbols
        norm_fno = {s.upper().replace(".NS", "") for s in fno_universe_set}
        if fno_universe_set:
            self.expected_fno_coverage = len(fno_universe_set)

        # Sort all sectors by Groww's official 1D price change percent
        sorted_bullish = sorted([s for s in sectors_raw if s.get('dayChangePercent', 0) > 0], key=lambda x: x.get('dayChangePercent', 0), reverse=True)
        sorted_bearish = sorted([s for s in sectors_raw if s.get('dayChangePercent', 0) < 0], key=lambda x: x.get('dayChangePercent', 0))
        # Neutral / Flat sectors (|dayChangePercent| <= 0.3% or near flat)
        sorted_neutral = sorted([s for s in sectors_raw if abs(s.get('dayChangePercent', 0)) <= 0.35], key=lambda x: abs(x.get('dayChangePercent', 0)))

        sector_results = {}
        # Build local dictionary first to ensure atomic replacement on completion
        new_stock_to_sector_map = {}
        unique_fno_covered = set()

        # 1. Evaluate Bullish Sectors
        bull_count = 0
        for s in sorted_bullish:
            s_name = s['sectorName']
            s_code = s['industryCode']
            change_pct = round(s.get('dayChangePercent', 0.0), 2)
            
            constituents = self._fetch_sector_constituents(s_name, s_code, norm_fno, is_bullish=True, sec_pct=change_pct)
            if constituents:
                bull_count += 1
                for c in constituents:
                    unique_fno_covered.add(c["symbol"].replace(".NS", "").upper())
                sector_results[s_name] = {
                    "ticker": s_code,
                    "groww_code": s_code,
                    "change": change_pct,
                    "is_bullish": True,
                    "is_neutral": False,
                    "gainers": s.get('positiveStocks', len(constituents)),
                    "losers": s.get('negativeStocks', 0),
                    "top_5": constituents[:5],
                    "all_constituents": constituents,
                    "fno_coverage_count": len(constituents),
                    "last_updated": now,
                    "is_stale": False
                }
                for stk in constituents:
                    new_stock_to_sector_map[stk["symbol"]] = {
                        "sector": s_name,
                        "bias": "BULLISH",
                        "sector_change": change_pct,
                        "relative_strength": stk.get("relative_strength", 0.0)
                    }
                if bull_count >= 3:
                    break

        # 2. Evaluate Bearish Sectors
        bear_count = 0
        for s in sorted_bearish:
            s_name = s['sectorName']
            s_code = s['industryCode']
            change_pct = round(s.get('dayChangePercent', 0.0), 2)
            
            constituents = self._fetch_sector_constituents(s_name, s_code, norm_fno, is_bullish=False, sec_pct=change_pct)
            if constituents:
                bear_count += 1
                for c in constituents:
                    unique_fno_covered.add(c["symbol"].replace(".NS", "").upper())
                sector_results[s_name] = {
                    "ticker": s_code,
                    "groww_code": s_code,
                    "change": change_pct,
                    "is_bullish": False,
                    "is_neutral": False,
                    "gainers": s.get('positiveStocks', 0),
                    "losers": s.get('negativeStocks', len(constituents)),
                    "top_5": constituents[:5],
                    "all_constituents": constituents,
                    "fno_coverage_count": len(constituents),
                    "last_updated": now,
                    "is_stale": False
                }
                for stk in constituents:
                    new_stock_to_sector_map[stk["symbol"]] = {
                        "sector": s_name,
                        "bias": "BEARISH",
                        "sector_change": change_pct,
                        "relative_strength": stk.get("relative_strength", 0.0)
                    }
                if bear_count >= 3:
                    break

        # 3. Evaluate Neutral Sectors (Top 2 Neutral / Sideways Sectors for stock-level discovery)
        neutral_count = 0
        for s in sorted_neutral:
            s_name = s['sectorName']
            if s_name in sector_results:
                continue
            s_code = s['industryCode']
            change_pct = round(s.get('dayChangePercent', 0.0), 2)
            
            constituents = self._fetch_sector_constituents(s_name, s_code, norm_fno, is_bullish=True, sec_pct=change_pct)
            if constituents:
                neutral_count += 1
                for c in constituents:
                    unique_fno_covered.add(c["symbol"].replace(".NS", "").upper())
                sector_results[s_name] = {
                    "ticker": s_code,
                    "groww_code": s_code,
                    "change": change_pct,
                    "is_bullish": False,
                    "is_neutral": True,
                    "gainers": s.get('positiveStocks', 0),
                    "losers": s.get('negativeStocks', 0),
                    "top_5": constituents[:5],
                    "all_constituents": constituents,
                    "fno_coverage_count": len(constituents),
                    "last_updated": now,
                    "is_stale": False
                }
                for stk in constituents:
                    if stk["symbol"] not in new_stock_to_sector_map:
                        new_stock_to_sector_map[stk["symbol"]] = {
                            "sector": s_name,
                            "bias": "NEUTRAL",
                            "sector_change": change_pct,
                            "relative_strength": stk.get("relative_strength", 0.0)
                        }
                if neutral_count >= 2:
                    break

        self.actual_fno_coverage = len(unique_fno_covered)
        # Multi-Factor Sector Health & Coverage Validation (P1):
        # 1. Total unique F&O stocks across priority sectors must meet minimum depth (>= 25 stocks).
        # 2. Minimum representation per sector: At least 5 of the 8 sectors must have valid F&O constituents.
        # 3. Overall ratio check: F&O coverage must be statistically representative.
        active_sectors_with_fno = sum(1 for s in sector_results.values() if s.get("fno_coverage_count", 0) > 0)
        
        has_healthy_coverage = (
            self.actual_fno_coverage >= 25 and 
            active_sectors_with_fno >= 5 and 
            len(sector_results) >= 6
        )

        if not has_healthy_coverage:
            self.is_stale = True
            self.is_usable = False
            self.last_error = f"UNHEALTHY_SECTOR_MAP: {self.actual_fno_coverage} stocks, {active_sectors_with_fno}/8 sectors active"
            print(f"[GROWW SECTORS] Coverage rejected: Only {self.actual_fno_coverage} stocks across {active_sectors_with_fno} sectors. Unhealthy map.")
            return {}  # Strictly reject incomplete/fragile results
        else:
            self.is_stale = False
            self.is_usable = True

        # Atomic replacement: assign once rebuild is completely finished
        self.stock_to_sector_map = new_stock_to_sector_map
        self.cached_sectors = sector_results
        self.last_fetch_time = now
        self.last_error = None
        return sector_results

    def _fetch_sector_constituents(self, sector_name, industry_code, norm_fno_set, is_bullish, sec_pct):
        """
        Fetches individual stocks from Groww sector page and extracts ALL tradeable F&O constituents.
        Does NOT drop flat, quiet, or coiling stocks! Preserves all F&O constituents for pre-move discovery.
        """
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

        all_fno_constituents = []
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
            
            # Keep ALL valid F&O constituents regardless of daily sign so quiet/flat pre-breakout coils are NEVER discarded!
            alpha_over_sector = pct - sec_pct
            momentum_score = round(abs(pct) * 1.5 + abs(alpha_over_sector) * 1.2, 2)
            all_fno_constituents.append({
                "symbol": f"{code}.NS",
                "price": ltp,
                "change": pct,
                "velocity": round(max(0.8, abs(pct) * 1.3), 1),
                "relative_strength": round(alpha_over_sector, 2),
                "momentum_score": momentum_score,
                "is_directional_aligned": (pct > 0 if is_bullish else pct < 0)
            })

        if is_bullish:
            all_fno_constituents.sort(key=lambda x: (x.get("is_directional_aligned", False), x.get("momentum_score", 0.0), x["change"]), reverse=True)
        else:
            all_fno_constituents.sort(key=lambda x: (x.get("is_directional_aligned", False), x.get("momentum_score", 0.0), -x["change"]), reverse=True)

        return all_fno_constituents
