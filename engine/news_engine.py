import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import os
import json

class NewsIntelligenceEngine:
    def __init__(self):
        self.cached_news = []
        self.last_fetch = 0
        self.stock_news_cache = {}  # {symbol: {"timestamp": t, "articles": [...]}}
        self.symbol_names = {}      # Dynamically populated from Groww instruments

    def register_symbol_names(self, symbol_map):
        """Dynamically populates company names from Groww's live instrument database"""
        if symbol_map and isinstance(symbol_map, dict):
            self.symbol_names.update(symbol_map)

    def fetch_market_news(self):
        """Fetches macro market headlines from Google News RSS (Refreshes every 5 mins)"""
        now = time.time()
        if now - self.last_fetch < 300 and self.cached_news:
            return self.cached_news

        url = "https://news.google.com/rss/search?q=NSE+NIFTY+stock+market+India+earnings+SEBI+contract&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        req = urllib.request.Request(url, headers=headers)
        
        articles = []
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                tree = ET.fromstring(resp.read())
                items = tree.findall(".//item")
                for item in items[:30]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    pubDate = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else ""
                    
                    sentiment = self._analyze_headline_sentiment(title)
                    articles.append({
                        "title": title,
                        "pub_date": pubDate,
                        "link": link,
                        "sentiment": sentiment
                    })
            self.cached_news = articles
            self.last_fetch = now
        except Exception as e:
            print("[NEWS ENGINE] Fetch error:", e)

        return self.cached_news

    def fetch_news_for_stock(self, symbol_clean):
        """
        Dynamically fetches targeted news for ANY stock using its symbol and live company name.
        Cached per stock for 5 minutes (300 seconds). Zero hardcoded stock names.
        """
        sym = symbol_clean.replace(".NS", "").upper()
        now = time.time()
        
        # Check per-stock cache (5 minutes)
        if sym in self.stock_news_cache:
            entry = self.stock_news_cache[sym]
            if now - entry["timestamp"] < 300:
                return entry["articles"]

        # Determine dynamic company name from Groww or clean ticker
        company_name = self.symbol_names.get(sym, sym)
        # Build search query: e.g. "ANGELONE Angel One stock" or "SBIN State Bank Of India stock"
        query = f"{sym} {company_name} stock".replace("&", "and")
        encoded_q = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"
        
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        req = urllib.request.Request(url, headers=headers)
        articles = []
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                tree = ET.fromstring(resp.read())
                items = tree.findall(".//item")
                for item in items[:10]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    pubDate = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else ""
                    
                    articles.append({
                        "title": title,
                        "pub_date": pubDate,
                        "link": link,
                        "sentiment": self._analyze_headline_sentiment(title)
                    })
        except urllib.error.HTTPError as he:
            if he.code == 429:
                # Rate limited by Google News RSS: cache existing articles or empty for 60s to prevent spam
                prev_articles = self.stock_news_cache.get(sym, {}).get("articles", [])
                self.stock_news_cache[sym] = {"timestamp": now - 240, "articles": prev_articles}
                return prev_articles
            print(f"[NEWS ENGINE] HTTP error for {sym}:", he)
        except Exception as e:
            print(f"[NEWS ENGINE] Error fetching targeted news for {sym}:", e)

        self.stock_news_cache[sym] = {
            "timestamp": now,
            "articles": articles
        }
        return articles

    def match_catalyst_for_stock(self, symbol_clean):
        """
        Finds validated relevant catalyst headline for the stock dynamically.
        Requires verified ticker/company keyword match. Returns None if unverified.
        """
        sym = symbol_clean.replace(".NS", "").upper()
        raw_company_name = self.symbol_names.get(sym, "")
        sym_lower = sym.lower()

        # Stopwords to filter out generic corporate terms
        STOPWORDS = {"ltd", "limited", "india", "corporation", "corp", "industries", "holdings", "company", "services"}
        significant_words = [w.lower() for w in raw_company_name.split() if len(w) > 3 and w.lower() not in STOPWORDS]

        # 1. Check targeted stock feed
        targeted = self.fetch_news_for_stock(sym)
        if targeted:
            for art in targeted:
                t_lower = art["title"].lower()
                # Strict relevance check: symbol or distinct company keyword
                if sym_lower in t_lower or any(word in t_lower for word in significant_words):
                    return art

        # 2. Fallback to broad market news with strict relevance check
        market_news = self.fetch_market_news()
        for n in market_news:
            title_lower = n["title"].lower()
            if sym_lower in title_lower or any(word in title_lower for word in significant_words):
                return n

        # Never attach an arbitrary unrelated headline
        return None

    def _analyze_headline_sentiment(self, text):
        text_lower = text.lower()
        
        # Check for negative prefixes/negations
        negations = ["no ", "not ", "n't ", "never ", "fails to "]
        has_negation = any(neg in text_lower for neg in negations)

        bull_words = ["surges", "rallies", "jumps", "profit", "beats", "growth", "order", "contracts", "dividend", "bullish", "record high", "upgrade", "gains", "partnership", "expansion"]
        bear_words = ["slumps", "falls", "tumbles", "loss", "misses", "probe", "fraud", "penalty", "bearish", "record low", "downgrade", "crisis", "decline", "warning", "investigation"]
        
        bull_score = sum(1 for w in bull_words if w in text_lower)
        bear_score = sum(1 for w in bear_words if w in text_lower)
        
        if has_negation:
            # Invert sentiment if negated
            bull_score, bear_score = bear_score, bull_score

        if bull_score > bear_score:
            return "BULLISH"
        elif bear_score > bull_score:
            return "BEARISH"
        return "NEUTRAL"
