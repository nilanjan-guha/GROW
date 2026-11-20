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
        except Exception as e:
            print(f"[NEWS ENGINE] Error fetching targeted news for {sym}:", e)

        self.stock_news_cache[sym] = {
            "timestamp": now,
            "articles": articles
        }
        return articles

    def match_catalyst_for_stock(self, symbol_clean):
        """
        Finds the most relevant catalyst headline for ANY stock dynamically.
        1. Queries targeted stock news (cached every 5 min).
        2. Falls back to general market news matching.
        """
        sym = symbol_clean.replace(".NS", "").upper()
        company_name = self.symbol_names.get(sym, sym).lower()
        sym_lower = sym.lower()

        # 1. Check targeted stock feed
        targeted = self.fetch_news_for_stock(sym)
        if targeted:
            for art in targeted:
                t_lower = art["title"].lower()
                # Confirm relevance by checking either symbol or company name in title
                if sym_lower in t_lower or any(word in t_lower for word in company_name.split() if len(word) > 3):
                    return art
            # If no exact match, the top targeted article is the closest catalyst
            return targeted[0]

        # 2. Fallback to broad market news
        market_news = self.fetch_market_news()
        for n in market_news:
            title_lower = n["title"].lower()
            if sym_lower in title_lower or any(word in title_lower for word in company_name.split() if len(word) > 3):
                return n

        return None

    def _analyze_headline_sentiment(self, text):
        text_lower = text.lower()
        bull_words = ["surges", "rallies", "jumps", "profit", "beats", "growth", "order", "contracts", "dividend", "bullish", "record high", "upgrade", "gains", "partnership", "expansion"]
        bear_words = ["slumps", "falls", "tumbles", "loss", "misses", "probe", "fraud", "penalty", "bearish", "record low", "downgrade", "crisis", "decline", "warning", "investigation"]
        
        bull_score = sum(1 for w in bull_words if w in text_lower)
        bear_score = sum(1 for w in bear_words if w in text_lower)
        
        if bull_score > bear_score:
            return "BULLISH"
        elif bear_score > bull_score:
            return "BEARISH"
        return "NEUTRAL"
