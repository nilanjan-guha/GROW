import time
import urllib.request
import xml.etree.ElementTree as ET
import os
import json

class NewsIntelligenceEngine:
    def __init__(self):
        self.cached_news = []
        self.last_fetch = 0
        self.groq_api_key = os.getenv("GROQ_API_KEY")

    def fetch_market_news(self):
        now = time.time()
        if now - self.last_fetch < 180 and self.cached_news:
            return self.cached_news

        url = "https://news.google.com/rss/search?q=NIFTY+stock+market+India+earnings+SEBI&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        req = urllib.request.Request(url, headers=headers)
        
        articles = []
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                tree = ET.fromstring(resp.read())
                items = tree.findall(".//item")
                for item in items[:25]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    pubDate = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    link = item.find("link").text if item.find("link") is not None else ""
                    
                    # Tag relevant stock symbols if mentioned
                    articles.append({
                        "title": title,
                        "pub_date": pubDate,
                        "link": link,
                        "sentiment": self._analyze_headline_sentiment(title)
                    })
            self.cached_news = articles
            self.last_fetch = now
        except Exception as e:
            print("[NEWS ENGINE] Fetch error:", e)

        return self.cached_news

    def _analyze_headline_sentiment(self, text):
        text_lower = text.lower()
        bull_words = ["surges", "rallies", "jumps", "profit", "beats", "growth", "order", "contracts", "dividend", "bullish", "record high", "upgrade"]
        bear_words = ["slumps", "falls", "tumbles", "loss", "misses", "probe", "fraud", "penalty", "bearish", "record low", "downgrade", "crisis"]
        
        bull_score = sum(1 for w in bull_words if w in text_lower)
        bear_score = sum(1 for w in bear_words if w in text_lower)
        
        if bull_score > bear_score:
            return "BULLISH"
        elif bear_score > bull_score:
            return "BEARISH"
        return "NEUTRAL"

    def match_catalyst_for_stock(self, symbol_clean):
        news = self.fetch_market_news()
        for n in news:
            if symbol_clean.lower() in n["title"].lower():
                return n
        return None
