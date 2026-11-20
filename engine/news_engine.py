import time
import urllib.request
import xml.etree.ElementTree as ET
import os
import json

class NewsIntelligenceEngine:
    def __init__(self):
        self.cached_news = []
        self.last_fetch = 0
        
        # Company name and symbol aliases for Indian F&O universe
        self.company_aliases = {
            "RELIANCE": ["reliance", "ril", "jio"],
            "HDFCBANK": ["hdfc bank", "hdfc", "hdfcbank"],
            "ICICIBANK": ["icici bank", "icici"],
            "SBIN": ["state bank of india", "sbi", "sbin"],
            "AXISBANK": ["axis bank", "axis"],
            "KOTAKBANK": ["kotak", "kotak mahindra"],
            "INDUSINDBK": ["indusind bank", "indusind"],
            "TCS": ["tata consultancy services", "tcs"],
            "INFY": ["infosys", "infy"],
            "WIPRO": ["wipro"],
            "HCLTECH": ["hcl tech", "hcl technologies", "hcltech"],
            "TECHM": ["tech mahindra", "techm"],
            "MARUTI": ["maruti", "maruti suzuki"],
            "M&M": ["mahindra & mahindra", "m&m", "mahindra"],
            "BAJAJ-AUTO": ["bajaj auto"],
            "TATAMOTORS": ["tata motors", "tatamotors"],
            "SUNPHARMA": ["sun pharma", "sun pharmaceutical"],
            "CIPLA": ["cipla"],
            "DRREDDY": ["dr reddy", "dr. reddy", "drreddy"],
            "APOLLOHOSP": ["apollo hospitals", "apollo hospital"],
            "ULTRACEMCO": ["ultratech", "ultratech cement"],
            "TATASTEEL": ["tata steel"],
            "JSWSTEEL": ["jsw steel"],
            "HINDALCO": ["hindalco"],
            "VEDL": ["vedanta", "vedl"],
            "ITC": ["itc"],
            "HINDUNILVR": ["hindustan unilever", "hul"],
            "BHARTIARTL": ["bharti airtel", "airtel"],
            "LT": ["larsen & toubro", "l&t", "lt"],
            "DLF": ["dlf"],
            "GODREJPROP": ["godrej properties", "godrej prop"],
            "KALYANKJIL": ["kalyan jewellers", "kalyan"]
        }

    def fetch_market_news(self):
        now = time.time()
        if now - self.last_fetch < 120 and self.cached_news:
            return self.cached_news

        url = "https://news.google.com/rss/search?q=NSE+NIFTY+stock+market+India+earnings+SEBI+contract&hl=en-IN&gl=IN&ceid=IN:en"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        req = urllib.request.Request(url, headers=headers)
        
        articles = []
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
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

    def match_catalyst_for_stock(self, symbol_clean):
        news = self.fetch_market_news()
        sym = symbol_clean.replace(".NS", "").upper()
        aliases = self.company_aliases.get(sym, [sym.lower()])
        
        for n in news:
            title_lower = n["title"].lower()
            if any(alias in title_lower for alias in aliases):
                return n
        return None
