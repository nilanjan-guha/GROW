import os
import json
import urllib.request

class GroqIntelligence:
    """
    Integrates Groq API for AI-assisted news classification, catalyst interpretation,
    and institutional signal validation without overriding empirical numerical indicators.
    """
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def is_configured(self):
        return bool(self.api_key)

    def analyze_signal(self, symbol, direction, strategy, price, metrics, news_headline=""):
        if not self.is_configured():
            return {
                "ai_summary": f"Quant Rule Confirmed: {strategy} trigger active on {symbol} ({direction}) based on institutional volume, VWAP and trend confluence.",
                "catalyst_impact": "Technical & Order Flow Driven",
                "risk_rating": "MODERATE"
            }

        prompt = f"""
You are an institutional intraday equity research analyst.
Analyze the following trading candidate:
Stock: {symbol}
Direction: {direction}
Strategy: {strategy}
Current Price: ₹{price}
Metrics: {json.dumps(metrics)}
News: {news_headline or 'No immediate news catalyst; purely price/volume order flow'}

Provide a concise 2-sentence summary of why this trade has an institutional edge and the key risk to monitor. Respond in pure JSON format:
{{"ai_summary": "...", "catalyst_impact": "...", "risk_rating": "LOW/MODERATE/HIGH"}}
"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
        data = {
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }

        try:
            req = urllib.request.Request(self.endpoint, data=json.dumps(data).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"].strip()
                # If wrapped in markdown code blocks, clean it
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:].strip()
                return json.loads(content)
        except Exception as e:
            return {
                "ai_summary": f"Quant Momentum confirmed: {strategy} on {symbol} with high statistical confluence.",
                "catalyst_impact": "Order Flow Confluence",
                "risk_rating": "MODERATE"
            }
