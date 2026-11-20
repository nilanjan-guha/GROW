# Institutional-Grade Indian F&O Intraday Intelligence & Momentum Scanner

Advanced multi-layer quantitative intraday stock discovery, momentum prediction, institutional order flow and Smart Money Concepts (SMC) ranking terminal for the National Stock Exchange (NSE) F&O universe.

## Core Features
1. **True Daily Session-Reset VWAP & Opening Range**: Accurately isolates each trading date, calculating intraday VWAP from the market open without multi-day volume contamination.
2. **15 Institutional Strategies**: Gap-and-Go, Gap-Fill Reversal, Opening Range Breakout/Breakdown, VWAP Reclaim/Rejection, EMA 9/21 Momentum, Flat-Open Sudden Momentum, Liquidity Sweeps, ICT Fair Value Gaps, and Demand/Supply Order Blocks.
3. **Anomaly Detection**: RVOL anomalies, sudden volume acceleration, and candle range expansion.
4. **Live Catalyst Intelligence**: Real-time corporate news & event sentiment matching.
5. **Groq AI Integration**: Institutional trade rationale generation and risk ratings.
6. **Persistent Trade Learning**: Full SQLite tracking of every signal, MFE, MAE, and automated outcome calculation.

## Quickstart
```bash
pip install -r requirements.txt
python groww_terminal_app.py
```
Open `http://127.0.0.1:5000` in your browser.
