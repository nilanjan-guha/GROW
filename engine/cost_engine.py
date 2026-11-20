class IndianMarketCosts:
    """
    Realistic Indian Equity and F&O Trading Transaction Cost Engine.
    Computes statutory, regulatory, and brokerage charges as per SEBI, NSE, and broker standards:
    - Brokerage: ₹20 per executed order (or 0.05% whichever is lower) = ₹40 for round trip
    - Securities Transaction Tax (STT):
        * Equity Intraday: 0.025% on sell side only
        * Futures: 0.02% on sell side only
        * Options: 0.1% on premium on sell side
    - Exchange Turnover Charges:
        * NSE Equity Intraday: 0.00345% (both sides)
        * NSE Futures: 0.002% (both sides)
    - SEBI Turnover Charges: ₹10 per crore (0.0001% both sides)
    - Stamp Duty:
        * Equity Intraday: 0.003% on buy side
        * Futures: 0.002% on buy side
    - GST: 18% on (Brokerage + Exchange Charges + SEBI Charges)
    - Estimated Slippage: 0.02% per leg on liquid F&O equities
    """

    @staticmethod
    def calculate_intraday_costs(entry_price: float, exit_price: float, quantity: int = 100, instrument_type: str = "EQUITY_INTRADAY") -> dict:
        entry = float(entry_price)
        exit_p = float(exit_price)
        qty = int(quantity)

        buy_turnover = entry * qty
        sell_turnover = exit_p * qty
        total_turnover = buy_turnover + sell_turnover

        # 1. Brokerage (Zerodha/Groww standard: min(₹20, 0.05% of turnover) per leg)
        buy_brokerage = min(20.0, buy_turnover * 0.0005)
        sell_brokerage = min(20.0, sell_turnover * 0.0005)
        total_brokerage = round(buy_brokerage + sell_brokerage, 2)

        # 2. STT (Securities Transaction Tax)
        if instrument_type == "EQUITY_INTRADAY":
            stt = round(sell_turnover * 0.00025, 2)  # 0.025% on sell
        elif instrument_type == "FUTURES":
            stt = round(sell_turnover * 0.00020, 2)  # 0.020% on sell
        else:
            stt = round(sell_turnover * 0.00025, 2)

        # 3. Exchange Turnover Charges
        if instrument_type == "EQUITY_INTRADAY":
            exch_charges = round(total_turnover * 0.0000345, 2)
        elif instrument_type == "FUTURES":
            exch_charges = round(total_turnover * 0.0000200, 2)
        else:
            exch_charges = round(total_turnover * 0.0000345, 2)

        # 4. SEBI Charges (₹10 per crore = 0.0001%)
        sebi_charges = round(total_turnover * 0.000001, 2)

        # 5. Stamp Duty (State Government - payable on buy turnover)
        if instrument_type == "EQUITY_INTRADAY":
            stamp_duty = round(buy_turnover * 0.00003, 2)  # 0.003%
        elif instrument_type == "FUTURES":
            stamp_duty = round(buy_turnover * 0.00002, 2)  # 0.002%
        else:
            stamp_duty = round(buy_turnover * 0.00003, 2)

        # 6. GST (18% on Brokerage + Exchange Charges + SEBI Charges)
        taxable_services = total_brokerage + exch_charges + sebi_charges
        gst = round(taxable_services * 0.18, 2)

        total_costs = round(total_brokerage + stt + exch_charges + sebi_charges + stamp_duty + gst, 2)

        # Cost per share
        cost_per_share = round(total_costs / qty, 2) if qty > 0 else 0.0

        return {
            "total_costs": total_costs,
            "cost_per_share": cost_per_share,
            "brokerage": total_brokerage,
            "stt": stt,
            "exchange_charges": exch_charges,
            "stamp_duty": stamp_duty,
            "sebi_charges": sebi_charges,
            "gst": gst,
            "buy_turnover": round(buy_turnover, 2),
            "sell_turnover": round(sell_turnover, 2),
            "total_turnover": round(total_turnover, 2)
        }
