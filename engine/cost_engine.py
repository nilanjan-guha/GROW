class IndianMarketCosts:
    """
    Realistic Indian Equity and F&O Trading Transaction Cost Engine.
    Computes statutory, regulatory, and brokerage charges as per SEBI, NSE, and broker standards:
    - Brokerage: ₹20 per executed order (or 0.05% whichever is lower) = ₹40 round trip
    - Securities Transaction Tax (STT / CTT):
        * Equity Intraday: 0.025% on sell side turnover
        * Futures: 0.02% on sell side turnover
        * Options: 0.1% on sell side premium turnover
    - Exchange Turnover Charges:
        * NSE Equity Intraday: 0.00345% (both sides)
        * NSE Futures: 0.002% (both sides)
        * NSE Options: 0.053% on premium turnover (both sides)
    - SEBI Turnover Charges: ₹10 per crore (0.0001% both sides)
    - Stamp Duty (State Government - buy turnover only):
        * Equity Intraday: 0.003%
        * Futures: 0.002%
        * Options: 0.003% on premium
    - GST: 18% on (Brokerage + Exchange Charges + SEBI Charges)
    - Estimated Slippage:
        * Equity Intraday: 0.02% per leg on turnover
        * Futures: 0.015% per leg on turnover
        * Options: 0.50% on option premium per leg
    """

    @staticmethod
    def calculate_equity_costs(entry_price: float, exit_price: float, quantity: int = 100, include_slippage: bool = True) -> dict:
        entry = float(entry_price)
        exit_p = float(exit_price)
        qty = max(1, int(quantity))

        buy_turnover = entry * qty
        sell_turnover = exit_p * qty
        total_turnover = buy_turnover + sell_turnover

        # 1. Brokerage (min(₹20, 0.05%) per order)
        buy_brokerage = min(20.0, buy_turnover * 0.0005)
        sell_brokerage = min(20.0, sell_turnover * 0.0005)
        total_brokerage = round(buy_brokerage + sell_brokerage, 2)

        # 2. STT: 0.025% on sell turnover
        stt = round(sell_turnover * 0.00025, 2)

        # 3. Exchange charges: 0.00345% on total turnover
        exch_charges = round(total_turnover * 0.0000345, 2)

        # 4. SEBI charges: 0.0001%
        sebi_charges = round(total_turnover * 0.000001, 2)

        # 5. Stamp duty: 0.003% on buy turnover
        stamp_duty = round(buy_turnover * 0.00003, 2)

        # 6. GST: 18% on taxable services
        gst = round((total_brokerage + exch_charges + sebi_charges) * 0.18, 2)

        # 7. Slippage: 0.02% per leg
        slippage = round(total_turnover * 0.0002, 2) if include_slippage else 0.0

        total_costs = round(total_brokerage + stt + exch_charges + sebi_charges + stamp_duty + gst + slippage, 2)
        cost_per_share = round(total_costs / qty, 2)

        return {
            "instrument": "EQUITY_INTRADAY",
            "quantity": qty,
            "total_costs": total_costs,
            "cost_per_share": cost_per_share,
            "brokerage": total_brokerage,
            "stt": stt,
            "exchange_charges": exch_charges,
            "stamp_duty": stamp_duty,
            "sebi_charges": sebi_charges,
            "gst": gst,
            "slippage": slippage,
            "buy_turnover": round(buy_turnover, 2),
            "sell_turnover": round(sell_turnover, 2),
            "total_turnover": round(total_turnover, 2)
        }

    @staticmethod
    def calculate_futures_costs(entry_price: float, exit_price: float, lot_size: int = 500, include_slippage: bool = True) -> dict:
        entry = float(entry_price)
        exit_p = float(exit_price)
        qty = max(1, int(lot_size))

        buy_turnover = entry * qty
        sell_turnover = exit_p * qty
        total_turnover = buy_turnover + sell_turnover

        # Brokerage: flat ₹20 per executed order = ₹40 round trip
        total_brokerage = 40.0

        # STT: 0.02% on sell turnover
        stt = round(sell_turnover * 0.00020, 2)

        # Exchange charges: 0.002% on total turnover
        exch_charges = round(total_turnover * 0.0000200, 2)

        # SEBI charges: 0.0001%
        sebi_charges = round(total_turnover * 0.000001, 2)

        # Stamp duty: 0.002% on buy turnover
        stamp_duty = round(buy_turnover * 0.00002, 2)

        # GST: 18% on taxable services
        gst = round((total_brokerage + exch_charges + sebi_charges) * 0.18, 2)

        # Slippage: 0.015% per leg
        slippage = round(total_turnover * 0.00015, 2) if include_slippage else 0.0

        total_costs = round(total_brokerage + stt + exch_charges + sebi_charges + stamp_duty + gst + slippage, 2)
        cost_per_share = round(total_costs / qty, 2)

        return {
            "instrument": "FUTURES",
            "quantity": qty,
            "total_costs": total_costs,
            "cost_per_share": cost_per_share,
            "brokerage": total_brokerage,
            "stt": stt,
            "exchange_charges": exch_charges,
            "stamp_duty": stamp_duty,
            "sebi_charges": sebi_charges,
            "gst": gst,
            "slippage": slippage,
            "buy_turnover": round(buy_turnover, 2),
            "sell_turnover": round(sell_turnover, 2),
            "total_turnover": round(total_turnover, 2)
        }

    @staticmethod
    def calculate_options_costs(entry_premium: float, exit_premium: float, lot_size: int = 500, include_slippage: bool = True) -> dict:
        entry = float(entry_premium)
        exit_p = float(exit_premium)
        qty = max(1, int(lot_size))

        buy_turnover = entry * qty
        sell_turnover = exit_p * qty
        total_turnover = buy_turnover + sell_turnover

        # Brokerage: flat ₹20 per executed order = ₹40 round trip
        total_brokerage = 40.0

        # STT: 0.1% on sell side option premium turnover
        stt = round(sell_turnover * 0.00100, 2)

        # Exchange charges: 0.053% on premium turnover
        exch_charges = round(total_turnover * 0.00053, 2)

        # SEBI charges: 0.0001%
        sebi_charges = round(total_turnover * 0.000001, 2)

        # Stamp duty: 0.003% on buy premium turnover
        stamp_duty = round(buy_turnover * 0.00003, 2)

        # GST: 18% on taxable services
        gst = round((total_brokerage + exch_charges + sebi_charges) * 0.18, 2)

        # Slippage: 0.5% of option premium (bid-ask spread)
        slippage = round(total_turnover * 0.005, 2) if include_slippage else 0.0

        total_costs = round(total_brokerage + stt + exch_charges + sebi_charges + stamp_duty + gst + slippage, 2)
        cost_per_share = round(total_costs / qty, 2)

        return {
            "instrument": "OPTIONS",
            "quantity": qty,
            "total_costs": total_costs,
            "cost_per_share": cost_per_share,
            "brokerage": total_brokerage,
            "stt": stt,
            "exchange_charges": exch_charges,
            "stamp_duty": stamp_duty,
            "sebi_charges": sebi_charges,
            "gst": gst,
            "slippage": slippage,
            "buy_turnover": round(buy_turnover, 2),
            "sell_turnover": round(sell_turnover, 2),
            "total_turnover": round(total_turnover, 2)
        }

    @staticmethod
    def calculate_multi_exit_costs(entry_price: float, exits: list, total_quantity: int = 100, instrument_type: str = "EQUITY_INTRADAY", include_slippage: bool = True) -> dict:
        """
        Accurately calculates total transaction friction for a trade with 1 entry order and N partial exit orders.
        Guarantees:
        - Exactly 1 entry leg (brokerage, stamp duty, entry slippage evaluated once across total_quantity).
        - Exactly 1 exit leg per partial exit (brokerage, STT, exit slippage evaluated per exit order/quantity).
        - Exchange turnover charges and SEBI charges evaluated on exact aggregate turnover.
        - GST computed on total brokerage + exchange charges + SEBI charges.
        - cost_per_share computed as total_costs / total_quantity.
        
        exits is a list of tuples/dicts: [(exit_price, quantity), ...] or [{'price': p, 'quantity': q}, ...]
        """
        entry = float(entry_price)
        tot_qty = max(1, int(total_quantity))
        inst = instrument_type.upper()

        clean_exits = []
        for e in exits:
            if isinstance(e, (list, tuple)):
                clean_exits.append((float(e[0]), int(e[1])))
            elif isinstance(e, dict):
                clean_exits.append((float(e.get("price", entry)), int(e.get("quantity", 0))))

        sum_exit_qty = sum(q for _, q in clean_exits)
        if sum_exit_qty <= 0:
            clean_exits = [(entry, tot_qty)]
            sum_exit_qty = tot_qty

        # 1. Entry side (evaluated once)
        buy_turnover = entry * tot_qty
        if "OPTION" in inst:
            buy_brokerage = 20.0
            stamp_duty = round(buy_turnover * 0.00003, 2)
            entry_slippage = round(buy_turnover * 0.005, 2) if include_slippage else 0.0
        elif "FUTURE" in inst:
            buy_brokerage = 20.0
            stamp_duty = round(buy_turnover * 0.00002, 2)
            entry_slippage = round(buy_turnover * 0.00015, 2) if include_slippage else 0.0
        else:
            buy_brokerage = min(20.0, buy_turnover * 0.0005)
            stamp_duty = round(buy_turnover * 0.00003, 2)
            entry_slippage = round(buy_turnover * 0.0002, 2) if include_slippage else 0.0

        # 2. Exit sides (evaluated per order)
        total_sell_turnover = 0.0
        total_sell_brokerage = 0.0
        total_stt = 0.0
        total_exit_slippage = 0.0

        for exit_p, q in clean_exits:
            q = max(1, int(q))
            sell_turnover = exit_p * q
            total_sell_turnover += sell_turnover

            if "OPTION" in inst:
                total_sell_brokerage += 20.0
                total_stt += round(sell_turnover * 0.00100, 2)
                if include_slippage:
                    total_exit_slippage += round(sell_turnover * 0.005, 2)
            elif "FUTURE" in inst:
                total_sell_brokerage += 20.0
                total_stt += round(sell_turnover * 0.00020, 2)
                if include_slippage:
                    total_exit_slippage += round(sell_turnover * 0.00015, 2)
            else:
                total_sell_brokerage += min(20.0, sell_turnover * 0.0005)
                total_stt += round(sell_turnover * 0.00025, 2)
                if include_slippage:
                    total_exit_slippage += round(sell_turnover * 0.0002, 2)

        total_turnover = buy_turnover + total_sell_turnover
        total_brokerage = round(buy_brokerage + total_sell_brokerage, 2)

        # 3. Exchange & regulatory charges
        if "OPTION" in inst:
            exch_charges = round(total_turnover * 0.00053, 2)
        elif "FUTURE" in inst:
            exch_charges = round(total_turnover * 0.0000200, 2)
        else:
            exch_charges = round(total_turnover * 0.0000345, 2)

        sebi_charges = round(total_turnover * 0.000001, 2)
        gst = round((total_brokerage + exch_charges + sebi_charges) * 0.18, 2)
        slippage = round(entry_slippage + total_exit_slippage, 2)

        total_costs = round(total_brokerage + total_stt + exch_charges + sebi_charges + stamp_duty + gst + slippage, 2)
        cost_per_share = round(total_costs / tot_qty, 2)

        return {
            "instrument": inst,
            "quantity": tot_qty,
            "total_costs": total_costs,
            "cost_per_share": cost_per_share,
            "brokerage": total_brokerage,
            "stt": round(total_stt, 2),
            "exchange_charges": exch_charges,
            "stamp_duty": stamp_duty,
            "sebi_charges": sebi_charges,
            "gst": gst,
            "slippage": slippage,
            "buy_turnover": round(buy_turnover, 2),
            "sell_turnover": round(total_sell_turnover, 2),
            "total_turnover": round(total_turnover, 2)
        }

    @staticmethod
    def calculate_intraday_costs(entry_price: float, exit_price: float, quantity: int = 100, instrument_type: str = "EQUITY_INTRADAY") -> dict:
        """Unified interface routing to specialized calculators"""
        inst = instrument_type.upper()
        if "OPTION" in inst:
            return IndianMarketCosts.calculate_options_costs(entry_price, exit_price, lot_size=quantity)
        elif "FUTURE" in inst:
            return IndianMarketCosts.calculate_futures_costs(entry_price, exit_price, lot_size=quantity)
        else:
            return IndianMarketCosts.calculate_equity_costs(entry_price, exit_price, quantity=quantity)
