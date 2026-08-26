"""Alpaca broker client with paper/live switch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from stockpro.config import Settings


@dataclass
class OrderRequest:
    symbol: str
    qty: int
    side: str  # buy | sell
    limit_price: float
    time_in_force: str = "day"
    position_intent: str | None = None  # buy_to_open / sell_to_close


@dataclass
class OrderResult:
    submitted: bool
    dry_run: bool
    order_id: str | None
    symbol: str
    qty: int
    side: str
    limit_price: float
    status: str
    raw: Any = None


class AlpacaBroker:
    """Thin wrapper around alpaca-py. Works without credentials in dry-run/mock mode."""

    def __init__(self, settings: Settings, dry_run: bool = True):
        self.settings = settings
        self.dry_run = dry_run
        self._trading = None
        self._option_data = None

    @property
    def paper(self) -> bool:
        return self.settings.paper

    def connect(self) -> None:
        if self._trading is not None:
            return
        if self.dry_run and not self.settings.alpaca_api_key:
            return
        self.settings.require_broker_credentials()
        from alpaca.trading.client import TradingClient

        self._trading = TradingClient(
            api_key=self.settings.alpaca_api_key,
            secret_key=self.settings.alpaca_secret_key,
            paper=self.settings.paper,
        )
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient

            self._option_data = OptionHistoricalDataClient(
                api_key=self.settings.alpaca_api_key,
                secret_key=self.settings.alpaca_secret_key,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[broker] option data client unavailable: {exc}")
            self._option_data = None

    def get_account(self) -> dict[str, Any]:
        if self._trading is None:
            return {
                "equity": 100000.0,
                "buying_power": 100000.0,
                "cash": 100000.0,
                "status": "ACTIVE",
                "mock": True,
                "options_approved_level": None,
                "options_trading_level": None,
            }
        acct = self._trading.get_account()
        return {
            "equity": float(acct.equity),
            "buying_power": float(acct.buying_power),
            "cash": float(acct.cash),
            "status": str(acct.status),
            "mock": False,
            "options_approved_level": getattr(acct, "options_approved_level", None),
            "options_trading_level": getattr(acct, "options_trading_level", None),
        }

    def list_positions(self) -> list[dict[str, Any]]:
        if self._trading is None:
            return []
        positions = self._trading.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": int(float(p.qty)),
                "side": str(p.side),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(getattr(p, "current_price", 0) or 0),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "asset_class": str(getattr(p, "asset_class", "")),
            }
            for p in positions
        ]

    def get_option_contracts(
        self,
        underlying: str,
        option_type: str,
        min_dte: int,
        max_dte: int,
    ) -> list[Any]:
        """Fetch option contracts in the DTE window (paginated)."""
        if self._trading is None:
            return []

        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import AssetStatus, ContractType

        today = date.today()
        exp_gte = today + timedelta(days=min_dte)
        exp_lte = today + timedelta(days=max_dte)
        collected: list[Any] = []
        page_token: str | None = None

        try:
            while True:
                req = GetOptionContractsRequest(
                    underlying_symbols=[underlying],
                    status=AssetStatus.ACTIVE,
                    type=ContractType.CALL if option_type == "call" else ContractType.PUT,
                    expiration_date_gte=exp_gte,
                    expiration_date_lte=exp_lte,
                    limit=1000,
                    page_token=page_token,
                )
                resp = self._trading.get_option_contracts(req)
                batch = list(getattr(resp, "option_contracts", None) or [])
                collected.extend(batch)
                page_token = getattr(resp, "next_page_token", None)
                if not page_token or not batch:
                    break
                if len(collected) >= 5000:
                    break

            filtered = []
            for c in collected:
                exp = c.expiration_date
                if hasattr(exp, "date"):
                    exp_d = exp.date()
                else:
                    exp_d = exp if isinstance(exp, date) else date.fromisoformat(str(exp)[:10])
                dte = (exp_d - today).days
                if min_dte <= dte <= max_dte:
                    filtered.append(c)
            return filtered
        except Exception as exc:  # noqa: BLE001
            print(f"[broker] option contracts error for {underlying}: {exc}")
            return []

    def get_option_quotes(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """Latest bid/ask for option symbols. Empty dict if data client unavailable."""
        if not symbols or self._option_data is None:
            return {}
        from alpaca.data.requests import OptionLatestQuoteRequest

        out: dict[str, dict[str, float]] = {}
        # Alpaca accepts batch; chunk to be safe
        chunk_size = 100
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            try:
                resp = self._option_data.get_option_latest_quote(
                    OptionLatestQuoteRequest(symbol_or_symbols=chunk)
                )
                # resp may be dict-like QuoteSet
                items = resp.items() if hasattr(resp, "items") else []
                for sym, quote in items:
                    bid = float(getattr(quote, "bid_price", 0) or 0)
                    ask = float(getattr(quote, "ask_price", 0) or 0)
                    out[str(sym)] = {"bid": bid, "ask": ask}
            except Exception as exc:  # noqa: BLE001
                print(f"[broker] quote error: {exc}")
        return out

    def submit_option_order(self, order: OrderRequest) -> OrderResult:
        if self.dry_run or self._trading is None:
            return OrderResult(
                submitted=False,
                dry_run=True,
                order_id=None,
                symbol=order.symbol,
                qty=order.qty,
                side=order.side,
                limit_price=order.limit_price,
                status="dry_run",
            )

        if "_SYN_" in order.symbol:
            raise RuntimeError(f"Refusing to submit synthetic symbol: {order.symbol}")

        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        req_kwargs: dict[str, Any] = {
            "symbol": order.symbol,
            "qty": order.qty,
            "side": side,
            "time_in_force": TimeInForce.DAY,
            "limit_price": round(order.limit_price, 2),
        }
        if order.position_intent:
            req_kwargs["position_intent"] = order.position_intent

        try:
            req = LimitOrderRequest(**req_kwargs)
        except Exception:
            # Older SDKs may not accept position_intent
            req_kwargs.pop("position_intent", None)
            req = LimitOrderRequest(**req_kwargs)

        submitted = self._trading.submit_order(req)
        return OrderResult(
            submitted=True,
            dry_run=False,
            order_id=str(submitted.id),
            symbol=order.symbol,
            qty=order.qty,
            side=order.side,
            limit_price=order.limit_price,
            status=str(submitted.status),
            raw=submitted,
        )

    def close_position(self, symbol: str, qty: int | None = None) -> OrderResult:
        """Sell-to-close a long option (or equity) position at market via close API."""
        if self.dry_run or self._trading is None:
            return OrderResult(
                submitted=False,
                dry_run=True,
                order_id=None,
                symbol=symbol,
                qty=qty or 0,
                side="sell",
                limit_price=0.0,
                status="dry_run",
            )
        if "_SYN_" in symbol:
            raise RuntimeError(f"Refusing to close synthetic symbol: {symbol}")

        try:
            if qty is None:
                raw = self._trading.close_position(symbol)
            else:
                from alpaca.trading.requests import ClosePositionRequest

                raw = self._trading.close_position(
                    symbol, close_options=ClosePositionRequest(qty=str(qty))
                )
            return OrderResult(
                submitted=True,
                dry_run=False,
                order_id=str(getattr(raw, "id", None)),
                symbol=symbol,
                qty=qty or 0,
                side="sell",
                limit_price=0.0,
                status=str(getattr(raw, "status", "submitted")),
                raw=raw,
            )
        except Exception as exc:  # noqa: BLE001
            return OrderResult(
                submitted=False,
                dry_run=False,
                order_id=None,
                symbol=symbol,
                qty=qty or 0,
                side="sell",
                limit_price=0.0,
                status=f"error:{exc}",
            )

    def smoke_test(self) -> dict[str, Any]:
        """Account + SPY chain probe for connectivity checks."""
        self.connect()
        account = self.get_account()
        contracts = self.get_option_contracts("SPY", "call", 14, 45)
        sample = None
        if contracts:
            c0 = contracts[0]
            sample = {
                "symbol": c0.symbol,
                "expiration": str(c0.expiration_date),
                "strike": float(c0.strike_price),
                "open_interest": getattr(c0, "open_interest", None),
            }
            quotes = self.get_option_quotes([c0.symbol])
            sample["quote"] = quotes.get(c0.symbol)
        return {
            "paper": self.paper,
            "dry_run": self.dry_run,
            "account": account,
            "spy_call_contracts": len(contracts),
            "sample_contract": sample,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
