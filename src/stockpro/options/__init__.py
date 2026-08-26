"""Universe liquidity filters and option contract selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class LiquidityReport:
    ticker: str
    passed: bool
    last_price: float
    avg_volume: float
    avg_dollar_volume: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class ContractCandidate:
    symbol: str
    underlying: str
    option_type: str  # call | put
    strike: float
    expiration: date
    dte: int
    bid: float
    ask: float
    mid: float
    spread_pct: float
    open_interest: int
    volume: int
    delta: float | None = None


def filter_underlying_liquidity(
    ticker: str,
    ohlcv: pd.DataFrame,
    min_price: float = 10.0,
    min_avg_volume: float = 2_000_000,
    min_avg_dollar_volume: float = 50_000_000,
    volume_lookback_days: int = 20,
) -> LiquidityReport:
    tail = ohlcv.tail(volume_lookback_days)
    last_price = float(tail["Close"].iloc[-1])
    avg_volume = float(tail["Volume"].mean())
    avg_dollar = float((tail["Close"] * tail["Volume"]).mean())
    reasons: list[str] = []

    if last_price < min_price:
        reasons.append(f"price {last_price:.2f} < min_price {min_price}")
    if avg_volume < min_avg_volume:
        reasons.append(f"avg_volume {avg_volume:.0f} < {min_avg_volume}")
    if avg_dollar < min_avg_dollar_volume:
        reasons.append(f"avg_dollar_volume {avg_dollar:.0f} < {min_avg_dollar_volume}")

    return LiquidityReport(
        ticker=ticker,
        passed=len(reasons) == 0,
        last_price=last_price,
        avg_volume=avg_volume,
        avg_dollar_volume=avg_dollar,
        reasons=reasons,
    )


def _to_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return pd.Timestamp(value).date()


def contracts_from_alpaca_payload(
    contracts: Iterable[Any],
    underlying: str,
    option_type: str,
    spot: float,
    today: date | None = None,
) -> list[ContractCandidate]:
    """Normalize Alpaca OptionContract objects or dict-like rows."""
    today = today or date.today()
    out: list[ContractCandidate] = []
    for c in contracts:
        if hasattr(c, "symbol"):
            symbol = c.symbol
            strike = float(c.strike_price)
            expiration = _to_date(c.expiration_date)
            oi_raw = getattr(c, "open_interest", None)
            oi = int(oi_raw) if oi_raw is not None else -1  # -1 = unknown
            close = float(getattr(c, "close_price", 0) or 0)
            bid = close
            ask = close
            delta = None
        else:
            symbol = c["symbol"]
            strike = float(c["strike"])
            expiration = _to_date(c["expiration"])
            oi = int(c.get("open_interest", 0) or 0)
            bid = float(c.get("bid", 0) or 0)
            ask = float(c.get("ask", 0) or 0)
            delta = c.get("delta")
            if delta is not None:
                delta = float(delta)

        mid = (bid + ask) / 2 if bid and ask else max(bid, ask, 0.0)
        spread_pct = ((ask - bid) / mid) if mid > 0 and ask >= bid else 1.0
        dte = (expiration - today).days
        out.append(
            ContractCandidate(
                symbol=symbol,
                underlying=underlying,
                option_type=option_type,
                strike=strike,
                expiration=expiration,
                dte=dte,
                bid=bid,
                ask=ask,
                mid=mid,
                spread_pct=spread_pct,
                open_interest=oi,
                volume=int(c.get("volume", 0) if isinstance(c, dict) else getattr(c, "volume", 0) or 0),
                delta=delta,
            )
        )
    return out


def select_contract(
    candidates: list[ContractCandidate],
    spot: float,
    min_dte: int = 14,
    max_dte: int = 45,
    target_delta_min: float = 0.30,
    target_delta_max: float = 0.50,
    otm_pct_min: float = 0.01,
    otm_pct_max: float = 0.08,
    min_open_interest: int = 500,
    min_option_volume: int = 50,
    max_spread_pct: float = 0.08,
    max_mid_price: float | None = 5.0,
    min_mid_price: float | None = None,
    target_delta: float = 0.40,
    rank: str = "cheap",
    target_dte: int | None = None,
) -> tuple[ContractCandidate | None, list[str]]:
    """Pick the best liquid contract; return (contract, rejection_notes).

    rank:
      - "cheap": prefer tighter spread then lower mid (legacy swing lane)
      - "atm": prefer near target delta / ATM, then tight spread (0DTE / directional)
    """
    notes: list[str] = []
    pool: list[ContractCandidate] = []

    for c in candidates:
        if c.dte < min_dte or c.dte > max_dte:
            continue
        # Liquidity: require OI or volume; unknown OI (-1) allowed if quoted
        if c.open_interest >= 0:
            if c.open_interest < min_open_interest and c.volume < min_option_volume:
                continue
        if c.mid <= 0:
            continue
        if min_mid_price is not None and c.mid < min_mid_price:
            continue
        if max_mid_price is not None and c.mid > max_mid_price:
            continue
        # Require a real two-sided quote for live-quality selection
        if c.bid <= 0 or c.ask <= 0 or c.ask < c.bid:
            continue
        if c.spread_pct > max_spread_pct:
            continue

        # Delta band if available; else OTM % of spot (negative otm = slight ITM)
        if c.delta is not None:
            abs_delta = abs(c.delta)
            if abs_delta < target_delta_min or abs_delta > target_delta_max:
                continue
        else:
            if c.option_type == "call":
                otm = (c.strike - spot) / spot
            else:
                otm = (spot - c.strike) / spot
            if otm < otm_pct_min or otm > otm_pct_max:
                continue

        pool.append(c)

    if not pool:
        notes.append("no contracts passed liquidity/DTE/delta/premium filters")
        return None, notes

    def _otm(c: ContractCandidate) -> float:
        if c.option_type == "call":
            return (c.strike - spot) / max(spot, 1e-9)
        return (spot - c.strike) / max(spot, 1e-9)

    target_dte_mid = float(target_dte) if target_dte is not None else (min_dte + max_dte) / 2.0

    def score_atm(c: ContractCandidate) -> tuple:
        if c.delta is not None:
            moneyness = abs(abs(c.delta) - target_delta)
        else:
            moneyness = abs(_otm(c))
        oi_rank = -c.open_interest if c.open_interest >= 0 else 0
        dte_rank = abs(c.dte - target_dte_mid)
        return (moneyness, dte_rank, c.spread_pct, -c.volume, oi_rank)

    def score_cheap(c: ContractCandidate) -> tuple:
        oi_rank = -c.open_interest if c.open_interest >= 0 else 0
        return (c.spread_pct, c.mid, abs(c.dte - target_dte_mid), oi_rank)

    scorer = score_atm if rank == "atm" else score_cheap
    best = sorted(pool, key=scorer)[0]
    notes.append(
        f"selected {best.symbol} mid={best.mid:.2f} strike={best.strike:.0f} "
        f"dte={best.dte} from {len(pool)} candidates (rank={rank})"
    )
    return best, notes


def synthetic_candidates_for_backtest(
    underlying: str,
    spot: float,
    option_type: str,
    as_of: date,
    premium_pct: float = 0.005,
    target_dte: int = 30,
) -> list[ContractCandidate]:
    """Create a liquid-looking synthetic contract for backtests without OPRA."""
    expiration = as_of + timedelta(days=int(target_dte))
    # Near ATM for short-dated; mild OTM for longer swing synthetics
    if target_dte <= 1:
        strike = round(spot)
    elif option_type == "call":
        strike = round(spot * 1.05, 2)
    else:
        strike = round(spot * 0.95, 2)
    # Cap synthetic premium so tiny risk budgets can still size 1 contract
    mid = float(np.clip(spot * premium_pct, 0.50, 4.00))
    spread = mid * 0.04
    return [
        ContractCandidate(
            symbol=f"{underlying}_SYN_{option_type}_{strike}_{expiration.isoformat()}",
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            expiration=expiration,
            dte=int(target_dte),
            bid=mid - spread / 2,
            ask=mid + spread / 2,
            mid=mid,
            spread_pct=spread / mid,
            open_interest=5000,
            volume=1000,
            delta=0.40 if option_type == "call" else -0.40,
        )
    ]
