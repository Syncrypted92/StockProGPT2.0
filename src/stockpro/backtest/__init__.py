"""Walk-forward directional options backtest with spread / TP-SL proxies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from stockpro.features import FEATURE_COLUMNS, add_forward_labels, add_indicators
from stockpro.models import build_model
from stockpro.options import select_contract, synthetic_candidates_for_backtest
from stockpro.signals import Signal, score_to_signal


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0


def _simulate_premium_path(
    closes: pd.Series,
    entry_ts: pd.Timestamp,
    entry_prem: float,
    spot: float,
    delta: float,
    *,
    horizon: int,
    profit_target_pct: float,
    stop_loss_pct: float,
    exit_days_before_expiry: int,
    dte: int,
    spread_penalty_pct: float,
    signal_side: int,
) -> tuple[float, str]:
    """Day-by-day option premium proxy with TP/SL/time exits."""
    if entry_ts not in closes.index:
        return entry_prem, "missing"
    loc = closes.index.get_loc(entry_ts)
    if isinstance(loc, slice):
        return entry_prem, "missing"
    prem = entry_prem
    side = 1 if signal_side >= 0 else -1
    max_i = min(horizon, len(closes) - loc - 1)
    for i in range(1, max_i + 1):
        prev = float(closes.iloc[loc + i - 1])
        cur = float(closes.iloc[loc + i])
        if prev <= 0:
            continue
        und_ret = (cur / prev - 1.0) * side
        prem_frac = max(prem / spot, 1e-4)
        day_opt = float(np.clip(delta * und_ret / prem_frac, -0.45, 0.80))
        prem = prem * (1.0 + day_opt)
        ret = prem / entry_prem - 1.0
        if ret >= profit_target_pct:
            exit_p = prem * (1 - spread_penalty_pct / 2)
            return exit_p, "profit_target"
        if ret <= -stop_loss_pct:
            exit_p = prem * (1 - spread_penalty_pct / 2)
            return exit_p, "stop_loss"
        remaining_dte = dte - i
        if exit_days_before_expiry >= 0 and remaining_dte <= exit_days_before_expiry:
            exit_p = prem * (1 - spread_penalty_pct / 2)
            return exit_p, "time_stop"
    exit_p = prem * (1 - spread_penalty_pct / 2)
    return exit_p, "horizon"


def run_walk_forward_backtest(
    frames: dict[str, pd.DataFrame],
    initial_equity: float = 100_000.0,
    spread_penalty_pct: float = 0.04,
    option_premium_pct_of_spot: float = 0.02,
    train_days: int = 504,
    test_days: int = 63,
    horizon: int = 5,
    flat_threshold: float = 0.005,
    probability_threshold: float = 0.55,
    max_risk_per_trade_pct: float = 0.01,
    max_notional_per_trade: float = 500.0,
    max_contracts: int = 1,
    *,
    spy_close: pd.Series | None = None,
    feature_columns: list[str] | None = None,
    binary_labels: bool = False,
    model_params: dict[str, Any] | None = None,
    profit_target_pct: float = 0.40,
    stop_loss_pct: float = 0.25,
    exit_days_before_expiry: int = 2,
    min_dte: int = 7,
    max_dte: int = 21,
    use_path_exits: bool = True,
) -> BacktestResult:
    """
    Walk-forward: train on trailing train_days, trade next test_days.
    Option PnL uses a day-path premium proxy with TP/SL/time stops when enabled.
    """
    feat_cols = list(feature_columns or FEATURE_COLUMNS)
    spy = spy_close
    if spy is None and "SPY" in frames:
        spy = frames["SPY"]["Close"]

    any_frame = next(iter(frames.values()))
    dates = list(any_frame.index)
    if len(dates) < train_days + test_days:
        train_days = max(60, len(dates) // 2)
        test_days = max(20, len(dates) // 5)

    trade_rows: list[dict[str, Any]] = []
    equity = initial_equity
    equity_rows: list[dict[str, Any]] = []
    target_dte = int((min_dte + max_dte) / 2)

    start = train_days
    while start + test_days <= len(dates):
        train_slice = dates[start - train_days : start]
        test_slice = dates[start : start + test_days]

        X_list = []
        y_list = []
        for _ticker, ohlcv in frames.items():
            panel = add_forward_labels(
                add_indicators(ohlcv, spy_close=spy),
                horizon=horizon,
                flat_threshold=flat_threshold,
                binary=binary_labels,
            )
            sub = panel.loc[panel.index.isin(train_slice)].dropna(
                subset=feat_cols + ["direction"]
            )
            if sub.empty:
                continue
            X_list.append(sub[feat_cols])
            y_list.append(sub["direction"])

        if not X_list:
            start += test_days
            continue

        X_train = pd.concat(X_list)
        y_train = pd.concat(y_list)
        params = dict(model_params or {})
        model: Pipeline = build_model(**params)
        model.fit(X_train, y_train)

        for ticker, ohlcv in frames.items():
            panel = add_indicators(ohlcv, spy_close=spy)
            panel = add_forward_labels(
                panel, horizon=horizon, flat_threshold=flat_threshold, binary=binary_labels
            )
            closes = panel["Close"]
            for ts in test_slice:
                if ts not in panel.index:
                    continue
                row = panel.loc[[ts]]
                if row[feat_cols].isna().any(axis=None):
                    continue
                fwd = row["forward_return"].iloc[0]
                if pd.isna(fwd):
                    continue

                sig = score_to_signal(
                    model,
                    row[feat_cols],
                    ticker,
                    probability_threshold=probability_threshold,
                )
                if sig.signal == Signal.FLAT:
                    continue

                spot = float(row["Close"].iloc[0])
                opt_type = "call" if sig.signal == Signal.BULLISH else "put"
                as_of = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
                cands = synthetic_candidates_for_backtest(
                    ticker,
                    spot,
                    opt_type,
                    as_of,
                    premium_pct=option_premium_pct_of_spot,
                    target_dte=target_dte,
                )
                contract, _notes = select_contract(
                    cands,
                    spot=spot,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    min_open_interest=0,
                    min_option_volume=0,
                    max_spread_pct=0.5,
                )
                if contract is None:
                    continue

                entry = contract.mid * (1 + spread_penalty_pct / 2)
                delta = abs(contract.delta or 0.4)
                side = 1 if sig.signal == Signal.BULLISH else -1

                if use_path_exits:
                    exit_prem, exit_reason = _simulate_premium_path(
                        closes,
                        pd.Timestamp(ts),
                        entry,
                        spot,
                        delta,
                        horizon=horizon,
                        profit_target_pct=profit_target_pct,
                        stop_loss_pct=stop_loss_pct,
                        exit_days_before_expiry=exit_days_before_expiry,
                        dte=int(contract.dte),
                        spread_penalty_pct=spread_penalty_pct,
                        signal_side=side,
                    )
                else:
                    prem_frac = entry / spot
                    approx_opt_ret = float(
                        np.clip(delta * (fwd * side) / max(prem_frac, 1e-4), -1.0, 5.0)
                    )
                    approx_opt_ret = float(
                        np.clip(approx_opt_ret, -stop_loss_pct, profit_target_pct)
                    )
                    exit_prem = entry * (1 + approx_opt_ret) * (1 - spread_penalty_pct / 2)
                    exit_reason = "horizon"

                pnl_per_contract = (exit_prem - entry) * 100
                risk_budget = min(equity * max_risk_per_trade_pct, max_notional_per_trade)
                qty = int(risk_budget // (entry * 100))
                qty = max(0, min(qty, max_contracts))
                if qty < 1:
                    continue

                pnl = pnl_per_contract * qty
                equity += pnl
                trade_rows.append(
                    {
                        "date": str(as_of),
                        "ticker": ticker,
                        "signal": sig.signal.value,
                        "contract": contract.symbol,
                        "qty": qty,
                        "entry": entry,
                        "exit": exit_prem,
                        "pnl": pnl,
                        "equity": equity,
                        "confidence": sig.confidence,
                        "exit_reason": exit_reason,
                    }
                )
                equity_rows.append({"date": str(as_of), "equity": equity})

        start += test_days

    trades = pd.DataFrame(trade_rows)
    curve = pd.DataFrame(equity_rows)
    if curve.empty:
        curve = pd.DataFrame([{"date": str(date.today()), "equity": initial_equity}])

    metrics = {
        "final_equity": float(curve["equity"].iloc[-1]),
        "total_return": float(curve["equity"].iloc[-1] / initial_equity - 1),
        "n_trades": int(len(trades)),
        "max_drawdown": _max_drawdown(curve["equity"]),
        "win_rate": float((trades["pnl"] > 0).mean()) if len(trades) else 0.0,
        "expectancy": float(trades["pnl"].mean()) if len(trades) else 0.0,
        "profit_factor": (
            float(
                trades.loc[trades["pnl"] > 0, "pnl"].sum()
                / abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
            )
            if len(trades) and (trades["pnl"] < 0).any()
            else (float("inf") if len(trades) and (trades["pnl"] > 0).any() else 0.0)
        ),
    }
    return BacktestResult(equity_curve=curve, trades=trades, metrics=metrics)


def _trade_metrics(pnls: list[float], initial_equity: float) -> dict[str, float]:
    if not pnls:
        return {
            "final_equity": float(initial_equity),
            "total_return": 0.0,
            "n_trades": 0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
        }
    eq = initial_equity
    curve = []
    for p in pnls:
        eq += p
        curve.append(eq)
    s = pd.Series(curve)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    return {
        "final_equity": float(eq),
        "total_return": float(eq / initial_equity - 1),
        "n_trades": int(len(pnls)),
        "max_drawdown": _max_drawdown(s),
        "win_rate": float(len(wins) / len(pnls)),
        "expectancy": float(sum(pnls) / len(pnls)),
        "profit_factor": float(pf) if pf != float("inf") else float("inf"),
    }


def _collect_signal_events(
    frames: dict[str, pd.DataFrame],
    *,
    horizon: int,
    flat_threshold: float,
    train_days: int,
    test_days: int,
    spy_close: pd.Series | None,
    feature_columns: list[str] | None,
    binary_labels: bool,
    model_params: dict[str, Any] | None,
    spread_penalty_pct: float = 0.04,
    option_premium_pct_of_spot: float = 0.02,
    max_risk_per_trade_pct: float = 0.01,
    max_notional_per_trade: float = 500.0,
    max_contracts: int = 1,
    target_dte: int = 14,
) -> list[dict[str, Any]]:
    """One walk-forward pass; keep all directional signals with confidence for offline grids."""
    feat_cols = list(feature_columns or FEATURE_COLUMNS)
    spy = spy_close
    if spy is None and "SPY" in frames:
        spy = frames["SPY"]["Close"]

    any_frame = next(iter(frames.values()))
    dates = list(any_frame.index)
    if len(dates) < train_days + test_days:
        train_days = max(60, len(dates) // 2)
        test_days = max(20, len(dates) // 5)

    events: list[dict[str, Any]] = []
    start = train_days
    while start + test_days <= len(dates):
        train_slice = dates[start - train_days : start]
        test_slice = dates[start : start + test_days]

        X_list, y_list = [], []
        for _ticker, ohlcv in frames.items():
            panel = add_forward_labels(
                add_indicators(ohlcv, spy_close=spy),
                horizon=horizon,
                flat_threshold=flat_threshold,
                binary=binary_labels,
            )
            sub = panel.loc[panel.index.isin(train_slice)].dropna(
                subset=feat_cols + ["direction"]
            )
            if sub.empty:
                continue
            X_list.append(sub[feat_cols])
            y_list.append(sub["direction"])
        if not X_list:
            start += test_days
            continue

        model: Pipeline = build_model(**dict(model_params or {}))
        model.fit(pd.concat(X_list), pd.concat(y_list))

        for ticker, ohlcv in frames.items():
            panel = add_indicators(ohlcv, spy_close=spy)
            panel = add_forward_labels(
                panel, horizon=horizon, flat_threshold=flat_threshold, binary=binary_labels
            )
            closes = panel["Close"]
            for ts in test_slice:
                if ts not in panel.index:
                    continue
                row = panel.loc[[ts]]
                if row[feat_cols].isna().any(axis=None):
                    continue
                if pd.isna(row["forward_return"].iloc[0]):
                    continue
                sig = score_to_signal(model, row[feat_cols], ticker, probability_threshold=0.0)
                if sig.signal == Signal.FLAT:
                    continue
                spot = float(row["Close"].iloc[0])
                opt_type = "call" if sig.signal == Signal.BULLISH else "put"
                as_of = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
                cands = synthetic_candidates_for_backtest(
                    ticker,
                    spot,
                    opt_type,
                    as_of,
                    premium_pct=option_premium_pct_of_spot,
                    target_dte=target_dte,
                )
                contract = cands[0]
                entry = contract.mid * (1 + spread_penalty_pct / 2)
                events.append(
                    {
                        "ts": pd.Timestamp(ts),
                        "ticker": ticker,
                        "confidence": float(sig.confidence),
                        "side": 1 if sig.signal == Signal.BULLISH else -1,
                        "entry": float(entry),
                        "spot": spot,
                        "delta": abs(contract.delta or 0.4),
                        "closes": closes,
                        "dte": int(contract.dte),
                        "horizon": horizon,
                        "spread_penalty_pct": spread_penalty_pct,
                        "max_risk_per_trade_pct": max_risk_per_trade_pct,
                        "max_notional_per_trade": max_notional_per_trade,
                        "max_contracts": max_contracts,
                    }
                )
        start += test_days
    return events


def optimize_trade_layer(
    frames: dict[str, pd.DataFrame],
    *,
    horizon: int = 5,
    flat_threshold: float = 0.008,
    train_days: int = 180,
    test_days: int = 30,
    spy_close: pd.Series | None = None,
    feature_columns: list[str] | None = None,
    binary_labels: bool = False,
    model_params: dict[str, Any] | None = None,
    min_trades: int = 40,
    verbose: bool = True,
    initial_equity: float = 100_000.0,
) -> dict[str, Any]:
    """Grid-search threshold / TP / SL / DTE using one cached walk-forward pass."""
    if verbose:
        print("[trade] collecting walk-forward signals (one model pass)...")
    events = _collect_signal_events(
        frames,
        horizon=horizon,
        flat_threshold=flat_threshold,
        train_days=train_days,
        test_days=test_days,
        spy_close=spy_close,
        feature_columns=feature_columns,
        binary_labels=binary_labels,
        model_params=model_params,
    )
    if verbose:
        print(f"[trade] cached {len(events)} raw signals")

    thresholds = [0.55, 0.60, 0.65, 0.70]
    tps = [0.30, 0.40, 0.50]
    sls = [0.20, 0.25, 0.30]
    dte_bands = [(7, 21), (14, 35)]

    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for thr in thresholds:
        for tp in tps:
            for sl in sls:
                if sl >= tp:
                    continue
                for min_dte, max_dte in dte_bands:
                    target_dte = int((min_dte + max_dte) / 2)
                    equity = initial_equity
                    pnls: list[float] = []
                    for ev in events:
                        if ev["confidence"] < thr:
                            continue
                        # Refresh synthetic dte for time-stop behaviour
                        exit_prem, _reason = _simulate_premium_path(
                            ev["closes"],
                            ev["ts"],
                            ev["entry"],
                            ev["spot"],
                            ev["delta"],
                            horizon=ev["horizon"],
                            profit_target_pct=tp,
                            stop_loss_pct=sl,
                            exit_days_before_expiry=2,
                            dte=target_dte,
                            spread_penalty_pct=ev["spread_penalty_pct"],
                            signal_side=ev["side"],
                        )
                        pnl_per = (exit_prem - ev["entry"]) * 100
                        risk_budget = min(
                            equity * ev["max_risk_per_trade_pct"],
                            ev["max_notional_per_trade"],
                        )
                        qty = int(risk_budget // (ev["entry"] * 100))
                        qty = max(0, min(qty, ev["max_contracts"]))
                        if qty < 1:
                            continue
                        pnl = pnl_per * qty
                        equity += pnl
                        pnls.append(pnl)

                    m = _trade_metrics(pnls, initial_equity)
                    trial = {
                        "probability_threshold": thr,
                        "profit_target_pct": tp,
                        "stop_loss_pct": sl,
                        "min_dte": min_dte,
                        "max_dte": max_dte,
                        **m,
                    }
                    trials.append(trial)
                    if verbose:
                        pf = m["profit_factor"]
                        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
                        print(
                            f"[trade] thr={thr:.2f} tp={tp:.2f} sl={sl:.2f} "
                            f"dte={min_dte}-{max_dte} "
                            f"E={m['expectancy']:.2f} PF={pf_s} "
                            f"n={m['n_trades']} wr={m['win_rate']:.2f}"
                        )
                    if m["n_trades"] < min_trades:
                        continue
                    score = (
                        float(m["expectancy"]),
                        float(m["profit_factor"]) if m["profit_factor"] != float("inf") else 99.0,
                        float(m["win_rate"]),
                    )
                    if best is None or score > best["_score"]:
                        best = {**trial, "_score": score}

    trials_sorted = sorted(
        trials,
        key=lambda t: (
            t["expectancy"] if t["n_trades"] >= min_trades else -1e9,
            t["profit_factor"] if t["profit_factor"] != float("inf") else 0.0,
        ),
        reverse=True,
    )
    winner = {k: v for k, v in (best or trials_sorted[0]).items() if k != "_score"}
    return {"best": winner, "trials": trials_sorted[:15], "n_trials": len(trials)}
