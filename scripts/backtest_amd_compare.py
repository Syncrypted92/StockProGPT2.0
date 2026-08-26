"""Compare AMD 0DTE vs short-DTE (and full stack) on SPY 5m."""

from __future__ import annotations

from dataclasses import replace

from stockpro.config import load_settings
from stockpro.data import get_spy_5m
from stockpro.spy_day.amd import configure_amd_params
from stockpro.spy_day.backtest import run_spy_day_backtest
from stockpro.spy_day.session import load_spy_day_config


def _run(label, bars, base, *, patterns, skip_friday=None, amd_as_0dte=False, max_trades=3):
    cfg = replace(base, patterns=list(patterns), max_trades_per_day=max_trades)
    amd = replace(
        cfg.amd,
        skip_friday=bool(skip_friday) if skip_friday is not None else cfg.amd.skip_friday,
    )
    cfg = replace(cfg, amd=amd)
    configure_amd_params(cfg.amd.detector_dict())
    r = run_spy_day_backtest(bars, cfg=cfg, amd_as_0dte=amd_as_0dte)
    m = r.metrics
    pnl = m["final_equity"] - 100_000.0
    print(label)
    print(
        f"  n={int(m['n_trades'])} WR={m['win_rate']*100:.1f}% "
        f"PF={m['profit_factor']:.2f} E=${m['expectancy']:.2f} "
        f"pnl=${pnl:.0f} DD={m['max_drawdown']*100:.2f}%"
    )
    for p, s in sorted(r.by_pattern.items()):
        print(
            f"    {p}: n={int(s['n_trades'])} WR={s['win_rate']*100:.1f}% "
            f"PF={s['profit_factor']:.2f} pnl=${s['total_pnl']:.0f}"
        )
    return r


def main() -> None:
    settings = load_settings()
    bars = get_spy_5m(settings, refresh=False, rth_only=True)
    base = load_spy_day_config(settings)
    print(f"bars={len(bars)} {bars.index.min().date()} -> {bars.index.max().date()}")
    print(
        f"spy_day.enabled={base.enabled} patterns={base.patterns} "
        f"amd.enabled={base.amd.enabled} skip_friday={base.amd.skip_friday}"
    )
    print()
    _run(
        "1) AMD alone OLD (0DTE vehicle)",
        bars,
        base,
        patterns=["amd"],
        skip_friday=False,
        amd_as_0dte=True,
        max_trades=1,
    )
    _run(
        "2) AMD alone NEW (short-DTE, no Fri skip)",
        bars,
        base,
        patterns=["amd"],
        skip_friday=False,
        amd_as_0dte=False,
        max_trades=1,
    )
    _run(
        "3) AMD alone NEW + Fri skip",
        bars,
        base,
        patterns=["amd"],
        skip_friday=True,
        amd_as_0dte=False,
        max_trades=1,
    )
    _run(
        "4) Full live stack (ORB+retest+PH+AMD short-DTE)",
        bars,
        base,
        patterns=base.patterns,
        skip_friday=False,
        amd_as_0dte=False,
        max_trades=base.max_trades_per_day,
    )


if __name__ == "__main__":
    main()
