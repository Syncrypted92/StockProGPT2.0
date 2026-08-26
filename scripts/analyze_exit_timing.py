"""Exit-timing analysis: hold duration + MFE/MAE vs actual exit (proxy + optional OPRA bars)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stockpro.broker import AlpacaBroker
from stockpro.config import load_settings
from stockpro.data import get_spy_5m
from stockpro.journal import Journal
from stockpro.spy_day.session import load_spy_day_config

ET = ZoneInfo("America/New_York")


TRADES_OF_INTEREST = [
    # recent SPY-day winners + losers for path study
    "SPY260810C00774000",  # today ORB
    "SPY260810C00772000",  # Aug7 AMD (Aug10 expiry name in journal was SPY260810C00772000)
    "SPY260807P00770000",
    "SPY260810P00771000",
    "SPY260806C00772000",
    "SPY260804C00764000",
    "SPY260731P00739000",
    "SPY260730C00740000",
]


@dataclass
class Fill:
    contract: str
    pattern: str
    side: str  # call|put
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_mid: float
    exit_mid: float
    qty: int
    pnl: float
    exit_reason: str
    tp_pct: float
    sl_pct: float


def _parse_side(contract: str) -> str:
    # OCC ...YYMMDD C/P strike
    import re

    m = re.search(r"\d{6}([CP])\d{8}$", contract)
    if not m:
        return "call"
    return "call" if m.group(1) == "C" else "put"


def _strike(contract: str) -> float:
    import re

    m = re.search(r"[CP](\d{8})$", contract)
    if not m:
        return float("nan")
    return int(m.group(1)) / 1000.0


def load_fills() -> list[Fill]:
    j = Journal(load_settings())
    t = j.load_trades().copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True, errors="coerce")
    for c in ("pnl", "limit_price", "entry_premium", "qty"):
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    live = t[~t["dry_run"].astype(str).str.lower().isin(["true", "1"])] if "dry_run" in t.columns else t
    d = pd.read_csv("data/journal/decisions.csv")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    orders = d[d.get("action").astype(str) == "order"].copy() if "action" in d.columns else d.iloc[0:0]

    cfg = load_spy_day_config()
    fills: list[Fill] = []
    exits = live[live["side"].astype(str).str.contains("sell|close", case=False, na=False)]
    entries = live[live["side"].astype(str).str.contains("buy", case=False, na=False)]

    for _, x in exits.iterrows():
        c = str(x["contract"])
        if not c.startswith("SPY"):
            continue
        # collapse spam flat zeros
        if x.get("exit_reason") == "session_flat" and float(x.get("pnl") or 0) >= -1.5:
            continue
        prior = entries[(entries["contract"].astype(str) == c) & (entries["timestamp"] <= x["timestamp"])]
        if not len(prior):
            continue
        ent = prior.iloc[-1]
        prem = ent["entry_premium"] if pd.notna(ent.get("entry_premium")) else ent["limit_price"]
        qty = int(ent["qty"]) if pd.notna(ent.get("qty")) else int(x["qty"] or 1)
        pat = "unknown"
        o = orders[orders["contract"].astype(str) == c] if len(orders) and "contract" in orders.columns else None
        if o is not None and len(o):
            det = str(o.iloc[-1].get("details") or "")
            pat = det.split(":")[0].strip() if ":" in det else "unknown"
        is_amd = pat == "amd"
        fills.append(
            Fill(
                contract=c,
                pattern=pat,
                side=_parse_side(c),
                entry_ts=pd.Timestamp(ent["timestamp"]).tz_convert(ET),
                exit_ts=pd.Timestamp(x["timestamp"]).tz_convert(ET),
                entry_mid=float(prem),
                exit_mid=float(x["limit_price"]) if pd.notna(x["limit_price"]) else float("nan"),
                qty=qty,
                pnl=float(x["pnl"]) if pd.notna(x["pnl"]) else float("nan"),
                exit_reason=str(x.get("exit_reason") or ""),
                tp_pct=cfg.amd.profit_target_pct if is_amd else cfg.profit_target_pct,
                sl_pct=cfg.amd.stop_loss_pct if is_amd else cfg.stop_loss_pct,
            )
        )
    # newest first, keep recent
    fills.sort(key=lambda f: f.exit_ts, reverse=True)
    return fills[:8]


def try_option_bars(broker: AlpacaBroker, symbol: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    try:
        from alpaca.data.requests import OptionBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        if broker._option_data is None:
            broker.connect()
        if broker._option_data is None:
            return None
        req = OptionBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
            end=end.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
        )
        bars = broker._option_data.get_option_bars(req)
        df = bars.df if hasattr(bars, "df") else pd.DataFrame(bars)
        if df is None or df.empty:
            return None
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol) if symbol in df.index.get_level_values(0) else df.reset_index(level=0, drop=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"  [opra] {symbol}: {exc}")
        return None


def proxy_path(spy: pd.DataFrame, fill: Fill) -> pd.DataFrame:
    """Reconstruct approx premium from SPY 5m after entry (same model as day backtest)."""
    day = spy[(spy.index >= fill.entry_ts.floor("5min")) & (spy.index <= fill.exit_ts.ceil("5min") + pd.Timedelta(hours=2))]
    if day.empty:
        # stretch to end of entry day
        end = fill.entry_ts.normalize() + pd.Timedelta(hours=16)
        day = spy[(spy.index >= fill.entry_ts.floor("5min")) & (spy.index <= end)]
    side = 1 if fill.side == "call" else -1
    spot0 = float(day["Close"].iloc[0]) if len(day) else fill.entry_mid
    prem = fill.entry_mid
    rows = []
    closes = day["Close"]
    for i in range(len(closes)):
        ts = closes.index[i]
        if i == 0:
            rows.append({"ts": ts, "spot": float(closes.iloc[i]), "prem": prem, "ret": 0.0})
            continue
        prev, cur = float(closes.iloc[i - 1]), float(closes.iloc[i])
        und_ret = (cur / prev - 1.0) * side if prev > 0 else 0.0
        prem_frac = max(prem / max(spot0, 1e-6), 1e-4)
        prem = prem * (1.0 + float(np.clip(0.40 * und_ret / prem_frac, -0.55, 1.20)))
        # rough overnight not needed intraday 0DTE
        rows.append({"ts": ts, "spot": cur, "prem": prem, "ret": prem / fill.entry_mid - 1.0})
    return pd.DataFrame(rows)


def analyze_fill(spy: pd.DataFrame, broker: AlpacaBroker, fill: Fill) -> dict:
    hold_min = (fill.exit_ts - fill.entry_ts).total_seconds() / 60.0
    actual_ret = fill.exit_mid / fill.entry_mid - 1.0 if fill.entry_mid else float("nan")

    opra = try_option_bars(
        broker,
        fill.contract,
        fill.entry_ts - timedelta(minutes=5),
        fill.entry_ts.normalize() + timedelta(hours=16),
    )
    source = "proxy"
    path = proxy_path(spy, fill)
    if opra is not None and len(opra):
        # use mid from H/L/C if available
        px = opra["close"] if "close" in opra.columns else opra.iloc[:, 0]
        after = px[px.index >= fill.entry_ts.floor("5min")]
        if len(after):
            source = "opra_5m"
            path = pd.DataFrame(
                {
                    "ts": after.index,
                    "prem": after.values.astype(float),
                    "ret": after.values.astype(float) / fill.entry_mid - 1.0,
                }
            )

    # window from entry to +2h or EOD for counterfactual
    window = path[path["ts"] >= fill.entry_ts.floor("5min")].copy()
    if window.empty:
        window = path
    # until session force flat-ish 15:45
    window = window[window["ts"].dt.time <= pd.Timestamp("15:45").time()]

    mfe_ret = float(window["ret"].max()) if len(window) else float("nan")
    mae_ret = float(window["ret"].min()) if len(window) else float("nan")
    mfe_i = int(window["ret"].values.argmax()) if len(window) else -1
    mae_i = int(window["ret"].values.argmin()) if len(window) else -1
    mfe_ts = window.iloc[mfe_i]["ts"] if mfe_i >= 0 else None
    mae_ts = window.iloc[mae_i]["ts"] if mae_i >= 0 else None
    mfe_prem = float(window.iloc[mfe_i]["prem"]) if mfe_i >= 0 else float("nan")
    mae_prem = float(window.iloc[mae_i]["prem"]) if mae_i >= 0 else float("nan")

    # pnl at MFE/MAE for our qty
    mfe_pnl = (mfe_prem - fill.entry_mid) * 100 * fill.qty
    mae_pnl = (mae_prem - fill.entry_mid) * 100 * fill.qty
    left_on_table = mfe_pnl - fill.pnl  # vs actual

    # would TP have hit earlier on path?
    tp_hit = window[window["ret"] >= fill.tp_pct]
    first_tp = tp_hit.iloc[0]["ts"] if len(tp_hit) else None
    sl_hit = window[window["ret"] <= -fill.sl_pct]
    first_sl = sl_hit.iloc[0]["ts"] if len(sl_hit) else None

    # after our exit: did it keep running?
    after = window[window["ts"] > fill.exit_ts]
    post_mfe = float(after["ret"].max()) if len(after) else None
    post_mae = float(after["ret"].min()) if len(after) else None

    return {
        "contract": fill.contract,
        "pattern": fill.pattern,
        "source": source,
        "entry_et": str(fill.entry_ts),
        "exit_et": str(fill.exit_ts),
        "hold_min": round(hold_min, 1),
        "entry": fill.entry_mid,
        "exit": fill.exit_mid,
        "actual_ret_pct": round(100 * actual_ret, 1),
        "pnl": fill.pnl,
        "exit_reason": fill.exit_reason,
        "tp_pct": fill.tp_pct,
        "mfe_ret_pct": round(100 * mfe_ret, 1),
        "mfe_prem": round(mfe_prem, 3),
        "mfe_pnl": round(mfe_pnl, 1),
        "mfe_et": str(mfe_ts) if mfe_ts is not None else None,
        "mae_ret_pct": round(100 * mae_ret, 1),
        "mae_pnl": round(mae_pnl, 1),
        "mae_et": str(mae_ts) if mae_ts is not None else None,
        "left_vs_mfe": round(left_on_table, 1),
        "first_tp_et": str(first_tp) if first_tp is not None else None,
        "first_sl_et": str(first_sl) if first_sl is not None else None,
        "post_exit_mfe_pct": round(100 * post_mfe, 1) if post_mfe is not None else None,
        "post_exit_mae_pct": round(100 * post_mae, 1) if post_mae is not None else None,
        "path": window,
    }


def main() -> None:
    settings = load_settings()
    spy = get_spy_5m(settings, refresh=False, rth_only=True)
    if spy.index.tz is None:
        spy.index = spy.index.tz_localize(ET)
    broker = AlpacaBroker(settings, dry_run=False)
    broker.connect()
    fills = load_fills()
    print(f"Analyzing {len(fills)} recent SPY-day exits\n")
    rows = []
    for f in fills:
        r = analyze_fill(spy, broker, f)
        path = r.pop("path")
        rows.append(r)
        print("=" * 72)
        print(f"{r['pattern'].upper()}  {r['contract']}  [{r['source']}]")
        print(f"  hold {r['hold_min']:.0f} min | entry ${r['entry']:.3f} -> exit ${r['exit']:.3f} ({r['actual_ret_pct']:+.1f}%) | pnl ${r['pnl']:.0f} ({r['exit_reason']})")
        print(f"  entry {r['entry_et']}")
        print(f"  exit  {r['exit_et']}")
        print(f"  MFE {r['mfe_ret_pct']:+.1f}% (~${r['mfe_pnl']:.0f}) @ {r['mfe_et']} | left vs MFE ${r['left_vs_mfe']:.0f}")
        print(f"  MAE {r['mae_ret_pct']:+.1f}% (~${r['mae_pnl']:.0f}) @ {r['mae_et']}")
        print(f"  first TP touch ({100*f.tp_pct:.0f}%): {r['first_tp_et']} | first SL: {r['first_sl_et']}")
        print(f"  after our exit — further MFE {r['post_exit_mfe_pct']}% / MAE {r['post_exit_mae_pct']}%")
        # today extra detail
        if f.contract.endswith("C00774000") and "0810" in f.contract.replace("SPY", ""):
            print("  --- path around exit (proxy/OPRA prem) ---")
            sub = path[(path["ts"] >= f.entry_ts - pd.Timedelta(minutes=5)) & (path["ts"] <= f.exit_ts + pd.Timedelta(hours=1))]
            if len(sub):
                for _, row in sub.iterrows():
                    mark = ""
                    if abs((row["ts"] - f.exit_ts).total_seconds()) < 180:
                        mark = "  << EXIT"
                    if abs((row["ts"] - f.entry_ts).total_seconds()) < 180:
                        mark = "  << ENTRY"
                    print(f"    {row['ts']}  prem~{row['prem']:.3f}  ret={100*row['ret']:+.1f}%{mark}")

    out = pd.DataFrame(rows)
    out.to_csv("artifacts/exit_timing_analysis.csv", index=False)
    print("\nWrote artifacts/exit_timing_analysis.csv")


if __name__ == "__main__":
    main()
