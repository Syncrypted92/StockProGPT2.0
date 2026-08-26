"""Summarize entry cost of winning SPY-day paper trades."""

from __future__ import annotations

import pandas as pd

from stockpro.config import load_settings
from stockpro.journal import Journal


def main() -> None:
    j = Journal(load_settings())
    t = j.load_trades()
    if t is None or len(t) == 0:
        print("no trades")
        return

    t = t.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True, errors="coerce")
    for col in ("pnl", "limit_price", "entry_premium", "qty"):
        if col in t.columns:
            t[col] = pd.to_numeric(t[col], errors="coerce")
    t["side"] = t["side"].astype(str)
    t["contract"] = t["contract"].astype(str)

    if "dry_run" in t.columns:
        live = t[~t["dry_run"].astype(str).str.lower().isin(["true", "1"])].copy()
    else:
        live = t.copy()

    exits = live[live["side"].str.contains("sell|close", case=False, na=False)].copy()
    wins = exits[exits["pnl"] > 0].sort_values("timestamp")
    entries = live[live["side"].str.contains("buy", case=False, na=False)].copy()

    try:
        d = pd.read_csv("data/journal/decisions.csv")
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
        orders = d[d.get("action") == "order"] if "action" in d.columns else d.iloc[0:0]
    except Exception:
        orders = pd.DataFrame()

    rows = []
    for _, x in wins.iterrows():
        c = x["contract"]
        prior = entries[(entries["contract"] == c) & (entries["timestamp"] <= x["timestamp"])]
        if len(prior) == 0:
            prior = entries[entries["contract"] == c]
        ent = prior.iloc[-1] if len(prior) else None

        qty = float(x["qty"]) if pd.notna(x.get("qty")) else None
        prem = None
        et = None
        sig = None
        if ent is not None:
            prem = ent["entry_premium"] if pd.notna(ent.get("entry_premium")) else ent["limit_price"]
            if not qty:
                qty = float(ent["qty"]) if pd.notna(ent.get("qty")) else 1.0
            et = ent["timestamp"]
            sig = ent.get("signal")

        cost = float(prem) * float(qty) * 100.0 if prem is not None and qty else None

        pat = None
        if len(orders) and "contract" in orders.columns:
            o = orders[orders["contract"].astype(str) == c]
            if len(o):
                det = str(o.iloc[-1].get("details", "") or "")
                pat = det.split(":")[0].strip() if ":" in det else (det[:40] or None)

        rows.append(
            {
                "exit_ts": str(x["timestamp"]),
                "contract": c,
                "pattern": pat,
                "side": sig,
                "entry_mid": round(float(prem), 3) if prem is not None else None,
                "qty": int(qty) if qty else None,
                "cost_usd": round(cost, 2) if cost is not None else None,
                "exit_mid": round(float(x["limit_price"]), 3) if pd.notna(x.get("limit_price")) else None,
                "pnl_usd": round(float(x["pnl"]), 2),
                "exit_reason": x.get("exit_reason"),
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 36)
    print(df.to_string(index=False))
    print()
    print(f"winners: {len(df)}")
    print(f"total entry debit (cost): ${df['cost_usd'].sum():.2f}")
    print(f"total win pnl: ${df['pnl_usd'].sum():.2f}")


if __name__ == "__main__":
    main()
