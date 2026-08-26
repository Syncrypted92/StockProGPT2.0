"""Trade and decision journaling with a stable CSV schema."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stockpro.config import Settings

DECISION_COLUMNS = [
    "timestamp",
    "ticker",
    "action",
    "reason",
    "signal",
    "confidence",
    "predicted_class",
    "probabilities",
    "contract",
    "qty",
    "limit_price",
    "bid",
    "ask",
    "dte",
    "dry_run",
    "status",
    "order_id",
    "rank",
    "details",
    "entry",
    "current",
]

TRADE_COLUMNS = [
    "timestamp",
    "ticker",
    "side",
    "contract",
    "qty",
    "limit_price",
    "status",
    "dry_run",
    "signal",
    "confidence",
    "pnl",
    "entry_premium",
    "expiration",
    "exit_reason",
]


class Journal:
    def __init__(self, settings: Settings):
        cfg = settings.get("journal", default={}) or {}
        self.dir = Path(cfg.get("directory", "data/journal"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.trades_path = self.dir / cfg.get("trades_csv", "trades.csv")
        self.decisions_path = self.dir / cfg.get("decisions_csv", "decisions.csv")
        self.daily_pnl_path = self.dir / cfg.get("daily_pnl_csv", "daily_pnl.csv")

    def _append(self, path: Path, row: dict[str, Any], columns: list[str]) -> None:
        normalized = {c: row.get(c, "") for c in columns}
        df = pd.DataFrame([normalized], columns=columns)
        if path.exists():
            try:
                existing = pd.read_csv(path)
                # Align to schema
                for c in columns:
                    if c not in existing.columns:
                        existing[c] = ""
                existing = existing[[c for c in columns if c in existing.columns]]
                df = pd.concat([existing[columns], df], ignore_index=True)
                df.to_csv(path, index=False)
                return
            except Exception:
                # Corrupt / legacy file — archive and start fresh
                archive = path.with_suffix(path.suffix + f".bak-{datetime.now().strftime('%Y%m%d%H%M%S')}")
                path.rename(archive)
        df.to_csv(path, index=False)

    def log_decision(self, **kwargs: Any) -> None:
        row = {"timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), **kwargs}
        self._append(self.decisions_path, row, DECISION_COLUMNS)

    def log_trade(self, **kwargs: Any) -> None:
        row = {"timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), **kwargs}
        self._append(self.trades_path, row, TRADE_COLUMNS)

    def log_daily_pnl(self, day: str, pnl: float, equity: float) -> None:
        path = self.daily_pnl_path
        df = pd.DataFrame([{"date": day, "pnl": pnl, "equity": equity}])
        if path.exists():
            prev = pd.read_csv(path)
            prev = prev[prev["date"].astype(str) != str(day)]
            df = pd.concat([prev, df], ignore_index=True)
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)

    def load_trades(self) -> pd.DataFrame:
        if not self.trades_path.exists():
            return pd.DataFrame(columns=TRADE_COLUMNS)
        try:
            return pd.read_csv(self.trades_path)
        except Exception:
            return pd.read_csv(self.trades_path, engine="python", on_bad_lines="skip")

    def load_decisions(self) -> pd.DataFrame:
        if not self.decisions_path.exists():
            return pd.DataFrame(columns=DECISION_COLUMNS)
        try:
            return pd.read_csv(self.decisions_path)
        except Exception:
            return pd.read_csv(self.decisions_path, engine="python", on_bad_lines="skip")

    def summary(self) -> dict[str, Any]:
        trades = self.load_trades()
        if trades.empty:
            return {"n_trades": 0}
        out: dict[str, Any] = {"n_trades": len(trades)}
        if "pnl" in trades.columns:
            pnls = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
            out["total_pnl"] = float(pnls.sum())
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            out["win_rate"] = float((pnls > 0).mean()) if len(pnls) else 0.0
            gross_win = float(wins.sum()) if len(wins) else 0.0
            gross_loss = float(losses.sum()) if len(losses) else 0.0
            out["profit_factor"] = (
                (gross_win / abs(gross_loss)) if gross_loss < 0 else float("inf")
            )
        return out
