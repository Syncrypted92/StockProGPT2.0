"""Configuration loading for StockPro."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    raw: dict[str, Any]
    paper: bool = True
    allow_live: bool = False
    trading_halted: bool = False
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    config_path: Path = field(default_factory=lambda: ROOT / "config" / "settings.yaml")

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def is_live(self) -> bool:
        return (not self.paper) and self.allow_live and not self.trading_halted

    def require_broker_credentials(self) -> None:
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            raise RuntimeError(
                "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. "
                "Copy .env.example to .env and fill in paper keys."
            )
        if not self.paper and not self.allow_live:
            raise RuntimeError(
                "Live trading blocked: set PAPER=false and ALLOW_LIVE=true explicitly."
            )
        if self.trading_halted:
            raise RuntimeError("Trading halted via TRADING_HALTED=true.")


def load_settings(config_path: str | Path | None = None) -> Settings:
    load_dotenv(ROOT / ".env")
    path = Path(
        config_path
        or os.getenv("STOCKPRO_CONFIG")
        or (ROOT / "config" / "settings.yaml")
    )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    paper_env = os.getenv("PAPER", "true").lower() in {"1", "true", "yes"}
    allow_live = os.getenv("ALLOW_LIVE", "false").lower() in {"1", "true", "yes"}
    halted = os.getenv("TRADING_HALTED", "false").lower() in {"1", "true", "yes"}

    # YAML broker flags are defaults; env wins
    paper = paper_env if os.getenv("PAPER") is not None else bool(
        raw.get("broker", {}).get("paper", True)
    )
    if os.getenv("ALLOW_LIVE") is None:
        allow_live = bool(raw.get("broker", {}).get("allow_live", False))

    return Settings(
        raw=raw,
        paper=paper,
        allow_live=allow_live,
        trading_halted=halted,
        alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        config_path=path,
    )
