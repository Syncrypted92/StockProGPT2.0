"""Alpaca connectivity smoke test (paper by default)."""

from __future__ import annotations

import json

from stockpro.broker import AlpacaBroker
from stockpro.config import load_settings


def main() -> None:
    settings = load_settings()
    dry = not bool(settings.alpaca_api_key)
    broker = AlpacaBroker(settings, dry_run=dry)
    result = broker.smoke_test()
    print(json.dumps(result, indent=2, default=str))
    if dry:
        print(
            "\nNo API keys found — ran mock smoke test. "
            "Copy .env.example to .env and add paper keys for a live paper probe."
        )


if __name__ == "__main__":
    main()
