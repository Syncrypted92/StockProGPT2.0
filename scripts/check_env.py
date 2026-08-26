"""Safe .env diagnostics — never prints secret values."""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(".env"))
key = os.getenv("ALPACA_API_KEY", "") or ""
secret = os.getenv("ALPACA_SECRET_KEY", "") or ""
paper = os.getenv("PAPER", "")


def flags(name: str, value: str) -> None:
    print(f"{name} set: {bool(value)}")
    print(f"{name} length: {len(value)}")
    print(f"{name} has_whitespace: {value != value.strip()}")
    print(f"{name} has_quotes: {value[:1] in {chr(34), chr(39)} and value[-1:] in {chr(34), chr(39)}}")
    print(f"{name} prefix4: {value[:4]!r}")


flags("ALPACA_API_KEY", key)
flags("ALPACA_SECRET_KEY", secret)
print(f"PAPER: {paper!r}")
