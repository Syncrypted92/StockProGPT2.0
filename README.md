# StockProGPT 2.0 — Directional Options + Alpaca

Python package for **directional single-leg options** (long calls/puts) with liquidity filters, risk limits, Alpaca paper trading, and a gated path to tiny live size.

> The original Jupyter notebook (`StockPredictionProject.ipynb`) is kept for reference. The runnable system lives under `src/stockpro/`.

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -e ".[dev]"

copy .env.example .env
# Add Alpaca paper API keys to .env (optional for dry-run)

python scripts/train.py
python scripts/smoke_test.py
python scripts/scan_and_trade.py --dry-run
python scripts/report.py --backtest
pytest
```

## Pipeline

1. **Universe filter** — min price, volume, dollar volume  
2. **Features** — technical indicators → directional model (gradient boosting)  
3. **Signal** — bullish → long call, bearish → long put, else flat  
4. **Contract picker** — DTE, OI, volume, bid-ask spread, delta/OTM band  
5. **Risk** — tiny notional, max positions, daily/weekly loss halt, kill switch  
6. **Broker** — Alpaca paper by default; live only if `PAPER=false` and `ALLOW_LIVE=true`

## Config

- [`config/settings.yaml`](config/settings.yaml) — tickers, filters, risk, backtest  
- [`.env.example`](.env.example) — `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `PAPER`, `ALLOW_LIVE`, `TRADING_HALTED`

## Paper trading (tomorrow)

```bash
.\.venv\Scripts\activate

# 1) Confirm Alpaca paper + options chain
python scripts/smoke_test.py

# 2) Preview signals/orders — no submits
python scripts/scan_and_trade.py --dry-run

# 3) During market hours: submit paper entries
python scripts/scan_and_trade.py --submit

# 4) Check / exit open positions (stops, targets, time)
python scripts/manage_positions.py --dry-run
python scripts/manage_positions.py --submit
```

Keep `PAPER=true` and `ALLOW_LIVE=false` in `.env`. Kill switch: `TRADING_HALTED=true`.

## Paper trading plan

Full day-by-day playbook, test length, and pass/fail gates:

→ **[docs/PAPER_TRADING_PLAN.md](docs/PAPER_TRADING_PLAN.md)**

End of each session:

```bash
python scripts/daily_grade.py
python scripts/daily_grade.py --summary
```

Reports land in `data/journal/reports/YYYY-MM-DD.md` and `data/journal/daily_grades.csv`.

## Disclaimer

No model here has a proven edge. Paper validates plumbing and discipline, not profitability. Options can expire worthless; hard daily loss limits are mandatory for live.
