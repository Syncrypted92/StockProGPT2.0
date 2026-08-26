# News data sources for StockPro

Recommended stack for this project (cheap → free first).

## Tier 1 — Primary (your setup)

| Source | Cost | Role |
|--------|------|------|
| **Tiingo News** | **~$30/mo** | Primary headlines when `TIINGO_API_KEY` is set |
| **Yahoo Finance** | Free | Fallback + OHLCV + earnings dates |
| **Weekend notes** | Free | Manual newspaper bias in `weekend_notes.yaml` |

## Tier 2 — Optional extras

| Source | Typical cost | Notes |
|--------|--------------|-------|
| **Finnhub** | Free / paid | Optional; free tier often lacks news. Set `FINNHUB_API_KEY` if you have access. |
| **Marketaux** | ~$29/mo | Alternative to Tiingo with built-in sentiment |
| **Alpha Vantage** | ~$50/mo | News + sentiment |

Fetch order in code: **Tiingo → Finnhub → Yahoo**.

## Tier 3 — Alpaca (if you already pay)

- **Alpaca Market Data** — check your plan for news endpoints. Same keys as paper trading; no extra vendor if included.

## Scraping — not recommended

| Site | Verdict |
|------|---------|
| **Finnhub website** | Use the official API instead — scraping hits Cloudflare. |
| **Yahoo Finance web** | `yfinance` already wraps it; direct scraping breaks often. |
| **Seeking Alpha / WSJ / NYT** | Paywalls + ToS; not worth automating for a trading bot. |
| **Google News RSS** | Possible for macro headlines only; noisy, hard to map to tickers. |

**Better pattern:** API headlines + your **weekend_notes.yaml** for newspaper insight the APIs miss.

## How StockPro uses news today

1. **Veto** — earnings window + bearish keywords (block bad entries)
2. **Score** — sentiment, headline velocity, weekend bias (rank entries)
3. **In the model** — Tiingo daily features are backfilled (~90d) and merged into training; `train.py --optimize` picks lookback / flat band / HGB / news via validation directional hit
4. **Log** — `data/journal/news_features.csv` (training + scan)

```bash
python scripts/optimize_system.py   # model + trade-layer expectancy search
python scripts/train.py --optimize  # model-only search
```
