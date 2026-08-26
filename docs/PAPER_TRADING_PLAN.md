# Paper Trading Plan — Profit Optimization Test

**Status:** Paper only (`PAPER=true`, `ALLOW_LIVE=false`)  
**Goal:** Measure whether the directional options system has a usable edge after costs, fills, and risk rules — not just model accuracy.

---

## Quick turnaround (post bloody-week rules)

Current paper lane (see `docs/WEEK_2026-07-13_BASELINE.md`):

| Lane | DTE | Underlyings | Status |
|------|-----|-------------|--------|
| **Swing** (morning scan) | **7–21** | Cheap liquid names | **Active** — confidence ≥ **0.65** |
| **0DTE** | **0–1** | SPY, QQQ, IWM | **Paused** (`zerodte.enabled: false`) |

Exits (hourly): swing **+40% / −25%**, time-stop **2 DTE** before expiry.  
Also enforced: **1 position per underlying**, **3-day cooldown after stop**, earnings/headline veto.

Your open INTC Jul-31 call (~18 DTE) was from older settings — new entries follow the table above.

---

## Universe (25+ mega-caps)

Paper practices the same liquid-options habit you’ll use live:

- Prefer liquid names; **option mid ≤ $5** (`max_mid_price`) → ≤ **$500** per contract  
- **Mega-caps included:** SPY, QQQ, AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA  
- Plus sector/cheaper names: IWM, XLF, XLE, XLI, XLU, EEM, BAC, WFC, KEY, PFE, T, INTC, AMD, MU, PLTR, SOFI, HOOD, PYPL, UBER, F, GM, RIVN, SNAP, NCLH, CCL (**34** total)

Edit `config/settings.yaml` → `universe.tickers` anytime, then `python scripts/train.py`.

---

## What “optimized for profit” means here

We are **not** optimizing for train accuracy. We grade **realized paper PnL** against hard gates:

| Metric | Pass (good) | Review | Fail / stop |
|--------|-------------|--------|-------------|
| Trading days with clean runs | ≥ 20 | 10–19 | Bugs / can't run |
| Max drawdown (equity) | ≤ 10% | 10–15% | > 15% |
| Profit factor (closed trades) | ≥ 1.3 | 1.0–1.3 | < 1.0 after 20+ trades |
| Win rate (closed) | ≥ 45% | 40–45% | < 40% with PF < 1.1 |
| Expectancy ($/trade) | > 0 | ~0 | Clearly negative |
| Critical bugs / bad fills | 0 | Rare | Recurring |

**Do not go live** until Pass column is mostly green after the full test window.

---

## How long to test

| Phase | Length | Purpose |
|-------|--------|---------|
| **Week 1** | 5 trading days | Plumbing: orders fill, exits work, journal grades daily |
| **Weeks 2–4** | ~15 more days (**20 total**) | First real performance sample |
| **Optional extend** | to **40–60 days** | If results are mixed but not broken |

**Minimum before any live discussion: 20 trading days** and at least **~15–25 closed option trades** (so stats aren't noise).

One day proves nothing. Options are noisy; 20 sessions is the floor in `config/settings.yaml` → `paper_gates.min_trading_days`.

---

## Tomorrow — day-1 playbook (market hours, ET)

Run from project root with venv active.

### Morning (after ~9:45 ET — let opening volatility settle)

```bash
python scripts/smoke_test.py
python scripts/scan_and_trade.py --dry-run
```

1. Confirm account ACTIVE, options level ≥ 2, SPY contracts > 0.  
2. Read **`confidence_board`** — only top signals above threshold.  
3. If ranks look sane (liquid names, confidence ≥ 0.65), then:

```bash
python scripts/scan_and_trade.py --submit
```

### Midday — hourly exit monitor (automatic)

Task `StockPro-Exits-Hourly` runs **every 60 minutes from 10:00–15:00 ET**:

```bash
python scripts/manage_positions.py --submit
```

Exits: +40% premium target, −25% stop, or time stop 2 days before expiry.

### End of day (after close ~16:15 ET)

```bash
python scripts/daily_grade.py
```

This pulls Alpaca account/positions + local journal, writes:

- `data/journal/daily_grades.csv` (append one row per day)  
- `data/journal/reports/YYYY-MM-DD.md` (human-readable grade)

Review the report. Fix bugs before next session if grade is F for ops reasons.

### Kill switch anytime

Set in `.env`:

```
TRADING_HALTED=true
```

---

## Daily operating rules (whole paper period)

1. **Always dry-run before submit** on entry days.  
2. **Max 2 open positions**, **1 contract** each, **one per underlying** (config).  
3. **No live keys**, no `ALLOW_LIVE=true`.  
4. **No changing the model mid-week** unless a bug blocks trading — log any change in the daily report notes.  
5. **Retarget model weekly** (optional Sunday): `python scripts/train.py` then note it in the grade report.  
6. If daily loss halt triggers, **stop new entries that day**.  
7. **3 trading-day cooldown** after a stop-out on that ticker; earnings/headline veto can skip entries.

---

## What we measure each day (`daily_grade.py`)

From **Alpaca** (source of truth for money):

- Equity, cash, buying power  
- Open positions + unrealized P/L  
- Closed activity / trade PnL when available  

From **journal** (source of truth for decisions):

- Signals scanned, flat vs traded  
- Confidence of entries  
- Exits and reasons  

From **model accuracy tracker** (`data/journal/predictions.csv`):

- Every scan logs a prediction (ticker, class, confidence, spot)  
- After **5 trading sessions**, the grader resolves actual forward return  
- Reports: 3-class accuracy, directional accuracy, high-confidence accuracy  
- Model lens gets its own letter grade (needs ≥20 resolved predictions)

**Letter grade** combines:

- **Ops grade** — did the bot run without errors?  
- **Risk grade** — drawdown / daily loss within limits?  
- **Edge grade** — rolling profit factor, win rate, expectancy  
- **Model grade** — realized prediction accuracy  

Overall: A/B/C/D/F with a one-line verdict.

---

## Scheduler (auto-run on Alpaca paper)

There was no scheduler before — runs were manual. Now:

### One-time install (Windows Task Scheduler)

Market times are **US Eastern**. The install script converts them to your PC's local timezone — you do **not** need Windows set to Eastern.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows_tasks.ps1
```

| Task | Eastern time | Action |
|------|--------------|--------|
| `StockPro-Morning` | 09:50 ET | smoke + `scan_and_trade --submit` |
| `StockPro-Exits-Hourly` | **every 60 min** 10:00–15:45 ET | `manage_positions --submit` (TP / SL / time stop) |
| `StockPro-EOD` | 16:20 ET | `daily_grade.py` (PnL + model accuracy) |

`StockPro-ZeroDTE` is **unregistered** while `zerodte.enabled: false`. Re-run the install script after re-enabling.

Exit checks run at **10, 11, 12, 13, 14, 15 ET** (9–2 CT on your machine). That is still not tick-by-tick, but stops/targets are reviewed hourly during the session.

### Manual / dry-run equivalents

```bash
python scripts/run_session.py morning --dry-run
python scripts/run_session.py morning --submit
python scripts/run_session.py afternoon --submit
python scripts/run_session.py eod
```

Keep the PC awake on market days (or use a small always-on box).  
`TRADING_HALTED=true` in `.env` still blocks trading even if tasks fire.  
Re-run `install_windows_tasks.ps1` after DST transitions if a trigger looks an hour off.

---

## Decision after 20 days

| Outcome | Next step |
|---------|-----------|
| Pass gates | Keep paper another 20 days **or** discuss tiny live with same risk caps |
| Mixed (PF ~1.0–1.2) | Extend paper; tighten threshold / universe; do **not** go live |
| Fail (DD > 15% or PF < 1) | Stop; revise signal/exits; re-paper from scratch |

---

## Commands cheat sheet

| When | Command |
|------|---------|
| Connectivity | `python scripts/smoke_test.py` |
| Preview entries | `python scripts/scan_and_trade.py --dry-run` |
| Paper entries | `python scripts/scan_and_trade.py --submit` |
| Preview exits | `python scripts/manage_positions.py --dry-run` |
| Paper exits | `python scripts/manage_positions.py --submit` |
| End-of-day grade | `python scripts/daily_grade.py` |
| **Evaluation report** | `python scripts/evaluate.py --from 2026-07-13 --to 2026-07-20` |
| **Train A/B (news)** | `python scripts/train.py --ab-test` |
| Rolling summary | `python scripts/daily_grade.py --summary` |
| Session runner | `python scripts/run_session.py morning\|afternoon\|eod\|all` |
| Install scheduler | `powershell -File scripts\install_windows_tasks.ps1` |
| Retrain (weekend) | `python scripts/train.py` |

---

## Disclaimer

Paper fills are simulated. A good paper grade is necessary but **not sufficient** for live. Options can go to zero; keep hard daily/weekly loss limits forever.
