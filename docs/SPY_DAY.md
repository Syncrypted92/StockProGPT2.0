## SPY Day (5m)

**Live paper**
- **ORB** — first 30m range break; skip 4H counter-trend; **0DTE**; TP 30% / SL 25%; force flat 15:45 ET
- **ORB retest** — pullback to OR after the break, continuation (0DTE)
- **Power hour** — 14:00–14:25 continuation of day/ORB bias (0DTE)
- **AMD** — OR sweep reclaim fade; **~3 DTE (2–5)**, not 0DTE; reclaim + HTF; 10:00–12:00; TP 35% / SL 30%; overnight OK (no same-day forced flat). All weekdays (no Friday skip).
  - **Scale-out (paper, live):** **3 contracts on every lane**. Lot 1 closes at coded TP (30% 0DTE / 35% AMD). Lot 2 runs to **+60%** (original SL still applies). Lot 3 runner: after TP2, **trail 12% off peak**; if it never trails, **SL to entry**. Exits poll **every 1m**. 0DTE leftover lots flatten **15:45 ET**. Only **AMD (~3 DTE)** can hold overnight. Replaces AMD I1 trail while `scale_out.enabled`.

Max **3** entries / day, **2** open SPY option books (e.g. 0DTE ORB + short-DTE AMD).

**Caveats (AMD)** — option-premium *proxy* in research, n≈40, still thin. Paper path uses real Alpaca quotes. Keep AMD off 0DTE ORB economics. NQ/MNQ futures vehicle looked stronger in-sample but **not available on Alpaca paper** — do not enable here.

**AMD walk-forward** (`artifacts/amd_walk_forward.csv`, locked recipe, no new search)
- Full-sample short-DTE proxy PF ~1.59 (n=40)
- Expanding-fold mean OOS PF ~1.38; 60/40 holdout PF ~1.46 (n=14)
- Friday skip was identical to include-Friday on this sample → removed from live

**AMD research** (`artifacts/amd_research.csv`, `artifacts/amd_vehicles_research.csv`, `artifacts/amd_short_dte_tune.csv`, `artifacts/amd_nq_mnq_research.csv`)
- Alone on **SPY 0DTE**: failed (PF ~0.7–0.8) — do not use
- **SPY ~3 DTE proxy + HTF** (simple recipe above): paper live
- **SPY futures-style points**: ~flat on long sample
- **NQ/MNQ 5m RTH** (~60d Yahoo): reclaim AMD PF ~2.0–2.5 in-sample — short history; Alpaca can’t trade futures

**Intraday research** (`artifacts/intraday_strategies_research.csv`, ~6m SPY)

| Strategy | Alone | +ORB | Note |
|----------|-------|------|------|
| power_hour | PF **1.40**, WR 62%, n=37 | PF **1.49**, pnl best | **live** (14:00–14:25) |
| spy_qqq_lead | PF 1.18, n=23 | PF **1.52** | Slight ORB filter, not mid-day |
| gap_and_go | n=3 only | = ORB | Too rare |
| opening_drive | PF 1.09 | PF 1.41 | Dilutes vs ORB alone |
| compression_break | PF 0.99 | PF 1.38 | No edge alone |
| trend_day_pullback | PF **0.73** | PF 1.36 | Lose alone |
| multi_day_break | PF 0.71 | PF 1.35 | Lose alone |
| failed_auction | PF 0.89 | PF 1.15 | Drags ORB |
| range_day_fade | n=1 | ~ORB | Dead |
| amd (0DTE) | PF **0.71–0.81** | PF 1.17–1.25 | Lose alone; dilutes ORB |

```bash
python scripts/research_amd.py
python scripts/research_amd_short_dte_tune.py
python scripts/research_amd_walk_forward.py
python scripts/research_intraday_strategies.py
python scripts/backtest_spy_day.py
python scripts/scan_spy_day.py --dry-run --refresh
```
