# Paper week baseline — Jul 13–16, 2026

Documented before the “Fix Bloody Paper Week” rule changes. Do not treat this as a live edge signal.

## Account

| | |
|---|---:|
| Start equity | ~$100,000 |
| End equity (Jul 16) | ~$99,685 |
| Week PnL | **−$315** |

## Closed trades

| Trade | Exit | PnL |
|---|---|---:|
| INTC Jul17 call | profit target | +$32 |
| NCLH Jul17 call | stop | −$20 |
| INTC Jul17 call | stop | −$125 |
| EEM Jul17 call | stop | −$30 |
| INTC Jul31 call | stop | −$172 |
| **Total** | | **−$315** |

## Summary stats

- Win rate: **20%** (1/5)
- Profit factor: **~0.09**
- Expectancy: **−$63 / trade**
- Ops grade: **A** (scheduler / entries / exits worked)
- Edge grade: **F**
- INTC concentration: **3 of 5** closed trades; same-underlying doubling
- Model OOS accuracy at train time: **~47%**
- Prediction tracker: **0 resolved** through Jul 16 (horizon not yet due / resolve bug)

## Root causes (process)

1. Short DTE (1–5) + wide stops → fast bleed when direction wrong  
2. Confidence threshold too low for a near coin-flip model  
3. No same-underlying cap or post-stop cooldown  
4. No earnings / headline veto  

## Post-fix success criteria (next ~10 sessions)

- ≤1 contract per underlying open  
- No re-entry within 3 trading days after a stop  
- Expectancy improves toward ≥ −$10, then > 0  
- Edge grade moves off F  
- `model_n_resolved` populates in daily grades  
