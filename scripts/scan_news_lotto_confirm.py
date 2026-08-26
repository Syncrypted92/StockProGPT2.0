"""News + market confirmation board for discretionary lotto ideas (research only).

Pairs Tiingo catalyst headlines with price/volume/RS/earnings checks from Yahoo.
Does NOT place trades or touch spy-day live. You decide yes/no.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from stockpro.config import ROOT, load_settings
from stockpro.news import next_earnings_date

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from scan_tiingo_lotto_news import _score, article_relevant, fetch_articles  # noqa: E402


@dataclass
class MarketConfirm:
    last: float
    ret_1d_pct: float
    ret_5d_pct: float
    gap_pct: float
    vol_vs_20d: float
    atr_pct: float
    rs_spy_1d_pct: float
    ok: bool
    detail: str


def _hist(ticker: str, days: int = 40) -> pd.DataFrame:
    end = datetime.now(timezone.utc).date() + timedelta(days=1)
    start = end - timedelta(days=days + 10)
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    return df.dropna(how="any")


def market_confirm(ticker: str, spy_1d: float | None) -> MarketConfirm:
    df = _hist(ticker)
    if len(df) < 22:
        return MarketConfirm(0, 0, 0, 0, 0, 0, 0, False, "insufficient_bars")
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)
    open_ = df["Open"].astype(float)
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    ret_1d = (last / prev - 1.0) * 100.0
    ret_5d = (last / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 else ret_1d
    gap = (float(open_.iloc[-1]) / prev - 1.0) * 100.0
    vol_avg = float(vol.iloc[-21:-1].mean()) or 1.0
    vol_vs = float(vol.iloc[-1]) / vol_avg
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.iloc[-15:].mean())
    atr_pct = (atr / last) * 100.0 if last else 0.0
    rs = ret_1d - (spy_1d or 0.0)
    return MarketConfirm(
        last=last,
        ret_1d_pct=ret_1d,
        ret_5d_pct=ret_5d,
        gap_pct=gap,
        vol_vs_20d=vol_vs,
        atr_pct=atr_pct,
        rs_spy_1d_pct=rs,
        ok=True,
        detail="ok",
    )


def earn_days(ticker: str) -> tuple[int | None, str]:
    earn = next_earnings_date(ticker)
    if earn is None:
        return None, "unknown"
    delta = (earn - date.today()).days
    return delta, earn.isoformat()


def reinforce_score(
    news_score: float,
    bias: str,
    m: MarketConfirm,
    earn_delta: int | None,
) -> tuple[float, list[str], str]:
    """Build confirmation layer score; return total, tags, grade."""
    tags: list[str] = []
    conf = 0.0

    bull = bias.startswith("bull")
    bear = bias.startswith("bear")
    mixed = bias.startswith("mixed")

    if not m.ok:
        return news_score, ["no_market_data"], "incomplete"

    if bull and m.ret_1d_pct >= 1.0:
        conf += 2.5
        tags.append("price_up_with_bull_news")
    elif bear and m.ret_1d_pct <= -1.0:
        conf += 2.5
        tags.append("price_down_with_bear_news")
    elif bull and m.ret_1d_pct <= -1.5:
        conf -= 2.0
        tags.append("conflict_price_vs_bull")
    elif bear and m.ret_1d_pct >= 1.5:
        conf -= 2.0
        tags.append("conflict_price_vs_bear")
    elif abs(m.ret_1d_pct) < 0.4:
        tags.append("price_quiet")

    if m.vol_vs_20d >= 1.8:
        conf += 2.0
        tags.append(f"volume_x{m.vol_vs_20d:.1f}")
    elif m.vol_vs_20d >= 1.25:
        conf += 1.0
        tags.append(f"volume_x{m.vol_vs_20d:.1f}")
    elif m.vol_vs_20d < 0.8:
        conf -= 0.5
        tags.append("volume_soft")

    if bull and m.gap_pct >= 1.0:
        conf += 1.5
        tags.append(f"gap_up_{m.gap_pct:.1f}%")
    elif bear and m.gap_pct <= -1.0:
        conf += 1.5
        tags.append(f"gap_dn_{m.gap_pct:.1f}%")

    if bull and m.rs_spy_1d_pct >= 1.0:
        conf += 1.5
        tags.append("rs_beat_spy")
    elif bear and m.rs_spy_1d_pct <= -1.0:
        conf += 1.5
        tags.append("rs_weak_vs_spy")
    elif bull and m.rs_spy_1d_pct <= -1.5:
        conf -= 1.0
        tags.append("lagging_spy")
    elif bear and m.rs_spy_1d_pct >= 1.5:
        conf -= 1.0
        tags.append("holding_vs_spy")

    if 1.5 <= m.atr_pct <= 6.0:
        conf += 0.5
        tags.append(f"atr_{m.atr_pct:.1f}%")
    elif m.atr_pct > 8.0:
        tags.append(f"atr_hot_{m.atr_pct:.1f}%")

    if earn_delta is not None:
        if 0 <= earn_delta <= 1:
            conf += 1.0
            tags.append(f"earnings_in_{earn_delta}d")
        elif -1 <= earn_delta < 0:
            conf += 0.5
            tags.append("post_earnings")
        elif abs(earn_delta) <= 5 and not mixed:
            tags.append(f"earn_{earn_delta}d")

    if mixed:
        conf -= 1.5
        tags.append("mixed_headlines")

    total = float(news_score) + conf
    if conf >= 4.0 and news_score >= 3 and not any(t.startswith("conflict") for t in tags):
        grade = "strong_confirm"
    elif conf >= 2.0 and news_score >= 3:
        grade = "mild_confirm"
    elif any(t.startswith("conflict") for t in tags):
        grade = "conflict"
    else:
        grade = "news_only"
    return total, tags, grade


def prefer_universe(tickers: list[str]) -> list[str]:
    prefer = [
        "TSLA",
        "NVDA",
        "AMD",
        "INTC",
        "MU",
        "PLTR",
        "SOFI",
        "HOOD",
        "RIVN",
        "SNAP",
        "NCLH",
        "CCL",
        "UBER",
        "PYPL",
        "META",
        "AMZN",
        "AAPL",
        "MSFT",
        "GOOGL",
        "BAC",
        "WFC",
        "KEY",
        "PFE",
        "F",
        "GM",
        "SPY",
        "QQQ",
        "IWM",
    ]
    return [t for t in prefer if t in tickers] + [t for t in tickers if t not in prefer]


def main() -> None:
    parser = argparse.ArgumentParser(description="News + market confirmation lotto board")
    parser.add_argument("--hours", type=int, default=36, help="News lookback hours")
    parser.add_argument("--min-news", type=int, default=3, help="Min headline catalyst score")
    parser.add_argument("--top", type=int, default=20, help="Max ideas to print")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    settings = load_settings()
    tickers = list(settings.get("universe", "tickers", default=[]) or [])
    ordered = prefer_universe(tickers)

    print(f"Fetching SPY baseline + Tiingo news ({args.hours}h) + Yahoo confirms...\n")
    spy = market_confirm("SPY", None)
    spy_1d = spy.ret_1d_pct if spy.ok else 0.0
    print(f"SPY 1d={spy_1d:+.2f}% vol_x={spy.vol_vs_20d:.2f}\n")

    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for i, t in enumerate(ordered, 1):
        arts = fetch_articles(t, args.hours)
        print(f"[{i}/{len(ordered)}] {t}: {len(arts)} articles", flush=True)
        for a in arts:
            if not article_relevant(t, a["title"], a.get("tagged") or []):
                continue
            score, side, tags = _score(a["title"])
            if score < args.min_news:
                continue
            by_ticker[t].append({**a, "news_score": score, "bias": side, "tags": tags})

    if not by_ticker:
        print("No catalyst headlines above threshold.")
        return

    rows = []
    for ticker, arts in by_ticker.items():
        arts = sorted(arts, key=lambda x: x["news_score"], reverse=True)
        sides = [a["bias"] for a in arts[:5]]
        if any(s.startswith("bull") for s in sides) and any(s.startswith("bear") for s in sides):
            bias = "mixed_catalyst"
        else:
            bias = arts[0]["bias"]
        news_score = float(max(a["news_score"] for a in arts) + min(2, len(arts) - 1) * 0.5)
        m = market_confirm(ticker, spy_1d)
        edelta, edate = earn_days(ticker)
        total, ctags, grade = reinforce_score(news_score, bias, m, edelta)
        headline = arts[0]["title"]
        rows.append(
            {
                "ticker": ticker,
                "grade": grade,
                "total": round(total, 2),
                "news_score": round(news_score, 2),
                "bias": bias,
                "n_headlines": len(arts),
                "headline": headline[:140],
                "last": round(m.last, 2) if m.ok else None,
                "ret_1d_pct": round(m.ret_1d_pct, 2) if m.ok else None,
                "gap_pct": round(m.gap_pct, 2) if m.ok else None,
                "vol_vs_20d": round(m.vol_vs_20d, 2) if m.ok else None,
                "rs_spy_1d_pct": round(m.rs_spy_1d_pct, 2) if m.ok else None,
                "atr_pct": round(m.atr_pct, 2) if m.ok else None,
                "earn_date": edate,
                "earn_days": edelta,
                "confirm_tags": "|".join(ctags),
                "source": arts[0].get("source", ""),
                "published": arts[0].get("published", ""),
                "url": arts[0].get("url", ""),
            }
        )

    df = pd.DataFrame(rows).sort_values(["total", "news_score"], ascending=False)
    out = ROOT / "artifacts" / "news_lotto_confirm_board.csv"
    df.to_csv(out, index=False)

    show = df.head(args.top)
    print("\n=== LOTTO IDEA BOARD (news + confirmation) — YOU decide ===\n")
    print("Grades: strong_confirm | mild_confirm | news_only | conflict | incomplete\n")
    for _, r in show.iterrows():
        print(
            f"[{r['grade']:14s}] total={r['total']:5.1f}  {r['ticker']:5s}  "
            f"{r['bias']:18s}  n={r['n_headlines']}"
        )
        print(f"  HEADLINE: {r['headline']}")
        if r["last"] is not None:
            print(
                f"  MKT: last={r['last']}  1d={r['ret_1d_pct']:+.2f}%  gap={r['gap_pct']:+.2f}%  "
                f"vol×{r['vol_vs_20d']:.2f}  RS_vs_SPY={r['rs_spy_1d_pct']:+.2f}%  ATR={r['atr_pct']:.1f}%"
            )
        else:
            print("  MKT: (no data)")
        ed = "n/a" if r["earn_days"] is None or (isinstance(r["earn_days"], float) and pd.isna(r["earn_days"])) else f"{int(r['earn_days'])}d"
        print(f"  EARN: {r['earn_date']} ({ed})  CONFIRM: {r['confirm_tags']}")
        print(f"  {r['published']}  {r['source']}")
        print()

    strong = df[df["grade"] == "strong_confirm"]
    mild = df[df["grade"] == "mild_confirm"]
    print("--- Summary ---")
    print(f"strong_confirm: {len(strong)}  mild_confirm: {len(mild)}  total scanned: {len(df)}")
    print(f"Wrote {out}")
    print(
        "\nNot wired to the bot. If you want one executed after your review, "
        "say the ticker + side — spy-day stays pattern-only."
    )


if __name__ == "__main__":
    main()
