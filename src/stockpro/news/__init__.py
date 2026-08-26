"""Lightweight news: veto, scoring, weekend notes, feature logging."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yaml
import yfinance as yf

from stockpro.config import Settings

NEWS_FEATURE_COLUMNS = [
    "date",
    "ticker",
    "headline_count",
    "bullish_hits",
    "bearish_hits",
    "news_score",
    "sentiment",
    "velocity",
    "earnings_adj",
    "weekend_bias",
    "top_headline",
    "logged_at",
]


@dataclass
class NewsVeto:
    blocked: bool
    reason: str  # "" | "earnings" | "news"
    detail: str = ""


@dataclass
class NewsScore:
    ticker: str
    score: float  # -1 to +1 combined
    sentiment: float = 0.0
    velocity: float = 0.0
    earnings_adj: float = 0.0
    weekend_bias: float = 0.0
    headline_count: int = 0
    bullish_hits: int = 0
    bearish_hits: int = 0
    top_headline: str = ""
    headlines: list[str] = field(default_factory=list)
    detail: str = ""


def _news_cfg(settings: Settings) -> dict[str, Any]:
    return dict(settings.get("news", default={}) or {})


def _finnhub_key() -> str:
    return (os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_TOKEN") or "").strip()


def _tiingo_key() -> str:
    return (os.getenv("TIINGO_API_KEY") or os.getenv("TIINGO_TOKEN") or "").strip()


def _fetch_tiingo_headlines(ticker: str, lookback_hours: int) -> list[str]:
    """Tiingo News API — primary paid feed when TIINGO_API_KEY is set."""
    key = _tiingo_key()
    if not key:
        return []
    headlines: list[str] = []
    try:
        now = datetime.now(timezone.utc)
        start = (now - timedelta(hours=lookback_hours)).date().isoformat()
        end = (now + timedelta(days=1)).date().isoformat()
        qs = urlencode(
            {
                "tickers": ticker.lower(),
                "startDate": start,
                "endDate": end,
                "limit": 50,
                "sortBy": "publishedDate",
            }
        )
        url = f"https://api.tiingo.com/tiingo/news?{qs}"
        req = Request(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Token {key}",
                "User-Agent": "StockProGPT/2.0",
            },
        )
        with urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, list):
            return []
        cutoff = now - timedelta(hours=lookback_hours)
        for item in payload[:50]:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            pub = item.get("publishedDate") or item.get("crawlDate")
            if pub:
                try:
                    ts = pd.to_datetime(pub, utc=True).to_pydatetime()
                    if ts < cutoff:
                        continue
                except Exception:  # noqa: BLE001
                    pass
            headlines.append(title)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return []
    return headlines


def _fetch_finnhub_headlines(ticker: str, lookback_hours: int) -> list[str]:
    key = _finnhub_key()
    if not key:
        return []
    headlines: list[str] = []
    try:
        now = datetime.now(timezone.utc)
        frm = (now - timedelta(hours=lookback_hours)).date().isoformat()
        to = now.date().isoformat()
        qs = urlencode({"symbol": ticker, "from": frm, "to": to, "token": key})
        url = f"https://finnhub.io/api/v1/company-news?{qs}"
        req = Request(url, headers={"User-Agent": "StockProGPT/2.0"})
        with urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if isinstance(payload, list):
            for item in payload[:40]:
                title = str(item.get("headline") or item.get("title") or "").strip()
                if title:
                    headlines.append(title)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return []
    return headlines


def _fetch_yahoo_headlines(ticker: str, lookback_hours: int) -> list[str]:
    headlines: list[str] = []
    try:
        t = yf.Ticker(ticker)
        news = getattr(t, "news", None) or []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        for item in news[:30]:
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, dict):
                title = str(content.get("title") or "").strip()
                pub = content.get("pubDate") or content.get("providerPublishTime")
            else:
                title = str((item or {}).get("title") or "").strip()
                pub = (item or {}).get("providerPublishTime")
            if not title:
                continue
            if pub is not None:
                try:
                    if isinstance(pub, (int, float)):
                        ts = datetime.fromtimestamp(float(pub), tz=timezone.utc)
                    else:
                        ts = pd.to_datetime(pub, utc=True).to_pydatetime()
                    if ts < cutoff:
                        continue
                except Exception:  # noqa: BLE001
                    pass
            headlines.append(title)
    except Exception:  # noqa: BLE001
        return headlines
    return headlines


def fetch_headlines(ticker: str, lookback_hours: int = 48) -> list[str]:
    """Headlines: Tiingo (if keyed) → Finnhub → Yahoo fallback."""
    for fetcher in (_fetch_tiingo_headlines, _fetch_finnhub_headlines, _fetch_yahoo_headlines):
        headlines = fetcher(ticker, lookback_hours)
        if headlines:
            return headlines
    return []


def news_source_status() -> dict[str, Any]:
    """Which news backends are configured (does not call APIs)."""
    return {
        "tiingo": bool(_tiingo_key()),
        "finnhub": bool(_finnhub_key()),
        "yahoo_fallback": True,
        "primary": (
            "tiingo" if _tiingo_key() else ("finnhub" if _finnhub_key() else "yahoo")
        ),
    }


def tiingo_configured() -> bool:
    return bool(_tiingo_key())


def _default_bearish_keywords() -> list[str]:
    return [
        "investigation",
        "downgrade",
        "bankruptcy",
        "offering",
        "dilution",
        "fraud",
        "delisting",
        "guidance cut",
        "sued",
        "lawsuit",
        "miss",
        "warning",
        "layoff",
        "recall",
    ]


def _default_bullish_keywords() -> list[str]:
    return [
        "upgrade",
        "beat",
        "record",
        "partnership",
        "buyback",
        "contract win",
        "approval",
        "breakthrough",
        "surge",
        "outperform",
        "raises guidance",
        "acquisition",
        " dividend",
    ]


def _default_gem_keywords() -> list[str]:
    return [
        "upgrade",
        "beat",
        "partnership",
        "buyback",
        "contract win",
        "hidden gem",
        "undervalued",
        "breakout",
    ]


def next_earnings_date(ticker: str) -> date | None:
    """Best-effort next/most-recent earnings date via Yahoo."""
    try:
        t = yf.Ticker(ticker)
        if hasattr(t, "get_earnings_dates"):
            ed = t.get_earnings_dates(limit=8)
            if ed is not None and not ed.empty:
                idx = pd.DatetimeIndex(pd.to_datetime(ed.index)).tz_localize(None)
                today = pd.Timestamp(date.today())
                future = idx[idx >= today - pd.Timedelta(days=5)]
                if len(future):
                    return future.min().date()
                return idx.max().date()
        cal = getattr(t, "calendar", None)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for key in ("Earnings Date", "Earnings Date.1"):
                if key in cal.index:
                    val = cal.loc[key].iloc[0]
                    ts = pd.to_datetime(val, errors="coerce")
                    if pd.notna(ts):
                        return ts.date()
        info = getattr(t, "info", None) or {}
        for key in ("earningsTimestamp", "earningsTimestampStart", "earningsDate"):
            raw = info.get(key)
            if raw is None:
                continue
            if isinstance(raw, (list, tuple)) and raw:
                raw = raw[0]
            if isinstance(raw, (int, float)) and raw > 0:
                return datetime.fromtimestamp(float(raw), tz=timezone.utc).date()
            ts = pd.to_datetime(raw, errors="coerce")
            if pd.notna(ts):
                return ts.date()
    except Exception:  # noqa: BLE001
        return None
    return None


def earnings_veto(
    ticker: str,
    *,
    as_of: date | None = None,
    days_before: int = 2,
    days_after: int = 1,
) -> NewsVeto:
    as_of = as_of or date.today()
    earn = next_earnings_date(ticker)
    if earn is None:
        return NewsVeto(False, "", "no_earnings_date")
    start = (pd.Timestamp(earn) - pd.tseries.offsets.BDay(days_before)).date()
    end = (pd.Timestamp(earn) + pd.tseries.offsets.BDay(days_after)).date()
    if start <= as_of <= end:
        return NewsVeto(
            True,
            "earnings",
            f"earnings={earn.isoformat()} window={start}..{end}",
        )
    return NewsVeto(False, "", f"earnings={earn.isoformat()} clear")


def earnings_proximity_score(
    ticker: str,
    *,
    as_of: date | None = None,
    days_before: int = 2,
    days_after: int = 1,
) -> float:
    """Small penalty near earnings; small boost when clear."""
    as_of = as_of or date.today()
    earn = next_earnings_date(ticker)
    if earn is None:
        return 0.0
    start = (pd.Timestamp(earn) - pd.tseries.offsets.BDay(days_before)).date()
    end = (pd.Timestamp(earn) + pd.tseries.offsets.BDay(days_after)).date()
    if start <= as_of <= end:
        return -0.25
    days_away = abs((earn - as_of).days)
    if days_away >= 10:
        return 0.05
    return 0.0


def fetch_headlines_extended(ticker: str, short_hours: int = 48, long_hours: int = 168) -> tuple[list[str], list[str]]:
    """Recent headlines + longer window for velocity baseline."""
    recent = fetch_headlines(ticker, lookback_hours=short_hours)
    long = fetch_headlines(ticker, lookback_hours=long_hours)
    return recent, long


def headline_veto(
    ticker: str,
    *,
    keywords: list[str] | None = None,
    lookback_hours: int = 48,
) -> NewsVeto:
    kws = [k.lower().strip() for k in (keywords or _default_bearish_keywords()) if k.strip()]
    if not kws:
        return NewsVeto(False, "", "no_keywords")
    titles = fetch_headlines(ticker, lookback_hours=lookback_hours)
    if not titles:
        return NewsVeto(False, "", "no_headlines")
    for title in titles:
        low = title.lower()
        for kw in kws:
            if kw in low:
                return NewsVeto(True, "news", f"keyword={kw!r} title={title[:120]}")
    return NewsVeto(False, "", f"headlines_checked={len(titles)}")


def _count_keyword_hits(titles: list[str], keywords: list[str]) -> int:
    hits = 0
    for title in titles:
        low = title.lower()
        for kw in keywords:
            if kw in low:
                hits += 1
                break
    return hits


def _headline_sentiment(
    titles: list[str],
    bullish: list[str],
    bearish: list[str],
) -> float:
    if not titles:
        return 0.0
    bull = _count_keyword_hits(titles, bullish)
    bear = _count_keyword_hits(titles, bearish)
    total = bull + bear
    if total == 0:
        return 0.0
    return float(bull - bear) / float(total)


def _headline_velocity(recent: list[str], baseline: list[str]) -> float:
    """Unusual headline activity: recent count vs 7-day average (scaled 0–1)."""
    if not baseline:
        return 0.0
    short_n = len(recent)
    # baseline includes recent; approximate daily rate over 7 days
    daily_avg = max(len(set(baseline)) / 7.0, 0.5)
    ratio = short_n / (daily_avg * 2)  # 48h window ≈ 2 days
    return float(min(1.0, max(0.0, (ratio - 1.0) / 2.0)))  # >avg activity → positive


def load_weekend_notes(settings: Settings, as_of: date | None = None) -> dict[str, Any]:
    cfg = _news_cfg(settings)
    rel = cfg.get("weekend_notes_path", "data/journal/weekend_notes.yaml")
    path = Path(rel)
    if not path.is_absolute():
        from stockpro.config import ROOT

        path = ROOT / path
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}
    note_date = data.get("as_of")
    if note_date and as_of:
        try:
            nd = date.fromisoformat(str(note_date)[:10])
            if (as_of - nd).days > 7:
                return {}  # stale notes
        except ValueError:
            pass
    return data


def weekend_bias_for_ticker(notes: dict[str, Any], ticker: str) -> float:
    if not notes:
        return 0.0
    bias = 0.0
    macro = str(notes.get("macro_bias", "neutral")).lower()
    if macro == "bullish":
        bias += 0.05
    elif macro == "cautious":
        bias -= 0.05
    ticker_bias = notes.get("ticker_bias") or {}
    if ticker in ticker_bias:
        bias += float(ticker_bias[ticker])
    sector_notes = notes.get("sector_notes") or {}
    # Direct ticker match in sector notes (user may put ticker as key)
    if ticker in sector_notes:
        text = str(sector_notes[ticker]).lower()
        if any(w in text for w in ("avoid", "caution", "volatile", "skip")):
            bias -= 0.15
        elif any(w in text for w in ("support", "bullish", "positive", "strong")):
            bias += 0.10
    return float(max(-1.0, min(1.0, bias)))


def score_ticker_news(
    settings: Settings,
    ticker: str,
    *,
    as_of: date | None = None,
    signal_direction: int = 0,
) -> NewsScore:
    """Compute news score for ranking. signal_direction: 1 bull, -1 bear, 0 flat."""
    cfg = _news_cfg(settings)
    as_of = as_of or date.today()
    lookback = int(cfg.get("headline_lookback_hours", 48))
    bullish = [k.lower() for k in (cfg.get("bullish_keywords") or _default_bullish_keywords())]
    bearish = [k.lower() for k in (cfg.get("headline_keywords") or _default_bearish_keywords())]

    recent, long = fetch_headlines_extended(ticker, short_hours=lookback, long_hours=168)
    sentiment = _headline_sentiment(recent, bullish, bearish)
    velocity = _headline_velocity(recent, long)
    earn_adj = earnings_proximity_score(
        ticker,
        as_of=as_of,
        days_before=int(cfg.get("earnings_block_days_before", 2)),
        days_after=int(cfg.get("earnings_block_days_after", 1)),
    )
    notes = load_weekend_notes(settings, as_of=as_of)
    wk_bias = weekend_bias_for_ticker(notes, ticker.upper())

    bull_hits = _count_keyword_hits(recent, bullish)
    bear_hits = _count_keyword_hits(recent, bearish)

    # Gem boost: unusual activity + bullish keyword
    gem_kws = [k.lower() for k in (cfg.get("gem_keywords") or _default_gem_keywords())]
    gem_hit = _count_keyword_hits(recent, gem_kws) > 0
    gem_boost = 0.15 if (gem_hit and velocity > 0.2) else 0.0

    raw = 0.50 * sentiment + 0.25 * velocity + earn_adj + wk_bias + gem_boost
    score = float(max(-1.0, min(1.0, raw)))
    top = recent[0] if recent else ""

    # Align score with signal direction for combined ranking
    if signal_direction == 1 and score < 0:
        detail = "bullish_signal_vs_negative_news"
    elif signal_direction == -1 and score > 0:
        detail = "bearish_signal_vs_positive_news"
    else:
        detail = f"sentiment={sentiment:.2f} velocity={velocity:.2f}"

    return NewsScore(
        ticker=ticker,
        score=score,
        sentiment=sentiment,
        velocity=velocity,
        earnings_adj=earn_adj,
        weekend_bias=wk_bias,
        headline_count=len(recent),
        bullish_hits=bull_hits,
        bearish_hits=bear_hits,
        top_headline=top[:200],
        headlines=recent[:5],
        detail=detail,
    )


def aligned_news_contribution(score: NewsScore, signal_direction: int, news_weight: float) -> float:
    """Direction-aligned news boost/penalty for combined ranking."""
    if signal_direction == 0:
        return 0.0
    # Positive when news agrees with signal direction
    aligned = score.score * signal_direction
    return news_weight * aligned


def combined_rank_score(confidence: float, news: NewsScore, signal_direction: int, news_weight: float) -> float:
    return confidence + aligned_news_contribution(news, signal_direction, news_weight)


def entry_news_veto(settings: Settings, ticker: str, as_of: date | None = None) -> NewsVeto:
    """Combined earnings + headline veto used before order sizing."""
    cfg = _news_cfg(settings)
    if not cfg.get("enabled", True):
        return NewsVeto(False, "", "news_disabled")

    earn = earnings_veto(
        ticker,
        as_of=as_of,
        days_before=int(cfg.get("earnings_block_days_before", 2)),
        days_after=int(cfg.get("earnings_block_days_after", 1)),
    )
    if earn.blocked:
        return earn

    news = headline_veto(
        ticker,
        keywords=list(cfg.get("headline_keywords") or _default_bearish_keywords()),
        lookback_hours=int(cfg.get("headline_lookback_hours", 48)),
    )
    if news.blocked:
        return news
    detail = "; ".join(x for x in (earn.detail, news.detail) if x)
    return NewsVeto(False, "", detail)


def news_features_path(settings: Settings) -> Path:
    journal_dir = Path((settings.get("journal", default={}) or {}).get("directory", "data/journal"))
    path = journal_dir / "news_features.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def log_news_features(settings: Settings, news: NewsScore, as_of: date | None = None) -> None:
    """Append one row for Phase 2 retrain dataset."""
    as_of = as_of or date.today()
    row = {
        "date": as_of.isoformat(),
        "ticker": news.ticker,
        "headline_count": news.headline_count,
        "bullish_hits": news.bullish_hits,
        "bearish_hits": news.bearish_hits,
        "news_score": round(news.score, 4),
        "sentiment": round(news.sentiment, 4),
        "velocity": round(news.velocity, 4),
        "earnings_adj": round(news.earnings_adj, 4),
        "weekend_bias": round(news.weekend_bias, 4),
        "top_headline": news.top_headline,
        "logged_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    path = news_features_path(settings)
    df = pd.DataFrame([row], columns=NEWS_FEATURE_COLUMNS)
    if path.exists():
        prev = pd.read_csv(path)
        for c in NEWS_FEATURE_COLUMNS:
            if c not in prev.columns:
                prev[c] = ""
        prev = prev[
            ~((prev["date"].astype(str) == row["date"]) & (prev["ticker"].astype(str) == row["ticker"]))
        ]
        df = pd.concat([prev[NEWS_FEATURE_COLUMNS], df], ignore_index=True)
    df.to_csv(path, index=False)


def load_news_features_table(settings: Settings) -> pd.DataFrame:
    path = news_features_path(settings)
    if not path.exists():
        return pd.DataFrame(columns=NEWS_FEATURE_COLUMNS)
    return pd.read_csv(path)


def _tiingo_news_articles(ticker: str, start: date, end: date) -> list[dict[str, Any]]:
    """Raw Tiingo news articles for ticker in [start, end)."""
    key = _tiingo_key()
    if not key:
        return []
    try:
        qs = urlencode(
            {
                "tickers": ticker.lower(),
                "startDate": start.isoformat(),
                "endDate": end.isoformat(),
                "limit": 1000,
                "sortBy": "publishedDate",
            }
        )
        url = f"https://api.tiingo.com/tiingo/news?{qs}"
        req = Request(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Token {key}",
                "User-Agent": "StockProGPT/2.0",
            },
        )
        with urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload if isinstance(payload, list) else []
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        print(f"[news] Tiingo backfill failed for {ticker}: {exc}")
        return []


def _daily_features_from_titles(
    titles: list[str],
    titles_7d: list[str],
    *,
    bullish: list[str],
    bearish: list[str],
    gem: list[str],
) -> dict[str, Any]:
    sentiment = _headline_sentiment(titles, bullish, bearish)
    velocity = _headline_velocity(titles, titles_7d)
    bull_hits = _count_keyword_hits(titles, bullish)
    bear_hits = _count_keyword_hits(titles, bearish)
    gem_hit = _count_keyword_hits(titles, gem) > 0
    gem_boost = 0.15 if (gem_hit and velocity > 0.2) else 0.0
    raw = 0.50 * sentiment + 0.25 * velocity + gem_boost
    score = float(max(-1.0, min(1.0, raw)))
    return {
        "headline_count": len(titles),
        "bullish_hits": bull_hits,
        "bearish_hits": bear_hits,
        "news_score": round(score, 4),
        "sentiment": round(sentiment, 4),
        "velocity": round(velocity, 4),
        "earnings_adj": 0.0,
        "weekend_bias": 0.0,
        "top_headline": (titles[0][:200] if titles else ""),
    }


def backfill_tiingo_news_features(
    settings: Settings,
    tickers: list[str],
    *,
    lookback_days: int = 90,
    persist: bool = True,
) -> pd.DataFrame:
    """Pull Tiingo news and build daily feature rows for model training.

    Individual Tiingo plans typically allow ~3 months of news history.
    """
    cfg = _news_cfg(settings)
    if not _tiingo_key():
        print("[news] TIINGO_API_KEY missing — cannot backfill")
        return pd.DataFrame(columns=NEWS_FEATURE_COLUMNS)

    bullish = [k.lower() for k in (cfg.get("bullish_keywords") or _default_bullish_keywords())]
    bearish = [k.lower() for k in (cfg.get("headline_keywords") or _default_bearish_keywords())]
    gem = [k.lower() for k in (cfg.get("gem_keywords") or _default_gem_keywords())]

    end = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=int(lookback_days))
    rows: list[dict[str, Any]] = []
    logged_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    for i, ticker in enumerate(tickers):
        articles = _tiingo_news_articles(ticker, start, end)
        by_day: dict[date, list[str]] = {}
        for item in articles:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            pub = item.get("publishedDate") or item.get("crawlDate")
            try:
                d = pd.to_datetime(pub, utc=True).date()
            except Exception:  # noqa: BLE001
                continue
            by_day.setdefault(d, []).append(title)

        # All calendar days in window so missing news = zeros (honest)
        days = pd.bdate_range(start, date.today())
        for ts in days:
            d = ts.date()
            titles = by_day.get(d, [])
            # trailing 7 calendar days of titles for velocity baseline
            titles_7d: list[str] = []
            for j in range(7):
                titles_7d.extend(by_day.get(d - timedelta(days=j), []))
            feats = _daily_features_from_titles(
                titles, titles_7d, bullish=bullish, bearish=bearish, gem=gem
            )
            rows.append(
                {
                    "date": d.isoformat(),
                    "ticker": ticker,
                    **feats,
                    "logged_at": logged_at,
                }
            )
        print(f"[news] backfill {ticker}: {len(articles)} articles ({i + 1}/{len(tickers)})")

    df = pd.DataFrame(rows, columns=NEWS_FEATURE_COLUMNS)
    if persist and not df.empty:
        path = news_features_path(settings)
        if path.exists():
            prev = pd.read_csv(path)
            for c in NEWS_FEATURE_COLUMNS:
                if c not in prev.columns:
                    prev[c] = ""
            # Replace overlapping (date, ticker) with backfill
            keys = set(zip(df["date"].astype(str), df["ticker"].astype(str)))
            mask = [
                (str(r["date"]), str(r["ticker"])) not in keys
                for _, r in prev.iterrows()
            ]
            prev = prev.loc[mask]
            df = pd.concat([prev[NEWS_FEATURE_COLUMNS], df], ignore_index=True)
        df.to_csv(path, index=False)
        print(f"[news] wrote {len(df)} rows -> {path}")
    return df


def news_feature_dict_for_model(news: NewsScore) -> dict[str, float]:
    """Map NewsScore to model NEWS_FEATURE_COLUMNS (from features module)."""
    return {
        "news_score": float(news.score),
        "sentiment": float(news.sentiment),
        "velocity": float(news.velocity),
        "headline_count": float(news.headline_count),
        "bullish_hits": float(news.bullish_hits),
        "bearish_hits": float(news.bearish_hits),
        "weekend_bias": float(news.weekend_bias),
    }
