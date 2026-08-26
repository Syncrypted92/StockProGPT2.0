"""Scan Tiingo news for speculative / event-driven 'lotto' catalysts (research only)."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv

from stockpro.config import ROOT, load_settings
from stockpro.news import _tiingo_key

# Speculative / event keywords (not investment advice)
BULLISHISH = [
    r"\bacqui",
    r"\bmerger\b",
    r"\btakeover\b",
    r"\bbid for\b",
    r"\bfda\b",
    r"\bapprov",
    r"\bclearance\b",
    r"\bbreakthrough\b",
    r"\bbeat(s|ing)?\b.*estimat",
    r"\braises?\b.*guidance\b",
    r"\braises?\b.*outlook\b",
    r"\bupgrade[ds]?\b",
    r"\bprice target raised\b",
    r"\bbuyout\b",
    r"\bpartnership\b",
    r"\bcontract win\b",
    r"\bawarded\b",
    r"\brecord (revenue|sales|profit)",
    r"\bshort squeeze\b",
    r"\bstock split\b",
]
BEARISHISH = [
    r"\bsec\b.*charg",
    r"\bfraud\b",
    r"\binvestigat",
    r"\blawsuit\b",
    r"\bsued\b",
    r"\brecall\b",
    r"\bmiss(es|ed)?\b.*estimat",
    r"\bcuts?\b.*guidance\b",
    r"\blower(s|ed)?\b.*outlook\b",
    r"\bdowngrade[ds]?\b",
    r"\bbankrupt",
    r"\blayoff",
    r"\bwarns?\b",
    r"\bprobe\b",
    r"\bhalt(ed|s)?\b.*trad",
    r"\bdelist",
    r"\bceo (steps down|resign|oust)",
]
VOL_EVENT = [
    r"\bearnings\b",
    r"\beps\b",
    r"\brevenue\b",
    r"\bfomc\b",
    r"\bcpi\b",
    r"\bjobs report\b",
    r"\bnfp\b",
    r"\brate cut\b",
    r"\brate hike\b",
    r"\bopec\b",
    r"\btariff",
    r"\bsanction",
    r"\bcrypto\b",
    r"\bbitcoin\b",
    r"\bai\b",
    r"\bgpu\b",
    r"\bchip\b",
]


def _score(title: str) -> tuple[int, str, list[str]]:
    t = title.lower()
    tags = []
    score = 0
    side = "neutral"
    for pat in BULLISHISH:
        if re.search(pat, t, re.I):
            score += 3
            tags.append(pat.strip("\\b").replace("\\", "")[:24])
            side = "bullish_catalyst"
    for pat in BEARISHISH:
        if re.search(pat, t, re.I):
            score += 3
            tags.append(pat.strip("\\b").replace("\\", "")[:24])
            side = "bearish_catalyst" if side == "neutral" else "mixed_catalyst"
    for pat in VOL_EVENT:
        if re.search(pat, t, re.I):
            score += 1
            tags.append("event:" + pat.strip("\\b").replace("\\", "")[:16])
    # lottery vibe: micro/meme-ish names get slight bump only if catalyst present
    return score, side, tags[:6]


def fetch_articles(ticker: str, lookback_hours: int = 48) -> list[dict]:
    key = _tiingo_key()
    if not key:
        return []
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
    try:
        with urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  [{ticker}] fail: {exc}")
        return []
    if not isinstance(payload, list):
        return []
    cutoff = now - timedelta(hours=lookback_hours)
    out = []
    for item in payload:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        pub = item.get("publishedDate") or item.get("crawlDate")
        ts = None
        if pub:
            try:
                ts = pd.to_datetime(pub, utc=True).to_pydatetime()
                if ts < cutoff:
                    continue
            except Exception:  # noqa: BLE001
                pass
        src = ""
        if isinstance(item.get("source"), str):
            src = item["source"]
        elif isinstance(item.get("source"), list) and item["source"]:
            src = str(item["source"][0])
        raw_tickers = item.get("tickers") or item.get("tags") or []
        if isinstance(raw_tickers, str):
            tagged = [raw_tickers.upper()]
        elif isinstance(raw_tickers, list):
            tagged = [str(x).upper() for x in raw_tickers]
        else:
            tagged = []
        out.append(
            {
                "ticker": ticker,
                "title": title,
                "published": str(ts) if ts else str(pub),
                "source": src,
                "url": str(item.get("url") or "")[:120],
                "tagged": tagged,
            }
        )
    return out


# Common alias so "Tesla …" counts for TSLA when symbol missing from title
_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "TSLA": ("TESLA",),
    "NVDA": ("NVIDIA", "NVIDA"),
    "GOOGL": ("GOOGLE", "ALPHABET"),
    "GOOG": ("GOOGLE", "ALPHABET"),
    "META": ("FACEBOOK", "META PLATFORMS"),
    "AMZN": ("AMAZON",),
    "AAPL": ("APPLE",),
    "MSFT": ("MICROSOFT",),
    "AMD": ("ADVANCED MICRO",),
    "INTC": ("INTEL",),
    "MU": ("MICRON",),
    "PLTR": ("PALANTIR",),
    "PYPL": ("PAYPAL",),
    "UBER": ("UBER TECHNOLOGIES", "UBER "),
    "RIVN": ("RIVIAN",),
    "SOFI": ("SOFI ", "SOFI TECHNOLOGIES"),
    "HOOD": ("ROBINHOOD",),
    "NCLH": ("NORWEGIAN CRUISE",),
    "CCL": ("CARNIVAL",),
    "BAC": ("BANK OF AMERICA",),
    # WFC: do not alias "Wells Fargo" — floods with analyst PT spam; need WFC / $WFC in title
    "PFE": ("PFIZER",),
    "GM": ("GENERAL MOTORS",),
    "F": ("FORD MOTOR", " FORD "),
}


def article_relevant(ticker: str, title: str, tagged: list[str] | None = None) -> bool:
    """Drop mistags where Tiingo attaches an unrelated name (analyst spam, peers)."""
    t = ticker.upper()
    tagged = [x.upper() for x in (tagged or [])]
    title_u = f" {title.upper()} "
    if tagged and t not in tagged:
        return False
    title_has_sym = bool(re.search(rf"(?<![A-Z]){re.escape(t)}(?![A-Z])", title_u)) or (
        f"${t}" in title_u
    )
    aliases = _NAME_ALIASES.get(t, ())
    title_has_name = any(a in title_u for a in aliases)
    if title_has_sym or title_has_name:
        return True
    return False


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not _tiingo_key():
        raise SystemExit("TIINGO_API_KEY not set")
    settings = load_settings()
    tickers = list(settings.get("universe", "tickers", default=[]) or [])
    # Prefer names that can move hard on news for 'lotto' screen
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
    ordered = [t for t in prefer if t in tickers] + [t for t in tickers if t not in prefer]

    print(f"Scanning Tiingo news last 48h for {len(ordered)} tickers...\n")
    rows = []
    for i, t in enumerate(ordered, 1):
        arts = fetch_articles(t, 48)
        print(f"[{i}/{len(ordered)}] {t}: {len(arts)} articles", flush=True)
        for a in arts:
            if not article_relevant(t, a["title"], a.get("tagged") or []):
                continue
            score, side, tags = _score(a["title"])
            if score < 3:
                continue
            rows.append({**a, "score": score, "bias": side, "tags": ",".join(tags)})

    if not rows:
        print("No high-score catalyst headlines in last 48h.")
        return

    df = pd.DataFrame(rows).sort_values(["score", "published"], ascending=[False, False])
    # dedupe similar titles
    df = df.drop_duplicates(subset=["ticker", "title"])
    top = df.head(40)
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.width", 220)
    print("\n=== Top catalyst / lotto-flavored headlines ===\n")
    for _, r in top.iterrows():
        print(f"[{r['score']}] {r['ticker']:5s} {r['bias']:18s} | {r['title']}")
        print(f"      {r['published']}  {r['source']}")

    by = df.groupby("ticker").agg(n=("title", "count"), max_score=("score", "max")).sort_values(
        ["max_score", "n"], ascending=False
    )
    print("\n=== Tickers with strongest catalyst scores ===")
    print(by.head(15).to_string())

    out = ROOT / "artifacts" / "tiingo_lotto_news_scan.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(
        "\nNOTE: This is a headline screen only — not a trade signal. "
        "Your spy-day bot does not auto-trade these. Lotto options are high odds of zero."
    )


if __name__ == "__main__":
    main()
