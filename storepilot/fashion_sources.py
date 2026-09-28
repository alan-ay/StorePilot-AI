"""Bounded, read-only research tools. Provider responses never enter the journal raw."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd


class ResearchError(ValueError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_json(url, *, payload=None, headers=None):
    request = Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with build_opener(NoRedirect).open(request, timeout=35) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ResearchError("Response too large; try a narrower question.")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ResearchError("The provider returned an unexpected response.")
        return result
    except HTTPError as exc:
        # URLs and provider error bodies may include API keys. Never echo them.
        code = exc.code
        exc.close()
        raise ResearchError(f"Provider HTTP {code}. Check access, quota and credentials.") from None
    except ResearchError:
        raise
    except (URLError, TimeoutError, OSError, UnicodeError, ValueError):
        raise ResearchError("The provider could not be reached or returned invalid JSON.") from None


def public_url(value):
    try:
        parts = urlsplit(str(value))
        if (
            parts.scheme not in {"https", "http"}
            or not parts.hostname
            or parts.username
            or parts.password
        ):
            return ""
        if parts.hostname in {"localhost", "127.0.0.1", "::1"}:
            return ""
        query = [
            (k, v)
            for k, v in parse_qsl(parts.query)
            if k.lower() not in {"api_key", "key", "token", "access_token", "credential", "secret"}
        ]
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
    except ValueError:
        return ""


def clean_query(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 150:
        raise ResearchError("Use a search phrase of 1–150 characters.")
    return " ".join(value.split())


def records(value):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ResearchError("The data provider returned an unexpected result format.")
    return [row for row in value if isinstance(row, dict)]


def interest_summary(points, today):
    """Compare complete weekly windows within one independently scaled series."""
    rows = []
    for point in records(points):
        try:
            stamp = pd.Timestamp(int(point["timestamp"]), unit="s", tz="UTC").normalize()
            values = point.get("values", [])
            value = float(values[0]["extracted_value"])
            if point.get("isPartial") or point.get("is_partial") or values[0].get("is_partial"):
                continue
            if (
                math.isfinite(value)
                and 0 <= value <= 100
                and stamp <= pd.Timestamp(today, tz="UTC") - pd.Timedelta(days=7)
            ):
                rows.append((stamp, value))
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            continue
    frame = (
        pd.DataFrame(rows, columns=["date", "interest"]).drop_duplicates("date").sort_values("date")
    )
    result = {"points": [], "change_pct": None, "status": "insufficient_history"}
    if frame.empty:
        return result
    gaps = frame.date.diff().dt.days.dropna()
    # The tool requests 12 months, normally weekly. Do not mistake daily/monthly
    # points for weeks or quietly fill gaps in a returned series.
    if gaps.empty or not gaps.between(6, 8).all():
        result["status"] = "irregular_history"
        return result
    result["points"] = [
        {"date": row.date.date().isoformat(), "interest": row.interest}
        for row in frame.itertuples()
    ]
    if len(frame) < 8 or (pd.Timestamp(today, tz="UTC") - frame.date.max()).days > 21:
        return result
    previous, recent = frame.interest.iloc[-8:-4], frame.interest.iloc[-4:]
    result.update(
        previous_mean=float(previous.mean()), recent_mean=float(recent.mean()), status="low_volume"
    )
    if previous.mean() >= 5 and recent.gt(0).sum() >= 3:
        result.update(
            change_pct=round(float((recent.mean() / previous.mean() - 1) * 100), 1), status="ok"
        )
    return result


def price_currency(label, country):
    label = str(label).upper()
    for code, symbol in [("EUR", "€"), ("GBP", "£"), ("USD", "US$"), ("CHF", "CHF"), ("PLN", "ZŁ")]:
        if code in label or symbol in label:
            return code
    if "$" in label and country == "US":
        return "USD"
    return None


class FashionSources:
    def __init__(self, api_key, *, transport=request_json, today=None):
        if not api_key.strip():
            raise ResearchError("A SerpApi data key is required for live research.")
        self._key = api_key.strip()
        self.transport = transport
        self.today = today or datetime.now(UTC).date()

    def _search(self, **params):
        data = self.transport(
            "https://serpapi.com/search.json?" + urlencode(dict(params, api_key=self._key))
        )
        if data.get("error"):
            raise ResearchError(
                "The data provider could not complete this search. Check quota or narrow the query."
            )
        return data

    def web(self, query, profile, channel="web"):
        query = clean_query(query)
        if channel not in {"web", "instagram"}:
            raise ResearchError("Unknown search channel.")
        if channel == "instagram":
            query += " site:instagram.com"
        data = self._search(
            engine="google",
            q=query,
            gl=profile["country"].lower(),
            hl=profile["search_language"],
            tbs="qdr:m3",
            num=8,
        )
        items, seen = [], set()
        for row in records(data.get("organic_results"))[:8]:
            url = public_url(row.get("link", ""))
            if not url or url in seen:
                continue
            seen.add(url)
            items.append(
                {
                    "kind": "web_excerpt",
                    "title": str(row.get("title", ""))[:200],
                    "url": url,
                    "excerpt": str(row.get("snippet", ""))[:350],
                    "published_label": str(row.get("date", ""))[:60],
                    "query": query,
                    "country": profile["country"],
                    "coverage": "Public search-index excerpts only; not platform-wide engagement or sales.",
                }
            )
        return items

    def interest(self, query, country):
        query = clean_query(query)
        if len(query) > 100 or "," in query:
            raise ResearchError("Interest checks accept one phrase, up to 100 characters.")
        if not re.fullmatch(r"[A-Z]{2}(?:-[A-Z0-9]{1,5})?", country):
            raise ResearchError("Use a valid country or Google Trends region code.")
        data = self._search(
            engine="google_trends",
            q=query,
            geo=country,
            date="today 12-m",
            data_type="TIMESERIES",
            tz=0,
        )
        summary = interest_summary(
            data.get("interest_over_time", {}).get("timeline_data", []), self.today
        )
        return [
            {
                "kind": "search_interest",
                "title": f"{query} · {country}",
                "url": "https://trends.google.com/trends/explore?"
                + urlencode({"q": query, "geo": country, "date": "today 12-m"}),
                "query": query,
                "country": country,
                **summary,
                "coverage": "Normalized search interest, not sales or approval. Compare growth within this series, not raw levels across separate queries or countries. No arrival-time estimate is established.",
            }
        ]

    def prices(self, query, profile):
        query = clean_query(query)
        data = self._search(
            engine="google_shopping",
            q=query,
            gl=profile["country"].lower(),
            hl=profile["search_language"],
        )
        rows = records(data.get("shopping_results"))
        for group in records(data.get("categorized_shopping_results")):
            rows.extend(records(group.get("shopping_results")))
        items, seen = [], set()
        for row in rows[:30]:
            url = public_url(row.get("product_link") or row.get("link", ""))
            identity = (str(row.get("source", "")), str(row.get("title", "")))
            if not url or identity in seen:
                continue
            seen.add(identity)
            try:
                amount = float(row.get("extracted_price"))
                if not math.isfinite(amount) or amount <= 0:
                    continue
            except (ValueError, TypeError):
                continue
            currency = price_currency(row.get("price", ""), profile["country"])
            items.append(
                {
                    "kind": "retailer_listing",
                    "title": str(row.get("title", ""))[:200],
                    "url": url,
                    "retailer": str(row.get("source", ""))[:100],
                    "price": amount,
                    "price_label": str(row.get("price", ""))[:60],
                    "currency": currency,
                    "within_budget": profile["price_min"] <= amount <= profile["price_max"]
                    if currency == profile["currency"]
                    else None,
                    "query": query,
                    "country": profile["country"],
                    "coverage": "A retail asking price, not a supplier quote or proof of demand. Check product, sizes, shipping and currency yourself.",
                }
            )
            if len(items) == 6:
                break
        return items
