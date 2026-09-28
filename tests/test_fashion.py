import json
from datetime import UTC, date, datetime, timedelta
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from storepilot.fashion_agent import (
    FashionJournal,
    GeminiResearchModel,
    journal_path,
    research_turn,
    validate_answer,
    validate_profile,
)
from storepilot.fashion_sources import (
    FashionSources,
    ResearchError,
    interest_summary,
    public_url,
    request_json,
)


@pytest.fixture
def profile():
    return {
        "country": "DE",
        "currency": "EUR",
        "search_language": "de",
        "location": "Bonn",
        "customers": "Students",
        "focus": "Everyday clothing",
        "price_min": 20.0,
        "price_max": 60.0,
    }


def weekly_points(values, today=date(2026, 9, 27)):
    return [
        {
            "timestamp": str(
                int(
                    datetime.combine(
                        today - timedelta(weeks=len(values) - i), datetime.min.time(), tzinfo=UTC
                    ).timestamp()
                )
            ),
            "values": [{"query": "jeans", "extracted_value": value}],
        }
        for i, value in enumerate(values)
    ]


def test_interest_uses_complete_weeks_and_never_fills_gaps():
    today = date(2026, 9, 27)
    points = weekly_points([10] * 4 + [20] * 4)
    partial = {
        "timestamp": str(int(datetime(2026, 9, 27, tzinfo=UTC).timestamp())),
        "values": [{"extracted_value": 100}],
    }
    result = interest_summary(points + [partial], today)
    assert result["change_pct"] == 100
    assert len(result["points"]) == 8
    assert interest_summary(points[:3] + points[4:], today)["status"] == "irregular_history"
    assert interest_summary(weekly_points([0] * 4 + [1] * 4), today)["change_pct"] is None
    assert interest_summary(points, date(2027, 1, 1))["status"] == "insufficient_history"


def test_malformed_interest_cannot_become_a_trend():
    result = interest_summary(
        [{"timestamp": "garbage"}, {"timestamp": "123", "values": []}], date(2026, 9, 27)
    )
    assert result["change_pct"] is None
    assert not result["points"]


def test_sources_preserve_evidence_and_do_not_mix_currencies(profile):
    calls = []

    def transport(url):
        calls.append(parse_qs(urlsplit(url).query))
        return {
            "shopping_results": [
                {
                    "title": "Wide-leg jeans",
                    "source": "Shop A",
                    "price": "€39",
                    "extracted_price": 39,
                    "link": "https://example.com/jeans",
                },
                {
                    "title": "Wide-leg jeans",
                    "source": "Shop A",
                    "price": "€39",
                    "extracted_price": 39,
                    "link": "https://example.com/duplicate",
                },
                {
                    "title": "Wide-leg jeans",
                    "source": "Shop B",
                    "price": "$25",
                    "extracted_price": 25,
                    "link": "https://example.org/jeans",
                },
                {
                    "title": "Wide-leg jeans",
                    "source": "Shop C",
                    "price": "€190",
                    "extracted_price": 190,
                    "link": "https://example.net/jeans",
                },
            ]
        }

    rows = FashionSources("test-secret", transport=transport).prices("wide-leg jeans", profile)
    assert len(rows) == 3
    assert [row["within_budget"] for row in rows] == [True, None, False]
    assert calls[0]["gl"] == ["de"]
    assert calls[0]["api_key"] == ["test-secret"]
    assert "test-secret" not in json.dumps(rows)


def test_public_links_cannot_embed_credentials_or_scripts():
    assert not public_url("javascript:alert(1)")
    assert not public_url("https://user:password@example.com/")
    assert not public_url("http://localhost:8000/")
    assert (
        public_url("https://example.com/item?api_key=secret&id=12")
        == "https://example.com/item?id=12"
    )


def test_provider_errors_do_not_expose_credentials(profile):
    with patch("storepilot.fashion_sources.build_opener") as opener:
        opener.return_value.open.side_effect = HTTPError(
            "https://provider/?api_key=private", 401, "private", {}, None
        )
        with pytest.raises(ResearchError) as caught:
            request_json("https://provider/?api_key=private")
    assert "401" in str(caught.value)
    assert "private" not in str(caught.value)
    source = FashionSources("private", transport=lambda _: {"error": "bad key private"})
    with pytest.raises(ResearchError) as caught:
        source.web("jeans", profile)
    assert "private" not in str(caught.value)


def call(name, args):
    return {"role": "model", "parts": [{"functionCall": {"name": name, "args": args}}]}


class FakeSources:
    today = date(2026, 9, 27)
    calls = 0

    def web(self, query, profile, channel):
        self.calls += 1
        return [
            {
                "kind": "web_excerpt",
                "title": "Everyday jeans",
                "url": "https://example.com/jeans",
                "excerpt": "A new jeans range",
                "query": query,
            }
        ]


class FakeModel:
    model = "test-model"

    def __init__(self):
        self.calls = 0
        self.contents = None

    def respond(self, contents, language):
        self.calls += 1
        self.contents = contents
        if self.calls == 1:
            return call("search_fashion", {"query": "everyday jeans", "channel": "web"})
        observations = contents[-1]["parts"][0]["functionResponse"]["response"]["evidence"]
        ref = next(iter(observations))
        return call(
            "finish",
            {
                "reply": "Here is a starting point.",
                "claims": [
                    {
                        "kind": "observation",
                        "text": "This retailer shows jeans.",
                        "evidence_ids": [ref],
                    }
                ],
                "question": "What fit do your customers prefer?",
            },
        )


def test_agent_uses_tools_then_returns_checkable_claims(profile):
    source, model = FakeSources(), FakeModel()
    progress = []
    turn = research_turn(
        "What jeans should I investigate?",
        profile,
        [],
        model,
        source,
        on_progress=lambda *args: progress.append(args),
    )
    assert model.calls == 2 and source.calls == 1
    assert turn["data_calls"] == 1
    assert turn["trace"][0]["status"] == "ok"
    assert progress == [("search_fashion", "everyday jeans")]
    assert turn["answer"]["claims"][0]["evidence_ids"][0] in turn["evidence"]
    assert turn["mode"] == "live"


def test_fabricated_citations_and_unsourced_facts_are_rejected():
    answer = {
        "reply": "",
        "claims": [
            {"text": "Everyone loves it", "kind": "observation", "evidence_ids": ["invented"]}
        ],
        "question": "",
    }
    with pytest.raises(ResearchError, match="not retrieved"):
        validate_answer(answer, {})
    answer["claims"][0]["evidence_ids"] = []
    with pytest.raises(ResearchError, match="missing"):
        validate_answer(answer, {})
    assert validate_answer({"reply": "Who shops with you?", "claims": [], "question": ""}, {})


def test_agent_tool_and_cost_boundaries(profile):
    class LoopModel:
        model = "loop"
        calls = 0

        def respond(self, contents, language):
            self.calls += 1
            return call("search_fashion", {"query": f"jeans {self.calls}", "channel": "web"})

    model, sources = LoopModel(), FakeSources()
    with pytest.raises(ResearchError, match="limit"):
        research_turn("Research jeans", profile, [], model, sources)
    assert model.calls == 7
    assert sources.calls == 6


def test_unknown_tools_and_repeats_never_execute(profile):
    class BadModel:
        model = "test"
        calls = 0

        def respond(self, contents, language):
            self.calls += 1
            if self.calls == 1:
                return call("order_products", {"query": "jeans"})
            if self.calls < 4:
                return call("search_fashion", {"query": "jeans", "channel": "web"})
            return call(
                "finish", {"reply": "Evidence is limited.", "claims": [], "question": "Which fit?"}
            )

    sources = FakeSources()
    turn = research_turn("jeans", profile, [], BadModel(), sources)
    assert sources.calls == 1
    assert [entry["status"] for entry in turn["trace"]] == ["error", "ok", "error"]


def test_followup_receives_previous_evidence(profile):
    first = research_turn("jeans", profile, [], FakeModel(), FakeSources())
    model = FakeModel()
    decisions = [
        {
            "item": "Jeans",
            "decision": "pass",
            "note": "Customers need more sizes",
            "created_at": "2026-09-27",
        }
    ]
    research_turn(
        "Any cheaper options?", profile, [first], model, FakeSources(), owner_decisions=decisions
    )
    context = json.loads(model.contents[0]["parts"][0]["text"])
    assert context["conversation"][0]["question"] == "jeans"
    assert context["previous_evidence"] == first["evidence"]
    assert context["owner_decisions"] == decisions


def test_invalid_brief_is_repaired_before_it_is_published(profile):
    class RepairModel(FakeModel):
        def respond(self, contents, language):
            if self.calls == 1:
                self.calls += 1
                return call(
                    "finish",
                    {
                        "reply": "",
                        "claims": [
                            {
                                "kind": "observation",
                                "text": "Jeans are listed.",
                                "evidence_ids": ["made-up"],
                            }
                        ],
                        "question": "",
                    },
                )
            if self.calls == 2:
                self.calls += 1
                assert "error" in contents[-1]["parts"][0]["functionResponse"]["response"]
                evidence = contents[2]["parts"][0]["functionResponse"]["response"]["evidence"]
                return call(
                    "finish",
                    {
                        "reply": "Here is the source.",
                        "claims": [
                            {
                                "kind": "observation",
                                "text": "Jeans are listed.",
                                "evidence_ids": [next(iter(evidence))],
                            }
                        ],
                        "question": "",
                    },
                )
            return super().respond(contents, language)

    turn = research_turn("jeans", profile, [], RepairModel(), FakeSources())
    assert turn["trace"][-1]["tool"] == "finish"
    assert "made-up" not in json.dumps(turn["answer"])


def test_gemini_request_is_structured_and_has_no_key_in_conversation():
    calls = []

    def transport(url, **kwargs):
        calls.append((url, kwargs))
        return {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": call("finish", {"reply": "Hello", "claims": [], "question": ""}),
                }
            ]
        }

    model = GeminiResearchModel("private-secret", transport=transport)
    model.respond([{"role": "user", "parts": [{"text": "jeans"}]}], "en")
    url, request = calls[0]
    assert "private-secret" not in url
    assert request["headers"]["x-goog-api-key"] == "private-secret"
    assert request["payload"]["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert "private-secret" not in json.dumps(request["payload"])


def test_journal_is_scoped_and_decisions_do_not_create_orders(tmp_path, profile):
    path = journal_path(tmp_path / "storepilot.db")
    journal = FashionJournal(path)
    turn = research_turn("jeans", profile, [], FakeModel(), FakeSources())
    journal.save(turn)
    journal.save(turn)
    assert len(journal.history(profile)) == 1
    assert journal.history(dict(profile, country="FR")) == []
    journal.decide(turn, "Wide-leg jeans", "watch", "Check sizing with regulars")
    assert journal.decisions(profile)[0]["decision"] == "watch"
    assert not (tmp_path / "storepilot.db").exists()
    assert journal_path(tmp_path / "storepilot.db", demo=True) != path


@pytest.mark.parametrize(
    "changes",
    [
        {"country": ""},
        {"currency": "USD"},
        {"price_min": float("nan")},
        {"price_max": 10},
        {"customers": "x" * 801},
    ],
)
def test_invalid_profile(changes, profile):
    with pytest.raises(ResearchError):
        validate_profile(dict(profile, **changes))
