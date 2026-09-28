import json
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest, app_test

from storepilot.fashion_agent import FashionJournal, journal_path
from storepilot.fashion_page import demo_turn
from storepilot.fashion_sources import ResearchError

APP = Path(__file__).resolve().parents[1] / "app.py"


def teardown_module():
    # Streamlit creates a module-level TemporaryDirectory, even for file-based tests.
    directory = getattr(app_test, "TMP_DIR", None)
    if directory is not None:
        directory.cleanup()


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("STOREPILOT_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("SERPAPI_API_KEY", "")
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["language"] = "en"
    app.run()
    assert not app.exception
    return app


def live_setup(app):
    app.radio(key="fashion_mode").set_value("live").run()
    app.selectbox(key="fashion_country").set_value("DE").run()
    app.text_input(key="fashion_gemini_key").set_value("gemini-test-secret")
    app.text_input(key="fashion_serpapi_key").set_value("serp-test-secret").run()


def test_fashion_is_landing_page_without_inventory_training_or_network(app, tmp_path):
    assert app.radio(key="nav").value == "时尚侦察员"
    assert app.chat_input(key="fashion_chat").disabled
    assert not (tmp_path / "test.db").exists()
    assert not (tmp_path / "test-demo.db").exists()
    with patch("storepilot.fashion_page.research_turn", side_effect=AssertionError("network")):
        app.button(key="fashion_example").click().run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert any("invented" in value.value for value in app.markdown)
    assert app.chat_input(key="fashion_chat").disabled


def test_live_research_requires_keys_and_market(app):
    app.radio(key="fashion_mode").set_value("live").run()
    assert app.chat_input(key="fashion_chat").disabled
    app.selectbox(key="fashion_country").set_value("FR").run()
    assert app.chat_input(key="fashion_chat").disabled
    app.text_input(key="fashion_gemini_key").set_value("model-secret")
    app.text_input(key="fashion_serpapi_key").set_value("data-secret").run()
    assert not app.chat_input(key="fashion_chat").disabled
    assert not app.exception


def test_example_decision_is_explicit_and_survives_language_switch(app, tmp_path):
    app.button(key="fashion_example").click().run()
    turn = app.session_state["fashion_demo_turn"]
    journal = FashionJournal(journal_path(tmp_path / "test.db", demo=True))
    app.text_input(key=f"fashion_item:{turn['id']}").set_value("Wide-leg jeans")
    app.text_area(key=f"fashion_note:{turn['id']}").set_value("Ask about sizing")
    app.radio(key=f"fashion_decision:{turn['id']}").set_value("trial").run()
    assert not journal.decisions(turn["profile"])
    app.radio(key="language").set_value("zh").run()
    assert not app.exception
    choice = app.radio(key=f"fashion_decision:{turn['id']}")
    assert choice.value == "trial"
    assert choice.proto.set_value
    assert choice.proto.raw_value == "考虑小批量试卖"
    assert app.text_area(key=f"fashion_note:{turn['id']}").value == "Ask about sizing"
    app.button(key=f"fashion_save:{turn['id']}").click().run()
    assert not app.exception
    assert journal.decisions(turn["profile"])[0]["note"] == "Ask about sizing"
    assert not (tmp_path / "test.db").exists()


def test_live_chat_persists_and_never_exports_keys(app, tmp_path):
    live_setup(app)
    captured = []

    def fake_research(question, profile, history, model, sources, **kwargs):
        captured.append((question, history, kwargs))
        turn = demo_turn(profile, "en")
        turn.update(
            id=f"fixture-{len(captured)}", question=question, mode="live", model="test-model"
        )
        return turn

    with patch("storepilot.fashion_page.research_turn", side_effect=fake_research):
        app.chat_input(key="fashion_chat").set_value("Check affordable jeans").run()
        assert not app.exception
        app.chat_input(key="fashion_chat").set_value("What about a cheaper option?").run()
    assert not app.exception
    assert captured[1][1][0]["question"] == "Check affordable jeans"
    journal = FashionJournal(journal_path(tmp_path / "test.db"))
    profile = app.session_state["fashion_profile_draft"]
    saved = journal.history(profile)
    assert len(saved) == 2
    assert "gemini-test-secret" not in json.dumps(saved)
    assert "serp-test-secret" not in json.dumps(saved)
    with patch(
        "storepilot.fashion_page.research_turn",
        side_effect=AssertionError("unexpected live request"),
    ):
        app.radio(key="language").set_value("zh").run()
        assert not app.exception
        assert app.selectbox(key="fashion_country").proto.raw_value == "德国"
    assert len(journal.history(profile)) == 2
    app.selectbox(key="fashion_country").set_value("FR").run()
    assert not list(app.chat_message)


def test_live_failure_does_not_save_invented_reply(app, tmp_path):
    live_setup(app)
    with patch(
        "storepilot.fashion_page.research_turn", side_effect=ResearchError("Provider HTTP 429")
    ):
        app.chat_input(key="fashion_chat").set_value("Check jeans").run()
    assert not app.exception
    assert any("429" in value.value for value in app.error)
    journal = FashionJournal(journal_path(tmp_path / "test.db"))
    assert not journal.history(app.session_state["fashion_profile_draft"])
