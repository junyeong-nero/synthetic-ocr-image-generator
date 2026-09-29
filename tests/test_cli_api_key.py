"""API key resolution for the evaluate CLI."""

from src.cli.evaluate import get_api_key


def test_google_prefers_google_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    assert get_api_key("google") == "google-key"


def test_google_falls_back_to_gemini_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    assert get_api_key("google") == "gemini-key"


def test_google_without_any_key_returns_none(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert get_api_key("google") is None


def test_other_backends_ignore_gemini_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    assert get_api_key("openai") is None
    assert get_api_key("unknown") is None
