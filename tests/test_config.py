from __future__ import annotations

from agent_search.config import AppConfig


def test_from_env_prefers_openai_endpoint_alias(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_ENDPOINT", "https://example-resource.openai.azure.com/openai/v1/")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    config = AppConfig.from_env()

    assert config.openai_base_url == "https://example-resource.openai.azure.com/openai/v1/"


def test_from_env_does_not_default_to_openrouter_for_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_ENDPOINT", raising=False)

    config = AppConfig.from_env()

    assert config.openai_base_url is None


def test_from_env_ignores_openrouter_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    config = AppConfig.from_env()

    assert config.openai_api_key is None
    assert config.enable_llm is False
    assert config.openai_base_url is None


def test_from_env_defaults_to_openai_model(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    config = AppConfig.from_env()

    assert config.model_name == "gpt-5.4-mini"
