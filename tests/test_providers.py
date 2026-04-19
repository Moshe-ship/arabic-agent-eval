"""Tests for provider config and env-var overrides."""

import os

from arabic_agent_eval.providers import (
    PROVIDER_CONFIGS,
    get_base_url,
    get_default_model,
)


def test_base_url_defaults_to_config(monkeypatch):
    monkeypatch.delenv("HERMES_BASE_URL", raising=False)
    assert get_base_url("hermes") == PROVIDER_CONFIGS["hermes"]["base_url"]


def test_hermes_base_url_respects_env(monkeypatch):
    monkeypatch.setenv("HERMES_BASE_URL", "https://my.hermes.example/v1")
    assert get_base_url("hermes") == "https://my.hermes.example/v1"


def test_hermes_model_respects_env(monkeypatch):
    monkeypatch.setenv("HERMES_MODEL", "NousResearch/Hermes-4-405B")
    assert get_default_model("hermes") == "NousResearch/Hermes-4-405B"


def test_openrouter_base_url_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.local/v1")
    assert get_base_url("openrouter") == "https://proxy.local/v1"


def test_base_url_unknown_provider_empty(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    assert get_base_url("not-a-real-provider") == ""


def test_openai_has_no_env_override(monkeypatch):
    """OpenAI doesn't need a base_url override — it uses a fixed OpenAI URL."""
    monkeypatch.setenv("OPENROUTER_BASE_URL", "ignored")
    assert get_base_url("openai") == PROVIDER_CONFIGS["openai"]["base_url"]
