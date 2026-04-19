"""LLM provider integrations for Arabic Agent Eval."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


PROVIDER_CONFIGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-6",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.0-flash",
        "env_key": "GOOGLE_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-large-latest",
        "env_key": "MISTRAL_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "env_key": "QWEN_API_KEY",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "model": "grok-2",
        "env_key": "XAI_API_KEY",
    },
    "cohere": {
        "base_url": "https://api.cohere.com/v2",
        "model": "command-r-plus",
        "env_key": "COHERE_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "env_key": "TOGETHER_API_KEY",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "model": "accounts/fireworks/models/qwen2p5-72b-instruct",
        "env_key": "FIREWORKS_API_KEY",
    },
    # OpenRouter — the pragmatic route to test NousResearch Hermes 4 family.
    # Model slugs: nousresearch/hermes-4-70b, nousresearch/hermes-4-405b,
    # nousresearch/hermes-4-14b. Override via --model on the CLI.
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "nousresearch/hermes-4-70b",
        "env_key": "OPENROUTER_API_KEY",
    },
    # Direct Hermes endpoint — set HERMES_BASE_URL to your Nous/self-hosted URL
    # (e.g. a Hermes-Function-Calling Inference server). Defaults to OpenAI-compatible.
    "hermes": {
        "base_url": "https://hermes.nousresearch.com/v1",
        "model": "NousResearch/Hermes-4-70B",
        "env_key": "HERMES_API_KEY",
    },
}

CONFIG_DIR = os.path.expanduser("~/.aae")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


def save_config(config: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    os.chmod(CONFIG_PATH, 0o600)


def get_api_key(provider: str) -> str | None:
    """Get API key from config or environment."""
    config = load_config()
    pconf = PROVIDER_CONFIGS.get(provider, {})
    env_key = pconf.get("env_key", "")

    # Environment first, then config
    key = os.environ.get(env_key) or config.get("keys", {}).get(provider)
    return key


# Per-provider environment variables that override the hard-coded base_url /
# model defaults. Useful for self-hosted endpoints (e.g. a private Hermes
# Inference server) without patching code.
_BASE_URL_OVERRIDES: dict[str, str] = {
    "hermes": "HERMES_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
}

_MODEL_OVERRIDES: dict[str, str] = {
    "hermes": "HERMES_MODEL",
    "openrouter": "OPENROUTER_MODEL",
}


def get_base_url(provider: str) -> str:
    """Return the effective base_url for a provider.

    Resolves an env-var override when one is declared (e.g. HERMES_BASE_URL);
    falls back to the PROVIDER_CONFIGS entry otherwise.
    """
    env_name = _BASE_URL_OVERRIDES.get(provider)
    if env_name:
        override = os.environ.get(env_name)
        if override:
            return override
    return PROVIDER_CONFIGS.get(provider, {}).get("base_url", "")


def get_default_model(provider: str) -> str:
    """Return the effective default model for a provider (env override → config)."""
    env_name = _MODEL_OVERRIDES.get(provider)
    if env_name:
        override = os.environ.get(env_name)
        if override:
            return override
    return PROVIDER_CONFIGS.get(provider, {}).get("model", "")


def get_available_providers() -> list[str]:
    """Return providers that have API keys configured."""
    return [p for p in PROVIDER_CONFIGS if get_api_key(p)]


def call_openai_compatible(
    instruction: str,
    tools: list[dict],
    functions: list[dict],
    provider: str,
    model: str | None = None,
) -> dict:
    """Call an OpenAI-compatible API with function calling.

    Returns {"calls": [...], "raw": str}
    """
    pconf = PROVIDER_CONFIGS[provider]
    api_key = get_api_key(provider)
    if not api_key:
        raise ValueError(f"No API key for {provider}. Set {pconf['env_key']} or run: aae config")

    base_url = get_base_url(provider) or pconf["base_url"]
    model_name = model or get_default_model(provider) or pconf["model"]

    system_prompt = (
        "أنت مساعد ذكي يستخدم الأدوات المتاحة للإجابة على طلبات المستخدم. "
        "استخدم الأدوات المناسبة بناءً على الطلب. "
        "حافظ على النص العربي كما هو ولا تترجمه للإنجليزية."
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ],
        "tools": tools,
        "tool_choice": "auto",
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    # Extract tool calls
    calls = []
    message = data.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    for tc in tool_calls:
        func = tc.get("function", {})
        args = func.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({
            "function": func.get("name", ""),
            "arguments": args,
        })

    raw = message.get("content", "")
    return {"calls": calls, "raw": raw or ""}


def call_anthropic(
    instruction: str,
    tools: list[dict],
    functions: list[dict],
    model: str | None = None,
) -> dict:
    """Call Anthropic API with tool use."""
    api_key = get_api_key("anthropic")
    if not api_key:
        raise ValueError("No API key for Anthropic. Set ANTHROPIC_API_KEY or run: aae config")

    model_name = model or PROVIDER_CONFIGS["anthropic"]["model"]

    # Convert OpenAI tools format to Anthropic format
    anthropic_tools = []
    for tool in tools:
        func = tool["function"]
        anthropic_tools.append({
            "name": func["name"],
            "description": func["description"],
            "input_schema": func["parameters"],
        })

    system_prompt = (
        "أنت مساعد ذكي يستخدم الأدوات المتاحة للإجابة على طلبات المستخدم. "
        "استخدم الأدوات المناسبة بناءً على الطلب. "
        "حافظ على النص العربي كما هو ولا تترجمه للإنجليزية."
    )

    payload = {
        "model": model_name,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": instruction}],
        "tools": anthropic_tools,
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    calls = []
    raw_parts = []
    for block in data.get("content", []):
        if block.get("type") == "tool_use":
            calls.append({
                "function": block["name"],
                "arguments": block.get("input", {}),
            })
        elif block.get("type") == "text":
            raw_parts.append(block.get("text", ""))

    return {"calls": calls, "raw": "\n".join(raw_parts)}


def make_call_fn(provider: str, model: str | None = None):
    """Create a call function for a provider."""
    if provider == "anthropic":
        return lambda inst, tools, funcs: call_anthropic(inst, tools, funcs, model)
    return lambda inst, tools, funcs: call_openai_compatible(inst, tools, funcs, provider, model)
