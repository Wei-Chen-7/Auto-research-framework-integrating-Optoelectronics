"""Provider adapter tests: OpenAI-compatible client (DeepSeek et al.)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from picfix.agents.llm import MockLLM, OpenAICompatibleClient
from picfix.core.config import ExperimentConfig, LLMConfig
from picfix.experiments.run import build_llm


def _llm_config(**overrides) -> LLMConfig:
    base = dict(
        mode="api",
        provider="openai_compatible",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        model="deepseek-chat",
        temperature=0.0,
        max_tokens=1024,
        seed=7,
    )
    base.update(overrides)
    return LLMConfig.model_validate(base)


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"gap_nm": 1}'))],
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=7),
        )


def test_openai_compatible_adapter_request_and_response_mapping() -> None:
    fake = _FakeCompletions()
    stub = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    client = OpenAICompatibleClient(_llm_config(), client=stub)

    response = client.complete("system prompt", "user prompt", context={"ignored": True})

    assert response.text == '{"gap_nm": 1}'
    assert (response.tokens_in, response.tokens_out) == (42, 7)
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["model"] == "deepseek-chat"
    assert fake.last_kwargs["temperature"] == 0.0  # sent as-is on this provider
    assert fake.last_kwargs["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]


def test_openai_compatible_adapter_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        OpenAICompatibleClient(_llm_config())


def test_build_llm_dispatch(cfg: ExperimentConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cfg = cfg.model_copy(deep=True)
    mock_cfg.llm.mode = "mock"
    assert isinstance(build_llm(mock_cfg), MockLLM)

    api_cfg = cfg.model_copy(deep=True)
    api_cfg.llm.mode = "api"
    api_cfg.llm.provider = "openai_compatible"
    api_cfg.llm.api_key_env = "DEEPSEEK_API_KEY"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-real")
    assert isinstance(build_llm(api_cfg), OpenAICompatibleClient)


def test_config_rejects_unknown_provider() -> None:
    with pytest.raises(Exception):
        _llm_config(provider="totally_new_lab")
