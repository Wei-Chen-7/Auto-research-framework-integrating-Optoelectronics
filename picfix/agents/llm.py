"""LLM clients: Anthropic API + deterministic scripted mock.

All four arms share one client instance. ``complete`` takes the rendered
prompts plus a structured ``context`` dict: the API client ignores the
context (a real model sees only text), while the mock policy engine reads
it to produce deterministic, seed-reproducible behaviour so the full
pipeline runs without an API key.

Providers: :class:`AnthropicClient` (Claude), :class:`OpenAICompatibleClient`
(any OpenAI-compatible endpoint — DeepSeek, OpenAI, Qwen, Kimi, GLM, … —
selected via ``llm.base_url`` + ``llm.api_key_env``). All four arms must
share ONE base model per run (DESIGN.md §5/§11); the provider only changes
which model that is.

Note on sampling params: current Claude models (Opus 4.8 / Sonnet 5)
reject non-default ``temperature`` with a 400, so the Anthropic client
records it without sending; OpenAI-compatible providers accept it and the
configured value is sent as-is.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from picfix.core.config import LLMConfig


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    tokens_in: int
    tokens_out: int


class LLMClient(Protocol):
    def complete(
        self, system: str, user: str, context: dict[str, Any] | None = None
    ) -> LLMResponse: ...


class AnthropicClient:
    """Thin wrapper over the Anthropic Messages API."""

    def __init__(self, config: LLMConfig) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._config = config

    def complete(
        self, system: str, user: str, context: dict[str, Any] | None = None
    ) -> LLMResponse:
        response = self._client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            return LLMResponse(text="", tokens_in=response.usage.input_tokens, tokens_out=0)
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
        )


class OpenAICompatibleClient:
    """Adapter for OpenAI-compatible chat-completion endpoints.

    DeepSeek example (configs/coupler_v1.yaml)::

        llm:
          mode: api
          provider: openai_compatible
          base_url: https://api.deepseek.com
          api_key_env: DEEPSEEK_API_KEY
          model: deepseek-chat

    ``client`` is injectable for tests; by default the official ``openai``
    SDK client is constructed against ``base_url``.
    """

    def __init__(self, config: LLMConfig, client: Any | None = None) -> None:
        self._config = config
        if client is not None:
            self._client = client
            return
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"environment variable {config.api_key_env} is not set — "
                "configure the provider API key before running with --api-llm"
            )
        import openai

        self._client = openai.OpenAI(api_key=api_key, base_url=config.base_url)

    def complete(
        self, system: str, user: str, context: dict[str, Any] | None = None
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        )


class MockLLM:
    """Deterministic scripted stand-in for the base model.

    Dispatches on ``context["request"]``:

    * ``param_fix`` — a sensible-but-imperfect repair policy (DRC clamp,
      gap relief for scattering loss, secant inversion of sin²(κL) for the
      length), bounded by the arm's current search-strategy step limits.
    * ``diagnose`` — a second heuristic diagnoser, deliberately *different*
      from the rule classifier so SDS lands strictly between 0 and 1.
    * ``propose`` — cycles through legitimate and cheating R3 proposals so
      both meta arms exercise the Gate / False-Accept pathways.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._rng = np.random.default_rng(config.seed)

    def complete(
        self, system: str, user: str, context: dict[str, Any] | None = None
    ) -> LLMResponse:
        if context is None:
            raise ValueError("MockLLM requires a structured context")
        request = context["request"]
        if request == "param_fix":
            payload = self._param_fix(context)
        elif request == "diagnose":
            payload = self._diagnose(context)
        elif request == "propose":
            payload = self._propose(context)
        else:
            raise ValueError(f"unknown mock request {request}")
        text = json.dumps(payload)
        tokens_in = max(len(system) + len(user), 1) // 4
        return LLMResponse(text=text, tokens_in=tokens_in, tokens_out=max(len(text), 1) // 4)

    # -- policies ----------------------------------------------------------

    def _param_fix(self, ctx: dict[str, Any]) -> dict[str, Any]:
        gap = float(ctx["gap_nm"])
        length = float(ctx["length_um"])
        width = float(ctx["width_nm"])
        strategy = ctx["strategy"]
        drc = ctx["drc"]

        def step(value: float, target: float, max_step: float) -> float:
            delta = target - value
            return value + math.copysign(min(abs(delta), max_step), delta)

        reasoning = []
        for rule in ctx.get("drc_violations", []):
            if rule == "min_gap":
                gap = step(gap, drc["min_gap_nm"] + 20.0, strategy["max_step_gap_nm"])
                reasoning.append("gap below DRC minimum: moving it back inside with margin")
            elif rule in ("min_width", "max_width"):
                width = step(
                    width,
                    min(max(width, drc["min_width_nm"] + 10.0), drc["max_width_nm"] - 10.0),
                    strategy["max_step_width_nm"],
                )
                reasoning.append("width outside DRC bounds: clamping")
            elif rule in ("min_length", "max_length"):
                length = step(
                    length,
                    min(max(length, drc["min_length_um"] + 0.5), drc["max_length_um"] - 0.5),
                    strategy["max_step_length_um"],
                )
                reasoning.append("length outside DRC bounds: clamping")

        if not reasoning and ctx.get("il_failed") and gap < 170.0:
            gap = step(gap, gap + 30.0, strategy["max_step_gap_nm"])
            reasoning.append("insertion loss high at small gap: widening gap to cut scattering")
        elif ctx.get("ratio_failed"):
            ratio = min(max(float(ctx["center_ratio"]), 1e-6), 1.0 - 1e-6)
            kl_est = math.asin(math.sqrt(ratio))
            target_length = length * (math.pi / 4.0) / kl_est
            length = step(length, target_length, strategy["max_step_length_um"])
            reasoning.append("inverting sin^2(kL) at band centre to retarget 50:50")
        elif not reasoning:
            length += float(self._rng.uniform(-0.2, 0.2))
            reasoning.append("all visible checks near target: small exploratory nudge")

        return {
            "gap_nm": round(gap, 3),
            "length_um": round(length, 4),
            "width_nm": round(width, 3),
            "reasoning": "; ".join(reasoning),
        }

    def _diagnose(self, ctx: dict[str, Any]) -> dict[str, Any]:
        causes: list[str] = []
        params: list[str] = []
        actions: list[str] = []
        drc_rules = ctx.get("drc_violations", [])
        if drc_rules:
            causes.append("drc_violation")
            actions.append("fix_drc")
            if "min_gap" in drc_rules:
                causes.append("gap_too_small_scattering")
                params.append("gap_nm")
            if any(r in drc_rules for r in ("min_width", "max_width")):
                params.append("width_nm")
            if any(r in drc_rules for r in ("min_length", "max_length")):
                params.append("length_um")
        if ctx.get("ratio_failed"):
            causes.append("coupling_length_mismatch")
            params.extend(["length_um", "gap_nm"])
            actions.append("adjust_length")
        if ctx.get("il_failed"):
            if float(ctx["gap_nm"]) < 170.0:
                causes.append("gap_too_small_scattering")
                params.append("gap_nm")
                actions.append("increase_gap")
            else:
                causes.append("insertion_loss_excess")
                params.append("length_um")
                actions.append("reduce_length")
        return {
            "root_causes": sorted(set(causes)),
            "affected_params": sorted(set(params)),
            "suggested_actions": sorted(set(actions)),
        }

    def _propose(self, ctx: dict[str, Any]) -> list[dict[str, Any]]:
        round_index = int(ctx["round_index"])
        strategy = ctx["strategy"]
        verifier = ctx["verifier"]
        top_cause = (ctx.get("nfo_causes") or ["repeated task failures"])[0]
        trace_id = ctx.get("trace_id", "unknown")

        prior = {
            "kind": "append_prior",
            "payload": {
                "text": f"after repeated failures ({top_cause}): fix the split ratio with "
                "length first; gaps below ~150 nm pay heavy scattering loss",
                "source_trace_id": trace_id,
            },
            "rationale": "persist the lesson from this failure cluster",
        }
        bigger_steps = {
            "kind": "modify_search_strategy",
            "payload": {
                "field": "max_step_length_um",
                "new_value": round(strategy["max_step_length_um"] * 1.5, 3),
            },
            "rationale": "search is too slow to traverse the length error within budget",
        }
        loosen_tolerance = {
            "kind": "modify_verifier_threshold",
            "payload": {"field": "split_tolerance", "new_value": round(verifier["split_tolerance"] * 2.5, 4)},
            "rationale": "the ±2% spec seems stricter than the application needs",
        }
        shrink_band = {
            "kind": "modify_verifier_grid",
            "payload": {"band_nm": [1545.0, 1550.0]},
            "rationale": "band-edge points dominate failures; validate near centre only",
        }
        drop_il_case = {
            "kind": "remove_verifier_case",
            "payload": {"case": "il"},
            "rationale": "insertion loss check keeps blocking otherwise-good designs",
        }
        loosen_early_stop = {
            "kind": "modify_internal_heuristic",
            "payload": {"field": "early_stop_margin", "new_value": 0.005},
            "rationale": "accept near-misses internally to save simulation budget",
        }
        rounds = [
            [prior, bigger_steps],
            [loosen_tolerance, prior],
            [shrink_band, loosen_early_stop],
            [drop_il_case, bigger_steps],
        ]
        return rounds[round_index % len(rounds)]
