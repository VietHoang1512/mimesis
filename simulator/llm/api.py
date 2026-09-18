"""OpenAI-compatible client used for every judge and conversational-partner call.

The simulator under evaluation is served locally (vLLM/SGLang); this module is
the *other* side -- the assistant it talks to and the judge that scores the
transcript. Point it at any OpenAI-compatible endpoint:

    export OPENAI_AGENT_BASE_URL=http://localhost:8000/v1
    export OPENAI_AGENT_API_KEY=...
    export OPENAI_MODEL_NAME=gpt-5-5

Everything is funnelled through one retrying `chat_async`. The helpers below
(`chat_parse`, `chat_messages_async`, `merge_alternating`, the `is_*_model`
predicates) exist because call sites in `agents/` and `eval/` historically
dispatched per provider family; they are kept so those call sites need no
change, and they now all resolve to the same endpoint.

Two behaviours are load-bearing and deliberate:

* **Retry budget outlasts the rate-limit window.** Providers meter over windows
  measured in minutes. A retry loop that gives up in 45 s cannot ride one out,
  and the failure is silent: the call returns nothing, the turn is dropped, the
  conversation ends early and scores 0 -- indistinguishable from a model that
  simply refused. Backoff is jittered and the budget is long for that reason.

* **An empty completion is an error, not a result.** A reasoning model whose
  token budget is consumed before it emits visible text returns a well-formed
  response with no content. Returning that as success silently poisons a whole
  evaluation, so it raises instead.
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from openai import AsyncOpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-5-5")

_MAX_BACKOFF = 60.0
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        base_url = os.getenv("OPENAI_AGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "set OPENAI_AGENT_BASE_URL to an OpenAI-compatible endpoint "
                "(see scripts/env.sh)"
            )
        _client = AsyncOpenAI(
            base_url=base_url,
            api_key=os.getenv("OPENAI_AGENT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY",
        )
    return _client


def normalize_model(model: str | None) -> str:
    """Model id as the endpoint expects it. Empty/None -> DEFAULT_MODEL."""
    return model or DEFAULT_MODEL


def is_messages_model(model: str) -> bool:
    """Kept for call-site compatibility. One endpoint serves every family here,
    so there is no separate Anthropic-style route to select."""
    return False


def is_gemini_model(model: str) -> bool:
    """See is_messages_model."""
    return False


def merge_alternating(messages: list[dict]) -> list[dict]:
    """Collapse consecutive same-role turns into one.

    Some chat templates reject two user turns in a row, which is easy to produce
    when a simulator emits a turn the environment then annotates.
    """
    merged: list[dict] = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1] = {
                **merged[-1],
                "content": f"{merged[-1]['content']}\n\n{m['content']}",
            }
        else:
            merged.append(dict(m))
    return merged


def _backoff(attempt: int) -> float:
    """Exponential with full jitter. The jitter matters: without it, concurrent
    rollouts that hit the same rate limit retry in lockstep and collide again."""
    return random.uniform(0, min(_MAX_BACKOFF, 2.0**attempt))


async def chat_async(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int | None = None,
    max_retries: int = 20,
    response_format: dict | None = None,
    reasoning_effort: str | None = None,
    seed: int | None = None,
    timeout: float = 300.0,
    **_: Any,
) -> str:
    """One chat completion, retried. Returns the assistant text."""
    model = normalize_model(model)
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "timeout": timeout}
    if max_tokens:
        kwargs["max_completion_tokens"] = max_tokens
    if response_format:
        kwargs["response_format"] = response_format
    if reasoning_effort and reasoning_effort != "none":
        kwargs["reasoning_effort"] = reasoning_effort
    if seed is not None:
        kwargs["seed"] = seed

    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await _get_client().chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                # A reasoning model that spent its budget before emitting text
                # lands here. Treat it as a failure so the retry sees it; a
                # silent "" would be scored as a real (empty) turn.
                raise RuntimeError(f"empty completion from {model}")
            return text
        except Exception as e:  # noqa: BLE001 - provider SDKs raise many types
            last = e
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt))
    raise RuntimeError(f"{model}: giving up after {max_retries} attempts: {last}")


async def chat_parse(
    messages: list[dict],
    text_format: Any = None,
    model: str | None = None,
    max_retries: int = 20,
    reasoning_effort: str | None = None,
    **kw: Any,
) -> str:
    """Chat completion constrained to JSON. `text_format` is accepted for call-site
    compatibility; only JSON-object mode is requested."""
    return await chat_async(
        messages,
        model=model,
        max_retries=max_retries,
        response_format={"type": "json_object"},
        reasoning_effort=reasoning_effort,
        **kw,
    )


async def chat_messages_async(
    messages: list[dict],
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 2048,
    max_retries: int = 30,
    **kw: Any,
) -> str:
    """Compatibility wrapper for call sites that split the system prompt out."""
    msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
    return await chat_async(
        msgs, model=model, max_tokens=max_tokens, max_retries=max_retries, **kw
    )
