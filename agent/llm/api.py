"""OpenAI-compatible client for every user-simulator and judge call."""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from openai import AsyncOpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL_NAME", "default")

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
    """Kept for call-site compatibility."""
    return False


def is_gemini_model(model: str) -> bool:
    """See is_messages_model."""
    return False


def merge_alternating(messages: list[dict]) -> list[dict]:
    """Collapse consecutive same-role turns into one."""
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
    """Exponential with full jitter."""
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
    """Chat completion constrained to JSON."""
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
