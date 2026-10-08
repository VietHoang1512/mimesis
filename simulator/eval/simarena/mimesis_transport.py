"""MIMESIS transport for SimulatorArena."""
from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import Dict, List, Optional

_ODYSSIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ODYSSIM_ROOT not in sys.path:
    sys.path.insert(0, _ODYSSIM_ROOT)

GATEWAY_MODELS = [
    "gpt-5.5",
    "claude-opus-5", "claude-sonnet-5", "claude-opus-4.8",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.5-flash",
    "gemini-3.8-flash",
]

LOCAL_PREFIX = "local/"


def is_mimesis_model(model_name: str) -> bool:
    return model_name in GATEWAY_MODELS or model_name.startswith(LOCAL_PREFIX)


def _local_client(model_name: str):
    """OpenAI SDK client for a locally served checkpoint."""
    import httpx
    from openai import AsyncOpenAI

    port = os.environ.get("LOCAL_VLLM_PORT", "8100")
    return AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
        max_retries=3,
        http_client=httpx.AsyncClient(trust_env=False, timeout=600.0),
    )


async def _one_gateway(messages, model_name, max_tokens, sem, reasoning_effort):
    import api

    async with sem:
        try:
            return await api.chat_async(
                messages,
                model=model_name,
                max_tokens=max_tokens,
                max_retries=20,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            print(f"[mimesis_transport] {model_name} failed: {type(e).__name__}: {str(e)[:160]}")
            return ""


async def _one_local(client, messages, served, temperature, max_tokens, sem):
    async with sem:
        try:
            extra = {}
            if os.environ.get("ODYSSIM_ENABLE_THINKING") is not None:
                extra["extra_body"] = {"chat_template_kwargs": {
                    "enable_thinking": os.environ["ODYSSIM_ENABLE_THINKING"] == "1"}}
            rp = float(os.environ.get("ODYSSIM_REPETITION_PENALTY", "1.0") or 1.0)
            if rp != 1.0:
                extra.setdefault("extra_body", {})["repetition_penalty"] = rp
            r = await client.chat.completions.create(
                model=served,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
            return (r.choices[0].message.content or "") if r.choices else ""
        except Exception as e:
            print(f"[mimesis_transport] local/{served} failed: {type(e).__name__}: {str(e)[:160]}")
            return ""


def _strip_reasoning(text: str) -> str:
    """Remove chain of thought before the turn leaves this module."""
    if not text:
        return text
    try:
        from realusersim_eval import strip_reasoning
        out = strip_reasoning(text)
    except Exception:
        out = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        out = re.sub(r"^.*?</think>", "", out, flags=re.S).strip()
    if not out and text.strip():
        print(f"[mimesis_transport] WARNING reasoning-only turn dropped "
              f"({len(text)} chars): {text[:100]!r}")
    return out


async def generate_from_mimesis(
    full_contexts: List[List[Dict[str, str]]],
    model_name: str,
    temperature: float,
    max_tokens: int,
    n: int = 1,
    show_progress: bool = True,
    max_concurrent: Optional[int] = None,
) -> List[List[str]]:
    """Return one list of n strings per context, matching the other branches."""
    if model_name.startswith(LOCAL_PREFIX):
        served = model_name[len(LOCAL_PREFIX):]
        limit = max_concurrent or int(os.environ.get("ODYSSIM_LOCAL_CONCURRENCY", "32"))
        client = _local_client(model_name)
        sem = asyncio.Semaphore(limit)
        tasks = [
            _one_local(client, ctx, served, temperature, max_tokens, sem)
            for ctx in full_contexts for _ in range(n)
        ]
    else:
        limit = max_concurrent or int(os.environ.get("ODYSSIM_GATEWAY_CONCURRENCY", "16"))
        effort = os.environ.get("OPENAI_AGENT_REASONING_EFFORT", "none")
        sem = asyncio.Semaphore(limit)
        tasks = [
            _one_gateway(ctx, model_name, max_tokens, sem, effort)
            for ctx in full_contexts for _ in range(n)
        ]

    if show_progress:
        from tqdm.asyncio import tqdm_asyncio
        flat = await tqdm_asyncio.gather(*tasks, desc=f"[{model_name}]")
    else:
        flat = await asyncio.gather(*tasks)

    return [
        [_strip_reasoning(flat[i * n + j] or "").strip() for j in range(n)]
        for i in range(len(full_contexts))
    ]
