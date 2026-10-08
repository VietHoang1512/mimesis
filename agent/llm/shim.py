"""Local OpenAI-compatible proxy sitting between the gyms and the user simulator."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
UPSTREAM = os.environ.get("SHIM_UPSTREAM") or os.environ.get("OPENAI_BASE_URL")
UPSTREAM_KEY = os.environ.get("SHIM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
FORCE_MODEL = os.environ.get("SHIM_MODEL") or os.environ.get("SHIM_FORCE_MODEL")
STRIP_THINK = os.environ.get("SHIM_STRIP_THINK", "1") == "1"

THINK_RETRY = int(os.environ.get("SHIM_THINK_RETRY", "1"))
THINK_RETRY_GROWTH = float(os.environ.get("SHIM_THINK_RETRY_GROWTH", "2.0"))

THINK_BUF_TURNS = int(os.environ.get("SHIM_THINK_BUF_TURNS", "8"))
THINK_BUF_EPISODES = int(os.environ.get("SHIM_THINK_BUF_EPISODES", "2048"))
THINK_BUF_CHARS = int(os.environ.get("SHIM_THINK_BUF_CHARS", "4000"))

STATS = {"started": 0, "ok": 0, "failed": 0, "empty_after_strip": 0}

# --------------------------------------------------------------------------- #
# reasoning blocks
# --------------------------------------------------------------------------- #
_THINK_RE = [
    (re.compile(r"<seed:think>.*?</seed:think>", re.S), ""),
    (re.compile(r"<think>.*?</think>", re.S), ""),
    (re.compile(r"^.*?</seed:think>", re.S), ""),
    (re.compile(r"^.*?</think>", re.S), ""),
    (re.compile(r"<seed:think>.*$", re.S), ""),
    (re.compile(r"<think>.*$", re.S), ""),
]

_THINK_EXTRACT = [
    re.compile(r"<seed:think>(.*?)</seed:think>", re.S),
    re.compile(r"<think>(.*?)</think>", re.S),
    re.compile(r"^(.*?)</seed:think>", re.S),
    re.compile(r"^(.*?)</think>", re.S),
    re.compile(r"<seed:think>(.*)$", re.S),
    re.compile(r"<think>(.*)$", re.S),
]


def remove_think(text: str) -> str:
    if not text:
        return text
    out = text
    for rx, sub in _THINK_RE:
        out = rx.sub(sub, out, count=1 if rx.pattern.startswith("^") else 0)
    return out.strip()


def extract_think(text: str) -> str:
    """Return the model's reasoning block, or "" if it did not emit one."""
    if not text:
        return ""
    for rx in _THINK_EXTRACT:
        m = rx.search(text)
        if m:
            return (m.group(1) or "").strip()
    return ""


# --------------------------------------------------------------------------- #
# privileged channel
# --------------------------------------------------------------------------- #
_think_buf: collections.OrderedDict[str, collections.deque] = collections.OrderedDict()


def _remember_think(episode: str | None, raw: str) -> None:
    """Stash this call's reasoning block against its episode."""
    if not episode:
        return
    think = extract_think(raw)
    if not think:
        return
    buf = _think_buf.get(episode)
    if buf is None:
        buf = _think_buf[episode] = collections.deque(maxlen=THINK_BUF_TURNS)
        while len(_think_buf) > THINK_BUF_EPISODES:
            _think_buf.popitem(last=False)
    else:
        _think_buf.move_to_end(episode)
    buf.append(think[:THINK_BUF_CHARS])


# --------------------------------------------------------------------------- #
# upstream
# --------------------------------------------------------------------------- #
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not UPSTREAM:
            raise RuntimeError("set SHIM_UPSTREAM to an OpenAI-compatible endpoint")
        _client = AsyncOpenAI(base_url=UPSTREAM, api_key=UPSTREAM_KEY)
    return _client


def _max_tokens_of(body: dict) -> int:
    return int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)


async def _complete_once(body: dict, max_tokens: int) -> str:
    kwargs = {
        "model": FORCE_MODEL or body.get("model") or "default",
        "messages": body["messages"],
        "max_tokens": max_tokens,
        "timeout": float(os.environ.get("SHIM_TIMEOUT", "300")),
    }
    for k in ("temperature", "top_p", "seed", "stop", "response_format"):
        if body.get(k) is not None:
            kwargs[k] = body[k]
    resp = await _get_client().chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    raw = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning and "</think>" not in raw:
        raw = f"<think>{reasoning}</think>{raw}"
    return raw


def _envelope(text: str, model: str) -> dict:
    return {
        "id": "chatcmpl-shim",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
app = FastAPI()


@app.get("/health")
async def health():
    return {
        "upstream": UPSTREAM,
        "model": FORCE_MODEL,
        "strip_think": STRIP_THINK,
        "episodes_buffered": len(_think_buf),
        **STATS,
    }


@app.get("/ep/{episode}/privileged")
async def privileged(episode: str):
    """Read side of the privileged channel, for interact_tool.py."""
    buf = _think_buf.get(episode)
    return {
        "episode": episode,
        "turns": len(buf) if buf else 0,
        "think": list(buf) if buf else [],
    }


@app.post("/ep/{episode}/v1/chat/completions")
@app.post("/ep/{episode}/chat/completions")
async def chat_completions_scoped(episode: str, request: Request):
    return await _handle(request, episode=episode)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    return await _handle(request, episode=None)


async def _handle(request: Request, episode: str | None):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "request body is not valid JSON",
                               "type": "invalid_request_error"}},
        )
    if body.get("stream"):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "streaming is not supported by this shim",
                               "type": "invalid_request_error"}},
        )

    model = FORCE_MODEL or body.get("model") or "default"
    STATS["started"] += 1
    budget = _max_tokens_of(body)
    try:
        raw = await _complete_once(body, budget)
        text = remove_think(raw) if STRIP_THINK else raw
        for _ in range(THINK_RETRY):
            if text or not STRIP_THINK:
                break
            budget = int(budget * THINK_RETRY_GROWTH)
            raw = await _complete_once(body, budget)
            text = remove_think(raw)
        if STRIP_THINK and not text:
            STATS["empty_after_strip"] += 1
        _remember_think(episode, raw)
        STATS["ok"] += 1
        return JSONResponse(content=_envelope(text, model))
    except Exception as e:  # noqa: BLE001 - provider SDKs raise many types
        STATS["failed"] += 1
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"{type(e).__name__}: {e}",
                               "type": "upstream_error"}},
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8710)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"shim :{args.port} -> {UPSTREAM} model={FORCE_MODEL} "
          f"strip_think={STRIP_THINK}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
