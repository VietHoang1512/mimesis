"""Local OpenAI-compatible proxy sitting between the gyms and the user simulator.

Every gym talks to its simulated user with the plain OpenAI SDK. This shim is
what they point at. It does three jobs, and the first two are load-bearing for
correctness rather than convenience:

1. **Strip reasoning before the gym sees it.** A gym uses `message.content`
   verbatim as the user's utterance. Hand it a `<think>` block and the agent
   reads a reasoning trace as if the user had said it, the judge grades against
   it, and the episode is corrupted in a way nothing downstream detects.

2. **Hold the simulator's reasoning as a privileged channel.** That same block,
   deleted from the reply, is one of the two privileged signals SDPO builds a
   hint from (the other is the user's next reply). The strip site is the only
   place in the system where the block is still whole and in memory, so this is
   where it is captured. `verl/tools/interact_tool.py` reads it back over
   `GET /ep/<episode>/privileged`.

3. Forward everything else to an OpenAI-compatible upstream.

**Episode scoping is not optional.** `verl/tools/env_manager.py` gives each gym
instance a base_url of `/ep/<request_id>/v1` precisely so reasoning can be
attributed to a trajectory. An unscoped call is not stashed at all: guessing
would pair one episode's reasoning with another's turn, which produces plausible
hints and no error.

Usage:

    export SHIM_UPSTREAM=http://localhost:8000/v1   # the served simulator
    export SHIM_MODEL=my-simulator                  # override the requested model
    python -m llm.shim --port 8710

The experiments reported in the paper used a larger version of this file, which
added rate limiting, admission control and multi-endpoint fan-out so a single
process could drive a full agent x simulator sweep against capacity-limited
endpoints. None of that affects the method or the numbers, and it is omitted
here.
"""

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

# A simulator that spends its whole budget reasoning returns a well-formed reply
# with nothing said. Retry once with a larger budget before accepting the empty
# turn -- see remove_think() for why the empty turn is nonetheless the honest
# outcome when the retry also fails.
THINK_RETRY = int(os.environ.get("SHIM_THINK_RETRY", "1"))
THINK_RETRY_GROWTH = float(os.environ.get("SHIM_THINK_RETRY_GROWTH", "2.0"))

# Bounded on three axes because this is a long-lived training process: 512
# trajectories/step x hundreds of steps is >100k episodes, and a thinking
# simulator emits ~9 KB per call. Unbounded that is tens of GB in a process whose
# job is to answer in 2 s.
THINK_BUF_TURNS = int(os.environ.get("SHIM_THINK_BUF_TURNS", "8"))
THINK_BUF_EPISODES = int(os.environ.get("SHIM_THINK_BUF_EPISODES", "2048"))
THINK_BUF_CHARS = int(os.environ.get("SHIM_THINK_BUF_CHARS", "4000"))

STATS = {"started": 0, "ok": 0, "failed": 0, "empty_after_strip": 0}

# --------------------------------------------------------------------------- #
# reasoning blocks
# --------------------------------------------------------------------------- #
# Three shapes, because a reasoning block does not always arrive whole:
#   <think>r</think>answer   model emitted its own opener (Qwen3, Seed)
#   r</think>answer          opener was PRE-FILLED by the chat template, so the
#                            completion starts inside the block (Qwen3.5/R1)
#   answer<think>r           generation hit the token cap before closing
# Not recoverable: a pre-filled opener that is also truncated before the close
# tag carries no tag at all and is indistinguishable from a plain answer -- serve
# with thinking disabled if that matters.
_THINK_RE = [
    (re.compile(r"<seed:think>.*?</seed:think>", re.S), ""),
    (re.compile(r"<think>.*?</think>", re.S), ""),
    (re.compile(r"^.*?</seed:think>", re.S), ""),
    (re.compile(r"^.*?</think>", re.S), ""),
    (re.compile(r"<seed:think>.*$", re.S), ""),
    (re.compile(r"<think>.*$", re.S), ""),
]

# The complement of remove_think(): what the model THOUGHT rather than what it
# said. Same six shapes in the same order, so a completion splits identically
# under both -- capture the group instead of deleting the span.
_THINK_EXTRACT = [
    re.compile(r"<seed:think>(.*?)</seed:think>", re.S),
    re.compile(r"<think>(.*?)</think>", re.S),
    re.compile(r"^(.*?)</seed:think>", re.S),
    re.compile(r"^(.*?)</think>", re.S),
    re.compile(r"<seed:think>(.*)$", re.S),
    re.compile(r"<think>(.*)$", re.S),
]


def remove_think(text: str) -> str:
    """Return only what the model SPEAKS -- "" if it never got to speaking.

    Deliberately does NOT fall back to the raw text. Falling back hands the gym a
    reasoning trace as the simulated user's utterance: the agent reads it, the
    judge grades against it, and the episode is corrupted undetectably. An empty
    return lands in the gym's ordinary missing-response path, which is the honest
    outcome -- the simulator did not produce an utterance.
    """
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
# interact_tool.py reads this synchronously right after each env.step(), so it
# only ever needs the most recent few entries -- but it reads over HTTP and can
# miss one, so keep a short tail rather than popping.
_think_buf: collections.OrderedDict[str, collections.deque] = collections.OrderedDict()


def _remember_think(episode: str | None, raw: str) -> None:
    """Stash this call's reasoning block against its episode.

    No-op without an episode id: an unscoped call cannot be attributed to a
    trajectory, and guessing would silently pair one episode's reasoning with
    another's turn.
    """
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
    # Some servers return the reasoning out of band instead of inline. Re-inline
    # it so the two paths below see one shape; without this the privileged
    # channel is silently empty for those servers.
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
    """Read side of the privileged channel, for interact_tool.py.

    Returns the simulator's recent reasoning blocks oldest-first; the caller
    wants [-1], the thought behind the reply it just received.

    `turns` is how many calls this episode has made -- if it disagrees with the
    rollout's turn index, thoughts and turns are misaligned and every hint built
    from them is attached to the wrong response. That is invisible downstream, so
    it is returned rather than merely logged.
    """
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
        # No gym streams, and implementing SSE over a non-streaming upstream
        # would only pretend to. Fail loudly instead of hanging.
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
        # Empty after stripping means the simulator reasoned past its budget and
        # never spoke. Grow the budget and ask again before accepting that.
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
