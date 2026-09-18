"""MIMESIS transport for SimulatorArena.

SimulatorArena calls providers directly (OpenAI, Anthropic, Google, Azure,
Mistral SDKs). None of those are reachable from the the compute cluster: outbound
HTTPS to api.openai.com / api.anthropic.com is VPN-gated, and
`generate_from_openai_chat_completion` builds its AsyncOpenAI client with no
base_url, so every call lands on api.openai.com and hangs.

Two routes DO work here, and this module exposes both behind one function:

  * frontier models -> repo-root api.py, the hosted endpoint curl transport already
    used by the SOUL / tau-USI / Turing evals. It handles the outbound proxy, the
    per-family routing (gpt-* Azure Responses, claude-* Anthropic /v1/messages,
    gemini-* Vertex), jittered retries and the up-anthropic quota backoff.
  * local checkpoints -> an OpenAI-compatible `vllm serve` on localhost, named
    "local/<served-model-name>". SimulatorArena has an in-process vLLM path
    (utils.generate_from_vllm_local) but user_simulation_document_creation.py
    never populates `vllm_models`, so wiring a served endpoint here is both less
    invasive and lets the user simulator and the assistant sit on different
    backends in the same run.

The contract matches what utils.generate_responses_in_batch expects from every
other provider branch: a list (one entry per context) of lists of n strings.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import Dict, List, Optional

# Repo-root api.py, two levels up from simulation/.
_ODYSSIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ODYSSIM_ROOT not in sys.path:
    sys.path.insert(0, _ODYSSIM_ROOT)

# Frontier names routed through the hosted endpoint. Spelled as the gateway registry
# expects; api.normalize_model() fixes the punctuation differences per family.
#
# Unlike api.py's GEMINI_MODELS (a record), this list is a GATE: is_mimesis_model()
# tests membership, and a name absent from here falls through to SimulatorArena's
# own Google/OpenAI SDK branch, which has no base_url and hangs on this cluster.
# Add the name here before running any new frontier model.
GATEWAY_MODELS = [
    "gpt-5.5",
    "claude-opus-5", "claude-sonnet-5", "claude-opus-4.8",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.5-flash",
    "gemini-3.8-flash",
]

# Anything starting with this prefix is served locally by `vllm serve`; the rest
# of the name is the --served-model-name. The port comes from LOCAL_VLLM_PORT so
# one run script can start the server and point the sim at it.
LOCAL_PREFIX = "local/"


def is_mimesis_model(model_name: str) -> bool:
    return model_name in GATEWAY_MODELS or model_name.startswith(LOCAL_PREFIX)


def _local_client(model_name: str):
    """OpenAI SDK client for a locally served checkpoint.

    trust_env=False so the inherited https_proxy (which points at the SUBMITTING
    host's outbound proxy listener and is dead on a compute node) cannot capture a request
    to localhost -- the same failure that silently broke the Turing judge.
    """
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
        except Exception as e:  # one bad turn must not sink the batch
            print(f"[mimesis_transport] {model_name} failed: {type(e).__name__}: {str(e)[:160]}")
            return ""


async def _one_local(client, messages, served, temperature, max_tokens, sem):
    async with sem:
        try:
            extra = {}
            # MIMESIS thinking checkpoints render a template that pre-fills <think>.
            # ODYSSIM_ENABLE_THINKING=0 turns that off at the template level, which is
            # the only way to stop generation from spending the whole budget on the
            # reasoning block for a task whose prompt already asks for explicit CoT
            # ("Thought: ... Message: ...").
            if os.environ.get("ODYSSIM_ENABLE_THINKING") is not None:
                extra["extra_body"] = {"chat_template_kwargs": {
                    "enable_thinking": os.environ["ODYSSIM_ENABLE_THINKING"] == "1"}}
            # Repetition penalty. Measured on the 459-topic run, our checkpoints
            # re-issue a near-identical request on 36-80% of turns (>=0.6 Jaccard with
            # an earlier turn) against 0-2% for frontier models, and corr(cap%,
            # repeat%) = 0.79 -- the conversations hit the 15-turn cap because the
            # simulator loops, not because it declines to stop. >1 discourages that at
            # the sampler. Off by default: it changes the decoding distribution, so a
            # run with it set is not comparable to one without, and the label must say so.
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
    """Remove chain of thought before the turn leaves this module.

    THIS IS LOAD-BEARING, not cosmetic. SimulatorArena hides the simulator's chain
    of thought from the assistant by splitting on the literal strings "Thought:" /
    "Message:" (utils.py, the `data['assistant_messages'].append({"role": "user",
    "content": query})` path). Anything that uses a different marker falls through
    to `query = user_query` and the assistant receives the private reasoning
    verbatim.

    TWO shapes get through that guard, and only one is a tag:
      * <think>...</think>
      * UNTAGGED PROSE -- "Thinking Process:\\n\\n1. **Analyze the Request:**...".
        Measured on the 51-topic run: 35.8% of Qwen3.5-4B turns and 26.8% of
        Qwen3.5-9B turns, which is why their median "message" was 542 and 88 words
        against a human 25. Tag-stripping cannot catch this by construction.

    Delegates to realusersim_eval.strip_reasoning so there is ONE definition of
    what a reasoning block is across both benchmarks. Note the regex deliberately
    does not match a bare "Thought:" -- that is SimulatorArena's own requested CoT
    format, which its Thought:/Message: split already handles downstream.
    """
    if not text:
        return text
    try:
        from realusersim_eval import strip_reasoning
        out = strip_reasoning(text)
    except Exception:
        out = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        out = re.sub(r"^.*?</think>", "", out, flags=re.S).strip()
    if not out and text.strip():
        # Loud, because the alternative is an empty user turn that silently
        # degrades this simulator's score with no trace of why.
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
    """Return one list of n strings per context, matching the other branches.

    n > 1 is served by repeating the call: neither the hosted endpoint curl transport
    nor a plain vLLM chat completion takes an `n`, and silently returning fewer
    than n strings would corrupt the caller's indexing.
    """
    if model_name.startswith(LOCAL_PREFIX):
        served = model_name[len(LOCAL_PREFIX):]
        # Local vLLM is ours alone, so concurrency is bounded by the server, not
        # a provider quota.
        limit = max_concurrent or int(os.environ.get("ODYSSIM_LOCAL_CONCURRENCY", "32"))
        client = _local_client(model_name)
        sem = asyncio.Semaphore(limit)
        tasks = [
            _one_local(client, ctx, served, temperature, max_tokens, sem)
            for ctx in full_contexts for _ in range(n)
        ]
    else:
        # Frontier: the binding constraint is the provider quota. claude-* share
        # the up-anthropic bucket (475/60s, 2450/600s) that throttled the SOUL
        # evals, so keep this well under it.
        limit = max_concurrent or int(os.environ.get("ODYSSIM_GATEWAY_CONCURRENCY", "16"))
        # API models: reasoning OFF. Unlike a local checkpoint, there is no chat
        # template to consult and no way to inspect what the provider does by
        # default, so "let the model choose" is not available here -- the only
        # options are a value we pick or a provider default we cannot see. Pick
        # the one that matches what this benchmark asks for: a user's chat message,
        # not a deliberation.
        #   gpt-*    -> reasoning.effort = "none" in the Responses payload
        #   claude-* -> api.py already sends thinking={"type":"disabled"} by default
        #   gemini-* -> effort flows into generationConfig
        # "none" is also the only value for which _to_responses_payload skips the
        # 8192-token reasoning headroom; that is safe here because SIMARENA_MAX_TOKENS
        # is 8192, whereas the 2048-token budget on the RealUserSim assistant made
        # the same setting starve and triggered a retry storm.
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
