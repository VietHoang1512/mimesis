# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tau-USI benchmark for Harmony.

This module runs live TauBench rollouts and computes aggregate USI-style
metrics against human baseline annotations from tau_bench_tasks_unified.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

from openai import AsyncOpenAI

from llm import api
from agents.env_utils import (
    BaseEnv,
    RuntimeServiceError,
    extract_fn_call,
)
from agents.tau_usi.reward import FeatureStatsBuffer, compute_distributional_reward
from agents.tau_usi.utils import FIELD_ORDINAL, extract_conversation_features
from agents.tool_prompt import TOOL_PROMPT, convert_tools_to_description
from agents.utils import Agent, _get_openai_client, call_openai, process_post_chat, remove_think

# Agent (the assistant the user-sim talks to) — matched to AgentArena's fixed
# tau eval agent in agent_service/tau_agent.py so USI numbers are comparable.
# On the leaderboard the agent was gpt-5.2 via the Responses API; on clusters without that endpoint
#  that model/endpoint is unreachable, so the agent runs through the shared API client
# chat-completions model (default gpt-5-5) — held constant across all user-sims.
AGENT_MODEL = os.getenv("TAU_USI_AGENT_MODEL", "gpt-5.2")
AGENT_REASONING_EFFORT = os.getenv("TAU_USI_AGENT_REASONING_EFFORT", "low")
# Completion cap for the agent turn (its user-facing message or one tool call).
AGENT_MAX_TOKENS = int(os.getenv("TAU_USI_AGENT_MAX_TOKENS", "8192"))
# Retries for a failed agent the shared API client call (api.chat_async's own default is 50). The
# earlier "hang" was a starved max_tokens causing a *persistent* error, not the retries.
AGENT_MAX_RETRIES = int(os.getenv("TAU_USI_AGENT_MAX_RETRIES", "50"))

# Role under which tool observations are fed back to the agent. tau_agent.py (the
# gpt-5.2/Responses reference agent) uses 'system', and the shared API client chat-completions
# tolerates a mid-conversation system message. But a LOCAL agent rendered through
# its OWN chat template cannot: Qwen's template hard-raises on a non-leading
# 'system' ("System message must be at the beginning.") and instead represents
# tool output as role 'tool' (rendered as <tool_response> user turns). So use the
# native 'tool' role for Qwen agents, and 'system' otherwise (leaderboard parity).
AGENT_OBS_ROLE = "tool" if "qwen" in AGENT_MODEL.lower() else "system"

# Cached tokenizer for the LOCAL agent, used only to budget max_tokens so that
# input + output stays within the served context window (see _agent_respond).
_AGENT_TOKENIZER: Any = None


def _agent_input_tokens(chat) -> int:
    """Best-effort input-token count for the local agent. Uses the agent's OWN
    tokenizer (exact, matches what vLLM computes) when it can be loaded; otherwise
    falls back to a conservative ~2.5 chars/token estimate (which over-counts, so we
    never request more output than the context can hold)."""
    global _AGENT_TOKENIZER
    if _AGENT_TOKENIZER is None:
        try:
            from transformers import AutoTokenizer

            _AGENT_TOKENIZER = AutoTokenizer.from_pretrained(AGENT_MODEL, trust_remote_code=True)
        except Exception:
            _AGENT_TOKENIZER = False
    if _AGENT_TOKENIZER:
        try:
            return len(_AGENT_TOKENIZER.apply_chat_template(chat, add_generation_prompt=True, tokenize=True))
        except Exception:
            try:
                return sum(len(_AGENT_TOKENIZER.encode(m.get("content") or "")) for m in chat)
            except Exception:
                pass
    return int(sum(len(m.get("content") or "") for m in chat) / 2.5)


async def _agent_respond(chat):
    """Generate the agent's turn (the assistant the user-sim talks to), matched to
    AgentArena's fixed tau eval agent (``agent_service/tau_agent.py``). Returns the
    response *text* only; tool-calling then uses the shared ``extract_fn_call`` +
    ``tau_env.step``, so nothing downstream depends on the transport.

    Three transports:
    - **the hosted Responses API (default).** On the compute cluster the OpenAI SDK is
      unreachable — connection-reset by the egress proxy — and the configured endpoint passthrough
      is VPN-gated (403). So route through ``api.chat_async`` (via llm/api.py),
      the same path the judges use. An unset ``TAU_USI_AGENT_MODEL`` falls back to
      ``api.DEFAULT_MODEL`` (gpt-5.5); since the endpoint migration gpt-5.2 is
      routable there too. api.chat_async strips an unsupported ``reasoning_effort``
      itself.
    - **Local OpenAI-compatible server.** When ``TAU_USI_AGENT_BASE_URL`` is set, the
      agent runs as a LOCAL model over ``/v1/chat/completions`` (e.g. a ``vllm serve``
      box) — the same transport the user-sim's ``CallAPI`` uses (agents/utils.py).
      ``TAU_USI_AGENT_MODEL`` is the served name; ``TAU_USI_AGENT_API_KEY`` optional.
    - **Direct OpenAI Responses API (SDK).** Used only when ``OPENAI_BASE_URL`` is explicitly set (e.g. the
      AgentArena box), reproducing the leaderboard's gpt-5.2 agent exactly.
    """
    if os.getenv("OPENAI_BASE_URL"):
        client = _get_openai_client()
        resp = await client.responses.create(
            model=AGENT_MODEL,
            input=chat,
            reasoning={"summary": "detailed", "effort": AGENT_REASONING_EFFORT},
        )
        answer = ""
        for item in resp.output:
            if getattr(item, "type", None) == "message" and getattr(item, "content", None):
                answer += ("\n\n" if answer else "") + item.content[0].text
        return answer

    # Local OpenAI-compatible agent (vLLM/SGLang/etc). When TAU_USI_AGENT_BASE_URL is
    # set, run the agent as a LOCAL model over /v1/chat/completions instead of the shared API client
    # gpt-5-5 — mirroring the user-sim's CallAPI OpenAI-SDK path (agents/utils.py).
    # AGENT_MODEL (=TAU_USI_AGENT_MODEL) must match the server's --served-model-name.
    agent_base_url = os.getenv("TAU_USI_AGENT_BASE_URL")
    if agent_base_url:
        client = AsyncOpenAI(
            api_key=os.getenv("TAU_USI_AGENT_API_KEY") or "EMPTY",
            base_url=agent_base_url,
        )
        # reasoning_effort is only valid for reasoning models; opt in explicitly via
        # TAU_USI_AGENT_REASONING_EFFORT_LOCAL so a plain instruct agent doesn't 400.
        extra: dict[str, Any] = {}
        if os.getenv("TAU_USI_AGENT_REASONING_EFFORT_LOCAL"):
            extra["reasoning_effort"] = AGENT_REASONING_EFFORT
        # Qwen3 & similar emit <think>...</think> by default; the agent turn is parsed
        # for tool calls and reused verbatim as a message, so keep it non-thinking (via
        # vLLM's chat_template_kwargs hook) unless TAU_USI_AGENT_ENABLE_THINKING=1. Any
        # think tags that still slip through are stripped from the returned text below.
        if os.getenv("TAU_USI_AGENT_ENABLE_THINKING", "0").lower() not in ("1", "true", "yes"):
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        # Local servers reject input_tokens + max_tokens > max_model_len. The agent prompt
        # (wiki + tools + growing transcript) is large for a small-context local model, so
        # cap the output to what's left. TAU_USI_AGENT_MAX_MODEL_LEN is the served context
        # (exported by run_tau_usi_eval_local_agent.sh); input tokens are counted with the
        # agent's own tokenizer (exact) when available. Unset (0) => send the full
        # AGENT_MAX_TOKENS (e.g. a big-context API agent).
        max_out = AGENT_MAX_TOKENS
        max_ctx = int(os.getenv("TAU_USI_AGENT_MAX_MODEL_LEN", "0") or 0)
        if max_ctx > 0:
            n_in = _agent_input_tokens(chat)
            max_out = max(256, min(AGENT_MAX_TOKENS, max_ctx - n_in - 128))
            print(f"[_agent_respond local] calling {AGENT_MODEL} (input~{n_in} tok, max_out={max_out}, ctx={max_ctx})")
        else:
            print(f"[_agent_respond local] calling {AGENT_MODEL} (max_out={max_out})")
        for attempt in range(10):
            try:
                resp = await client.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=chat,
                    max_completion_tokens=max_out,
                    **extra,
                )
                return remove_think(resp.choices[0].message.content or "")
            except Exception as error:
                # A 400 (e.g. the prompt alone exceeds the context window) is deterministic
                # — retrying only burns backoff, so give up on it immediately.
                status = getattr(error, "status_code", None)
                if status == 400 or attempt == 9:
                    print(f"[_agent_respond local] giving up (status={status}): {error}")
                    return ""
                await asyncio.sleep(min(2**attempt, 30))

    model = api.normalize_model(os.getenv("TAU_USI_AGENT_MODEL") or api.DEFAULT_MODEL)
    answer = await api.chat_async(
        chat,
        model=model,
        max_tokens=AGENT_MAX_TOKENS,
        reasoning_effort=AGENT_REASONING_EFFORT,
        max_retries=AGENT_MAX_RETRIES,
    )
    return answer or ""


class TauUSIEnv(BaseEnv):
    """Tau environment client with automatic recovery via conversation replay."""

    def __init__(self, env_name: str, task_index: int):
        super().__init__(env_name=env_name, task_index=task_index)

    def get_system_prompt(self) -> str:
        """Build tool-augmented system prompt from environment metadata."""
        if not self.meta_info:
            ping_result = self.ping()
            if ping_result["exists"]:
                self.meta_info = ping_result["meta_info"]

        if not self.meta_info:
            return ""

        tools_info = self.meta_info.get("tools_info", [])
        wiki = self.meta_info.get("wiki", "")
        tool_description = TOOL_PROMPT.format(description=convert_tools_to_description(tools_info))
        return wiki + "\n\n" + tool_description


def _to_positive_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
        return parsed if parsed > 0 else None
    except Exception:
        return None


def _remaining_timeout_seconds(started_at: float, total_timeout: float | None) -> float | None:
    if total_timeout is None:
        return None
    return max(0.0, total_timeout - (time.monotonic() - started_at))


async def _await_with_remaining_timeout(
    awaitable: Any,
    started_at: float,
    total_timeout: float | None,
) -> Any:
    """
    Await a coroutine with shared wall-clock timeout budget.
    """
    remaining_timeout = _remaining_timeout_seconds(started_at, total_timeout)
    if remaining_timeout is not None and remaining_timeout <= 0:
        raise asyncio.TimeoutError("Tau-USI overall timeout exhausted.")
    if remaining_timeout is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=remaining_timeout)


SURVEY_QUESTION_TEXT = {
    "task_success": "Did the agent successfully complete your task?",
    "efficiency": "How efficient was the agent in completing the task?",
    "question_amount_preference": "How did the number of clarifying questions feel to you?",
    "answer_effort_time": "How much time/effort did it take to answer the agent's clarifying questions?",
    "human_like": "Does the agent feel human-like?",
    "interaction_flow": "How smooth was the overall interaction during clarification?",
    "overall_score": "Overall agent performance score (1-5)",
    "reuse": "If you encounter similar problems in life, would you like to reuse this agent?",
}


def _default_structured_survey() -> dict[str, dict[str, str]]:
    return {
        field: {"question": SURVEY_QUESTION_TEXT.get(field, field), "answer": "no answer"} for field in FIELD_ORDINAL
    }


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1).strip())
    brace_match = re.search(r"\{.*\}", text, flags=re.S)
    if brace_match:
        candidates.append(brace_match.group(0).strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _structure_survey_answers(raw_answers: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    structured = _default_structured_survey()
    if not isinstance(raw_answers, dict):
        return structured

    for field, options_map in FIELD_ORDINAL.items():
        answer_value = raw_answers.get(field)
        if isinstance(answer_value, dict):
            answer_value = answer_value.get("answer")
        if isinstance(answer_value, str) and answer_value in options_map:
            structured[field]["answer"] = answer_value
    return structured


async def parse_survey_response(survey_response):
    parsed = _parse_json_object(survey_response)
    if parsed is None:
        question_ids = list(FIELD_ORDINAL.keys())
        reformat_prompt = (
            "A user just filled out a survey but the output was not valid JSON. "
            "Extract the survey answers and return ONLY a JSON object with these keys: "
            f"{question_ids}\n"
            '- If an answer is unrelated, use "no answer".\n'
            '- If the text is gibberish or unreadable, use "no answer".\n\n'
            f"{survey_response}"
        )
        try:
            reformatted = await call_openai(reformat_prompt)
            parsed = _parse_json_object(reformatted)
        except Exception:
            parsed = None
    return _structure_survey_answers(parsed or {})


async def rollout_one_task(data, context):
    """Run ONE live TauBench task with the model as user simulator.

    This is the pure rollout: it returns the per-task evaluation record and has
    NO dependency on verl or the RL output type. It can therefore be driven
    standalone (see ``run_eval.py``) to evaluate external/API user-sim models
    without verl/torch, OR wrapped by :func:`agent_loop` for the verl RL path.

    Needs only ``context.llm_client`` (the user-sim), ``context.tokenizer``,
    ``context.config`` and a reachable runtime service (``RUNTIME_SERVICE_URL``).

    Returns ``(record, user_agent)``: ``record`` is the eval row
    (``instance_id, conversation, chat, survey, reward, features,
    termination_reason``); ``user_agent`` is returned so the verl path can build
    token-level outputs — standalone eval ignores it.
    """
    task_index = int(data["task_index"])
    env_name = str(data["env_name"])
    instance_id = str(data.get("instance_id") or f"{env_name}_{task_index}")

    tau_env = TauUSIEnv(env_name=env_name, task_index=task_index)
    try:
        await tau_env.initialize()
    except RuntimeServiceError as error:
        raise RuntimeError("Tau runtime service unavailable during initialization.") from error

    # AgentArena eval_tau opens with the agent greeting; the user-sim replies first.
    GREETING = "Hi! How can I help you today?"
    system_prompt = tau_env.get_system_prompt()
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": GREETING},
    ]
    user_system_prompt = f"""{tau_env.meta_info["instruction"] if tau_env.meta_info else ""}

Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisfied, generate '###STOP###' as a standalone message without anything else to end the conversation.
- If transferring to a human, after the agent confirms the transfer is successful, you must generate '###STOP###' immediately to end the conversation with the agent.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""
    user_history: list[dict[str, str]] = [
        {"role": "system", "content": user_system_prompt},
        {"role": "user", "content": GREETING},
    ]
    user_agent = Agent(context.llm_client, user_history, context.tokenizer, context.config, prompt_turn=2)

    # User simulator replies to the greeting first.
    response = await user_agent.step()
    response = remove_think(response)
    chat.append({"role": "user", "content": response})

    # Turn structure mirrors AgentArena eval_tau (<=80 user turns) + tau_agent.py
    # (each agent turn does <=64 tool-call rounds before yielding to the user).
    termination_reason = "max_user_turns"
    content = ""
    for _user_turn in range(80):
        for _ in range(64):
            content = await _agent_respond(chat)
            chat.append({"role": "assistant", "content": content})
            fn_call = extract_fn_call(content)
            observation = None

            if isinstance(fn_call, dict) and "error" in fn_call:
                # Wrong tool-call format: tau_agent.py appends the hint as 'user'
                # and lets the agent retry.
                chat.append({"role": "user", "content": fn_call["error"]})
                continue

            if isinstance(fn_call, list) and fn_call:
                try:
                    chunks = []
                    for call_item in fn_call:
                        chunk = await tau_env.step(call_item, conversation=chat)
                        chunks.append(chunk)
                    observation = "\n\n".join(chunks)
                except Exception as error:
                    observation = f"Error executing tool: {error}"

            if observation is None:
                break  # no tool call -> user-facing message; hand off to user-sim
            # Tool observation. tau_agent.py appends as role 'system'; for a local
            # Qwen agent that becomes role 'tool' (see AGENT_OBS_ROLE) so its chat
            # template accepts it instead of 400ing on a non-leading system turn.
            chat.append({"role": AGENT_OBS_ROLE, "content": observation})

        # User simulator responds to the agent's message.
        user_agent.append({"role": "user", "content": content})
        user_response = await user_agent.step()
        if not user_response:
            termination_reason = "empty_user_response"
            break
        user_response = remove_think(user_response)
        if "###STOP###" in user_response:
            termination_reason = "stop"
            break
        chat.append({"role": "user", "content": user_response})

    questions_block_lines = []
    for field, options_map in FIELD_ORDINAL.items():
        options = list(options_map.keys())
        questions_block_lines.append(f"- {field}: {SURVEY_QUESTION_TEXT.get(field, field)}")
        questions_block_lines.append(f"  Options: {options}")
    questions_block = "\n".join(questions_block_lines)

    survey_prompt = f"""Based on the above conversation, fill out the survey below.

{questions_block}

Respond ONLY with a JSON object mapping each survey field id to exactly one listed option string.
Do not include any text outside JSON."""
    user_agent.append({"role": "user", "content": survey_prompt})

    response = await user_agent.step()
    response = remove_think(response)
    survey = await parse_survey_response(response)

    # Clean transcript for USI scoring: agent (assistant) + user-sim (user) turns
    # only — drop the system prompt and tool/developer observations. Built before
    # get_reward so it can be forwarded: tau-bench reward = DB-state check (from the
    # write actions already applied via tau_env.step) + output-string check (required
    # strings must appear in the agent's messages, which only the transcript carries).
    conversation = [m for m in chat[1:] if m.get("role") in ("user", "assistant")]

    model_reward = 0.0
    try:
        # get_reward is async; awaiting asyncio.to_thread(get_reward) returns the
        # *coroutine* (to_thread is for sync fns) -> float() on it blew up. Await directly.
        model_reward = await tau_env.get_reward(messages=conversation)
    except Exception:
        model_reward = 0.0

    features = extract_conversation_features(chat[1:], source="llm") or {}

    record = {
        "instance_id": instance_id,
        "env_name": env_name,
        "domain": env_name,
        "task_index": task_index,
        "conversation": conversation,
        "chat": chat,
        "survey": survey,
        "reward": float(model_reward),
        "features": features,
        "termination_reason": termination_reason,
    }
    return record, user_agent


async def agent_loop(data, context):
    """verl RL path: run one task, compute the distributional proxy reward, and
    return the verl ``AgentLoopOutput``. Thin wrapper over :func:`rollout_one_task`
    (the rollout itself needs no verl).

    For EVALUATION call :func:`rollout_one_task` directly, not this. This builds a
    verl ``AgentLoopOutput`` (so it imports verl) and requires RL-only inputs
    (``context["feature_stats_buffer"]``, ``data["human_feature_targets"]``);
    eval just wants the transcript/reward/survey/features that rollout_one_task
    already returns. See ``run_eval.py``."""
    record, user_agent = await rollout_one_task(data, context)
    features = record["features"]
    survey = record["survey"]
    model_reward = record["reward"]

    # --- Distributional reward (D1–D4 moment matching) ---
    # Requires context["feature_stats_buffer"] (FeatureStatsBuffer) and
    # data["human_feature_targets"] (dict of precomputed human baseline means
    # in the same units as extract_conversation_features output, i.e. rates in
    # [0,1], NOT the ×100 scaled values from build_row).
    dist_reward: dict[str, float] = {
        "D1_conv": 0.0,
        "D2_info": 0.0,
        "D3_clarif": 0.0,
        "D4_react": 0.0,
        "ece": 0.0,
        "eval_agreement": 0.0,
        "total": 0.0,
    }
    buf: FeatureStatsBuffer | None = context.get("feature_stats_buffer")
    human_targets: dict[str, float] | None = data.get("human_feature_targets")
    human_instance: dict[str, Any] | None = data.get("human_instance_data")
    buf.add(features)
    dist_reward = compute_distributional_reward(
        features=features,
        human_targets=human_targets,
        mu_estimates=buf.read(),
        mode="mse",
        model_reward=model_reward,
        human_reward=float(human_instance["reward"]) if human_instance else None,
        model_survey=survey,
        human_survey=human_instance.get("survey") if human_instance else None,
    )
    reward = dist_reward["total"]
    output = await user_agent.get_agent_output(
        reward, extra_info={"tau_usi/reward": dist_reward, **{f"tau_usi/{k}": v for k, v in dist_reward.items()}}
    )

    await process_post_chat(data, context, user_agent.chat, output)
    return output
