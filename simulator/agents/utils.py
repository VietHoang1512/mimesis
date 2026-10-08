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

import asyncio
import contextvars
import copy
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass

from omegaconf import DictConfig
from openai import AsyncOpenAI
from transformers import AutoTokenizer, PreTrainedTokenizer

from llm import api


async def post_chat(messages, folder, title):
    return


def _save_conversation_to_disk(
    context,
    experiment_name,
    data_source,
    index,
    step,
    date_tag,
    reward,
    reward_extra_info,
    chat,
    extra,
):
    if os.getenv("SAVE_CONVERSATIONS", "1").lower() in ("0", "false", "no"):
        return
    try:
        save_root = os.getenv("CONVERSATION_LOG_DIR")
        if not save_root:
            try:
                save_root = context.config.trainer.default_local_dir
            except Exception:
                save_root = os.path.join("outputs", str(experiment_name))
        conv_dir = os.path.join(save_root, "conversations")
        os.makedirs(conv_dir, exist_ok=True)

        try:
            reward_val = float(reward)
        except (TypeError, ValueError):
            reward_val = reward

        record = {
            "experiment_name": experiment_name,
            "data_source": data_source,
            "index": index,
            "global_step": step,
            "date": date_tag,
            "reward": reward_val,
            "reward_extra_info": reward_extra_info,
            "extra": extra,
            "messages": chat,
        }

        fname = re.sub(r"[^A-Za-z0-9._-]", "_", f"{data_source}-{index}.json")
        path = os.path.join(conv_dir, fname)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)
    except Exception as e:
        print(f"[save_conversation] failed for {data_source}-{index}: {e}")


async def process_post_chat(
    data, context, chat, output, format_think=False, extra=None
):
    if not context.is_train:
        import datetime

        step = context.global_step
        experiment_name = context.config.trainer.experiment_name
        now = datetime.datetime.now()
        date_tag = f"{now.month}.{now.day}.{now.hour}"
        index = data["extra_info"]["index"]
        data_source = data.get("data_source", "unknown")
        extra_info = output.extra_fields.get("reward_extra_info") or {}

        _save_conversation_to_disk(
            context,
            experiment_name,
            data_source,
            index,
            step,
            date_tag,
            output.reward_score,
            extra_info,
            chat,
            extra,
        )

        result_rows = "\n".join(f"| {k} | {v} |" for k, v in extra_info.items())

        if not extra:
            extra = ""
        elif isinstance(extra, dict):
            extra = json.dumps(extra, indent=4)
        else:
            extra = str(extra)

        canvas = (
            f"<|canvas|>"
            f"## System Prompt\n"
            f"{chat[0]['content']}\n"
            f"## Meta\n"
            f"| Key | Value |\n|---|---|\n"
            f"| data | {data_source} |\n"
            f"| index | {index} |\n"
            f"| step | {step} |\n"
            f"| date | {date_tag} |\n"
            f"## Result\n"
            f"| Key | Value |\n|---|---|\n"
            f"{result_rows}\n"
            f"## Extra\n"
            f"{extra}"
            f"<|/canvas|>"
        )
        reward = output.reward_score
        messages = chat[1:]
        if format_think:
            processed = []
            for m in messages:
                if m["role"] == "assistant":
                    think, content = split_think(m["content"])
                    new_content = (
                        f"<|think|>{think}<|/think|>{content}" if think else content
                    )
                    processed.append({**m, "content": new_content})
                else:
                    processed.append(m)
            messages = processed
        first_assistant = next((m for m in messages if m["role"] == "assistant"), None)
        if first_assistant:
            first_assistant["content"] += canvas
        await post_chat(
            messages,
            folder=f"{experiment_name}-{step}-({date_tag})",
            title=f"val-{data_source}-{index}-({reward})",
        )


_openai_client: AsyncOpenAI | None = None


OPENAI_TIMEOUT = 120

LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY", "")
LLAMA_BASE_URL = os.getenv("OPENAI_FALLBACK_BASE_URL", "http://localhost:8000/v1")


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or LLAMA_API_KEY,
            base_url=os.getenv("OPENAI_BASE_URL") or LLAMA_BASE_URL,
            timeout=OPENAI_TIMEOUT,
        )
    return _openai_client


def get_judge_model(default: str) -> str:
    """Return JUDGE_MODEL_NAME env override if set, else the default."""
    return os.environ.get("JUDGE_MODEL_NAME") or default


def get_judge_reasoning(default):
    """Return JUDGE_MODEL_REASONING env override if set, else the default."""
    val = os.environ.get("JUDGE_MODEL_REASONING")
    return val if val else default


def _resolve_model() -> str:
    return api.normalize_model(os.environ.get("OPENAI_MODEL_NAME"))


def _extract_reasoning_effort(reasoning_effort, kwargs):
    reasoning = kwargs.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        return reasoning["effort"]
    return reasoning_effort


async def call_openai(
    messages,
    model="gpt-5-nano",
    max_retries=3,
    response_format=None,
    reasoning_effort="low",
):
    model = _resolve_model()
    return await api.chat_async(
        messages,
        model=model,
        max_retries=50,
        response_format=response_format,
        reasoning_effort=reasoning_effort,
    )


async def call_openai_parse(
    messages,
    text_format,
    model="gpt-5-nano",
    max_retries=3,
    reasoning_effort="low",
    **kwargs,
):
    model = _resolve_model()
    effort = _extract_reasoning_effort(reasoning_effort, kwargs)
    return await api.chat_parse(
        messages,
        text_format,
        model=model,
        max_retries=50,
        reasoning_effort=effort,
    )



async def editlens_score(
    text: str, base_url: str | None = None, timeout: float = 10.0, max_retries: int = 2
) -> float | None:
    base_url = base_url or os.getenv("EDITLENS_BASE_URL")
    if not base_url:
        return None
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=base_url, api_key="EMPTY", timeout=timeout, max_retries=0
    )
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            completion = await client.chat.completions.create(
                model="editlens-llama",
                max_tokens=20,
                temperature=0.0,
                messages=[{"role": "user", "content": text}],
            )
            return float(
                json.loads(completion.choices[0].message.content)["score_pred"]
            )
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (attempt + 1))
    logging.getLogger(__name__).warning(
        f"[editlens_score] API call failed after {max_retries + 1} attempts, returning None: {last_err}"
    )
    return None


def remove_think(text: str) -> str:
    """Remove thinking blocks from model response."""
    if not text:
        return text
    text = re.sub(r"<seed:think>.*?</seed:think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</seed:think>", "", text, count=1, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, count=1, flags=re.DOTALL)
    text = re.sub(r"<seed:think>.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def split_think(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    think_parts = []
    for pattern in (r"<seed:think>(.*?)</seed:think>", r"<think>(.*?)</think>"):
        for m in re.finditer(pattern, text, flags=re.DOTALL):
            think_parts.append(m.group(1).strip())
    content = remove_think(text)
    return "\n".join(think_parts), content


MIN_THINK_TOKENS = int(os.getenv("MIN_THINK_TOKENS", "4"))

_THINK_TAGS = (("<think>", "</think>"), ("<seed:think>", "</seed:think>"))


def has_think_tag(text: str) -> bool:
    """True if ``text`` carries any reasoning tag, opening or closing, in either convention."""
    t = text or ""
    return any(tag in t for pair in _THINK_TAGS for tag in pair)

IS_TRAIN = contextvars.ContextVar("mimesis_is_train", default=False)


def split_think_response(text: str) -> tuple[str | None, str]:
    """Split a completion into (reasoning, answer)."""
    text = text or ""
    for open_tag, close_tag in _THINK_TAGS:
        n_open, n_close = text.count(open_tag), text.count(close_tag)
        if n_close == 0:
            continue
        if n_close != 1 or n_open > 1:
            return None, ""
        if n_open == 1 and not text.lstrip().startswith(open_tag):
            return None, ""
        reasoning, _, answer = text.partition(close_tag)
        if n_open == 1:
            reasoning = reasoning.split(open_tag, 1)[1]
        return reasoning.strip(), answer.strip()
    return None, ""


def decode_conversation(
    input_ids: list[int], tokenizer
) -> tuple[list[dict[str, str]], str]:
    decoded_str = tokenizer.decode(input_ids, skip_special_tokens=False)
    pattern = re.compile(
        re.escape(tokenizer.bos_token)
        + r"(system|user|assistant|tool)\n"
        + r"(.*?)"
        + r"(?="
        + re.escape(tokenizer.eos_token)
        + r")",
        re.DOTALL,
    )
    matches = pattern.findall(decoded_str)
    conversation = [{"role": role, "content": content} for role, content in matches]
    return conversation, decoded_str


def truncate_text(text: str, n_words: int) -> str:
    if not text:
        return text
    matches = list(re.finditer(r"\S+", text))
    if len(matches) <= n_words:
        return text
    end = matches[n_words - 1].end()
    return text[:end] + " [...]"


def truncate_text_left(text: str, n_words: int) -> str:
    """Keep the last n_words words (drop from the left). For conversation history."""
    if not text:
        return text
    matches = list(re.finditer(r"\S+", text))
    if len(matches) <= n_words:
        return text
    start = matches[-n_words].start()
    return "[...] " + text[start:]


def truncate_turns_for_reference(
    turns: list[dict],
    content_key: str = "content",
    max_words_per_turn: int = 500,
) -> list[dict]:
    """Truncate turn contents for use as references in teacher prompts."""
    return [
        {**t, content_key: truncate_text(t.get(content_key, ""), max_words_per_turn)}
        for t in turns
    ]


def _eval_seed():
    """Optional per-run sampling seed for reproducible multi-seed eval."""
    v = os.getenv("EVAL_SEED")
    if v is None or not v.strip():
        return None
    try:
        return int(v)
    except ValueError:
        return None


class LLMClass:
    async def create_completion(self, input_ids, **kwargs):
        raise NotImplementedError


class CallLLM(LLMClass):
    def __init__(self, url, tokenizer, config, loop, **kwargs):
        self.server_manager = url
        self.tokenizer = tokenizer
        self.config = config
        self.loop = loop

    async def _create_completion(self, input_ids, **kwargs):
        from uuid import uuid4

        max_len = (
            kwargs.pop("max_len", None)
            or self.config.prompt_length + self.config.response_length
        )
        max_len = min(max_len, self.config.prompt_length + self.config.response_length)
        max_new_tokens = max_len - len(input_ids)
        if "max_new_tokens" in kwargs:
            max_new_tokens = min(max_new_tokens, kwargs["max_new_tokens"])

        if max_new_tokens < 10:
            return None

        uid = kwargs.pop("uid", None) or uuid4().hex

        sampling_params = kwargs.pop("sampling_params", None) or {}
        seed = sampling_params.get("seed", _eval_seed())
        sampling_params = {
            "temperature": sampling_params.get(
                "temperature", getattr(self.config, "temperature", 1.0)
            ),
            "top_p": sampling_params.get("top_p", getattr(self.config, "top_p", 1.0)),
            "top_k": sampling_params.get("top_k", getattr(self.config, "top_k", -1)),
            "repetition_penalty": sampling_params.get(
                "repetition_penalty", getattr(self.config, "repetition_penalty", 1.0)
            ),
            "logprobs": sampling_params.get(
                "logprobs", getattr(self.config, "calculate_log_probs", True)
            ),
            "max_tokens": max_new_tokens,
        }
        if seed is not None:
            sampling_params["seed"] = seed

        output = await self.server_manager.generate(
            request_id=uid,
            prompt_ids=input_ids,
            sampling_params=sampling_params,
            image_data=None,
        )

        if output is None or len(output.token_ids) == 0:
            return None

        response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True),
        )

        return {
            "choices": [
                {
                    "message": {
                        "content": response_text,
                        "raw_output_ids": output.token_ids,
                        "response_log_probs": (
                            output.log_probs
                            if hasattr(output, "log_probs")
                            else [0.0] * len(output.token_ids)
                        ),
                        "routed_experts": (
                            output.routed_experts
                            if hasattr(output, "routed_experts")
                            else None
                        ),
                        "extra_data": {"input_ids": input_ids},
                        "metrics": {},
                    }
                }
            ]
        }

    async def create_completion(self, input_ids, **kwargs):
        completion = await self._create_completion(input_ids, **kwargs)
        return completion


class CallAPI(LLMClass):
    def __init__(self, url, tokenizer, config, **kwargs):
        self.tokenizer = tokenizer
        self.config = config
        self.model = url
        self.base_url = os.getenv("OPENAI_AGENT_BASE_URL")
        self.transport = os.getenv("OPENAI_AGENT_TRANSPORT", "openai")
        self.client = (
            AsyncOpenAI(
                api_key=os.getenv("OPENAI_AGENT_API_KEY"),
                base_url=self.base_url,
            )
            if self.transport == "openai"
            else None
        )

    async def _api_completion(self, messages, max_tokens, input_ids):
        """Route the evaluated model through the configured OpenAI-compatible endpoint."""
        model = api.normalize_model(self.model)
        reasoning_effort = os.getenv("OPENAI_AGENT_REASONING_EFFORT", None)
        try:
            text = await api.chat_async(
                messages,
                model=model,
                max_tokens=max_tokens,
                max_retries=20,
                reasoning_effort=reasoning_effort,
                seed=_eval_seed(),
            )
        except Exception as e:
            print(f"[CallAPI] completion failed: {e}")
            return None

    async def create_completion(self, input_ids, **kwargs):
        max_len = (
            kwargs.pop("max_len", None)
            or self.config.prompt_length + self.config.response_length
        )
        max_tokens = max_len - len(input_ids)

        kwargs.pop("uid", None)
        kwargs.pop("sampling_params", None)

        if "max_new_tokens" in kwargs:
            max_tokens = min(max_tokens, kwargs.pop("max_new_tokens"))

        if max_tokens < 10:
            print(
                f"[CallAPI] context exhausted, skipping rollout: max_tokens={max_tokens} "
                f"(max_len={max_len}, prompt={len(input_ids)} tokens)"
            )
            return None

        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = decode_conversation(input_ids, self.tokenizer)[0]

        if self.transport != "openai":
            return await self._api_completion(messages, max_tokens, input_ids)
        reasoning_effort = os.getenv("OPENAI_AGENT_REASONING_EFFORT", None)
        extra_kwargs = {}
        if reasoning_effort:
            extra_kwargs["reasoning_effort"] = reasoning_effort
        seed = _eval_seed()
        if seed is not None:
            extra_kwargs["seed"] = seed

        extra_body = {}
        rep_pen = float(getattr(self.config, "repetition_penalty", 1.0) or 1.0)
        if rep_pen != 1.0:
            extra_body["repetition_penalty"] = rep_pen
        freq_pen = float(getattr(self.config, "frequency_penalty", 0.0) or 0.0)
        if freq_pen != 0.0:
            extra_body["frequency_penalty"] = freq_pen
        pres_pen = float(getattr(self.config, "presence_penalty", 0.0) or 0.0)
        if pres_pen != 0.0:
            extra_body["presence_penalty"] = pres_pen
        norep = int(getattr(self.config, "no_repeat_ngram_size", 0) or 0)
        if norep > 0:
            extra_body["no_repeat_ngram_size"] = norep
        if self.transport == "openai" and os.getenv(
            "USER_SIM_ENABLE_THINKING", "0"
        ).lower() not in ("1", "true", "yes"):
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if extra_body:
            extra_kwargs["extra_body"] = extra_body

        needs_stream = "Qwen3.6-Plus" in self.model

        for attempt in range(10):
            try:
                if needs_stream:
                    stream = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        stream=True,
                        stream_options={"include_usage": True},
                        **extra_kwargs,
                    )
                    text = ""
                    usage = None
                    async for chunk in stream:
                        if chunk.usage is not None:
                            usage = chunk.usage
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                text += delta.content

                    class _U:
                        prompt_tokens = (
                            getattr(usage, "prompt_tokens", 0) if usage else 0
                        )
                        completion_tokens = (
                            getattr(usage, "completion_tokens", 0) if usage else 0
                        )
                        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

                    response = type("R", (), {"usage": _U()})()
                else:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                        **extra_kwargs,
                    )
                    text = response.choices[0].message.content or ""

                text_ids = self.tokenizer.encode(text, add_special_tokens=False)

                return {
                    "choices": [
                        {
                            "message": {
                                "content": text,
                                "raw_output_ids": text_ids,
                                "response_log_probs": [0.0] * len(text_ids),
                                "routed_experts": None,
                                "extra_data": {"input_ids": input_ids},
                                "metrics": {
                                    "usage": {
                                        "prompt_tokens": (
                                            response.usage.prompt_tokens
                                            if response.usage
                                            else 0
                                        ),
                                        "completion_tokens": (
                                            response.usage.completion_tokens
                                            if response.usage
                                            else len(text_ids)
                                        ),
                                        "total_tokens": (
                                            response.usage.total_tokens
                                            if response.usage
                                            else 0
                                        ),
                                    }
                                },
                            }
                        }
                    ]
                }
            except Exception as e:
                if attempt == 9:
                    print(f"[CallAPI ERROR] Failed after 10 attempts: {e}")
                    return None
                wait_time = min(2**attempt, 60)
                print(
                    f"[CallAPI] Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
        return None


def truncate_prompt(
    chat, prompt_length, tokenizer, prompt_turn, apply_chat_template=None
):
    _apply = apply_chat_template or tokenizer.apply_chat_template
    exceed_len = len(_apply(chat[:prompt_turn], tokenize=True)) + 8 - prompt_length
    _cut_idx = 0
    while exceed_len > 0:
        print("[PROMPT] now exceed", exceed_len, "work on cut turn", _cut_idx)
        chat[_cut_idx]["content"] = tokenizer.decode(
            tokenizer.encode(chat[_cut_idx]["content"], add_special_tokens=False)[
                exceed_len + 4 :
            ],
            add_special_tokens=False,
        )
        exceed_len = len(_apply(chat[:prompt_turn], tokenize=True)) + 8 - prompt_length
        _cut_idx = _cut_idx + 1
        if _cut_idx >= prompt_turn:
            break
    return chat


_CHATML_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + (message['content'] or '') + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant' + '\\n' }}"
    "{% endif %}"
)


def _to_ids(result):
    """Normalize apply_chat_template output to list[int]."""
    if isinstance(result, (str, list)):
        return result
    try:
        return result["input_ids"]
    except (KeyError, TypeError):
        return result


class AgentContext:
    # Manage context of an agent
    def __init__(
        self,
        chat,
        tokenizer,
        config,
        prompt_turn=2,
        enable_think=None,
        reserve_length=0,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.init_len = len(chat)
        self.prompt_turn = prompt_turn
        self.reserve_length = reserve_length

        if os.getenv("USE_MODEL_CHAT_TEMPLATE", "0").lower() in ("0", "false", "no"):
            tokenizer.chat_template = _CHATML_CHAT_TEMPLATE

        if enable_think is None:
            enable_think = os.getenv("TURNOFF_THINK", "1").lower() in (
                "0",
                "false",
                "no",
            )
        self.enable_think = enable_think
        _orig = tokenizer.apply_chat_template
        if enable_think:
            self._apply_chat_template = lambda *a, **k: _to_ids(_orig(*a, **k))
        else:

            def _apply_chat_template(*args, **kwargs):
                for extra in [{"thinking_budget": 0}, {"enable_thinking": False}, {}]:
                    try:
                        return _to_ids(_orig(*args, **{**extra, **kwargs}))
                    except TypeError:
                        continue
                return _to_ids(_orig(*args, **kwargs))

            self._apply_chat_template = _apply_chat_template

        # Support both inference and training config styles
        if hasattr(config, "actor_rollout_ref"):
            # Training config (VERL)
            self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
            self.response_length = config.actor_rollout_ref.rollout.response_length
        else:
            # Inference config
            self.prompt_length = config.prompt_length
            self.response_length = config.response_length

        self.context_uid = str(uuid.uuid4())

        self.chat = copy.deepcopy([turn for turn in chat])
        self.chat = truncate_prompt(
            self.chat,
            self.prompt_length,
            tokenizer,
            prompt_turn,
            self._apply_chat_template,
        )
        self.chat_completions = [None for _ in range(len(self.chat))]
        self.chat_ids = [self.get_turn_context(i) for i in range(len(self.chat))]
        self.log_probs = [[0.0] * len(turn) for turn in self.chat_ids]
        self.token_mask = [[False] * len(turn) for turn in self.chat_ids]
        self.turn_truncated = [False for _ in self.chat_ids]
        self.additional_info = [None for _ in self.chat_ids]
        self.generation_prompt = None
        self.last_raw = None
        self.metrics = None
        self.prompt_ids_len = len(sum(self.chat_ids[:prompt_turn], []))

    def get_turn_context(self, i):
        tokens = self._apply_chat_template(
            self.chat[: i + 1], add_generation_prompt=False, tokenize=True
        )
        prev = (
            self._apply_chat_template(
                self.chat[:i], add_generation_prompt=False, tokenize=True
            )
            if i > 0
            else []
        )
        turn_tokens = tokens[len(prev) :]
        return turn_tokens

    def get_generation_prompt(self):
        if self.generation_prompt is None:
            tokens = self._apply_chat_template(
                self.chat, add_generation_prompt=False, tokenize=True
            )
            add_tokens = self._apply_chat_template(
                self.chat, add_generation_prompt=True, tokenize=True
            )
            self.generation_prompt = add_tokens[len(tokens) :]
        return self.generation_prompt

    def messages(self):
        return self.chat

    def context_ids(self, messages=None):
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def context(self, turn_cut: int = None):
        if turn_cut is not None:
            return sum(self.chat_ids[:turn_cut], []) + self.get_generation_prompt()
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def append(self, turn, completion=None, additional_info=None):
        self.chat.append(turn)
        self.chat_completions.append(completion)
        self.additional_info.append(additional_info)
        if completion is None:
            self.chat_ids.append(self.get_turn_context(len(self.chat) - 1))
            self.log_probs.append([0.0] * len(self.chat_ids[-1]))
            self.token_mask.append([False] * len(self.chat_ids[-1]))
            self.turn_truncated.append(False)
        else:
            completion_tokens = completion["choices"][0]["message"]["raw_output_ids"]
            completion_log_probs = completion["choices"][0]["message"][
                "response_log_probs"
            ] or [0.0] * len(completion_tokens)
            self.chat_ids.append(self.get_generation_prompt() + completion_tokens)
            self.log_probs.append(
                [0.0] * len(self.get_generation_prompt()) + completion_log_probs
            )
            self.token_mask.append(
                [False] * len(self.get_generation_prompt())
                + [True] * len(completion_tokens)
            )
            # No natural EOS => generation hit the token cap.
            truncated = (
                len(completion_tokens) == 0
                or completion_tokens[-1] != self.tokenizer.eos_token_id
            )
            self.turn_truncated.append(truncated)
            if truncated:
                self.chat_ids[-1].append(self.tokenizer.eos_token_id)
                self.log_probs[-1].append(0.0)
                self.token_mask[-1].append(False)

    def clean_response(self, text):
        """The text handed to scorers, judges and counterpart models: reasoning removed."""
        if not self.enable_think:
            return remove_think(text or "")
        reasoning, answer = split_think_response(text)
        if reasoning is None and not has_think_tag(text):
            return remove_think(text or "")
        return answer

    def _count_tokens(self, text):
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return len(text.split())

    def think_format_correct(self):
        return self.think_stats()[0]

    def think_stats(self):
        """(format_correct, mean reasoning tokens) over generated assistant turns."""
        correct = 1
        think_tokens = []
        for i, turn in enumerate(self.chat):
            if turn["role"] != "assistant" or self.chat_completions[i] is None:
                continue
            content = turn.get("content", "") or ""
            try:
                if not self.enable_think:
                    if content.lstrip().startswith(("<think>", "<seed:think>")):
                        correct = 0
                    continue
                reasoning, answer = split_think_response(content)
                if reasoning is None or not answer:
                    correct = 0
                    continue
                n_tokens = self._count_tokens(reasoning)
                think_tokens.append(n_tokens)
                truncated = i < len(self.turn_truncated) and self.turn_truncated[i]
                if truncated or n_tokens < MIN_THINK_TOKENS:
                    correct = 0
            except Exception:
                correct = 0
        mean_tokens = sum(think_tokens) / len(think_tokens) if think_tokens else 0.0
        return correct, mean_tokens

    def fork(self):
        """Copy this context resetting all token masks and log probs to zero."""
        new = object.__new__(self.__class__)
        new.tokenizer = self.tokenizer
        new.config = self.config
        new.init_len = self.init_len
        new.prompt_turn = self.prompt_turn
        new._apply_chat_template = self._apply_chat_template
        new.prompt_length = self.prompt_length
        new.response_length = self.response_length
        new.context_uid = self.context_uid
        new.chat = copy.deepcopy(self.chat)
        new.chat_completions = copy.deepcopy(self.chat_completions)
        new.chat_ids = copy.deepcopy(self.chat_ids)
        new.log_probs = [[0.0] * len(ids) for ids in self.chat_ids]
        new.token_mask = [[False] * len(ids) for ids in self.chat_ids]
        new.turn_truncated = list(self.turn_truncated)
        new.additional_info = copy.deepcopy(self.additional_info)
        new.generation_prompt = self.generation_prompt
        new.last_raw = None
        new.metrics = None
        new.enable_think = self.enable_think
        new.reserve_length = 0
        new.prompt_ids_len = len(sum(new.chat_ids, []))
        return new

    def rollback(self, k=1):
        self.chat = self.chat[:-k]
        self.chat_completions = self.chat_completions[:-k]
        self.chat_ids = self.chat_ids[:-k]
        self.log_probs = self.log_probs[:-k]
        self.token_mask = self.token_mask[:-k]
        self.turn_truncated = self.turn_truncated[:-k]
        self.additional_info = self.additional_info[:-k]

    def get_metrics(self):
        if self.metrics is None:
            return {}
        return self.metrics

    async def get_data(self):
        prompt_turn = self.prompt_turn
        prompt_length = self.prompt_length
        response_length = self.response_length

        prompt_ids = sum(self.chat_ids[:prompt_turn], [])
        if len(prompt_ids) > prompt_length:
            print("[PROMPT] prompt truncated")
            prompt_ids = prompt_ids[-prompt_length:]

        response_ids = sum(self.chat_ids[prompt_turn:], [])[:response_length]
        response_logprobs = sum(self.log_probs[prompt_turn:], [])[:response_length]
        response_mask = [
            1 if m else 0 for turn in self.token_mask[self.prompt_turn :] for m in turn
        ][:response_length]
        process_reward_mask = sum(
            [
                [info.get("process_reward", 0) if isinstance(info, dict) else 0]
                * len(turn)
                for turn, info in zip(self.chat_ids, self.additional_info, strict=False)
            ][prompt_turn:],
            [],
        )
        process_reward_mask = [
            p * m for p, m in zip(process_reward_mask, response_mask, strict=False)
        ][:response_length]
        return {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_logprobs": response_logprobs,
            "response_mask": response_mask,
            "process_reward_mask": process_reward_mask,
            "num_turns": len(self.chat_ids),
            "messages": self.chat,
        }

    async def get_agent_output(
        self,
        agent_reward,
        extra_info=None,
        teacher_prompt=None,
        gen_uid=None,
        agent_role=None,
    ):
        extra_info = dict(extra_info or {})
        fmt_ok, think_tokens = self.think_stats()
        extra_info["all/format_correct"] = float(fmt_ok)
        extra_info["all/think_tokens"] = float(think_tokens)
        headline = next(
            (extra_info[k] for k in ("all/score", "all/score_v3") if isinstance(extra_info.get(k), (int, float))),  # noqa: UP038
            agent_reward,
        )
        if isinstance(headline, (int, float)):  # noqa: UP038
            extra_info["all/score_v2"] = float(headline)
        if IS_TRAIN.get() and self.enable_think and not fmt_ok:
            agent_reward = 0.0
        extra_fields = {"reward_extra_info": extra_info}
        if gen_uid is not None:
            extra_fields["gen_uid"] = gen_uid
        if agent_role is not None:
            extra_fields["agent_role"] = agent_role
        if teacher_prompt is not None:
            extra_fields["teacher_prompt_ids"] = self._apply_chat_template(
                teacher_prompt, add_generation_prompt=True, tokenize=True
            )

        from verl.experimental.agent_loop.agent_loop import (
            AgentLoopMetrics,
            AgentLoopOutput,
        )

        out = await self.get_data()
        out = AgentLoopOutput(
            prompt_ids=out["prompt_ids"],
            response_ids=out["response_ids"],
            response_mask=out["response_mask"],
            response_logprobs=out["response_logprobs"],
            multi_modal_data={},
            reward_score=agent_reward,
            num_turns=out["num_turns"],
            metrics=AgentLoopMetrics(),
            extra_fields=extra_fields,
        )
        return out


class Agent(AgentContext):
    # Agent utils
    def __init__(
        self,
        llm_client,
        conversations,
        tokenizer,
        config,
        prompt_turn=2,
        enable_think=None,
        reserve_length=0,
    ):
        super().__init__(
            conversations,
            tokenizer,
            config,
            prompt_turn=prompt_turn,
            enable_think=enable_think,
            reserve_length=reserve_length,
        )
        self.llm_client = llm_client
        self.info_cache = {}

    async def step(self, max_new_tokens=None):
        prompt = self.context()
        max_len = self.prompt_ids_len + self.response_length - self.reserve_length
        if max_new_tokens is not None:
            max_len = min(len(prompt) + max_new_tokens, 131072)
        completion = await self.llm_client.create_completion(
            prompt, uid=self.context_uid, max_len=max_len, messages=self.chat
        )
        if completion is None:
            return None
        response = completion["choices"][0]["message"]["content"]
        self.append({"role": "assistant", "content": response}, completion)
        self.last_raw = response
        return self.clean_response(response)

    async def react(
        self,
        run_action,
        max_turn=64,
        max_tokens=None,
        session_timeout=60 * 60,
        should_continue=None,
        summary_prompt=None,
        safe_finish=None,
        observation_prompt=None,
    ):
        # Run react for max_turn turn
        if should_continue is None:
            should_continue = lambda st: True
        session_start_time = time.time()
        iteration = 0
        if max_tokens is not None:
            max_tokens = max_tokens - 512
        else:
            max_tokens = self.response_length - 512

        last_response = None
        response = None
        init_len = len(self.context(turn_cut=self.prompt_turn))
        while iteration < max_turn:
            if (
                time.time() - session_start_time > session_timeout
            ):
                print("[SESSION] Session Timeout")
                break
            if len(self.context()) - init_len > max_tokens:
                break

            iteration += 1
            response = await self.step()
            if response is None:
                break

            if not should_continue(response):
                last_response = response
                break
            if safe_finish is not None and safe_finish(response) is not None:
                observation = safe_finish(response)
            else:
                observation = await run_action(response)
            if observation is None:
                break
            if observation_prompt:
                observation += "\n" + observation_prompt
            self.append(
                {
                    "role": "user",
                    "content": observation,
                }
            )

        if last_response is None and summary_prompt is not None:
            if len(self.context()) - init_len > self.response_length - 1024:
                self.rollback(k=2)
            if self.chat[-1]["role"] == "user":
                self.append(
                    {
                        "role": "assistant",
                        "content": "",
                    }
                )
            self.append(
                {
                    "role": "user",
                    "content": summary_prompt,
                }
            )
            last_response = await self.step(max_new_tokens=4096)
        elif last_response is None:
            last_response = str(response)

        return {"last_response": last_response, "iteration": iteration}

    def set_process_reward(self, turn, reward):
        if isinstance(turn, str) and turn.lower() == "all":
            turn = [i for i in range(len(self.chat))]
        if not isinstance(turn, list):
            turn = [turn]
        for i in turn:
            if i <= 0:
                continue
            if i > len(self.chat) - 1:
                continue
            if self.chat_completions[i] is None:
                continue
            if self.additional_info[i] is None:
                self.additional_info[i] = {}
            self.additional_info[i]["process_reward"] = reward

    def set_cache(self, key, value):
        self.info_cache[key] = value

    def fork(self):
        new = super().fork()
        new.llm_client = self.llm_client
        new.info_cache = {}
        return new


@dataclass
class TaskContext:
    config: DictConfig
    global_step: int
    is_train: bool
    tokenizer: PreTrainedTokenizer | AutoTokenizer | None = None
    llm_client: LLMClass = None


async def run_action(env, response):
    try:
        try:
            act = time.time()
            env_return = await asyncio.wait_for(env.run_action(response), timeout=120.0)
            if time.time() - act > 10:
                print("Action Cost", time.time() - act)
        except asyncio.TimeoutError:
            print("[ACTION] Action timed out after 120 seconds")
            env_return = {"observation": "Action timed out after 120 seconds"}
        if "action" in env_return:
            action, arguments = env_return["action"], env_return.get("arguments", {})  # noqa: F841
            if action == "finish":
                return None
        elif env_return.get("observation", None) == "finish":
            return None
        observation = env_return.pop("observation", "Empty")
    except Exception as e:
        observation = f"Error: {e}"
    return observation
