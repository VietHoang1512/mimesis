# Per-source SFT processors for the user-simulator mid-training mix. Each
# processor returns {input_ids, responses, response_mask} in the "old"
# schema; sft/dataset.py flattens this into the {input_ids, position_ids,
# loss_mask} that verl's sft_trainer expects.

import os
import re

import torch

from sft.chat_template import apply_chat_template, extract_system_prompt_and_generation, normalize_token_ids


def _has_consecutive_tool_turns(messages: list) -> bool:
    for i in range(len(messages) - 1):
        if messages[i].get("role") == "tool" and messages[i + 1].get("role") == "tool":
            return True
    return False


def _get_process_fn(data_source: str):
    _ROUTES = {
        # Character role-play
        "coser": process_coser,
        # Chat-style (human-AI) — all share wildchat-style simulating-user prompt
        "wildchat": process_swap,
        "wildchat_v1": process_swap,
        "lmsys": process_swap,
        "oasst1": process_swap,
        "oasst2": process_swap,
        "prism": process_swap,
        "alignx": process_swap,
        "nectar": process_swap,
        "hh_rlhf": process_swap,
        # Dialogue (human-human)
        "dialogue": process_dialogue,
        "dailydialog": process_dialogue,
        "cornell_movie": process_dialogue,
        "empathetic": process_dialogue,
        # Humanual-style
        "humanual": process_humanual,
        "humanual_book": process_humanual,
        "humanual_chat": process_humanual,
        "humanual_email": process_humanual,
        "humanual_news": process_humanual,
        "humanual_opinion": process_humanual,
        "humanual_politics": process_humanual,
        # Experiment
        "experiment": process_experiment,
        "psych101": process_psych101,
        "socsci210": process_experiment,
        # No-role-swap: assistant messages ARE the loss targets.
        "human_llm": process_no_swap,
        # ToM reasoning (instruction → answer format)
        "tom_from_coser": process_no_swap,
        "tom_socialiqa": process_no_swap,
        "tom_mindgames": process_no_swap,
        "tom_moralstories": process_no_swap,
        # Format-matched ToM with OpenAI CoT
        "tom_fantom": process_no_swap,
        "tom_hitom": process_no_swap,
        "tom_paratomi": process_no_swap,
        "tom_mmtom": process_no_swap,
        "tom_sotopia": process_no_swap,
        # Social-skills mega-sotopia
        "soc_sotopia_tom_silver": process_no_swap,
        "soc_haico": process_no_swap,
        "soc_persona_conflicts": process_no_swap,
        "soc_cornell": process_no_swap,
        "soc_sotopia_pi_bc": process_no_swap,
        # RM-R1 Distill SFT
        "rm_r1_sft": process_no_swap,
        # Additional ToM datasets
        "tom_characterllm": process_no_swap,
        "tom_grimulkan": process_no_swap,
        "tom_tominli": process_no_swap,
        # Education / tutoring
        "education_dialogue": process_swap,
        "mathdial": process_swap,
        "studychat": process_swap,
        # ConvoKit back-generated SFT corpora
        "convokit_IDEA-NTHU-unintended-offense-tweets": process_swap,
        "convokit_casino-corpus": process_swap,
        "convokit_chromium-corpus": process_swap,
        "convokit_conversations-gone-awry-cmv-corpus": process_swap,
        "convokit_conversations-gone-awry-cmv-corpus-large": process_swap,
        "convokit_conversations-gone-awry-corpus": process_swap,
        "convokit_emotional-support": process_swap,
        "convokit_friends-corpus": process_swap,
        "convokit_mediasum-corpus": process_swap,
        "convokit_npr-2p-corpus": process_swap,
        "convokit_parliament-corpus": process_swap,
        "convokit_persuasionforgood-corpus": process_swap,
        "convokit_reddit-coarse-discourse-corpus": process_swap,
        "convokit_reddit-corpus-small": process_swap,
        "convokit_small-pool": process_swap,
        "convokit_supreme-corpus": process_swap,
        "convokit_switchboard-corpus": process_swap,
        "convokit_tennis-corpus": process_swap,
        "convokit_wiki-articles-for-deletion-corpus": process_swap,
        "convokit_wiki-corpus": process_swap,
        "convokit_wikiconv-2018": process_swap,
        "convokit_winning-args-corpus": process_swap,
        # ppp-070 generic SFT processors:
        "sotopia": process_sotopia,
        "chat": process_chat,
        "step35_flash_sft": process_chat,
    }
    if data_source in _ROUTES:
        return _ROUTES[data_source]
    for key, fn in _ROUTES.items():
        if key in data_source:
            return fn
    raise NotImplementedError(
        f"No SFT data processor found for data_source={data_source!r}. Register one in sft/sft_data.py."
    )


def process(row: dict, tokenizer, config=None) -> dict:
    """Route one data row to the task-specific processor."""
    data_source = row.get("data_source", "")
    fn = _get_process_fn(data_source)
    return fn(row, tokenizer, config=config)


def _strip_system_flag(config) -> bool:
    if config is None:
        return False
    data_cfg = getattr(config, "data", config)
    try:
        return bool(data_cfg.get("strip_system", False))
    except AttributeError:
        return False


def _get_messages(row: dict, config=None) -> list:
    messages = row.get("messages")
    if messages is None:
        extra_info = row.get("extra_info") or {}
        messages = extra_info.get("messages")
    if messages is None:
        raise KeyError("SFT row must contain `messages` or `extra_info.messages`.")

    if _strip_system_flag(config):
        messages = [m for m in messages if m.get("role") != "system"]
    return messages


def truncate_text(text: str, n_words: int) -> str:
    matches = list(re.finditer(r"\S+", text))
    if len(matches) <= n_words:
        return text
    end = matches[n_words - 1].end()
    return text[:end] + " [...]"


def _fold_head(messages, keep_last=20):
    """Keep the last `keep_last` non-system turns; fold earlier turns into a
    single context string prepended to the first kept turn's content."""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if len(rest) <= keep_last:
        return messages
    head, tail = rest[:-keep_last], rest[-keep_last:]
    context = "[Earlier conversation]\n" + "\n\n".join(f"{m.get('role')}: {m.get('content') or ''}" for m in head)
    first = dict(tail[0])
    first["role"] = "assistant"
    first["content"] = context + "\n\n" + (first.get("content") or "")
    return system_msgs + [first] + tail[1:]


def _apply_chat_template(tokenizer, messages, **kwargs) -> list[int]:
    """Wrapper around verl's Qwen3.5-tolerant apply_chat_template that always
    returns a flat list[int] regardless of the transformers version. Silently
    drops thinking-toggle kwargs if the specific tokenizer doesn't accept them."""
    turnoff_think = os.getenv("TURNOFF_THINK", "1").lower() not in ("0", "false", "no")
    kwargs = dict(kwargs)
    kwargs.setdefault("tokenize", True)
    kwargs.setdefault("add_generation_prompt", False)
    # Try (thinking-disabling kwarg, then no kwarg). Qwen3.5 template accepts
    # `enable_thinking`; Seed/Gemma templates use `thinking_budget`.
    if turnoff_think:
        for extra in [{"enable_thinking": False}, {"thinking_budget": 0}, {}]:
            try:
                out = apply_chat_template(tokenizer, messages, **{**extra, **kwargs})
                break
            except TypeError:
                continue
        else:
            out = apply_chat_template(tokenizer, messages, **kwargs)
    else:
        out = apply_chat_template(tokenizer, messages, **kwargs)
    return normalize_token_ids(out)


def _tokenize_chat(messages: list, tokenizer, generation_role: str = "assistant", config=None, start_from=0) -> dict:
    _, generation_prompt = extract_system_prompt_and_generation(tokenizer)
    generation_prompt_len = len(generation_prompt)

    full_ids = []
    loss_mask = []
    prompt_length = None

    prev_tokens: list[int] = []
    for i, msg in enumerate(messages):
        tokens = _apply_chat_template(tokenizer, messages[: i + 1])
        turn_tokens = tokens[len(prev_tokens) :]
        prev_tokens = tokens

        full_ids.extend(turn_tokens)

        if msg["role"] == generation_role and i >= start_from:
            n = len(turn_tokens)
            header_len = min(generation_prompt_len, n)
            loss_mask.extend([0] * header_len + [1] * (n - header_len))
        else:
            loss_mask.extend([0] * len(turn_tokens))

        if i == 1:
            prompt_length = len(full_ids)

    if prompt_length is None:
        prompt_length = len(full_ids)

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    loss_mask = torch.tensor(loss_mask, dtype=torch.float32)

    full_tokens = _apply_chat_template(tokenizer, messages)
    if input_ids.tolist() != full_tokens:
        if _has_consecutive_tool_turns(messages):
            # Benign: VL-Instruct merges consecutive tool messages; per-turn closes/reopens.
            pass
        else:
            raise AssertionError(
                "Per-turn token concat does not match apply_chat_template(all_messages). "
                "The chat template may have context-dependent behaviour. "
                f"Roles: {[m.get('role') for m in messages]}"
            )

    nonzero = loss_mask.nonzero(as_tuple=True)[0]
    first_loss_pos = int(nonzero[0]) if len(nonzero) > 0 else len(full_ids)
    prompt_length = min(prompt_length, first_loss_pos)

    data_cfg = getattr(config, "data", config) if config is not None else {}
    max_prompt_length = data_cfg.get("max_prompt_length") if data_cfg else None
    max_response_length = data_cfg.get("max_response_length") if data_cfg else None

    prompt_ids = input_ids[:prompt_length]
    responses = input_ids[prompt_length:]
    response_mask = loss_mask[prompt_length:]

    if max_prompt_length is not None:
        prompt_ids = prompt_ids[-int(max_prompt_length) :]
    if max_response_length is not None:
        responses = responses[: int(max_response_length)]
        response_mask = response_mask[: int(max_response_length)]

    input_ids = torch.cat([prompt_ids, responses], dim=0)

    return {
        "input_ids": input_ids,
        "responses": responses,
        "response_mask": response_mask,
    }


# ---------------------------------------------------------------------------
# Per-task processors
# ---------------------------------------------------------------------------


def process_sotopia(row, tokenizer, config=None):
    messages = _get_messages(row)
    return _tokenize_chat(messages, tokenizer, generation_role="assistant", config=config)


def process_chat(row, tokenizer, config=None):
    messages = _get_messages(row)
    return _tokenize_chat(messages, tokenizer, generation_role="assistant", config=config)


def process_coser(row, tokenizer, config=None):
    messages = _get_messages(row, config=config)
    corrected_messages = []
    for msg in messages:
        msg = dict(msg)
        if msg["role"] == "assistant":
            msg["role"] = "user"
            corrected_messages.append(msg)
        elif msg["role"] == "user":
            msg["role"] = "assistant"
            corrected_messages.append(msg)
        else:
            corrected_messages.append(msg)
    return _tokenize_chat(corrected_messages, tokenizer, generation_role="assistant", config=config)


def process_swap(row, tokenizer, config=None):
    """Chat-style datasets. Keep system prompt, swap user↔assistant. start_from=0 so
    cold-open user turns (what user-simulator inference needs) contribute to loss."""
    messages = _get_messages(row, config=config)
    corrected_messages = []
    for msg in messages:
        msg = dict(msg)
        if msg["role"] == "system":
            corrected_messages.append(msg)
        elif msg["role"] == "assistant":
            msg["role"] = "user"
            corrected_messages.append(msg)
        elif msg["role"] == "user":
            msg["role"] = "assistant"
            corrected_messages.append(msg)
        else:
            corrected_messages.append(msg)
    return _tokenize_chat(corrected_messages, tokenizer, generation_role="assistant", config=config, start_from=0)


def process_dialogue(row, tokenizer, config=None):
    messages = _get_messages(row, config=config)
    corrected_messages = []
    for msg in messages:
        msg = dict(msg)
        if msg["role"] == "system":
            corrected_messages.append(msg)
        elif msg["role"] == "assistant":
            msg["role"] = "user"
            corrected_messages.append(msg)
        elif msg["role"] == "user":
            msg["role"] = "assistant"
            corrected_messages.append(msg)
        else:
            corrected_messages.append(msg)
    return _tokenize_chat(
        corrected_messages,
        tokenizer,
        generation_role="assistant",
        config=config,
        start_from=0,
    )


def process_humanual(row, tokenizer, config=None):
    messages = _get_messages(row, config=config)
    corrected_messages = []
    for msg in messages:
        msg = dict(msg)
        if msg["role"] == "system":
            corrected_messages.append(msg)
        elif msg["role"] == "assistant":
            msg["role"] = "user"
            corrected_messages.append(msg)
        elif msg["role"] == "user":
            msg["role"] = "assistant"
            corrected_messages.append(msg)
        else:
            corrected_messages.append(msg)
    return _tokenize_chat(
        corrected_messages,
        tokenizer,
        generation_role="assistant",
        config=config,
        start_from=0,
    )


def process_experiment(row, tokenizer, config=None):
    messages = _get_messages(row, config=config)
    corrected_messages = []
    for msg in messages:
        msg = dict(msg)
        if msg["role"] == "system":
            corrected_messages.append(msg)
        elif msg["role"] == "assistant":
            msg["role"] = "user"
            corrected_messages.append(msg)
        elif msg["role"] == "user":
            msg["role"] = "assistant"
            corrected_messages.append(msg)
        else:
            corrected_messages.append(msg)
    return _tokenize_chat(
        corrected_messages,
        tokenizer,
        generation_role="assistant",
        config=config,
        start_from=0,
    )


def process_psych101(row, tokenizer, config=None):
    """psych101 rows can have thousands of turns; fold all but the last 20 into
    a context string prepended to the first kept assistant msg."""
    folded = _fold_head(_get_messages(row, config=config), keep_last=20)
    return process_experiment({**row, "messages": folded}, tokenizer, config=config)


def process_no_swap(row, tokenizer, config=None):
    """Generic processor: `assistant` messages are loss targets as-is (no swap).
    Prepends an empty system message if none present."""
    messages = _get_messages(row, config=config)
    corrected = []
    if len(messages) == 0 or dict(messages[0]).get("role") != "system":
        corrected.append({"role": "system", "content": ""})
    corrected.extend(dict(m) for m in messages)
    return _tokenize_chat(
        corrected,
        tokenizer,
        generation_role="assistant",
        config=config,
        start_from=0,
    )


# Backward-compat aliases.
process_human_llm = process_no_swap
process_wildchat = process_swap
