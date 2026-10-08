"""Hindsight hints and teacher-sequence construction for SDPO."""
import os
import re
import random

import httpx

HINT_BODY = (
    "=== HINDSIGHT CONTEXT ===\n"
    "[The following is guidance about the user's needs. "
    "Use this to guide your answer to the user prompt.]\n"
    "{hint}"
)

HINT_BLOCK = "<|im_start|>user\n" + HINT_BODY + "<|im_end|>\n"

HINT_SYSTEM_PROSPECTIVE = """You are writing a short briefing that will be shown to an AI assistant as a user turn, immediately before it writes its next message. It is used only during training.

You see the conversation so far plus PRIVILEGED material about the user -- their private reasoning and what they say next. The assistant has NOT replied yet and you will not be shown any reply.

Write at most 3 sentences stating, concretely, what this user needs from the next message.

THE TEST YOUR NOTE MUST PASS. Your note is used to score a reply that already exists but that you cannot see. A good reply should look MORE likely after reading your note; a poor one should look LESS likely. So be specific enough to separate them. "Be helpful and ask clarifying questions" fails this test -- it fits every reply. "The user has twice steered away from abstract motives toward the physical scene; the next question should name a concrete object" passes it.

SCOPE -- write about the USER: what they are after, what they are stuck on, what they need asked or pursued now. Speak in the present, about the message about to be written.

FORWARD-LOOKING ONLY. There is no previous reply in view, so never evaluate or correct one. Write "The user needs X" or "Ask about X", never "You should have" or "instead of".

DO NOT write about tool syntax, output format, or protocol. Those are visible without privileged information and the reward already penalises them. If the privileged material says nothing about the user beyond format, reply with exactly: NONE

HARD CONSTRAINT -- every claim must be derivable from the conversation so far. Never state or hint at a fact that appears only in the privileged material: no dates, destinations, budgets, names, numbers, or preferences the user has not already said out loud. Say what to ASK or CLARIFY, never the answer to guess.

Good:  "The user has given four separate needs and wants the cheapest option for each; the next message should handle one of them concretely rather than restating the list."
Good:  "The user keeps rejecting broad category questions as too vague; ask a yes/no question about one specific attribute."
Bad:   "The user is travelling in June, so suggest summer options."   (leaks a private fact)
Bad:   "You should have asked about timing first."                    (refers to a reply not in view)
Bad:   "Be specific and address the user's concerns."                 (fits any reply -- fails the test)

If the privileged material reveals nothing actionable about the user, reply with exactly: NONE"""

HINT_USER_PROSPECTIVE = """## Conversation so far
{history}

## PRIVILEGED -- the user's private reasoning at this point
{think}

## PRIVILEGED -- what the user says next
{next_reply}

Write the briefing for the assistant's next message, or NONE."""


HINT_SYSTEM_LOCAL = """You are a transcript analyst. You never act, never call tools, never role-play, and never continue the conversation. You write exactly one sentence of analysis and nothing else.

Below is a finished transcript between a PERSON and an AGENT, plus private notes the agent could not see. Write ONE sentence saying what the AGENT should have asked or pursued at its last turn.

THE ONE RULE THAT MATTERS: your sentence may only use facts the PERSON said out loud in the TRANSCRIPT. The private notes tell you WHAT WENT WRONG, but never repeat a name, place, date, month, or number that appears only there. Refer to it indirectly -- "the dates they gave", "the city they named" -- never the value.

BE CONCRETE: your sentence must name something the person actually said -- the need, object, or constraint they raised, in their own words. A sentence that would fit any transcript is useless.

LENGTH: at most 40 words. One sentence. No preamble, no heading, no quotes, no reasoning.

If the private notes reveal nothing useful about the person, output exactly: NONE"""

HINT_USER_LOCAL = """Example 1
TRANSCRIPT
PERSON: I need help planning a trip.
AGENT: Here is a 5-day itinerary starting Monday.
PRIVATE - reasoning: I have not said when I am travelling.
PRIVATE - said next: You are assuming dates I never gave.
One-sentence analysis: The agent proposed a dated itinerary before establishing when the person is travelling; it should have asked about timing first.

Example 2
TRANSCRIPT
PERSON: Book me a restaurant near my hotel.
AGENT: I searched for restaurants.
PRIVATE - reasoning: I want it for November 16, which I already told them.
PRIVATE - said next: You ignored the date I gave you.
One-sentence analysis: The person had already supplied the dining date, so the agent should have carried that date into the restaurant search instead of running an undated query.
(Note: says "the dining date", NOT "November 16" -- never repeat a value that appears only in the private notes.)

Example 3
TRANSCRIPT
PERSON: Is it something you'd find in a kitchen?
AGENT: Is it related to the concept of nourishment?
PRIVATE - reasoning: They keep going abstract when I want a physical object.
PRIVATE - said next: That's too vague again.
One-sentence analysis: The person's answers pushed toward a physical object while the agent kept probing abstract concepts; it should have named a concrete item instead.

Now do the same for this one.

TRANSCRIPT
{history}

AGENT'S LAST TURN
{reply}

PRIVATE - the person's reasoning
{think}

PRIVATE - what the person said next
{next_reply}

One-sentence analysis:"""

HINT_STOP_LOCAL = ["<tool_call>", "[tool_call]", "\nTRANSCRIPT", "\nPRIVATE",
                   "\nExample", "\nAGENT'S", "<think>", "</think>", "\n\n"]

_EP_MARKUP = re.compile(r"</?tool_call>|</?tool_response>|</?think>")
_EP_SYSTEM = re.compile(r"\[system\].*?(?=\[user\]|\[assistant\]|\[tool\]|$)", re.S)

def _sanitize_episode(t):
    if not t:
        return t
    t = _EP_MARKUP.sub("", t)
    t = _EP_SYSTEM.sub("", t)
    t = (t.replace("[assistant]", "AGENT:").replace("[user]", "PERSON:")
          .replace("[tool]", "RESULT:").replace("interact_with_env", "act"))
    return re.sub(r"\n{3,}", "\n\n", t).strip()


HINT_STYLE = os.environ.get("SDPO_HINT_STYLE", "corrective")

HINT_SYSTEM = """You are preparing a coaching note for an AI assistant, to be used only during training.

You will see: the conversation so far, the assistant's reply, the user's private reasoning, and the user's next message. The last two are PRIVILEGED -- the assistant did not have them when it replied.

Write a short note (at most 3 sentences) telling the assistant what it should have done differently at this turn.

SCOPE -- the note must be about the USER: what they were actually after, what they were confused by, what the assistant should have asked or pursued instead. It exists to convey what the privileged material reveals about the user's state.

DO NOT write about tool syntax, output format, or protocol. If the assistant failed to call a tool, emitted malformed arguments, or only narrated an intended action, that is a format error -- it is visible without any privileged information, the training reward already penalises it, and a note about it teaches nothing this method is for. When the only thing wrong with the turn is format, reply with exactly: NONE

DO NOT write feedback that would fit any conversation. Ground every sentence in something specific to THIS exchange -- a claim that was made, an option that was skipped, a direction the user kept steering toward. Adapted from the reference's rule "Do NOT give general feedback that does not relate to the USER PROFILE" (auxiliary/user_simulator.py:165).

HARD CONSTRAINT -- every claim in your note must be derivable from the conversation so far. Never state or hint at a specific fact that appears only in the privileged material: no dates, destinations, budgets, names, numbers, or preferences the user has not already said out loud. Describe what the assistant should ASK or CLARIFY, never the answer it should have guessed.

Good:  "You proposed an itinerary before establishing the traveller's dates; ask about timing first."
Good:  "Your last three guesses all probed abstract concepts while the user kept steering toward something physical; narrow to concrete objects."
Bad:   "The user is travelling in June, so suggest summer options."   (leaks a private fact)
Bad:   "You did not call interact_with_env with an answer argument."  (format, not user state)
Bad:   "Be more specific and address the user's concerns."            (generic; fits any conversation)

If the privileged material reveals nothing about the user that the assistant could have acted on, reply with exactly: NONE"""

HINT_USER = """## Conversation so far
{history}

## The assistant's reply at this turn
{reply}

## PRIVILEGED -- the user's private reasoning before that reply
{think}

## PRIVILEGED -- the user's next message, reacting to that reply
{next_reply}

Write the coaching note, or NONE."""

HINT_MODEL = os.environ.get("SDPO_HINT_MODEL", "default")
HINT_MAX_CHARS = int(os.environ.get("SDPO_HINT_MAX_CHARS", "600"))
HINT_TIMEOUT = float(os.environ.get("SDPO_HINT_TIMEOUT", "120"))
TURNS_PER_EPISODE = int(os.environ.get("SDPO_TURNS_PER_EPISODE", "16"))
ABLATE_PRIVILEGED = os.environ.get("SDPO_ABLATE_PRIVILEGED") == "1"
NULL_PROBE = os.environ.get("SDPO_NULL_PROBE") == "1"
DUMP_ROW = os.environ.get("SDPO_DUMP_ROW") == "1"
HINT_INLINE = os.environ.get("SDPO_HINT_INLINE", "0") == "1"
REQUIRE_REPLY = os.environ.get("SDPO_REQUIRE_REPLY", "0") == "1"
_DUMPED = False

STATS = {"asked": 0, "ok": 0, "none": 0, "failed": 0, "no_channel": 0}

_PLACEBO_POOL: "list[str]" = []
PLACEBO_POOL_MAX = int(os.environ.get("SDPO_PLACEBO_POOL", "256"))
DEBIAS = os.environ.get("SDPO_DEBIAS", "1") == "1"


def _remember_for_placebo(hint):
    if not hint:
        return
    _PLACEBO_POOL.append(hint)
    if len(_PLACEBO_POOL) > PLACEBO_POOL_MAX:
        del _PLACEBO_POOL[0]


def placebo_for(hints, rng=None):
    """An unrelated hint per hinted turn, drawn from other trajectories."""
    if not DEBIAS or len(_PLACEBO_POOL) < 8:
        return {}
    rng = rng or random
    out = {}
    for t, h in hints.items():
        pool = [p for p in _PLACEBO_POOL if p != h]
        if pool:
            out[t] = rng.choice(pool)
    return out

_client: "httpx.AsyncClient | None" = None

HINT_LOG = os.environ.get("SDPO_HINT_LOG", "")
_log_fh = None
_log_failed = False


def _log_hint(tag, history, reply, think, next_reply, hint):
    """Append one hint and everything it was built from. Never raises."""
    global _log_fh, _log_failed
    if not HINT_LOG or _log_failed:
        return
    try:
        import json
        if _log_fh is None:
            os.makedirs(os.path.dirname(HINT_LOG) or ".", exist_ok=True)
            _log_fh = open(HINT_LOG, "a")
        _log_fh.write(json.dumps({
            "ablate": ABLATE_PRIVILEGED,
            "k": TURNS_PER_EPISODE,
            "hint_model": HINT_MODEL,
            "tag": tag,
            "hint": hint,                       # None when the model said NONE
            "think": (think or "")[:2000],
            "next_reply": (next_reply or "")[:2000],
            "reply": (reply or "")[:2000],
            "history_tail": (history or "")[-2000:],
        }, ensure_ascii=False) + "\n")
        _log_fh.flush()
    except Exception as e:                                   # noqa: BLE001
        _log_failed = True
        print(f"[SDPO] hint logging disabled: {type(e).__name__}: {e}")


def _judge_url() -> "str | None":
    url = os.environ.get("SDPO_HINT_BASE_URL") or os.environ.get("JUDGE_BASE_URL", "")
    if not url:
        return None
    return (url.rstrip("/") if url.endswith("/v1") else url.rstrip("/") + "/v1") + "/chat/completions"


def select_turns(privileged, n_turns, k=None, rng=None):
    """Which turns to hint."""
    k = TURNS_PER_EPISODE if k is None else k
    def _priv(t):
        return (privileged[t] if 0 <= t < len(privileged) else None) or {}

    if REQUIRE_REPLY:
        eligible = [t for t in range(n_turns) if _priv(t).get("reply")]
    else:
        eligible = list(range(n_turns))
    if not eligible:
        return []
    rng = rng or random
    rich = [t for t in eligible if t >= 1 and _priv(t - 1).get("think")]
    poor = [t for t in eligible if t not in set(rich)]
    picked = rng.sample(rich, min(k, len(rich)))
    if len(picked) < k and poor:
        picked += rng.sample(poor, min(k - len(picked), len(poor)))
    return sorted(picked)


async def build_hint(history: str, reply: str, think: str, next_reply: str,
                     tag: str = "") -> "str | None":
    """One hint-writer call. Returns None when there is nothing actionable."""
    global _client
    if ABLATE_PRIVILEGED:
        think, next_reply = "", "(withheld: ablation)"
    url = _judge_url()
    if url is None:
        STATS["no_channel"] += 1
        return None
    if not think:
        think = "(not available for this turn)"
    if not next_reply:
        next_reply = "(not available -- the user has not responded yet)"
    STATS["asked"] += 1
    _local = HINT_STYLE == "corrective_local"
    if _local:
        history = _sanitize_episode(history)
        reply = _sanitize_episode(reply)
    body = {
        "model": HINT_MODEL,
        "messages": [
            {"role": "system", "content": (
                HINT_SYSTEM_LOCAL if _local else
                HINT_SYSTEM_PROSPECTIVE if HINT_STYLE == "prospective" else HINT_SYSTEM)},
            {"role": "user", "content": (
                HINT_USER_LOCAL.format(
                    history=history, reply=reply,
                    think=think or "(the simulator emitted no reasoning)",
                    next_reply=next_reply or "(the conversation ended here)")
                if _local else
                HINT_USER_PROSPECTIVE.format(
                    history=history,
                    think=think or "(the simulator emitted no reasoning)",
                    next_reply=next_reply or "(the conversation ended here)")
                if HINT_STYLE == "prospective" else
                HINT_USER.format(
                    history=history, reply=reply,
                    think=think or "(the simulator emitted no reasoning)",
                    next_reply=next_reply or "(the conversation ended here)"))},
        ],
        "max_tokens": 110 if _local else 300,
    }
    if _local:
        body["stop"] = HINT_STOP_LOCAL
        body["temperature"] = float(os.environ.get("SDPO_HINT_TEMP", "0.7"))
    try:
        if _client is None:
            _client = httpx.AsyncClient(timeout=HINT_TIMEOUT)
        r = await _client.post(url, json=body)
        r.raise_for_status()
        text = (r.json()["choices"][0]["message"]["content"] or "").strip()
        if _local:
            text = re.sub(r"(?s)<think>.*?</think>", "", text).strip().strip('"')
    except Exception as e:                                   # noqa: BLE001
        STATS["failed"] += 1
        if STATS["failed"] % 50 == 1:
            print(f"[SDPO] hint call failed ({STATS['failed']}x): "
                  f"{type(e).__name__}: {str(e)[:160]}")
        return None
    if not text or text.strip().upper().startswith("NONE"):
        STATS["none"] += 1
        _log_hint(tag, history, reply, think, next_reply, None)
        return None
    STATS["ok"] += 1
    hint = text[:HINT_MAX_CHARS]
    _log_hint(tag, history, reply, think, next_reply, hint)
    _remember_for_placebo(hint)
    return hint


def turn_spans(loss_mask, turn_boundaries):
    """(start, end) of each assistant turn's content in the full token sequence."""
    spans = []
    for start in turn_boundaries:
        if start >= len(loss_mask) or not loss_mask[start]:
            spans.append(None)          # truncated mid-turn; not hintable
            continue
        end = start
        while end < len(loss_mask) and loss_mask[end]:
            end += 1
        spans.append((start, end))
    return spans


def build_teacher_rows(messages, input_ids, loss_mask, turn_boundaries,
                       prompt_len, hints, tokenizer, width, tool_schemas=None):
    """One teacher sequence per hinted turn, built the way the reference does."""
    spans = turn_spans(loss_mask, turn_boundaries)
    asst = [i for i, m in enumerate(messages) if getattr(m, "role", None) == "assistant"]
    tools = ([t.model_dump() for t in tool_schemas] if tool_schemas else None)

    n_resp = len(input_ids) - prompt_len
    index_map = [0] * n_resp
    row_idx = [0] * n_resp
    sdpo_mask = [0] * n_resp
    rows, skipped = [], 0

    for t, hint in sorted(hints.items()):
        if not hint or t >= len(spans) or spans[t] is None or t >= len(asst):
            continue
        start, end = spans[t]

        ctx_msgs = [{"role": m.role, "content": m.content or ""}
                    for m in messages[:asst[t]]]
        if not NULL_PROBE:
            blk = HINT_BODY.format(hint=hint)
            if HINT_INLINE and ctx_msgs:
                ctx_msgs[-1] = dict(ctx_msgs[-1])
                ctx_msgs[-1]["content"] = (ctx_msgs[-1]["content"] or "") + "\n\n" + blk
            else:
                ctx_msgs.append({"role": "user", "content": blk})
        ctx_text = tokenizer.apply_chat_template(
            ctx_msgs, tools=tools, add_generation_prompt=True, tokenize=False)
        ctx_ids = tokenizer.encode(ctx_text, add_special_tokens=False)

        if NULL_PROBE and ctx_ids != list(input_ids[:start]):
            raise ValueError(
                f"SDPO null probe: re-rendered context for turn {t} differs from "
                f"the rollout prefix ({len(ctx_ids)} vs {start} tokens)")

        y_ids = list(input_ids[start:end])
        row = ctx_ids + y_ids
        if len(row) > width:
            skipped += 1
            continue

        k = len(rows)
        base = len(ctx_ids)
        for j in range(max(start, prompt_len), end):
            i = j - prompt_len
            index_map[i] = base + (j - start)
            row_idx[i] = k
            sdpo_mask[i] = 1
        rows.append(row)

        if DUMP_ROW and not globals().get("_DUMPED"):
            globals()["_DUMPED"] = True
            d = tokenizer.decode
            print("[SDPO ROW DUMP] ----------------------------------------")
            print(f"  turn={t} ctx={len(ctx_ids)} y_t={len(y_ids)} row={len(row)}")
            print(f"  tail of context : {d(ctx_ids[-60:])!r}")
            print(f"  y_t in row      : {d(row[base:base+40])!r}")
            print(f"  y_t in student  : {d(input_ids[start:start+40])!r}")
            print(f"  MATCH: {row[base:base+40] == list(input_ids[start:start+40])}")
            print("[SDPO ROW DUMP] ----------------------------------------")

    return rows, index_map, row_idx, sdpo_mask, skipped
