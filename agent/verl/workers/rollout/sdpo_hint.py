"""Hindsight hints and teacher-sequence construction for SDPO.

`L = L_RLVR + alpha * L_SDPO`, where L_SDPO is the per-turn reverse KL between
the base policy pi(.|H_t) and the hindsight policy pi(.|H_t + hint_t), evaluated
on the agent's OWN response y_t. Nothing is generated here: y_t already exists in
the trajectory and is merely re-scored under a second context.

Two pieces live here because both are pure functions of an already-finished
trajectory, and neither needs the rollout engine:

  build_hint()             one hint-writer call turning two privileged channels
                           into a hint the agent could plausibly have acted on
  build_teacher_rows()     one hindsight sequence per hinted turn, plus an index
                           map back to the student's response positions

Kept out of sglang_rollout_customized.py so the splice can be unit-tested against
synthetic token sequences, which is the only practical way to check the index map
-- an off-by-one there produces a plausible but meaningless KL with no external
symptom.
"""
import os
import re
import random

import httpx

# The paper's Table 1 wording, which the reference implementation also uses
# verbatim (online_sdpo_updater_config.py:52-54). Kept identical so a result here
# is comparable to theirs rather than confounded by prompt drift.
#
# Injected as its own user turn immediately before y_t rather than appended to
# the preceding message's content. Appending inside would mean locating that
# message's <|im_end|> by arithmetic on token offsets; inserting a whole turn
# needs only turn_boundaries[t] and generation_prompt_ids, both of which the
# rollout already tracks exactly.
#
# INVARIANT: the hint is a MESSAGE BODY, never hand-written markup --
# apply_chat_template owns both ChatML boundaries around it.
#
# Failure mode this removes: hand-writing the markup means reproducing, byte for
# byte, what the template would have emitted on BOTH sides of the inserted turn,
# and getting either end wrong is silent. A leading '\n' merges with the
# previous turn's into the single token 'ĊĊ', so the boundary token itself
# changes rather than just the whitespace; a missing trailing '\n' leaves
# '<|im_end|><|im_start|>assistant', a seam the model never saw in training, at
# exactly the token where y_t begins. Either error makes the teacher score y_t
# in an out-of-distribution context, and SDPO then pulls the student toward a
# corrupted distribution -- worse the larger alpha, which is the monotone harm
# measured before the message-level rewrite (0.01->0.151, 0.05->0.127,
# 0.15->0.092 against controls at 0.162/0.219).
#
# Wording is the reference's (online_sdpo_updater_config.py:52-54) except that
# ours announces a coaching note rather than "a future user message", because
# ours IS a note. Three designs were measured head to head, scored on
# informative signal = real absmean MINUS placebo absmean:
#
#   design                       actionable  real   placebo  informative
#   coach note (this one)           57%      0.329   0.187     0.142
#   redact the user's real message  67%      0.240   0.188     0.052
#   compose a user message          10%      0.119   0.200    -0.081
#
# The reference's own register is the third and it loses here: their user has a
# persistent STYLE PROFILE ("too long", "no emojis") that always supplies
# something specific and non-leaking, while ours has a hidden TASK GOAL, so
# composing a legitimate utterance is usually impossible and the hint writer
# abstains.
#
# PLACEBO IS ~0.187 IN ALL THREE -- it does not move with wording, because it is
# not caused by wording: inserting ANY extra user turn shifts the conditional by
# about that much. Only a control variate removes it.
HINT_BODY = (
    "=== HINDSIGHT CONTEXT ===\n"
    "[The following is guidance about the user's needs. "
    "Use this to guide your answer to the user prompt.]\n"
    "{hint}"
)

# Legacy ChatML-wrapped form of HINT_BODY, kept for reference and derived from
# it so the two cannot drift. Not used on the live path: the teacher's context
# is rendered by apply_chat_template (see build_teacher_rows), which is what
# makes the boundary bug described above unrepresentable.
HINT_BLOCK = "<|im_start|>user\n" + HINT_BODY + "<|im_end|>\n"

# The filter's job is ACHIEVABILITY, not relevance. Irrelevant content barely
# moves the hindsight conditional and costs nothing; the harmful content is
# relevant, useful, and unavailable -- an undisclosed budget, an unnamed
# destination. A relevance filter keeps exactly that and drops the harmless
# remainder, which is backwards. If a hint asserts a fact the agent could not
# re-derive from H_t, the student is trained to produce that fact from a context
# that does not contain it: a hallucination, rewarded at train time and wrong at
# test time.
# ---------------------------------------------------------------------------
# PROSPECTIVE hint (SDPO_HINT_STYLE=prospective). The module-level default
# below is "corrective"; the shipped training pipeline selects prospective
# (scripts/_train_common.sh exports SDPO_HINT_STYLE=prospective unless it is
# already set).
#
# The corrective prompt below was conceptually wrong for what the teacher does.
# The teacher scores y_t as if generating it FRESH from H_t: in the teacher's
# context the assistant has not spoken yet. Telling it "you should have asked
# about budget" references a reply that does not exist in that conditioning,
# and marks the upcoming generation as wrong before it happens. Measured
# consequence: Delta = log pi_teacher - log pi_student ran -0.21..-0.35 on
# every arm, with only 14-18% of tokens endorsed -- the teacher was WORSE than
# the student, so the SDAR gate had nothing to strengthen.
#
# Prospective guidance is the coherent form. If y_t already does what the
# guidance asks, Delta goes positive; if it does not, negative. That sign
# variance IS the signal.
#
# The assistant's reply is deliberately NOT shown. If the hint writer cannot see
# y_t it cannot write a correction, and the note cannot be covertly tailored to
# one rollout's specific mistakes -- which would smuggle information about y_t
# into a context that must not contain it.
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


# ---------------------------------------------------------------------------
# LOCAL hint writer (SDPO_HINT_STYLE=corrective_local).
#
# The templates above were written for a frontier hint writer and are UNUSABLE
# with a local agent-SFT model: replayed over 100 real turns taken from a
# training run, the SFT'd Qwen3-8B scored 3/100 -- it emits
# `<tool_call> interact_with_env(...)` or echoes the `##` headings back, because
# any input shaped like a UserRL episode puts it back in agent mode. It is the
# policy, not an instruction-follower.
#
# Three changes fix it, measured cumulatively on the same 100 cases:
#     production templates                        3/100
#     + third-person framing, sanitised history  33/100
#     + two-shot                                 53/100
#     + explicit leak rule, three-shot           63/100
#     + "name what they said", 40-word cap       74/100
#     (frontier writer on the same grader:       79/100)
# A fourth example and temperatures 0.3/0.5 were tried and were worse
# (68-73), so this is the operating point.
#
# _sanitize_episode is load-bearing, not cosmetic: stripping <tool_call>,
# <tool_response>, the gym system prompt and the tool name is what stops the
# model continuing the episode instead of analysing it.
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

# Stop sequences catch the residual relapses; <think> is also stripped after the
# fact, because a stop sequence can only fire once generation reaches it and the
# tag may already be in the text by then.
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

# Which model writes the hints. scripts/_train_common.sh always exports this
# (defaulting it to the user simulator), so the fallback below is only reached
# if sdpo_hint is driven directly.
HINT_MODEL = os.environ.get("SDPO_HINT_MODEL", "default")
HINT_MAX_CHARS = int(os.environ.get("SDPO_HINT_MAX_CHARS", "600"))
HINT_TIMEOUT = float(os.environ.get("SDPO_HINT_TIMEOUT", "120"))
# Default to every turn. max_turns is 16, and the packer clamps
# sdpo_k = min(this, max_turns), so 16 means "no cap" without over-allocating
# teacher rows. Setting K above max_turns reserves rows no turn can ever fill
# -- K=99 reserves 198 rows for ~9 real ones -- and every reserved row is
# forwarded regardless, which is enough wasted collective work to trip the
# NCCL watchdog.
#
# The cap was never principled: the reference hints EVERY interaction with no
# selection (eval_online_sdpo.py:441-473). It existed to bound hint-writer calls
# and teacher rows, and both are now handled -- rows are chunked
# rank-invariantly, and the row budget is bounded by max_turns.
#
# Cost is real though: turn count is skewed, so raising 8 -> 16 lifts selected
# from ~5.6 to ~8 per trajectory (+43% hint calls), and hint generation is
# already the rollout bottleneck.
TURNS_PER_EPISODE = int(os.environ.get("SDPO_TURNS_PER_EPISODE", "16"))
ABLATE_PRIVILEGED = os.environ.get("SDPO_ABLATE_PRIVILEGED") == "1"
NULL_PROBE = os.environ.get("SDPO_NULL_PROBE") == "1"
DUMP_ROW = os.environ.get("SDPO_DUMP_ROW") == "1"
# Paper Table 1 form: hint INSIDE the last context message, no extra turn.
HINT_INLINE = os.environ.get("SDPO_HINT_INLINE", "0") == "1"
# Old behaviour (the 0.312 at step 25 configuration): only turns with a
# recorded simulator reaction are hintable. Costs the last turn of each
# trajectory.
REQUIRE_REPLY = os.environ.get("SDPO_REQUIRE_REPLY", "0") == "1"
_DUMPED = False

STATS = {"asked": 0, "ok": 0, "none": 0, "failed": 0, "no_channel": 0}

# Ring buffer of recent hints from OTHER trajectories, for the placebo control
# variate. Inserting any extra user turn moves the teacher's conditional by
# ~0.187 nats/token regardless of what the turn says -- measured identical
# across three completely different hint designs. That component is pure noise
# in the loss and prompt wording cannot remove it.
#
# So we score y_t under a SECOND, unrelated hint and subtract:
#     delta_debiased = (teacher_real - student) - (teacher_placebo - student)
#                    =  teacher_real - teacher_placebo
# The student's log-prob cancels, so this costs one extra no-grad teacher
# forward and no extra bookkeeping on the student side. It is the same
# variance-reduction logic as a baseline in policy gradient: subtract an
# expectation that carries no signal.
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
    """An unrelated hint per hinted turn, drawn from other trajectories.

    Returns {} until the pool has enough distinct material, so early steps
    simply run undebiased rather than pairing a hint against itself -- which
    would subtract the real signal instead of the noise.
    """
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

# Hints are spliced into token ids and otherwise vanish. Without a record, a
# smoke run reporting sdpo_delta ~= 0 cannot distinguish "the hint writer
# returned NONE for most turns" from "hints were generic" from "the thought
# channel was empty because the simulator does not think" -- four different
# problems with one loss curve. Log the inputs and the output together so the
# diagnosis is a grep.
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
            # Stamp the arm into every record, so which variant produced a
            # given log is recoverable from the log alone.
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
    """Endpoint for the HINT WRITER, which is not the gyms' reward judge.

    Those were one variable until now, and conflating them silently replaced the
    reward function: the working baseline grades with the simulator itself, while
    pinning the hint writer as JUDGE_MODEL_NAME made every gym score 0.000 on
    every step. SDPO needs a strong model to write feedback; it has no business
    changing how trajectories are scored.
    """
    url = os.environ.get("SDPO_HINT_BASE_URL") or os.environ.get("JUDGE_BASE_URL", "")
    if not url:
        return None
    return (url.rstrip("/") if url.endswith("/v1") else url.rstrip("/") + "/v1") + "/chat/completions"


def select_turns(privileged, n_turns, k=None, rng=None):
    """Which turns to hint.

    Uniform sampling, deliberately. The obvious gate -- skip turns that already
    scored full credit -- does not work here: the per-turn reward is a
    floor-clamped progress delta (TurtleGym story_env.py:182-201 is
    max(0, score - best_score) - step_penalty clamped to [0,1]), so it reads 0.0
    on most turns including good ones, and a reward threshold would admit
    essentially everything while looking selective.

    EVERY turn is eligible -- there is no reply gate. Turns whose predecessor
    carried a thought are preferred:
    turn 0 never has one (no preceding tool call produced the message it replied
    to), and with short trajectories uniform sampling picks turn 0 constantly --
    an early run got a thought on only 89/379 hints that way, starving half the
    method. Sample from thought-bearing turns first, then top up.
    """
    k = TURNS_PER_EPISODE if k is None else k
    # EVERY turn is eligible. The old gate required privileged[t]["reply"],
    # which cost exactly one turn per trajectory -- measured, the census showed
    # n_turns - priv_records = 1.0 in every arm. That turn is the last one:
    # nothing followed it, so no tool call recorded a reaction.
    #
    # A missing reaction does not make a turn unhintable. The hint writer still
    # has the full history and the assistant's own reply, which is enough to say
    # what should have been done -- from foresight rather than hindsight for
    # that turn, but a hint either way.
    #
    # The old gate is kept behind SDPO_REQUIRE_REPLY rather than deleted, and
    # defaults OFF, so every turn is hintable. It is retained because the
    # every-turn variant shipped alongside inline placement and k=16, and the
    # three together regressed ref/0.01 from 0.312 to 0.115 at step 25; turning
    # the gate back on isolates its contribution from the other two.
    #
    # Bounds-safe: `privileged` has n_turns-1 entries, not n_turns. The last
    # turn has no recorded reaction (nothing followed it), so a bare
    # privileged[t] over range(n_turns) throws on every trajectory -- which it
    # did, ~900 times per arm, silently zeroing sdpo_loss in an earlier set of
    # runs.
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
    """One hint-writer call. Returns None when there is nothing actionable.

    Input is deliberately narrow: H_t, y_t, and the two privileged channels for
    THIS turn -- never later turns. Achievability at turn t is a question about
    the prefix H_t, so a whole-trajectory call would see u_{t+1..T} and widen the
    privileged channel past the two signals the method specifies.
    """
    global _client
    # Ablation arm: blank BOTH privileged channels, so the hint writer writes
    # its note from H_t and y_t alone. If these hints match the full method's,
    # the privileged channels contribute nothing and the whole thing is frontier
    # distillation with a hindsight story attached -- a possibility raised by an
    # earlier run in which 54 of the hints were pure format corrections.
    if ABLATE_PRIVILEGED:
        think, next_reply = "", "(withheld: ablation)"
    url = _judge_url()
    # Only a missing endpoint stops the call. A turn with neither privileged
    # channel is still hintable: the hint writer has the history and the
    # assistant's own reply, which is enough to say what should have been done.
    # For such turns the note is coaching rather than hindsight -- weaker, but
    # it is the only way to reach the last turn of every trajectory (measured:
    # exactly one per trajectory has no follow-up).
    if url is None:
        STATS["no_channel"] += 1
        return None
    # Tell the hint writer a channel is absent rather than handing it a blank
    # heading, which reads as "the user said nothing" and reliably draws a NONE.
    if not think:
        think = "(not available for this turn)"
    if not next_reply:
        next_reply = "(not available -- the user has not responded yet)"
    STATS["asked"] += 1
    _local = HINT_STYLE == "corrective_local"
    if _local:
        # Must happen before formatting: the raw episode markup is exactly what
        # flips a local agent-SFT writer back into emitting tool calls.
        history = _sanitize_episode(history)
        reply = _sanitize_episode(reply)
    body = {
        "model": HINT_MODEL,
        "messages": [
            {"role": "system", "content": (
                HINT_SYSTEM_LOCAL if _local else
                HINT_SYSTEM_PROSPECTIVE if HINT_STYLE == "prospective" else HINT_SYSTEM)},
            # `reply` is intentionally absent from the prospective template: the
            # teacher's context has no y_t, so the briefing must not be written
            # against one.
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
            # A think block that PREFIXES real content survives the strip shim.
            text = re.sub(r"(?s)<think>.*?</think>", "", text).strip().strip('"')
    except Exception as e:                                   # noqa: BLE001
        # Degrades to no hindsight block for this turn, so its KL is identically
        # zero -- invisible in the loss. Counted so a dead judge endpoint shows
        # up as a number rather than as a method that mysteriously does nothing.
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
    """(start, end) of each assistant turn's content in the full token sequence.

    Derived from loss_mask, which is 1 only on assistant-generated tokens
    (schemas.py:184 vs :192), and cross-checked against turn_boundaries, which is
    recorded independently at the top of each RUNNING state. Two derivations of
    the same spans; disagreement means the geometry assumption below is wrong,
    and it is far better to hear that here than to silently train on a shifted
    index map.
    """
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
    """One teacher sequence per hinted turn, built the way the reference does.

        turn t:  apply_chat_template(messages[:t] + [hint_msg], add_gen=True)
                 ++ y_t's EXACT rollout token ids

    Message level, not token level: the hint is handed to the template as a
    message body and never written out as markup, so the ChatML boundaries
    around it cannot be got wrong (see HINT_BODY for the failure mode this
    removes). The reference builds its hindsight context the same way
    (online_sdpo_updater.py:392-407).

    y_t lands at the TAIL of the row, so its index map is arithmetic
    (len(ctx) + offset) rather than a shift that has to track the hint's token
    length. That removes the "+ len(block)" term whose correctness the null
    probe structurally could not check (it runs with an empty block).

    The context is re-rendered rather than reused. Safe here: verl builds
    input_ids incrementally (schemas.py:186-210) and one-shot
    apply_chat_template over the same messages was verified to produce
    byte-identical tokens. y_t itself is never re-tokenised -- its ids are
    copied straight from the rollout, as the reference insists
    (sdpo_signal_analysis.py:526, "Keep exact generated ids").

    Returns (rows, index_map, row_idx, sdpo_mask, skipped), unchanged, so the
    packer and dp_actor need no edits.
    """
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
                # PAPER FORM (Table 1): the hindsight context lives INSIDE the
                # last message of the context, not in a turn of its own --
                #   User: <history> <hindsight context> o
                #   Assistant: y
                # The reference does the same (_build_hindsight_messages:
                # conditional[i]["content"] += block).
                #
                # Why it matters: a separate turn makes the teacher see N+1
                # turns where the student saw N. That structural change is
                # IDENTICAL for every possible hint, so it is exactly the kind
                # of content-free difference that shows up as placebo -- an
                # unrelated hint moved our teacher 0.187 nats/token against the
                # real hint's 0.329 (ratio 1.76x, where the authors' diagnostic
                # wants ~0). Appending inline leaves the hint TEXT as the only
                # difference between x and (x,o).
                #
                # The paper walks back to the last USER message; that works
                # because their setting is single-turn. Here the message before
                # y_t is a tool response, so appending to the last user message
                # would strand the hint many turns from where it is needed.
                # Appending to the last message of ANY role preserves both the
                # turn count and the adjacency.
                ctx_msgs[-1] = dict(ctx_msgs[-1])
                ctx_msgs[-1]["content"] = (ctx_msgs[-1]["content"] or "") + "\n\n" + blk
            else:
                ctx_msgs.append({"role": "user", "content": blk})
        ctx_text = tokenizer.apply_chat_template(
            ctx_msgs, tools=tools, add_generation_prompt=True, tokenize=False)
        ctx_ids = tokenizer.encode(ctx_text, add_special_tokens=False)

        # With no hint the rendered context must reproduce the student's own
        # prefix exactly. That is the null probe, now enforced at construction
        # instead of needing a separate training run to observe.
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
