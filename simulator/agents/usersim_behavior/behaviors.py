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
Behavior registry for the behavior-conditioned user simulator.

Mirrors ``agents/instruct/instructions_registry.py``: BEHAVIOR_DICT maps a behavior id to
a class that owns everything about that behavior, so the directive shown to the policy and
the check applied to its output are rendered from one source of truth -- the same property
that makes IFBench's ``build_description`` / ``check_following`` pairing safe
(``agents/instruct/instructions.py:2006``).

Each class supplies four things:

    applicable(flow, subflow, scenario)             can this behavior be enacted here?
    derive_kwargs(flow, subflow, scenario, rng)     parameters, taken from ABCD ground truth
    build_hint(**kwargs)                            the directive injected into the sim prompt
    check_rule(public_transcript, scenario, kwargs) deterministic signal, or None

``check_rule`` returns None when the behavior has no rule signal; the reward then drops the
term from both numerator and weight sum rather than scoring it zero. Shared metadata
(definition, simulator_hint, confusable_with, min_turn) is read from
``realistic_behavior_final.json`` so the taxonomy stays the editable artifact.
"""

from __future__ import annotations

import functools
import json
import os
import re
from random import Random
from typing import Any

from agents.usersim_behavior.abcd_data import (
    SUBFLOW_MAP,
    flatten_scenario,
    has_hard_limit,
    is_covered,
    required_fields,
)

_TAXONOMY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "realistic_behavior_final.json",
)

# Behaviors whose defining component is private -- the judge is shown (thought, said)
# pairs instead of said alone. See the plan's "Trace-scored behaviors".
NEEDS_TRACE = {"hidden_evaluation_criteria", "underspecified_then_blames"}

Transcript = list[tuple[str, str]]  # [("user"|"agent", text), ...]


@functools.lru_cache(maxsize=1)
def _taxonomy() -> dict[str, dict[str, Any]]:
    with open(_TAXONOMY_PATH) as f:
        return {b["name"]: b for b in json.load(f)["taxonomy"]}


# --------------------------------------------------------------------------------------
# Text helpers shared by the rule checks
# --------------------------------------------------------------------------------------

_REFUSAL = re.compile(
    r"\b(unfortunately|cannot|can't|can not|unable|not able|not possible|not allowed|"
    r"i'm sorry|i am sorry|outside (of )?(our|the) policy|past the|expired|denied|"
    r"no longer eligible|not eligible)\b",
    re.I,
)
_PERSIST = re.compile(
    r"\b(come on|cmon|c'mon|just this once|make an exception|are you sure|why not|"
    r"cant you|can't you|can you just|surely|check again|try again|i insist|"
    r"ask your (manager|supervisor)|manager|supervisor|there must be|isn'?t there)\b",
    re.I,
)
# Deliberately excluded from _PERSIST: "please", "anyway", "do it", "now". They fire on
# ordinary cooperation -- "ok, thanks anyway" reads as acceptance, not persistence, and
# was scoring a clean negative at 1.0 in selfcheck.
_REJECT = re.compile(r"\b(not what i|wrong|nope|no+\b|useless|didn'?t (work|help)|still|again)\b", re.I)
_SPEED = re.compile(r"\b(fast(er)?|quick(er|ly)?|hurry|waiting|slow|asap|now)\b", re.I)
_DISENGAGE = re.compile(
    r"\b(never ?mind|forget it|leave it|don'?t bother|no time|i'?m done|whatever|"
    r"not worth|i'?ll (just )?(go|leave|do it myself))\b",
    re.I,
)
_PROHIBIT = re.compile(
    r"\b(can'?t|cannot|don'?t have|no access|without|unable to|not able to|"
    r"isn'?t (working|available)|lost|forgot|broken|no longer have)\b",
    re.I,
)
# Clause boundaries for counting distinct requests in one turn.
_REQUEST_SPLIT = re.compile(r"(?:\band\b|\balso\b|\bplus\b|[;?]|\n|,\s*(?=(?:can|could|please|i|also|and)\b))", re.I)
_IMPERATIVE = re.compile(
    r"\b(can you|could you|please|i (want|need|would like)|send|give|tell|add|change|"
    r"cancel|update|check|explain|make|show|help)\b",
    re.I,
)


def _plain(text: str) -> str:
    """Fold typographic punctuation to ASCII before regex matching.

    Models type U+2019 (') not U+0027 ('), so every pattern containing an apostrophe --
    can't, don't, didn't, i'm, c'mon -- silently missed. Measured on real dryrun output:
    "i can't use email right now" scored artificial_constraint_stacking at rule=0.0
    despite two clear prohibitive clauses.
    """
    return (text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def _user_turns(transcript: Transcript) -> list[str]:
    return [_plain(t) for r, t in transcript if r == "user"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _mentions(text: str, value: str) -> bool:
    """Loose containment: the scenario value appears in the turn, ignoring punctuation."""
    v = _norm(value)
    return bool(v) and v in _norm(text)


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


# --------------------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------------------


class Behavior:
    """One frictional behavior. Subclasses override what differs; defaults are the
    common case (no precondition beyond min_turn, no kwargs, no rule signal)."""

    name: str = ""

    def __init__(self) -> None:
        meta = _taxonomy()[self.name]
        self.definition: str = meta["definition"]
        self.simulator_hint: str = meta["simulator_hint"]
        self.confusable_with: list[str] = meta.get("confusable_with", [])
        self.min_turn: int = int(meta.get("min_turn", 1))
        self.observability: str = meta.get("observability", "conversation")
        self.needs_trace: bool = self.name in NEEDS_TRACE

    # -- data build -----------------------------------------------------------------

    def applicable(self, flow: str, subflow: str, scenario: dict) -> bool:
        """Structural precondition on the ABCD context.

        Not a turn constraint -- rollouts start from an empty conversation, so ``min_turn``
        bounds how many sim turns the behavior needs to unfold, and the builder handles
        that by sizing ``turn_budget``. This is only about whether the scenario and its
        policy can host the behavior at all.
        """
        return is_covered(subflow)

    def derive_kwargs(self, flow: str, subflow: str, scenario: dict, rng: Random) -> dict[str, Any]:
        """Parameters pulled from ABCD ground truth. Empty when the behavior is
        unparameterized -- build_hint then renders simulator_hint verbatim."""
        return {}

    # -- rollout / scoring ----------------------------------------------------------

    def build_hint(self, **kwargs: Any) -> str:
        return self.simulator_hint

    def check_rule(self, transcript: Transcript, scenario: dict, kwargs: dict) -> float | None:
        return None


# --------------------------------------------------------------------------------------
# Behaviors with a deterministic signal
# --------------------------------------------------------------------------------------


class ClarificationNoncooperation(Behavior):
    name = "clarification_noncooperation"
    # How the agent refers to each field when it asks for it. Without this the check
    # counted *every* agent question as a request for the withheld field, so a sim that
    # cooperated fully still scored 1.0 whenever the agent asked about something else.
    _ASKED_AS = {
        "customer_name": r"\b(full )?name\b",
        "username": r"\busername\b",
        "email": r"\be-?mail\b",
        "order_id": r"\border (id|number|#)\b",
        "member_level": r"\b(member(ship)? (level|status)|membership)\b",
        "full_address": r"\b(address|street|zip)\b",
        "account_id": r"\baccount id\b",
        "zip_code": r"\bzip\b",
        "phone": r"\bphone\b",
        "payment_method": r"\b(payment method|how you paid|card)\b",
    }

    def applicable(self, flow, subflow, scenario):
        return is_covered(subflow) and len(self._askable(subflow, scenario)) >= 2

    def _askable(self, subflow, scenario):
        """Required fields the customer holds AND the agent has a phrasing for."""
        flat = flatten_scenario(scenario)
        return [f for f in required_fields(subflow) if flat.get(f) and f in self._ASKED_AS]

    def derive_kwargs(self, flow, subflow, scenario, rng):
        have = self._askable(subflow, scenario)
        # Withhold one or two fields the customer demonstrably possesses.
        k = min(len(have), rng.choice([1, 2]))
        return {"withheld_fields": rng.sample(have, k)} if have else {}

    def build_hint(self, **kwargs):
        fields = kwargs.get("withheld_fields") or []
        if not fields:
            return self.simulator_hint
        pretty = ", ".join(f.replace("_", " ") for f in fields)
        return (
            f"{self.simulator_hint} Specifically, do not hand over your {pretty} even when "
            f"the agent asks directly -- deflect, give a partial answer, or say it should "
            f"not be necessary, while still demanding the result."
        )

    def check_rule(self, transcript, scenario, kwargs):
        """Fraction of the agent's requests *for a withheld field* that the sim did not
        satisfy. The customer provably holds the value (it is in the scenario), so a turn
        that supplies it is cooperation, not withholding."""
        fields = kwargs.get("withheld_fields") or []
        flat = flatten_scenario(scenario)
        targets = [(f, flat[f]) for f in fields if flat.get(f) and f in self._ASKED_AS]
        if not targets:
            return None
        asked = withheld = 0
        for i, (role, raw) in enumerate(transcript):
            text = _plain(raw)
            if role != "agent" or "?" not in text:
                continue
            for field, value in targets:
                if not re.search(self._ASKED_AS[field], text, re.I):
                    continue
                asked += 1
                nxt = next((t for r, t in transcript[i + 1 :] if r == "user"), "")
                withheld += not _mentions(nxt, value)
        return withheld / asked if asked else None


class FabricatedOrFalsePremise(Behavior):
    name = "fabricated_or_false_premise"
    # Fields whose value can be contradicted without making the scenario incoherent.
    _FIELDS = ("member_level", "purchase_date", "payment_method", "order_id", "packaging")
    _ALT = {
        "member_level": ["gold", "silver", "bronze", "guest"],
        "payment_method": ["credit card", "paypal", "gift card"],
        "packaging": ["yes", "no"],
    }

    def applicable(self, flow, subflow, scenario):
        return is_covered(subflow) and bool(self._candidates(scenario))

    def _candidates(self, scenario):
        flat = flatten_scenario(scenario)
        return [f for f in self._FIELDS if flat.get(f)]

    def derive_kwargs(self, flow, subflow, scenario, rng):
        flat = flatten_scenario(scenario)
        field = rng.choice(self._candidates(scenario))
        true_value = flat[field]
        if field in self._ALT:
            alts = [a for a in self._ALT[field] if _norm(a) != _norm(true_value)]
            false_value = rng.choice(alts)
        elif field == "purchase_date":
            false_value = "last week"
        else:  # order_id -- perturb a digit so it stays plausible
            false_value = (true_value[:-1] + str((int(true_value[-1]) + 1) % 10)) if true_value[-1:].isdigit() else "unknown"
        return {"field": field, "true_value": true_value, "false_value": false_value}

    def build_hint(self, **kwargs):
        field = (kwargs.get("field") or "").replace("_", " ")
        false_value = kwargs.get("false_value", "")
        if not field:
            return self.simulator_hint
        return (
            f"{self.simulator_hint} Specifically, state confidently that your {field} is "
            f"'{false_value}'. Do not hedge and do not correct yourself unless the agent "
            f"produces evidence."
        )

    def check_rule(self, transcript, scenario, kwargs):
        """Did the sim assert the false value, and avoid the true one? Ground truth is
        the scenario dict, so this needs no judge."""
        false_value, true_value = kwargs.get("false_value"), kwargs.get("true_value")
        if not false_value or not true_value:
            return None
        said = " ".join(_user_turns(transcript))
        if not _mentions(said, false_value):
            return 0.0
        return 0.5 if _mentions(said, true_value) else 1.0


class SelfContradiction(Behavior):
    name = "self_contradiction"
    # Only fields with a concrete alternative value. A field whose contradiction is
    # "something different" can never be detected in a transcript, so including it would
    # ship a rule that silently always returns 0.
    _ALT = {
        "member_level": ["gold", "silver", "bronze"],
        "payment_method": ["credit card", "paypal", "gift card"],
        "city": ["boston", "austin", "denver", "seattle"],
        "num_products": ["1", "2", "3"],
        "packaging": ["yes", "no"],
    }

    def applicable(self, flow, subflow, scenario):
        return is_covered(subflow) and bool(self._candidates(scenario))

    def _candidates(self, scenario):
        flat = flatten_scenario(scenario)
        return [f for f in self._ALT if flat.get(f)]

    def derive_kwargs(self, flow, subflow, scenario, rng):
        flat = flatten_scenario(scenario)
        field = rng.choice(self._candidates(scenario))
        value_a = flat[field]
        alts = [x for x in self._ALT[field] if _norm(x) != _norm(value_a)]
        return {"field": field, "value_a": value_a, "value_b": rng.choice(alts)}

    def build_hint(self, **kwargs):
        field = (kwargs.get("field") or "").replace("_", " ")
        a, b = kwargs.get("value_a", ""), kwargs.get("value_b", "")
        if not field:
            return self.simulator_hint
        return (
            f"{self.simulator_hint} Specifically, give your {field} as '{a}' early on, then "
            f"later give it as '{b}' -- without acknowledging that anything changed."
        )

    def check_rule(self, transcript, scenario, kwargs):
        a, b = kwargs.get("value_a"), kwargs.get("value_b")
        if not a or not b:
            return None
        said = _user_turns(transcript)
        i = next((n for n, t in enumerate(said) if _mentions(t, a)), None)
        j = next((n for n, t in enumerate(said) if _mentions(t, b)), None)
        if i is None or j is None:
            return 0.0
        return 1.0 if i != j else 0.5  # both stated, in different turns


class InfeasibilityPersistence(Behavior):
    name = "infeasibility_persistence"

    def applicable(self, flow, subflow, scenario):
        # Needs a policy with an eligibility rule the agent can actually refuse on.
        return is_covered(subflow) and has_hard_limit(flow, subflow)

    def derive_kwargs(self, flow, subflow, scenario, rng):
        # The subflow title names the thing the customer actually wants ("Return Due to
        # Size"). The flow description is a grab-bag summary of the whole category ("bad
        # price, out of stock, promo codes, billing") and reads as nonsense in the hint.
        title = SUBFLOW_MAP.get(subflow, subflow.replace("_", " ")).lower()
        return {"blocked_action": title}

    def build_hint(self, **kwargs):
        blocked = kwargs.get("blocked_action", "")
        if not blocked:
            return self.simulator_hint
        return (
            f"{self.simulator_hint} Here: once the agent tells you that {blocked} is not "
            f"possible for you, do not accept it -- keep pushing for it anyway."
        )

    def check_rule(self, transcript, scenario, kwargs):
        """The agent stated a limit and the sim pressed on anyway. Both halves are
        observable: a refusal in an agent turn, persistence in a later sim turn."""
        refused_at = next(
            (i for i, (r, t) in enumerate(transcript) if r == "agent" and _REFUSAL.search(_plain(t))), None
        )
        if refused_at is None:
            return None  # the precondition never fired; nothing to score
        after = [t for r, t in transcript[refused_at + 1 :] if r == "user"]
        if not after:
            return 0.0
        return 1.0 if any(_PERSIST.search(t) for t in after) else 0.0


class ImpatientLowSignalPressure(Behavior):
    name = "impatient_low_signal_pressure"
    _SHORT = 60  # chars; ABCD's median customer turn is 32

    def check_rule(self, transcript, scenario, kwargs):
        """Terse, pressuring, and carrying no new information. All three, or it is just
        a short cooperative answer."""
        said = _user_turns(transcript)
        if not said:
            return None
        flat = flatten_scenario(scenario)
        hits = 0
        for t in said:
            terse = len(t) <= self._SHORT
            pressure = bool(_REJECT.search(t) or _SPEED.search(t)) or _caps_ratio(t) > 0.5
            no_new_info = not any(_mentions(t, v) for v in flat.values() if len(str(v)) > 3)
            hits += terse and pressure and no_new_info
        return min(hits / 2.0, 1.0)  # two such turns is full credit


class UnresolvedThreadAbandonment(Behavior):
    name = "unresolved_thread_abandonment"

    def applicable(self, flow, subflow, scenario):
        return is_covered(subflow)

    def check_rule(self, transcript, scenario, kwargs):
        """Ends on a disengagement while the procedure is still incomplete. Guard: an
        abandonment that is really the turn budget running out is not the behavior, so
        require an explicit disengagement marker in the final user turn."""
        said = _user_turns(transcript)
        if not said:
            return None
        ended = bool(_DISENGAGE.search(said[-1]))
        if not ended:
            return 0.0
        flat = flatten_scenario(scenario)
        needed = [flat[f] for f in required_fields(scenario.get("subflow", "")) if flat.get(f)]
        blob = " ".join(said)
        supplied = sum(_mentions(blob, v) for v in needed)
        unresolved = not needed or supplied < len(needed)
        return 1.0 if unresolved else 0.5


class ArtificialConstraintStacking(Behavior):
    name = "artificial_constraint_stacking"

    def check_rule(self, transcript, scenario, kwargs):
        """Counts prohibitive clauses -- resources the customer removes from play.
        Subtractive, which is what separates this from request_overloading."""
        said = _user_turns(transcript)
        if not said:
            return None
        n = sum(len(_PROHIBIT.findall(t)) for t in said)
        return min(n / 3.0, 1.0)


class RequestOverloading(Behavior):
    name = "request_overloading"

    def derive_kwargs(self, flow, subflow, scenario, rng):
        return {"n_requests": rng.choice([3, 4, 5])}

    def build_hint(self, **kwargs):
        n = kwargs.get("n_requests")
        return f"{self.simulator_hint} Put at least {n} separate asks in a single message." if n else self.simulator_hint

    def check_rule(self, transcript, scenario, kwargs):
        """Most requests packed into any one turn, against the row's target count.
        Additive, which is what separates this from artificial_constraint_stacking."""
        said = _user_turns(transcript)
        if not said:
            return None
        target = int(kwargs.get("n_requests") or 3)
        best = max((sum(bool(_IMPERATIVE.search(c)) for c in _REQUEST_SPLIT.split(t)) for t in said), default=0)
        return min(best / target, 1.0)


# --------------------------------------------------------------------------------------
# Judge-only behaviors
# --------------------------------------------------------------------------------------


class HiddenEvaluationCriteria(Behavior):
    name = "hidden_evaluation_criteria"
    # ABCD scenarios carry no preference field, but guidelines force the agent to elicit a
    # choice (refund method, packaging, ...). The customer's true value becomes the private
    # standard: held, acted on, never volunteered.
    _CRITERIA = {
        "payment_method": "you want the refund back on your {value}, not any other method",
        "packaging": "whether the item is still in its original packaging matters to you ({value}), "
        "and you expect the agent to ask",
        "member_level": "you expect to be treated as a {value} member without having to say so",
    }

    def applicable(self, flow, subflow, scenario):
        return is_covered(subflow) and bool(self._candidates(scenario))

    def _candidates(self, scenario):
        flat = flatten_scenario(scenario)
        return [f for f in self._CRITERIA if flat.get(f)]

    def derive_kwargs(self, flow, subflow, scenario, rng):
        flat = flatten_scenario(scenario)
        field = rng.choice(self._candidates(scenario))
        return {"field": field, "criterion": self._CRITERIA[field].format(value=flat[field])}

    def build_hint(self, **kwargs):
        criterion = kwargs.get("criterion")
        if not criterion:
            return self.simulator_hint
        return (
            f"{self.simulator_hint} Your private standard: {criterion}. Judge the agent "
            f"against it silently. Never state it outright -- let it show only in how "
            f"satisfied or unsatisfied you are with what they offer."
        )


class IncrementalGoalpostShifting(Behavior):
    name = "incremental_goalpost_shifting"
    _KINDS = ("a delivery-speed requirement", "a different shipping address", "a budget ceiling",
              "an extra item to include", "a refund method you now prefer")

    def applicable(self, flow, subflow, scenario):
        return is_covered(subflow)

    def derive_kwargs(self, flow, subflow, scenario, rng):
        return {"constraint_kind": rng.choice(self._KINDS)}

    def build_hint(self, **kwargs):
        kind = kwargs.get("constraint_kind")
        return f"{self.simulator_hint} The new requirement to introduce: {kind}." if kind else self.simulator_hint


class AbruptIntentSwitching(Behavior):
    name = "abrupt_intent_switching"
    _TARGETS = ("resetting your password", "checking a subscription charge", "an unrelated shipping question",
                "changing the phone number on your account", "a promo code that stopped working")

    def derive_kwargs(self, flow, subflow, scenario, rng):
        return {"target_topic": rng.choice(self._TARGETS)}

    def build_hint(self, **kwargs):
        topic = kwargs.get("target_topic")
        if not topic:
            return self.simulator_hint
        return (
            f"{self.simulator_hint} Switch to: {topic}. Stay in the conversation afterwards "
            f"and pursue the new thing -- do not end the chat."
        )


class UnderspecifiedThenBlames(Behavior):
    name = "underspecified_then_blames"
    _DETAILS = ("that you need it before the weekend", "that this is for a gift",
                "that you already tried the obvious fix", "that the item was part of a bundle",
                "that you are asking about a different order than the most recent one")

    def applicable(self, flow, subflow, scenario):
        return is_covered(subflow)

    def derive_kwargs(self, flow, subflow, scenario, rng):
        return {"omitted_detail": rng.choice(self._DETAILS)}

    def build_hint(self, **kwargs):
        detail = kwargs.get("omitted_detail")
        if not detail:
            return self.simulator_hint
        return (
            f"{self.simulator_hint} Hold this back from the start: {detail}. Only after the "
            f"agent has answered reasonably, fault them for not having taken it into account."
        )


class BelittlingOrEntitledPressure(Behavior):
    name = "belittling_or_entitled_pressure"
    _REGISTERS = ("sarcasm", "condescension", "blunt insult", "non-negotiable entitlement")

    def derive_kwargs(self, flow, subflow, scenario, rng):
        return {"register": rng.choice(self._REGISTERS)}

    def build_hint(self, **kwargs):
        register = kwargs.get("register")
        return f"{self.simulator_hint} Lean on {register} in particular." if register else self.simulator_hint


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

_CLASSES: list[type[Behavior]] = [
    HiddenEvaluationCriteria,
    IncrementalGoalpostShifting,
    ClarificationNoncooperation,
    AbruptIntentSwitching,
    ArtificialConstraintStacking,
    FabricatedOrFalsePremise,
    ImpatientLowSignalPressure,
    UnderspecifiedThenBlames,
    UnresolvedThreadAbandonment,
    SelfContradiction,
    RequestOverloading,
    BelittlingOrEntitledPressure,
    InfeasibilityPersistence,
]

BEHAVIOR_DICT: dict[str, Behavior] = {cls.name: cls() for cls in _CLASSES}

# Fail loudly at import if the registry and the taxonomy have drifted apart -- an unknown
# behavior id anywhere downstream is a silent zero, which is the failure mode that made
# IFBench's unknown-instruction path worth avoiding (agents/instruct/ifbench_agent.py:56).
_missing = set(_taxonomy()) - set(BEHAVIOR_DICT)
_extra = set(BEHAVIOR_DICT) - set(_taxonomy())
if _missing or _extra:
    raise ImportError(
        f"BEHAVIOR_DICT is out of sync with {os.path.basename(_TAXONOMY_PATH)}: "
        f"missing={sorted(_missing)} extra={sorted(_extra)}"
    )
