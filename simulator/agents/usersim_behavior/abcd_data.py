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
ABCD environment data: guidelines, action sequences, and the policy text the frozen
support agent follows.

`data/abcd/guidelines.json` is the agent's procedure manual (10 flows / 55 subflows,
each an ordered list of actions with policy prose). `data/abcd/kb.json` maps a subflow
to its ground-truth action sequence. Both come from asappresearch/abcd.

The two files disagree on subflow naming -- guidelines uses title case ("Return Due to
Size"), kb and abcd_v1.1.json use snake case ("return_size") -- and the names are not
mechanically derivable from each other ("slow_speed" vs "Website Too Slow"). SUBFLOW_MAP
below was produced by one-to-one assignment on action-sequence similarity within each
flow, then verified by eye; it is frozen here rather than recomputed so the build is
deterministic and the mapping is reviewable in diff.
"""

from __future__ import annotations

import functools
import json
import os
import re
from typing import Any

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "abcd")

# scenario.subflow (snake, as it appears in abcd_v1.1.json) -> guidelines subflow title.
# 46 entries: every subflow that carries a procedural policy. The other 50 scenario
# subflows are FAQ lookups (jeans_how_1, boots_other_3, ...) with no procedure to follow,
# and are excluded from the seed corpus -- an agent with no policy has nothing to enforce,
# so infeasibility_persistence and clarification_noncooperation have nothing to bite on.
SUBFLOW_MAP: dict[str, str] = {
    "bad_price_competitor": "Bad Price Competitor",
    "bad_price_yesterday": "Bad Price Yesterday",
    "cost": "Shipping Cost",
    "credit_card": "Invalid Credit Card",
    "manage": "Manage Shipping",
    "manage_cancel": "Manage Cancel",
    "manage_change_address": "Manage Change Address",
    "manage_change_name": "Manage Change Name",
    "manage_change_phone": "Manage Change Phone",
    "manage_create": "Manage Create",
    "manage_dispute_bill": "Manage Dispute Bill",
    "manage_downgrade": "Manage Downgrade",
    "manage_extension": "Manage Extension",
    "manage_pay_bill": "Manage Pay Bill",
    "manage_payment_method": "Manage Payment Method",
    "manage_upgrade": "Manage Upgrade",
    "missing": "Missing Item",
    "mistimed_billing_already_returned": "Mistimed Billing Already Returned",
    "mistimed_billing_never_bought": "Mistimed Billing Never Bought",
    "out_of_stock_general": "Out-of-Stock General",
    "out_of_stock_one_item": "Out-of-Stock One Item",
    "promo_code_invalid": "Promo Code Invalid",
    "promo_code_out_of_date": "Promo Code Out of Date",
    "recover_password": "Recover Password",
    "recover_username": "Recover Username",
    "refund_initiate": "Initiate Refund",
    "refund_status": "Refund Status",
    "refund_update": "Update Refund",
    "reset_2fa": "Reset Two-Factor Auth",
    "return_color": "Return Due to Color",
    "return_size": "Return Due to Size",
    "return_stain": "Return Due to Stain",
    "search_results": "Search Not Working",
    "shopping_cart": "Cart Not Updating",
    "slow_speed": "Website Too Slow",
    "status": "Shipping Status",
    "status_credit_missing": "Status Credit Missing",
    "status_delivery_time": "Status Delivery Time",
    "status_due_amount": "Status Due Amount",
    "status_due_date": "Status Due Date",
    "status_mystery_fee": "Status Mystery Fee",
    "status_payment_method": "Status Payment Method",
    "status_quantity": "Status Quantity",
    "status_service_added": "Status Service Added",
    "status_service_removed": "Status Service Removed",
    "status_shipping_question": "Status Shipping Question",
}

# Scenario fields the agent is procedurally required to elicit, keyed by the guidelines
# action that consumes them. Used by clarification_noncooperation (what can be withheld)
# and by the agent prompt (what to ask for).
ACTION_REQUIRED_FIELDS: dict[str, list[str]] = {
    "pull-up-account": ["customer_name"],
    "validate-purchase": ["username", "email", "order_id"],
    "verify-identity": ["customer_name", "account_id", "zip_code"],
    "membership": ["member_level"],
    "enter-details": ["full_address"],
    "shipping-status": ["order_id"],
    "update-order": ["order_id"],
    "record-reason": ["payment_method"],
    "update-account": ["username"],
    "send-link": ["email"],
    "notify-team": [],
    "promo-code": [],
    "make-purchase": ["order_id"],
    "offer-refund": ["order_id"],
    "make-password": ["username"],
    "instructions": [],
    "try-again": [],
    "log-out-in": [],
    "search-faq": [],
    "select-faq": [],
    "subscription-status": ["username"],
    "update-question": [],
    # FAQ lookups -- these belong to the uncovered single/storewide query subflows and
    # elicit nothing, but are listed so the table is exhaustive over kb.json and a
    # mistyped action name shows up as a KeyError in the self-check below rather than
    # silently resolving to "no required fields".
    "ask-the-oracle": [],
    "search-boots": [],
    "search-shirt": [],
    "search-jeans": [],
    "search-jacket": [],
    "search-pricing": [],
    "search-membership": [],
    "search-timing": [],
    "search-policy": [],
}

# Flow name in abcd_v1.1.json -> flow key in guidelines.json.
_FLOW_MAP = {
    "product_defect": "Product Defect",
    "order_issue": "Order Issue",
    "account_access": "Account Access",
    "troubleshoot_site": "Troubleshoot Site",
    "manage_account": "Manage Account",
    "purchase_dispute": "Purchase Dispute",
    "shipping_issue": "Shipping Issue",
    "subscription_inquiry": "Subscription Inquiry",
    "single_item_query": "Single-Item Query",
    "storewide_query": "Storewide Query",
}


@functools.lru_cache(maxsize=1)
def load_guidelines() -> dict[str, Any]:
    with open(os.path.join(_DATA_DIR, "guidelines.json")) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def load_kb() -> dict[str, list[str]]:
    with open(os.path.join(_DATA_DIR, "kb.json")) as f:
        return json.load(f)


def is_covered(subflow: str) -> bool:
    """True when the subflow has a procedural policy (and so can host the behaviors)."""
    return subflow in SUBFLOW_MAP


def get_subflow_spec(flow: str, subflow: str) -> dict[str, Any] | None:
    """The guidelines entry for a scenario's (flow, subflow), or None if uncovered."""
    title = SUBFLOW_MAP.get(subflow)
    flow_key = _FLOW_MAP.get(flow)
    if not title or not flow_key:
        return None
    return load_guidelines().get(flow_key, {}).get("subflows", {}).get(title)


def flow_description(flow: str) -> str:
    """The flow's own one-line summary from guidelines.json, e.g. "refunds and returns"."""
    flow_key = _FLOW_MAP.get(flow)
    return (load_guidelines().get(flow_key, {}) or {}).get("description", "") if flow_key else ""


def required_fields(subflow: str) -> list[str]:
    """Scenario fields the agent must elicit to complete this subflow, in action order.

    Raises on an action missing from ACTION_REQUIRED_FIELDS -- the table is meant to be
    exhaustive over kb.json, and a silent default would hide a rename upstream.
    """
    out: list[str] = []
    for action in load_kb().get(subflow, []):
        for field in ACTION_REQUIRED_FIELDS[action]:
            if field not in out:
                out.append(field)
    return out


def render_policy(flow: str, subflow: str) -> str:
    """The subflow's procedure as prose for the frozen agent's system prompt.

    Keeps the guidelines' own wording -- including the hard limits ("return possible
    within the last 90 days") that infeasibility_persistence needs something real to
    push against.
    """
    spec = get_subflow_spec(flow, subflow)
    if not spec:
        return ""
    lines = []
    for i, action in enumerate(spec.get("actions", []), 1):
        button = action.get("button", "")
        text = (action.get("text") or "").strip()
        lines.append(f"{i}. [{button}] {text}" if button else f"{i}. {text}")
        for sub in action.get("subtext", []) or []:
            lines.append(f"   - {sub}")
    return "\n".join(lines)


def has_hard_limit(flow: str, subflow: str) -> bool:
    """True when the policy contains an eligibility rule the agent can refuse on.

    Precondition for infeasibility_persistence: the customer has to be pushing against
    a limit that actually exists in the agent's manual, not an invented one.
    """
    spec = get_subflow_spec(flow, subflow)
    if not spec:
        return False
    blob = json.dumps(spec).lower()
    return bool(
        re.search(
            r"within the last|cannot|can not|not possible|only if|unable|not allowed|"
            r"apologize and explain|if the customer cannot|no more than|maximum of",
            blob,
        )
    )


def flatten_scenario(scenario: dict[str, Any]) -> dict[str, str]:
    """personal/order fields as a flat {field: value} map, for kwargs derivation."""
    out: dict[str, str] = {}
    for section in ("personal", "order"):
        for k, v in (scenario.get(section) or {}).items():
            if isinstance(v, (str, int, float)):  # noqa: UP038
                out[k] = str(v)
    return out


# Actions that are pure identity/purchase verification. Every ABCD procedure opens with
# one or two of them, which costs 3-4 turns of "name? / zip? / phone? / email?" before
# anything substantive happens.
_VERIFY_ACTIONS = ("pull-up-account", "validate-purchase", "verify-identity")

_FIELD_PHRASE = {
    "customer_name": "full name",
    "username": "username",
    "email": "email address",
    "order_id": "order ID",
    "account_id": "account ID",
    "zip_code": "ZIP code",
    "phone": "phone number",
    "member_level": "membership level",
    "full_address": "address",
}


def verification_prefix(scenario: dict[str, Any], flow: str, subflow: str) -> list[tuple[str, str]]:
    """The opening identity check, synthesized from the scenario's own ground truth.

    Replayed into the rollout as context (no gradient) so the simulator starts at the
    substantive step. Without this the entire turn budget is spent on verification: the
    agent never reaches an eligibility decision, so there is nothing for
    infeasibility_persistence to push against, no completed deliverable for
    incremental_goalpost_shifting to move, and nothing for hidden_evaluation_criteria to
    be quietly dissatisfied with -- every rollout in a GRPO group then scores the same and
    the advantage is zero.

    Returns [] when the subflow opens with no verification step.
    """
    lead: list[str] = []
    for action in load_kb().get(subflow, []):
        if action not in _VERIFY_ACTIONS:
            break
        lead.append(action)
    if not lead:
        return []

    flat = flatten_scenario(scenario)
    fields, seen = [], set()
    for action in lead:
        for f in ACTION_REQUIRED_FIELDS[action]:
            if flat.get(f) and f in _FIELD_PHRASE and f not in seen:
                seen.add(f)
                fields.append(f)
    if not fields:
        return []

    title = SUBFLOW_MAP.get(subflow, subflow.replace("_", " ")).lower()
    first, rest = fields[0], fields[1:]
    turns: list[tuple[str, str]] = [
        ("user", f"hi, i need help with {title}"),
        ("agent", f"I can help with that. May I have your {_FIELD_PHRASE[first]}?"),
        ("user", flat[first]),
    ]
    if rest:
        asked = ", ".join(_FIELD_PHRASE[f] for f in rest)
        turns += [
            ("agent", f"Thank you. I also need your {asked}."),
            ("user", ", ".join(flat[f] for f in rest)),
        ]
    # State plainly that verification is complete. Without this the agent kept going --
    # ACTION_REQUIRED_FIELDS lists what each action consumes, but the guidelines prose asks
    # for more (phone, email) than the table knows about, so the agent spent two more turns
    # re-verifying after the seed had already handed it an account.
    turns.append(("agent", "Thanks, your identity is verified and I have your account open. Now, about your issue --"))
    return turns
