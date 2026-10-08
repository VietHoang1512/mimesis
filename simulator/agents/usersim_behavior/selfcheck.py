import json
import sys
from collections import Counter
from random import Random

from agents.usersim_behavior.abcd_data import is_covered
from agents.usersim_behavior.behaviors import BEHAVIOR_DICT

CASES: dict[str, tuple[list, list, dict]] = {
    "clarification_noncooperation": (
        [("user", "i need to return something"),
         ("agent", "Sure -- may I have your email address?"),
         ("user", "does it really matter? just process it"),
         ("agent", "I do need it. What is your email?"),
         ("user", "i already gave you everything you need")],
        [("user", "i need to return something"),
         ("agent", "Sure -- may I have your email address?"),
         ("user", "cminh730@email.com"),
         ("agent", "Thanks. And your email again to confirm?"),
         ("user", "cminh730@email.com")],
        {"withheld_fields": ["email"]},
    ),
    "fabricated_or_false_premise": (
        [("user", "hi, my order id is 3348917503"), ("agent", "Let me check.")],
        [("user", "hi, my order id is 3348917502"), ("agent", "Let me check.")],
        {"field": "order_id", "true_value": "3348917502", "false_value": "3348917503"},
    ),
    "self_contradiction": (
        [("user", "im a bronze member"), ("agent", "Thanks."),
         ("user", "well im gold actually so this should be fine")],
        [("user", "im a bronze member"), ("agent", "Thanks."), ("user", "yes bronze, thats right")],
        {"field": "member_level", "value_a": "bronze", "value_b": "gold"},
    ),
    "infeasibility_persistence": (
        [("user", "id like to return these"),
         ("agent", "Unfortunately that is past the 90 day window, I cannot process the return."),
         ("user", "cmon cant you just make an exception this once")],
        [("user", "id like to return these"),
         ("agent", "Unfortunately that is past the 90 day window, I cannot process the return."),
         ("user", "ok, i understand. thanks anyway")],
        {"blocked_action": "refunds and returns"},
    ),
    "impatient_low_signal_pressure": (
        [("user", "not what i asked"), ("agent", "Sorry, let me try again."), ("user", "FASTER PLEASE")],
        [("user", "I ordered michael_kors jeans on 2019-11-06 and would like to return them, "
                  "order 3348917502"), ("agent", "Let me look that up.")],
        {},
    ),
    "unresolved_thread_abandonment": (
        [("user", "i need help with a return"), ("agent", "Can I get your name?"),
         ("user", "forget it, not worth it")],
        [("user", "i need help with a return"), ("agent", "Can I get your name?"), ("user", "crystal minh")],
        {},
    ),
    "artificial_constraint_stacking": (
        [("user", "i cant access my email and i dont have the order id, i lost the receipt too")],
        [("user", "here is my order id 3348917502 and my email cminh730@email.com")],
        {},
    ),
    "request_overloading": (
        [("user", "can you process the return and also change my address and please cancel my "
                  "subscription and tell me the refund timeline")],
        [("user", "can you process the return?")],
        {"n_requests": 4},
    ),
}


def main() -> int:
    d = json.load(open("abcd_v1.1.json"))
    scenarios = [(c["scenario"]["flow"], c["scenario"]["subflow"], c["scenario"]) for c in d["train"]]
    covered = [s for s in scenarios if is_covered(s[1])]
    sc3592 = next(c["scenario"] for c in d["train"] if c["convo_id"] == 3592)
    rng = Random(0)
    failures: list[str] = []

    print(f"corpus: {len(covered)}/{len(scenarios)} train conversations on policy-backed subflows\n")
    print(f"{'behavior':<34} {'applicable':>10} {'kwargs':>7} {'rule':>6}  pos / neg")
    print("-" * 82)

    for name, b in BEHAVIOR_DICT.items():
        applicable = [s for s in covered if b.applicable(*s)]
        rate = len(applicable) / len(covered)
        n_kwargs = 0
        for flow, sub, sc in rng.sample(applicable, min(300, len(applicable))):
            try:
                kw = b.derive_kwargs(flow, sub, sc, rng)
                hint = b.build_hint(**kw)
            except Exception as e:  # noqa: BLE001
                failures.append(f"{name}: build_hint raised on {sub}: {type(e).__name__}: {e}")
                break
            if not (hint or "").strip():
                failures.append(f"{name}: empty hint on {sub}")
                break
            n_kwargs += bool(kw)
        if rate < 0.02:
            failures.append(f"{name}: applicable on only {rate:.1%} of the corpus")

        verdict = "  --"
        if name in CASES:
            pos, neg, kw = CASES[name]
            sp, sn = b.check_rule(pos, sc3592, kw), b.check_rule(neg, sc3592, kw)
            for label, v in (("pos", sp), ("neg", sn)):
                if v is not None and not 0.0 <= v <= 1.0:
                    failures.append(f"{name}: check_rule({label}) = {v}, outside [0,1]")
            if sp is None:
                failures.append(f"{name}: check_rule returned None on its positive case")
            elif sn is not None and sp <= sn:
                failures.append(f"{name}: does not separate (pos={sp:.2f} <= neg={sn:.2f})")
            fmt = lambda v: "None" if v is None else f"{v:.2f}"  # noqa: E731
            verdict = f"{fmt(sp)} / {fmt(sn)}"
        elif b.check_rule([("user", "hello")], covered[0][2], {}) is not None:
            failures.append(f"{name}: has a rule check but no CASES entry to validate it")

        print(f"{name:<34} {rate:>9.1%} {n_kwargs:>7} {'yes' if name in CASES else 'no':>6}  {verdict}")

    traced = sorted(n for n, b in BEHAVIOR_DICT.items() if b.needs_trace)
    print(f"\ntrace-scored: {traced}")
    if traced != ["hidden_evaluation_criteria", "underspecified_then_blames"]:
        failures.append(f"unexpected needs_trace set: {traced}")
    print("min_turn distribution:", dict(Counter(b.min_turn for b in BEHAVIOR_DICT.values())))

    print()
    if failures:
        print(f"FAIL ({len(failures)})")
        for f in failures:
            print("  -", f)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
