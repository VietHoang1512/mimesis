#!/usr/bin/env python3
"""Aggregate SimulatorArena document-creation runs across simulators.

WHAT THIS COMPUTES, AND WHAT IT DELIBERATELY DOES NOT

SimulatorArena scores a simulator on two axes:
  (1) how closely its messages resemble real user messages;
  (2) how well assistant ratings derived from it correlate with HUMAN ratings of
      those assistants (Spearman rho, the headline, rho~0.7 for profiled sims).

Axis (2) is not computable here and no amount of re-running fixes it. The human
ratings are ratings OF nine specific 2024-era assistants (gpt-4o, gpt-4-turbo,
claude-3-5-sonnet-20240620, llama-3-1-8b/70b, mistral-large-2407,
phi-3-small/medium, gpt-4o-mini); none resolve on this gateway, so our runs use a
substitute assistant that no annotator ever rated. show_simulator_performance.py
looks the human rating up BY assistant model and drops the instance when it comes
back None -- with a substituted assistant that is every instance.

So this reports axis (1) only, from behavioural statistics that need no judge:
turn count and message length against the real human on the SAME topic. That is a
real, on-protocol signal -- SimulatorArena's own message-similarity analysis is
built on exactly these quantities -- but it is descriptive, not the LLM-judged
Turing test, and it carries no human-correlation validity claim.

The judged half (extract document -> rate document -> rate interaction) is a
separate ~3.5k-call stage that additionally needs OpenAI's Batch API replaced
with our gateway transport. Left for when quota allows.

  python simarena_aggregate.py
  python simarena_aggregate.py --annotation-id document_creation_annotations_full
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "SimulatorArena"

# Grouping + display names, mirroring tau_usi_prism_score.py so the tables in the
# paper stay row-aligned across benchmarks.
GROUP = {
    "gemini-3.1-pro-preview": ("Frontier", "Gemini-3.1-Pro-Preview"),
    "gemini-3.6-flash": ("Frontier", "Gemini-3.6-Flash"),
    "claude-opus-5": ("Frontier", "Claude-Opus-5"),
    "claude-sonnet-5": ("Frontier", "Claude-Sonnet-5"),
    "gpt-5.5": ("Frontier", "GPT-5.5"),
    "Qwen3.5-9B-CHATML-290213-original-chat-thoughttrace-fix-426855-all-428848": ("Ours", "Qwen3.5-9B 428848 (paper)"),
    "Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-fix-329200-all-427969": ("Ours", "Qwen3.5-4B 427969 (paper)"),
    "Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-fix-329200-all-427969-400": ("Ours", "Qwen3.5-4B 427969-400"),
    "Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-329201-adv-370850": ("Ours", "Qwen3.5-4B adv-370850"),
    "Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-329200-all-337915": ("Ours", "Qwen3.5-4B 337915"),
    "Qwen3.5-4B-CHATML-279674-original-chat-thoughttrace-329200-all-386391": ("Ours", "Qwen3.5-4B 386391"),
    "Qwen3.5-4B-CHATML-279674-ditto-rl-all-288414": ("Ours", "Qwen3.5-4B ditto-rl 288414"),
    "Qwen3.5-4B-CHATML-279674-ditto-rl-all-291189": ("Ours", "Qwen3.5-4B ditto-rl 291189"),
    "Qwen3.5-4B-CHATML-291301-full-ditto-rl-all-300775": ("Ours", "Qwen3.5-4B ditto-rl 300775"),
    "Qwen3.5-9B-CHATML-279675-ditto-rl-all-290192": ("Ours", "Qwen3.5-9B ditto-rl 290192"),
    "Ditto-8B": ("Released sim", "Ditto-8B"),
    "osim-8b": ("Released sim", "OSim-8B"),
    "osim-4b": ("Released sim", "OSim-4B"),
    "humanlm-opinion": ("Released sim", "HumanLM-Opinion"),
    "sotopia-rl-qwen-2.5-7B-grpo": ("Released sim", "Sotopia-RL-Qwen2.5-7B"),
    "Qwen3.5-9B": ("Pretrained", "Qwen3.5-9B"),
    "Qwen3.5-4B": ("Pretrained", "Qwen3.5-4B"),
    "Qwen3-8B": ("Pretrained", "Qwen3-8B"),
}
GROUP_ORDER = ["Frontier", "Ours", "Released sim", "Pretrained", "other"]


def walk_convs(obj, out):
    """Yield every conversation dict, whatever the nesting depth.

    The layout is {assistant: {doc_type: {intent: {uuid: conv}}}} but the depth is
    not worth hard-coding -- one wrong guess silently returns zero conversations
    and reads as "no data" rather than "wrong key".
    """
    if isinstance(obj, dict):
        if any(isinstance(v, list) and v and isinstance(v[0], dict) and "role" in v[0]
               for v in obj.values()):
            out.append(obj)
            return
        for v in obj.values():
            walk_convs(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk_convs(v, out)


def user_turns(conv):
    """The simulator's turns AS THE ASSISTANT SAW THEM.

    Read from assistant_messages, not user_messages: user_messages is the
    simulator's own prompt context (it opens with a ~60k-char instruction block),
    whereas assistant_messages holds what was actually transmitted -- post
    Thought:/Message: and <think> stripping. Scoring the former would measure the
    prompt, not the model.
    """
    msgs = conv.get("assistant_messages") or []
    return [str(m.get("content", "")) for m in msgs if m.get("role") == "user"]


# utils.py:1378 -- the harness stops a conversation at 15 user turns. A simulator
# that hits it on every conversation never chose to stop, so its turn count is
# CENSORED: the true value is >=15, not 15. Reporting a Delta against the human
# mean without flagging that would read as "8 turns too talkative" when the honest
# statement is "does not terminate".
MAX_TURNS = 15


def words(s):
    return len(re.findall(r"\S+", s))


def load_ratings(annotation_id, sim, file_stem):
    """Mean judge rating per aspect, or None if that stage has not been run.

    Returns None rather than 0.0 for a missing stage: a simulator that was never
    judged and one the judge scored zero must not collide in the same column.
    """
    # annotation_id MUST be in the path. The rating scripts key their output on
    # file_name = "<simulator>/<stem>", and <stem> is identical for the 51-topic and
    # 459-topic sets -- so a flat layout makes the two datasets collide: the 459
    # table silently displayed 51-topic ratings, and running the judge on 459 would
    # have overwritten the 51-topic results outright. Namespacing by dataset is the
    # only thing separating them.
    base = (ROOT / "evaluation" / "document_creation" / "evaluation_outputs"
            / annotation_id)
    out = {}
    for aspect in ("document_rating", "interaction_rating"):
        f = base / aspect / sim / f"{file_stem}.json"
        if not f.exists():
            out[aspect] = None
            continue
        vals = []

        def walk(o):
            if isinstance(o, dict):
                if "extracted_rating" in o:
                    try:
                        vals.append(float(o["extracted_rating"]))
                    except (TypeError, ValueError):
                        pass
                else:
                    for v in o.values():
                        walk(v)
        walk(json.load(open(f)))
        out[aspect] = (st.mean(vals) if vals else None, len(vals))
    return out


def load_sim(path):
    convs = []
    walk_convs(json.load(open(path)), convs)
    return convs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotation-id", default="document_creation_annotations")
    ap.add_argument("--version", default="zero-shot-cot-user-profile",
                    help="prompt version; selects which <version>-up-*.json to read")
    ap.add_argument("--csv", default="simarena_results.csv")
    args = ap.parse_args()

    # Human reference: the real conversations behind these same topics.
    ann = json.load(open(ROOT / "data" / f"{args.annotation_id}_for_benchmarking.json"))
    h_turns = [len(r.get("user_queries") or []) for r in ann]
    h_words = [words(str(q)) for r in ann for q in (r.get("user_queries") or [])]
    H_T, H_W = st.mean(h_turns), st.mean(h_words)
    H_T_med, H_W_med = st.median(h_turns), st.median(h_words)

    base = ROOT / "simulation" / "output" / args.annotation_id
    rows = []
    for p in sorted(glob.glob(str(base / "**" / f"{args.version}-up-*.json"), recursive=True)):
        sim = str(Path(p).parent.relative_to(base))
        name = sim[len("local/"):] if sim.startswith("local/") else sim
        convs = load_sim(p)
        if not convs:
            continue
        per_conv_turns, all_words, empty, leaked, capped = [], [], 0, 0, 0
        for c in convs:
            ts = user_turns(c)
            per_conv_turns.append(len(ts))
            capped += len(ts) >= MAX_TURNS
            for t in ts:
                all_words.append(words(t))
                if not t.strip():
                    empty += 1
                if "<think>" in t or "</think>" in t:
                    leaked += 1
        rat = load_ratings(args.annotation_id, sim, Path(p).stem)
        grp, lbl = GROUP.get(name, ("other", name[:30]))
        mt, mw = st.mean(per_conv_turns), (st.mean(all_words) if all_words else 0.0)
        rows.append({
            "group": grp, "label": lbl, "n": len(convs),
            "turns": mt, "words": mw,
            # |deviation| from the real human on the same topics -- lower is more
            # human-like. This is the axis-(1) signal, stated as a gap rather than
            # a raw number so the sign of the error is not hidden by averaging.
            "d_turns": abs(mt - H_T), "d_words": abs(mw - H_W),
            "empty": empty, "leak": leaked,
            "cap_pct": 100.0 * capped / len(convs),
            "doc": rat["document_rating"], "inter": rat["interaction_rating"],
        })

    if not rows:
        print(f"no simulator outputs under {base}")
        return

    # RANKING METRIC -- read this before trusting the order.
    #
    # The previous version normalised each deviation by its COLUMN MAX. That was
    # broken: max Δwords is 602 (an untrained baseline writing essays), so every
    # sane model's word term was divided by 602 and collapsed to ~0.01, leaving
    # the composite 94% turn-count. It ranked Qwen3-8B above GPT-5.5 purely
    # because 5.63 turns is nearer 6.84 than 9.35 is -- an artifact, not a result.
    #
    # Use a scale-free log2 ratio instead: |log2(sim/human)| is 1.0 for "twice the
    # human" and equally 1.0 for "half the human", needs no cross-model
    # normaliser, and so cannot be hijacked by one outlier. A single model's value
    # no longer depends on which other models happen to be in the table.
    #
    # This is still OUR ranking, not SimulatorArena's. Their message-similarity
    # axis is an LLM-judged Likert scale plus a Turing test, neither of which is
    # in the public release. Treat lgap as a descriptive summary of Table-5-style
    # statistics, not as a reproduction of their metric.
    for r in rows:
        lt = abs(math.log2(max(r["turns"], 1e-6) / H_T))
        lw = abs(math.log2(max(r["words"], 1e-6) / H_W))
        r["gap"] = 100 * (lt + lw) / 2
        r["lg_turns"], r["lg_words"] = lt, lw

    w = 30
    hdr = (f"{'simulator':<{w}}{'n':>4}{'doc':>7}{'inter':>7}{'turns':>8}{'cap%':>6}"
           f"{'Δturns':>8}{'words/msg':>11}{'Δwords':>8}{'gap':>7}{'leak':>6}")
    print(f"\nSimulatorArena document-creation, {args.annotation_id} "
          f"({len(ann)} topics), assistant=gpt-5.5")
    print(f"HUMAN reference: {H_T:.2f} turns/conv (median {H_T_med:.0f}), "
          f"{H_W:.1f} words/msg (median {H_W_med:.0f})\n")
    print(hdr)
    print("-" * len(hdr))
    for grp in GROUP_ORDER:
        sel = sorted([r for r in rows if r["group"] == grp], key=lambda r: r["gap"])
        if not sel:
            continue
        print(grp)
        for r in sel:
            # >= rather than = on a fully censored row: the value is a floor.
            dt = f"{r['d_turns']:>7.2f}" + ("+" if r["cap_pct"] >= 99 else " ")
            doc = f"{r['doc'][0]:>7.2f}" if r["doc"] else f"{'-':>7}"
            inter = f"{r['inter'][0]:>7.2f}" if r["inter"] else f"{'-':>7}"
            print(f"  {r['label'][:w-2]:<{w-2}}{r['n']:>4}{doc}{inter}{r['turns']:>8.2f}"
                  f"{r['cap_pct']:>6.0f}{dt}{r['words']:>11.1f}{r['d_words']:>8.1f}"
                  f"{r['gap']:>7.1f}{r['leak']:>6}")

    print(f"\ncap% = conversations stopped by the harness at {MAX_TURNS} user turns "
          "(utils.py:1378),")
    print("i.e. the simulator never chose to end. A trailing '+' marks a fully")
    print("censored Δturns -- a floor, not an estimate.")
    print("\ndoc / inter = mean LLM-judge rating (1-10) of the produced document and")
    print("of the interaction. '-' = that simulator has not been judged yet.")
    print("\nΔ = |simulator - human| on the same topics; gap = mean of the two Δs,")
    print("each normalised by its column max. LOWER IS MORE HUMAN-LIKE.")
    print("This is SimulatorArena's message-similarity axis only, computed from")
    print("behavioural statistics -- NOT their LLM-judged Turing test, and NOT the")
    print("headline rho, which needs human ratings of 9 unreachable assistants.")

    cols = ["group", "label", "n", "doc_rating", "interaction_rating", "turns",
            "cap_pct", "d_turns", "words", "d_words", "gap", "empty", "leak"]
    for r in rows:
        r["doc_rating"] = f"{r['doc'][0]:.4f}" if r["doc"] else ""
        r["interaction_rating"] = f"{r['inter'][0]:.4f}" if r["inter"] else ""
    out = Path(args.csv)
    with out.open("w") as f:
        f.write(",".join(cols) + "\n")
        f.write(f"Human,HUMAN REFERENCE,{len(ann)},,,{H_T:.4f},0,0,{H_W:.4f},0,0,0,0\n")
        for r in sorted(rows, key=lambda r: r["gap"]):
            f.write(",".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                             for c in cols) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
