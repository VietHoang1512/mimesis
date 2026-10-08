#!/usr/bin/env python3
"""RealUserSim PT3: evaluate a user simulator against the real human it imitates."""
from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "RealUserSim" / "evaluation"
SPLITS = ["mixed_domain", "business_finance", "ecommerce",
          "medical_health", "technology_it", "travel_hospitality"]
CONDITIONS = ["task_only", "correct_profile", "shuffled_profile"]

DIMS = [
    "Persona_Affective_Traits_Match",
    "Linguistic_Style_Mechanics_Match",
    "Technical_Competency_Knowledge_Match",
    "Interaction_Data_Flow_Match",
    "Pacing_Action_Sequencing_Match",
]
SHORT = {
    "Persona_Affective_Traits_Match": "D1_persona",
    "Linguistic_Style_Mechanics_Match": "D2_style",
    "Technical_Competency_Knowledge_Match": "D3_technical",
    "Interaction_Data_Flow_Match": "D4_interaction",
    "Pacing_Action_Sequencing_Match": "D5_pacing",
}

MAX_AGENT_TURNS = 4
JUDGE_MSG_TRUNC = 9
LOCAL_PREFIX = "local/"

_PRIMER = "Start the conversation. Send your first message to the AI assistant."


# --------------------------------------------------------------------------- data

def load_cases(splits=None, limit=None, dedup=False):
    """Load PT3 cases, using the field names from the dataset."""
    out = []
    for sp in (splits or SPLITS):
        p = DATA / "test_sets" / f"{sp}.jsonl"
        for line in p.open():
            r = json.loads(line)
            prof = r.get("user_profile") or {}
            out.append({
                "case_id": f"{sp}-{r['test_id']}",
                "split": sp,
                "domain": r.get("domain", ""),
                "task_type": r.get("task_type", ""),
                "user_ip": r.get("user_ip", ""),
                "conversation_hash": r.get("conversation_hash", ""),
                "original_messages": r["original_messages"],
                "demographics": prof.get("demographics") or {},
                "linguistic_profile": prof.get("linguistic_profile") or "",
                "problem_desc": r.get("problem_desc") or r.get("user_goal") or "",
                "solution_conditions": r.get("solution_conditions") or r.get("key_context") or "",
            })
    if dedup:
        seen, keep = set(), []
        for c in out:
            if c["conversation_hash"] in seen:
                continue
            seen.add(c["conversation_hash"])
            keep.append(c)
        out = keep
    for c in out:
        missing = [k for k in ("problem_desc", "solution_conditions", "linguistic_profile")
                   if not c[k]]
        if missing:
            raise ValueError(f"{c['case_id']}: empty required field(s) {missing}")
    return out[:limit] if limit else out


def shuffle_map(cases, seed):
    """Derangement of case -> profile-donor case. No case keeps its own profile."""
    ids = [c["case_id"] for c in cases]
    order = list(ids)
    rng = random.Random(seed)
    for i in range(len(order) - 1, 0, -1):
        j = rng.randrange(i)
        order[i], order[j] = order[j], order[i]
    m = dict(zip(ids, order))
    assert all(k != v for k, v in m.items()), "derangement invariant broken"
    return m


# ------------------------------------------------------------------------ prompts

_DEMO_LABELS = {
    "age": "Age range", "gender": "Gender", "occupation": "Occupation",
    "education": "Education", "location": "Location/Nationality",
    "income": "Income level", "marital_status": "Marital status",
}


def _demographics_summary(demographics):
    parts = []
    for field, label in _DEMO_LABELS.items():
        info = demographics.get(field, {})
        value = info.get("value") if isinstance(info, dict) else info
        if value and str(value).lower() not in ("null", "none", "unknown", ""):
            parts.append(f"- {label}: {value}")
    return "\n".join(parts) or "No specific demographic information available."


def build_user_prompt(case, condition, donor=None):
    """The dataset card's build_user_prompt(), verbatim for correct_profile."""
    profiled = condition in ("correct_profile", "shuffled_profile")
    src = donor if condition == "shuffled_profile" else case
    if condition == "shuffled_profile" and donor is None:
        raise ValueError("shuffled_profile needs a donor case")

    if profiled:
        head = (
            "You are a linguistic mimic. Your goal is to follow a set of profile commands "
            "and mimic a USER's exact texting persona in a **new context**.\n"
            "You are not an AI assistant in this mode; you are a digital twin of this specific person.\n"
            "The new context contains a new request description and the solution conditions.\n\n"
            "Constraint Rules:\n"
            "- Do not \"clean up\" the writing. If the commands require poor grammar and frequent typos, "
            "your response must be equally messy. If you write perfectly, you have failed the task.\n"
            "- Do NOT directly copy artifacts from the examples in the commands. They are from a different context.\n"
        )
    else:
        head = (
            "You are simulating a USER seeking help from an AI assistant in a **new context**.\n"
            "You are not an AI assistant in this mode; you are the person asking for help.\n"
            "The new context contains a new request description and the solution conditions.\n"
        )

    body = (
        "\nBehavioral Instructions:\n"
        "- Stay in character. You are a USER seeking for help, not an assistant that helps others.\n"
        "- Adhere to available information in the request description. DO NOT make up information.\n"
        "- Make sure the information you share is accurate and consistent with the description.\n"
        "- DO NOT share or mention about the solution conditions to the assistant.\n"
        "- Use the solution conditions to decide if the request is addressed or not.\n"
        "- If request is addressed based on the solution conditions, please reply '/close' to end the chat.\n"
    )
    if profiled:
        body += (
            f"\nDemographic background:\n{_demographics_summary(src['demographics'])}\n"
            f"\nProfile Commands:\n{src['linguistic_profile']}\n"
        )
    body += (
        f"\n- New context (request description)\n{case['problem_desc']}\n"
        f"- New context (solution conditions)\n{case['solution_conditions']}"
    )
    return head + body


# ---------------------------------------------------------------------- transport

_PROSE_COT = re.compile(
    r"^\s{0,4}(?:\*{0,2})(thinking process|thought process|chain of thought|reasoning)"
    r"(?:\*{0,2})\s*:?\s*\n", re.I)
_COT_HANDOFF = re.compile(r"\n\s*(?:\*{0,2})(message|response|final answer|output|user message|reply)"
                          r"(?:\*{0,2})\s*:\s*", re.I)


class ReasoningOnlyTurn(RuntimeError):
    """The model emitted reasoning and no user message."""


def strip_reasoning(text, strict=False):
    """Remove a model's chain of thought before the turn leaves the simulator."""
    if not text:
        return ""
    raw = text
    try:
        from agents.utils import remove_think
        out = (remove_think(text) or "").strip()
    except Exception:
        out = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        out = re.sub(r"^.*?</think>", "", out, flags=re.S).strip()

    if _PROSE_COT.match(out):
        m = _COT_HANDOFF.search(out)
        out = out[m.end():].strip() if m else ""

    if not out and raw.strip():
        if strict:
            raise ReasoningOnlyTurn(
                f"turn was reasoning-only, nothing to send ({len(raw)} chars): {raw[:120]!r}")
        return ""
    return out


def _local_client():
    """OpenAI SDK client for a locally served checkpoint."""
    import httpx
    from openai import AsyncOpenAI
    port = os.environ.get("LOCAL_VLLM_PORT", "8300")
    return AsyncOpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1",
                       max_retries=3, http_client=httpx.AsyncClient(trust_env=False, timeout=600.0))


class Chat:
    """One call interface over both routes, so simulate/judge never branch on model."""

    def __init__(self, model, temperature=0.7, max_tokens=1024,
                 enable_thinking=None, reasoning_effort="none"):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.local = model.startswith(LOCAL_PREFIX)
        self.served = model[len(LOCAL_PREFIX):] if self.local else model
        self._client = _local_client() if self.local else None

    async def __call__(self, messages, max_tokens=None, temperature=None):
        mt = max_tokens or self.max_tokens
        tp = self.temperature if temperature is None else temperature
        if self.local:
            extra = {}
            if self.enable_thinking is not None:
                extra["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": bool(self.enable_thinking)}}
            r = await self._client.chat.completions.create(
                model=self.served, messages=messages,
                temperature=tp, max_tokens=mt, **extra)
            return (r.choices[0].message.content or "") if r.choices else ""
        import api
        return await api.chat_async(
            messages, model=self.served, max_tokens=mt, max_retries=20,
            reasoning_effort=self.reasoning_effort)


# --------------------------------------------------------------------- simulation

def _has_close(text):
    return "/close" in (text or "")


async def run_case(case, condition, sim, assistant, donor=None):
    """One simulated conversation. Mirrors the dataset card's simulate_conversation."""
    system = build_user_prompt(case, condition, donor)
    sim_msgs = [{"role": "system", "content": system},
                {"role": "user", "content": _PRIMER}]

    first = strip_reasoning(await sim(sim_msgs), strict=True)
    sim_msgs.append({"role": "assistant", "content": first})
    transcript = [{"role": "user", "content": first}]
    reason = "max_turns"

    if _has_close(first):
        reason = "close_token"
    else:
        for _ in range(MAX_AGENT_TURNS):
            reply = await assistant([{"role": m["role"], "content": m["content"]}
                                     for m in transcript])
            transcript.append({"role": "assistant", "content": reply})

            sim_msgs.append({"role": "user", "content": reply})
            nxt = strip_reasoning(await sim(sim_msgs), strict=True)
            sim_msgs.append({"role": "assistant", "content": nxt})
            transcript.append({"role": "user", "content": nxt})
            if _has_close(nxt):
                reason = "close_token"
                break

    n_user = sum(1 for m in transcript if m["role"] == "user")
    return {
        "case_id": case["case_id"], "split": case["split"], "domain": case["domain"],
        "user_ip": case["user_ip"], "conversation_hash": case["conversation_hash"],
        "condition": condition,
        "profile_case_id": (donor or case)["case_id"] if condition != "task_only" else None,
        "termination_reason": reason,
        "messages": transcript,
        "n_user_messages": n_user,
        "n_assistant_messages": len(transcript) - n_user,
        "n_user_chars": sum(len(m["content"]) for m in transcript if m["role"] == "user"),
        "empty_user_turns": sum(1 for m in transcript
                                if m["role"] == "user" and not m["content"].strip()),
        "system_prompt": system,
    }


# --------------------------------------------------------------------------- judge

def _judge_prompts():
    txt = (DATA / "judge_prompt.txt").read_text()
    system = txt.split("FORMATTING PROMPT")[0].strip()
    system = system.replace("SYSTEM PROMPT\n=============\n", "")
    fmt = txt.split(
        "FORMATTING PROMPT (applied to raw judge output)\n"
        "===============================================\n")[1].strip()
    return system, fmt


def prepare_trajectories(original_messages, synthetic_messages):
    """Released evaluate_fidelity() truncation, reproduced exactly."""
    orig = [m for m in original_messages[:JUDGE_MSG_TRUNC] if m["role"] == "user"]
    synth = [{"role": m["role"], "content": m["content"].replace("/close", "").strip()}
             for m in synthetic_messages[:JUDGE_MSG_TRUNC]
             if m["role"] == "user" and m["content"].replace("/close", "").strip()]
    return orig, synth


def _parse_verdict(text):
    """Parse the formatter's JSON into five booleans, or raise."""
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        raise ValueError(f"no JSON object in judge output: {s[:200]!r}")
    obj = json.loads(s[i:j + 1])
    out = {}
    for d in DIMS:
        if d not in obj:
            raise ValueError(f"missing dimension {d}")
        v = obj[d]
        v = v.get("is_matched") if isinstance(v, dict) else v
        if isinstance(v, str):
            v = {"true": True, "yes": True, "false": False, "no": False}.get(v.strip().lower())
        if not isinstance(v, bool):
            raise ValueError(f"{d}: non-boolean is_matched {obj[d]!r}")
        out[SHORT[d]] = v
    return out


async def judge_one(rec, case, judge, system, fmt):
    orig, synth = prepare_trajectories(case["original_messages"], rec["messages"])
    if not synth:
        return {"error": "empty_synthetic_trajectory"}
    user_msg = (
        "Here are the user's messages from two conversations:\n"
        f"CONVERSATION 1 (USER1's messages):\n{json.dumps(orig, indent=2)}\n\n"
        f"CONVERSATION 2 (USER2's messages):\n{json.dumps(synth, indent=2)}"
    )
    raw = await judge([{"role": "system", "content": system},
                       {"role": "user", "content": user_msg}],
                      max_tokens=4096, temperature=0.001)
    for attempt in range(2):
        note = "" if attempt == 0 else "\n\nReturn ONLY the JSON object, no prose."
        js = await judge([{"role": "system", "content": fmt + note},
                          {"role": "user", "content": raw}],
                         max_tokens=1024, temperature=0.001)
        try:
            v = _parse_verdict(js)
            v["_raw"] = raw
            v["_n_user_orig"] = len(orig)
            v["_n_user_synth"] = len(synth)
            return v
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
    return {"error": f"unparseable_judgment: {last}", "_raw": raw}


# ------------------------------------------------------------------------- jsonl io

def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    for line in p.open():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def dedup_cases(rows):
    """Collapse repeated case_ids, keeping the last write."""
    seen = {}
    for r in rows:
        seen[r.get("case_id")] = r
    return list(seen.values())


class Appender:
    """Append and fsync each record, so a preemption loses one case, not the run."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a")

    def write(self, rec):
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())

    def close(self):
        self.fh.close()


def stem_for(sim_model, condition, seed):
    return f"{sim_model.replace('/', '_').replace(':', '_')}-{condition}-seed{seed}"


# ------------------------------------------------------------------------ commands

def cmd_inspect(args):
    cases = load_cases()
    print(f"cases: {len(cases)}  splits: {len(SPLITS)}")
    print(f"unique conversation_hash: {len({c['conversation_hash'] for c in cases})}")
    print(f"unique user_ip: {len({c['user_ip'] for c in cases})}")
    nu = collections.Counter(
        len([m for m in c["original_messages"][:JUDGE_MSG_TRUNC] if m["role"] == "user"])
        for c in cases)
    print(f"human user-turns under [:{JUDGE_MSG_TRUNC}]: {sorted(nu.items())}")
    print(f"domains: {len({c['domain'] for c in cases})}")
    print("\n--- correct_profile system prompt, case 0 ---")
    print(build_user_prompt(cases[0], "correct_profile")[:1500])
    print("\n--- task_only system prompt, case 0 ---")
    print(build_user_prompt(cases[0], "task_only")[:900])


def cmd_simulate(args):
    cases = load_cases(args.splits, args.limit)
    donors = None
    if args.condition == "shuffled_profile":
        m = shuffle_map(cases, args.seed)
        by_id = {c["case_id"]: c for c in cases}
        donors = {k: by_id[v] for k, v in m.items()}
        mp = Path(args.out_dir) / f"profile_shuffle_seed{args.seed}.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(m, indent=2))

    stem = stem_for(args.sim_model, args.condition, args.seed)
    out = Path(args.out_dir) / f"{stem}_gen.jsonl"
    done = {r["case_id"] for r in read_jsonl(out)}
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"[simulate] {args.sim_model} / {args.condition} / seed{args.seed}: "
          f"{len(done)} done, {len(todo)} to go -> {out}")
    if not todo:
        return

    sim = Chat(args.sim_model, args.sim_temperature, args.sim_max_tokens,
               enable_thinking=args.enable_thinking, reasoning_effort=args.reasoning_effort)
    assistant = Chat(args.assistant_model, args.assistant_temperature,
                     args.assistant_max_tokens, reasoning_effort=args.reasoning_effort)
    app, errs = Appender(out), Appender(Path(args.out_dir) / "errors.jsonl")
    sem = asyncio.Semaphore(args.workers)
    t0, n_ok = time.time(), 0

    async def one(c):
        nonlocal n_ok
        async with sem:
            try:
                rec = await run_case(c, args.condition, sim, assistant,
                                     donors.get(c["case_id"]) if donors else None)
                rec.update(simulator_model=args.sim_model, assistant_model=args.assistant_model,
                           seed=args.seed,
                           metadata={"simulator_temperature": args.sim_temperature,
                                     "assistant_temperature": args.assistant_temperature,
                                     "max_agent_turns": MAX_AGENT_TURNS,
                                     "reasoning_effort": args.reasoning_effort})
                app.write(rec)
                n_ok += 1
                if n_ok % 25 == 0:
                    el = time.time() - t0
                    print(f"  {n_ok}/{len(todo)}  {el/60:.1f}m  "
                          f"eta {(len(todo)-n_ok)*el/max(n_ok,1)/60:.0f}m", flush=True)
            except Exception as e:
                errs.write({"case_id": c["case_id"], "condition": args.condition,
                            "stage": "simulate", "error": f"{type(e).__name__}: {str(e)[:400]}",
                            "ts": time.strftime("%F %T")})

    asyncio.run(_gather(todo, one))
    app.close()
    errs.close()
    print(f"[simulate] wrote {n_ok}/{len(todo)} in {(time.time()-t0)/60:.1f}m")


def cmd_judge(args):
    gen = Path(args.gen)
    recs = dedup_cases(read_jsonl(gen))
    by_case = {c["case_id"]: c for c in load_cases()}
    out = gen.with_name(gen.name.replace("_gen.jsonl", "_judged.jsonl"))
    done = {r["case_id"] for r in read_jsonl(out)}
    todo = [r for r in recs if r["case_id"] not in done]
    print(f"[judge] {gen.name}: {len(recs)} generations, {len(done)} judged, {len(todo)} to go")
    if not todo:
        return

    system, fmt = _judge_prompts()
    judge = Chat(args.judge_model, 0.001, 4096, reasoning_effort=args.reasoning_effort)
    app, errs = Appender(out), Appender(gen.parent / "errors.jsonl")
    sem = asyncio.Semaphore(args.workers)
    t0, n = time.time(), 0

    async def one(rec):
        nonlocal n
        async with sem:
            try:
                v = await judge_one(rec, by_case[rec["case_id"]], judge, system, fmt)
            except Exception as e:
                v = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            if "error" in v:
                errs.write({"case_id": rec["case_id"], "stage": "judge", "error": v["error"],
                            "ts": time.strftime("%F %T")})
            app.write({"case_id": rec["case_id"], "split": rec["split"], "domain": rec["domain"],
                       "user_ip": rec["user_ip"], "condition": rec["condition"],
                       "termination_reason": rec["termination_reason"],
                       "n_user_messages": rec["n_user_messages"],
                       "n_user_chars": rec["n_user_chars"],
                       "judge_model": args.judge_model, **v})
            n += 1
            if n % 25 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el/60:.1f}m  "
                      f"eta {(len(todo)-n)*el/max(n,1)/60:.0f}m", flush=True)

    asyncio.run(_gather(todo, one))
    app.close()
    errs.close()
    print(f"[judge] {n} judged in {(time.time()-t0)/60:.1f}m -> {out}")


async def _gather(items, fn):
    await asyncio.gather(*[fn(x) for x in items])


# ---------------------------------------------------------------------- aggregate

def cluster_bootstrap(rows, keyfn, n_boot=10000, seed=1234):
    """Percentile CI for the Fidelity Index, resampling users, not cases."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[keyfn(r)].append(r)
    keys = list(groups)
    if not keys:
        return (float("nan"),) * 2
    flat = [np.array([r["_hits"] for r in groups[k]]).sum() for k in keys]
    tot = [len(groups[k]) * 5 for k in keys]
    flat, tot = np.array(flat, float), np.array(tot, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    est = flat[idx].sum(1) / np.maximum(tot[idx].sum(1), 1)
    return float(np.percentile(est, 2.5)) * 100, float(np.percentile(est, 97.5)) * 100


def _load_judged(d):
    runs = {}
    for p in sorted(glob.glob(str(Path(d) / "*_judged.jsonl"))):
        rows = [r for r in dedup_cases(read_jsonl(p)) if "error" not in r]
        if not rows:
            continue
        for r in rows:
            r["_hits"] = sum(bool(r.get(SHORT[k])) for k in DIMS)
        stem = Path(p).name.replace("_judged.jsonl", "")
        runs[stem] = rows
    return runs


def cmd_aggregate(args):
    runs = _load_judged(args.dir)
    if not runs:
        print(f"no *_judged.jsonl under {args.dir}")
        return
    short = [SHORT[k] for k in DIMS]

    grouped = collections.defaultdict(dict)
    for stem, rows in runs.items():
        m = re.match(r"^(.*)-seed(\d+)$", stem)
        base, sd = (m.group(1), int(m.group(2))) if m else (stem, 0)
        grouped[base][sd] = rows

    out = []
    for base, per_seed in sorted(grouped.items()):
        fi, dims = [], collections.defaultdict(list)
        for rows in per_seed.values():
            fi.append(100 * sum(r["_hits"] for r in rows) / (5 * len(rows)))
            for s in short:
                dims[s].append(100 * np.mean([bool(r.get(s)) for r in rows]))
        allrows = [r for rows in per_seed.values() for r in rows]
        lo, hi = cluster_bootstrap(allrows, lambda r: r["user_ip"],
                                   args.bootstrap, args.bootstrap_seed)
        out.append({
            "run": base, "n_seeds": len(per_seed),
            "n_cases": int(np.mean([len(r) for r in per_seed.values()])),
            "FI": float(np.mean(fi)),
            "FI_sem": float(np.std(fi, ddof=1) / np.sqrt(len(fi))) if len(fi) > 1 else 0.0,
            "ci_lo": lo, "ci_hi": hi,
            **{s: float(np.mean(dims[s])) for s in short},
            "close_pct": 100 * np.mean([r["termination_reason"] == "close_token" for r in allrows]),
            "user_turns": float(np.mean([r["n_user_messages"] for r in allrows])),
        })
    out.sort(key=lambda r: -r["FI"])

    w = 46
    hdr = f"{'run':<{w}}{'n':>3}{'seeds':>6}{'FI':>8}{'95% CI':>16}" + \
          "".join(f"{s.split('_')[0]:>7}" for s in short) + f"{'close%':>8}{'turns':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in out:
        pm = f"±{r['FI_sem']:.1f}" if r["n_seeds"] > 1 else "    "
        ci = f"[{r['ci_lo']:.1f},{r['ci_hi']:.1f}]"
        print(f"{r['run'][:w-1]:<{w}}{r['n_cases']:>3}{r['n_seeds']:>6}"
              f"{r['FI']:>6.1f}{pm}{ci:>16}"
              + "".join(f"{r[s]:>7.1f}" for s in short)
              + f"{r['close_pct']:>8.1f}{r['user_turns']:>7.2f}")

    print("\nFI = Fidelity Index, % of dimension-judgments matched (5 per case).")
    print("95% CI = percentile cluster bootstrap over user_ip; ± on FI = SEM across seeds.")
    print("PUBLISHED REFERENCE (gpt-4o sim + gpt-4o assistant + gpt-4o judge, NOT run here):")
    print("  profile 45.3  (D1 39.0  D2 26.2  D3 93.2  D4 36.0  D5 32.0) | task-only 24.2")
    print("Our assistant/judge are gpt-5.5,")
    print("so absolute FI is not on the published scale. Compare rows here to each other.")

    csv = Path(args.dir) / "realusersim_results.csv"
    cols = ["run", "n_seeds", "n_cases", "FI", "FI_sem", "ci_lo", "ci_hi",
            *short, "close_pct", "user_turns"]
    with csv.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in out:
            f.write(",".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                             for c in cols) + "\n")
    print(f"\nwrote {csv}")


# ------------------------------------------------------------------------------ cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inspect", help="dataset stats + rendered prompts").set_defaults(fn=cmd_inspect)

    s = sub.add_parser("simulate", help="roll a simulator through PT3 conversations")
    s.add_argument("--sim-model", required=True,
                   help="gateway name (claude-opus-5) or local/<served-model-name>")
    s.add_argument("--condition", default="correct_profile", choices=CONDITIONS)
    s.add_argument("--assistant-model", default=os.environ.get("RUS_ASSISTANT", "gpt-5.5"))
    s.add_argument("--splits", nargs="*", default=None, choices=SPLITS)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--sim-temperature", type=float, default=0.7)
    s.add_argument("--sim-max-tokens", type=int, default=1024)
    s.add_argument("--assistant-temperature", type=float, default=0.7)
    s.add_argument("--assistant-max-tokens", type=int, default=8192)
    s.add_argument("--enable-thinking", type=lambda v: v == "1", default=None,
                   help="1/0 to force the chat template's thinking mode; omit to leave default")
    s.add_argument("--reasoning-effort", default="none",
                   help="gateway reasoning effort for every role. Default \"none\": gpt-4o,\n"
                        "which this substitutes for, does not reason.")
    s.add_argument("--out-dir", default="outputs/realusersim")
    s.set_defaults(fn=cmd_simulate)

    j = sub.add_parser("judge", help="run the released PT3 judge over generations")
    j.add_argument("--gen", required=True)
    j.add_argument("--judge-model", default=os.environ.get("RUS_JUDGE", "gpt-5.5"))
    j.add_argument("--workers", type=int, default=6)
    j.add_argument("--reasoning-effort", default="none",
                   help="gateway reasoning effort for every role. Default \"none\": gpt-4o,\n"
                        "which this substitutes for, does not reason.")
    j.set_defaults(fn=cmd_judge)

    a = sub.add_parser("aggregate", help="leaderboard from *_judged.jsonl")
    a.add_argument("--dir", default="outputs/realusersim")
    a.add_argument("--bootstrap", type=int, default=10000)
    a.add_argument("--bootstrap-seed", type=int, default=1234)
    a.set_defaults(fn=cmd_aggregate)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
