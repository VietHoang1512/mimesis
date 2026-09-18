#!/usr/bin/env python3
"""Score an MIMESIS user-simulator checkpoint with Turing-RL's pairwise Turing judge.

"Path B": turing-rl's data builder + turing-rl's judge, but OUR generation stack.
We deliberately do not go through their eval wrappers --
``turing-rl/shared/model_ids.py`` rejects any model outside {Qwen3-8B,
Qwen3.5-397B}, ``bash_scripts/eval/_eval_common.sh`` validates ``model`` against
a three-item list, and ``eval/generate_trained.py`` hunts for a PEFT
``adapter_config.json``. MIMESIS RL trains full-parameter (the LoRA flags in
run_rl.sh are commented out), so there is no adapter to find. Their judge, by
contrast, is model-agnostic: TURING_PROMPT takes four strings and nothing else.

Two stages:

  generate  Read turing-rl's heldout test.parquet, roll the served model over
            every (user_history, context) prompt, write generations JSON.
  score     Pair each generation against the real human utterance and hand both
            to the Turing judge, which rates 1-7 which one a human wrote.

The judge runs on the judge endpoint (gpt-5.6) instead of the paper's OpenRouter/Sonnet-4.6
setup, so absolute numbers are NOT comparable to the paper. They are comparable
across models scored by this script with the same judge.

Prereq -- build the heldout set once (PRISM is the cheap one: no 23GB ConvoKit
download, 880 test rows, and ``history`` skips the persona-induction LLM pass):

    cd turing-rl && bash_scripts/data/generate_data.sh prism qwen3-8b

Then:

    # 1) serve the checkpoint, e.g. vllm serve $CKPT --port 8100
    python turing_eval.py generate \
        --test-parquet turing-rl/data/prism/prism_history_s42_sft40_grpo60/test.parquet \
        --model "$CKPT" --base-url http://localhost:8100/v1 \
        --out outputs/turing/mymodel_gen.json

    # 2) score (needs LLAMA_API_KEY for the judge endpoint)
    python turing_eval.py score --gen outputs/turing/mymodel_gen.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TURING_RL = REPO_ROOT / "turing-rl"

# the judge endpoint's OpenAI-compatible endpoint, already redirect-resolved. The short form
# everything else uses (the configured endpoint see api.py and
# llm/api.py) answers 307 -> /experimental/compat/openai/v1/...;
# the OpenAI SDK follows that, but turing-rl's judge posts with urllib, which
# refuses to redirect a POST and raises HTTPError 307. So point straight at the
# target.
METAGEN_BASE_URL = os.getenv("JUDGE_BASE_URL", "http://localhost:8000/v1")
DEFAULT_JUDGE_MODEL = "gpt-5-6-sol-genai-responses"

# Ratings are 1-7 with 4 = "cannot tell". Canonicalized so HIGHER always means the
# generated response looked more human than the real one (see _canonical_rating).
TIE_RATING = 4


# ──────────────────────────────────────────────────────────────────────────────
# turing-rl bootstrap
# ──────────────────────────────────────────────────────────────────────────────
def _bootstrap_turing_rl(judge_model: str, api_key: str):
    """Import turing-rl's judge and repoint its transport at the judge endpoint.

    Two seams need patching; everything else (TURING_PROMPT, the rubric, the
    n-gram source-copy watchlist, rating parsing, retries) is used verbatim.

      1. ``shared.load_env.load_local_env`` raises FileNotFoundError unless a
         ``.env`` file exists, even when the vars are already exported. We set
         them from the environment, so make it a no-op.
      2. ``openrouter_request_extras`` injects OpenRouter-only routing fields
         (``provider: {order: [...]}`` and ``reasoning: {enabled: true}``) that
         the judge endpoint rejects. Swap in the OpenAI-style ``reasoning_effort``.

    Appended (not prepended) to sys.path: MIMESIS has no top-level ``shared`` /
    ``eval`` / ``data`` package, so this resolves without shadowing our own
    modules.
    """
    if not TURING_RL.is_dir():
        raise SystemExit(f"turing-rl checkout not found at {TURING_RL}")
    if str(TURING_RL) not in sys.path:
        sys.path.append(str(TURING_RL))

    os.environ.setdefault("OPENAI_API_BASE", METAGEN_BASE_URL)
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENROUTER_API_KEY"] = api_key
    os.environ["PERSONA_EVAL_JUDGE_MODEL"] = judge_model
    # the judge endpoint throttles gpt-5.6 on a shared tokens/minute budget (HTTP 429
    # ADMITTANCE_REJECTED), and a judge prompt is large -- rubric + history +
    # context + both responses. Pace request starts globally (see
    # shared/api_client._pace_request); without this a worker pool trips the
    # limit immediately, exactly as persona induction did.
    os.environ.setdefault("PERSONA_OPENAI_MIN_INTERVAL_SECONDS", "2")

    from shared import api_client, load_env

    load_env.load_local_env = lambda *a, **k: None

    effort = os.getenv("TURING_JUDGE_EFFORT", "medium")

    def _judge_request_extras(*, reasoning: bool) -> dict:
        return {"reasoning_effort": effort} if reasoning else {}

    api_client.openrouter_request_extras = _judge_request_extras

    import eval.metrics as metrics

    # 3. Transport. the judge endpoint serves ONLY gpt-5-6-sol-genai-responses -- every
    #    claude-* / gemini-* spelling returns 400 "Invalid model name" (verified
    #    2026-09-14). Both ARE reachable on the hosted endpoint, so a different judge
    #    gets routed through api.py instead. The rubric, TURING_PROMPT, parsing and
    #    n-gram watchlist are untouched -- only the HTTP call changes, so scores
    #    remain comparable across judges.
    if not judge_model.endswith("-genai-responses"):
        import api as _gw

        def _gateway_post(payload, **_kw):
            # chat_async, NOT chat: api.chat takes no max_retries/reasoning_effort
            # (that was a TypeError on all 880 calls) and its docstring says Claude
            # must go through chat_messages. chat_async routes all three families.
            # _score drives this from a ThreadPoolExecutor, so each worker thread
            # owns its own loop and asyncio.run is safe here.
            msgs = payload.get("messages") or []
            # response_format json_object has no portable equivalent across the
            # gateway families; the rubric already demands JSON in the prompt and
            # _parse_turing_response tolerates surrounding prose.
            return asyncio.run(_gw.chat_async(
                msgs, model=payload.get("model", judge_model),
                max_tokens=payload.get("max_completion_tokens", 2048),
                max_retries=20,
                reasoning_effort=os.getenv("TURING_JUDGE_EFFORT", "none"),
            ))

        api_client.post_chat_sync = _gateway_post
        metrics.post_chat_sync = _gateway_post
        print(f"[turing_eval] judge transport: llm/api.py for {judge_model}")

    # metrics.py did `from shared.api_client import openrouter_request_extras`,
    # so the bound name has to be patched too, not just the source module.
    metrics.openrouter_request_extras = _judge_request_extras
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# stage 1: generate
# ──────────────────────────────────────────────────────────────────────────────
def _load_rows(test_parquet: Path, limit: int | None) -> list[dict]:
    """Flatten turing-rl's parquet into the fields the judge needs.

    Schema comes from data/prism/build.py::build_prism_grpo_row --
    ``prompt`` is already a chat-message list (tokenizer-agnostic, so it can go
    straight to any served model), ``reward_model.ground_truth`` is the real
    human utterance, and extra_info carries the history/context plus the ids
    that seed the A/B position hash.
    """
    import pandas as pd

    df = pd.read_parquet(test_parquet)
    rows = []
    for _, r in df.iterrows():
        ei = r["extra_info"]
        rows.append(
            {
                "user_id": str(ei["user_id"]),
                "post_id": str(ei["post_id"]),
                "target_idx": int(ei["target_idx"]),
                "user_history": str(ei["user_history"] or ""),
                "context": str(ei.get("context") or ei.get("thread_context") or ""),
                "ground_truth": str(r["reward_model"]["ground_truth"] or ""),
                "prompt": [dict(m) for m in r["prompt"]],
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def _clean_generation(raw: str) -> tuple[str, str]:
    """Raw completion -> (reasoning, response) as the judge should see it.

    Two strippers, in order. remove_think() first because MIMESIS thinking
    models wrap reasoning in <think>...</think>, which turing-rl's parser knows
    nothing about -- leaving it in hands the judge a chain of thought and the
    Turing test is over before it starts. Then parse_sft_generation for their
    own <reasoning>...</reasoning> / "[HUMAN]:" convention, which also falls
    back sensibly when the model emits neither.
    """
    from shared.sft_prompt_utils import parse_sft_generation

    from agents.utils import remove_think

    parsed = parse_sft_generation(remove_think(raw or ""))
    return str(parsed.get("reasoning") or ""), str(parsed.get("response") or "")


async def _generate(args) -> None:
    sys.path.append(str(TURING_RL))  # for shared.sft_prompt_utils in _clean_generation

    rows = _load_rows(Path(args.test_parquet), args.limit)
    label = args.api_model or args.model
    print(f"[turing_eval] {len(rows)} heldout targets from {args.test_parquet}")

    sem = asyncio.Semaphore(args.workers)

    if args.api_model:
        # API user-sims go through the repo's the hosted endpoint transport, which already
        # routes per family (gpt-* -> Azure Responses, claude-* -> /v1/messages,
        # gemini-* -> Vertex). Same reason run_tau_usi_eval_api.sh reuses CallAPI
        # instead of speaking to each provider directly. No vLLM, no GPU.
        import api

        print(f"[turing_eval] API mode: {args.api_model} (the hosted endpoint)")

        async def complete(messages: list[dict]) -> str:
            return await api.chat_async(
                messages,
                model=args.api_model,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
            )
    else:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=args.api_key or "EMPTY", base_url=args.base_url, max_retries=3)
        print(f"[turing_eval] local mode: {args.model} @ {args.base_url}"
              + (f" seed={args.seed}" if args.seed is not None else " (unseeded)"))

        async def complete(messages: list[dict]) -> str:
            # seed is passed per REQUEST, not per run: vLLM derives each request's
            # sampler state from it, so the same seed reproduces the same 880
            # generations while different seeds give genuinely independent draws
            # at temperature 1.0. Omitted entirely when None so behaviour is
            # unchanged for callers that do not ask for a seed.
            extra = {"seed": args.seed} if args.seed is not None else {}
            resp = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_completion_tokens=args.max_tokens,
                temperature=args.temperature,
                **extra,
            )
            return resp.choices[0].message.content or ""

    async def one(row: dict) -> dict:
        async with sem:
            try:
                raw = await complete(row["prompt"])
            except Exception as e:  # one bad call must not sink the batch
                print(f"[turing_eval] ERROR {row['user_id']}/{row['target_idx']}: {type(e).__name__}: {e}")
                raw = ""
        reasoning, response = _clean_generation(raw)
        return {**row, "raw": raw, "gen_reasoning": reasoning, "generated": response}

    # Resume support, same rationale as _score: on the preemptible QOS a
    # generation pass is routinely killed part-way, and writing only at the end
    # meant a restart redid all 880 rows. Completed rows are checkpointed to
    # <out>.partial.json and skipped on the next attempt.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".partial.json")

    def _rk(r):
        return (str(r.get("user_id")), str(r.get("post_id")), int(r.get("target_idx", -1)))

    recs: list[dict] = []
    if part.exists():
        try:
            recs = json.loads(part.read_text())
            print(f"[turing_eval] resuming: {len(recs)} rows already generated")
        except Exception as e:
            print(f"[turing_eval] ignoring unreadable checkpoint: {type(e).__name__}: {e}")
            recs = []
    have = {_rk(r) for r in recs}
    pending = [r for r in rows if _rk(r) not in have]
    if len(pending) != len(rows):
        print(f"[turing_eval] {len(pending)} rows left to generate")

    # Checkpoint as results land rather than after gather(), so a kill keeps them.
    done_n = 0

    async def one_ckpt(row: dict) -> dict:
        nonlocal done_n
        r = await one(row)
        recs.append(r)
        done_n += 1
        if done_n % 50 == 0:
            tmp = part.with_suffix(".tmp")
            tmp.write_text(json.dumps(recs))
            tmp.replace(part)
            print(f"[turing_eval]   {done_n}/{len(pending)} generated "
                  f"({len(recs)} total, checkpointed)", flush=True)
        return r

    await asyncio.gather(*[one_ckpt(r) for r in pending])

    payload = {
        "model": label,
        "test_parquet": str(args.test_parquet),
        "n": len(recs),
        "records": recs,
    }
    out.write_text(json.dumps(payload))
    part.unlink(missing_ok=True)
    part.with_suffix(".tmp").unlink(missing_ok=True)
    empty = sum(1 for r in recs if not r["generated"].strip())
    print(f"[turing_eval] wrote {out}  ({len(recs)} records, {empty} empty after cleaning)")


# ──────────────────────────────────────────────────────────────────────────────
# stage 2: score
# ──────────────────────────────────────────────────────────────────────────────
def _canonical_rating(rating: int, generated_is_b: bool) -> int:
    """Fold the randomized A/B ordering back to "higher = model looks more human".

    Mirrors eval/metrics.py: when the generated response sat at B the raw rating
    already points that way; when it sat at A the scale is mirrored (8 - r).
    """
    return int(rating) if generated_is_b else 8 - int(rating)


def _score(args) -> None:
    api_key = os.getenv("LLAMA_API_KEY", "")
    if not api_key:
        raise SystemExit("LLAMA_API_KEY is not set (needed for the judge endpoint)")
    metrics = _bootstrap_turing_rl(args.judge_model, api_key)
    from shared.judge_utils import _stable_turing_generated_is_b

    payload = json.loads(Path(args.gen).read_text())
    records = payload["records"][: args.limit] if args.limit else payload["records"]

    # Resume support. A scoring pass is ~77 min and the preemptible QOS kills jobs
    # after ~50, so writing results only at the end meant every interrupted run
    # restarted from item 0 and could never finish. Judged items are checkpointed
    # to <stem>_turing_items.partial.json and reloaded here, keyed by the item id
    # triple, so a restart resumes instead of redoing.
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.gen).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.gen).name.replace("_gen.json", "")
    partial_path = out_dir / f"{stem}_turing_items.partial.json"

    def _key(r):
        return (str(r.get("user_id")), str(r.get("post_id")), int(r.get("target_idx", -1)))

    results: list[dict] = []
    done: set = set()
    if partial_path.exists():
        try:
            results = json.loads(partial_path.read_text())
            done = {_key(r) for r in results}
            print(f"[turing_eval] resuming: {len(done)} already judged "
                  f"(from {partial_path.name})")
        except Exception as e:  # a truncated checkpoint must not block the run
            print(f"[turing_eval] ignoring unreadable checkpoint: {type(e).__name__}: {e}")
            results, done = [], set()

    todo = [r for r in records if _key(r) not in done]
    print(f"[turing_eval] scoring {len(todo)} records "
          f"({len(done)} cached) with judge={args.judge_model}")

    def judge(rec: dict) -> dict:
        gen = rec["generated"]
        gt = rec["ground_truth"]
        if not rec["user_history"].strip():
            return {**rec, "skipped": "empty user_history"}
        # Deterministic per-item A/B assignment -- reruns see identical orderings.
        gen_is_b = _stable_turing_generated_is_b(
            gen, user_id=rec["user_id"], post_id=rec["post_id"], target_idx=rec["target_idx"]
        )
        a, b = (gt, gen) if gen_is_b else (gen, gt)
        try:
            det = metrics._turing_api_call(
                rec["context"], a, b, user_history=rec["user_history"], return_details=True
            )
        except Exception as e:
            return {**rec, "skipped": f"{type(e).__name__}: {e}"}
        raw_rating = int(det.get("rating", TIE_RATING))
        parse_error = bool(det.get("parse_error"))
        suffix = "b" if gen_is_b else "a"  # penalties are reported per position
        return {
            "user_id": rec["user_id"],
            "post_id": rec["post_id"],
            "target_idx": rec["target_idx"],
            "generated": gen,
            "ground_truth": gt,
            "generated_is_b": gen_is_b,
            "raw_rating": raw_rating,
            "rating": _canonical_rating(raw_rating, gen_is_b),
            "parse_error": parse_error,
            **{
                k: float(det.get(f"{k}_penalty_{suffix}", 0.0) or 0.0)
                for k in (
                    "assistant_like",
                    "source_copy",
                    "wrong_target_or_role",
                    "unsupported_adversarial_reframing",
                )
            },
            "judge_reasoning": str(det.get("reasoning") or "")[:1500],
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(judge, r) for r in todo]
        for i, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            if i % 50 == 0:
                # Checkpoint every 50. Written to a temp file and renamed so a kill
                # mid-write cannot leave a truncated JSON that the resume path would
                # then have to discard -- which would defeat the point.
                tmp = partial_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(results))
                tmp.replace(partial_path)
                print(f"[turing_eval]   {i}/{len(futures)} judged "
                      f"({len(results)} total, checkpointed)", flush=True)

    scored = [r for r in results if "skipped" not in r]

    # A total judge outage (the judge endpoint unreachable from the compute node, every call
    # exhausting its 8 retries) otherwise sails right through: every mean below
    # is nan, the summary gets written anyway, and run_turing_eval.sh then says
    # "skip score (exists)" on every future run -- so the model is pinned at nan
    # until someone deletes the file by hand. Fail instead of poisoning the dir.
    if not scored:
        reasons = Counter(str(r.get("skipped")).split(":")[0] for r in results)
        raise SystemExit(
            f"[turing_eval] all {len(results)} judge calls failed ({dict(reasons)}) -- "
            "refusing to write an all-nan summary. Fix the judge endpoint and re-run."
        )

    ok = [r for r in scored if not r["parse_error"]]
    ratings = [r["rating"] for r in ok]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    # GT accuracy: did the judge correctly finger the real human? Ties excluded,
    # matching how the paper binarizes for the human-vs-LLM comparison.
    decided = [r for r in ok if r["rating"] != TIE_RATING]
    summary = {
        "model": payload.get("model"),
        "judge_model": args.judge_model,
        "n_records": len(records),
        "n_scored": len(scored),
        "n_parse_error": len(scored) - len(ok),
        "n_skipped": len(results) - len(scored),
        # 1-7, higher = generated response looked more human than the real one
        "turing_rating_mean": mean(ratings),
        # the paper's training reward: (min(s,5)-1)/6, capped to punish "more human than human"
        "turing_reward_mean": mean([(min(s, 5) - 1) / 6 for s in ratings]),
        "tie_rate": mean([1.0 if r["rating"] == TIE_RATING else 0.0 for r in ok]),
        "gt_accuracy": mean([1.0 if r["rating"] < TIE_RATING else 0.0 for r in decided]),
        # D1 diagnostics: how often the judge flagged the generation as chatbot-like
        "assistant_like_penalty_mean": mean([r["assistant_like"] for r in ok]),
        "source_copy_penalty_mean": mean([r["source_copy"] for r in ok]),
        "wrong_target_or_role_penalty_mean": mean([r["wrong_target_or_role"] for r in ok]),
        "rating_histogram": {str(k): sum(1 for s in ratings if s == k) for k in range(1, 8)},
    }

    out_dir = Path(args.out_dir or Path(args.gen).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.gen).stem.replace("_gen", "")
    (out_dir / f"{stem}_turing_items.json").write_text(json.dumps(results, indent=2))
    (out_dir / f"{stem}_turing_summary.json").write_text(json.dumps(summary, indent=2))
    # The run is complete, so the resume checkpoint is now redundant; leaving it
    # would make a later re-score of this stem silently reuse stale judgements.
    partial_path.unlink(missing_ok=True)
    partial_path.with_suffix(".tmp").unlink(missing_ok=True)

    print("\n" + "=" * 62)
    for k, v in summary.items():
        print(f"  {k:36s} {v if not isinstance(v, float) else f'{v:.4f}'}")
    print("=" * 62)
    print(f"[turing_eval] wrote {out_dir}/{stem}_turing_{{items,summary}}.json")


# ──────────────────────────────────────────────────────────────────────────────
# stage 3: aggregate
# ──────────────────────────────────────────────────────────────────────────────
def _pm(row: dict, field: str, places: int) -> str:
    """`mean` for a single seed, `mean+/-std` once seeds have been merged."""
    v = row.get(field)
    if not isinstance(v, (int, float)):
        return "nan"
    sd = row.get(field + "_std")
    if row.get("_n_seeds", 1) > 1 and isinstance(sd, (int, float)):
        return f"{v:.{places}f}±{sd:.{places}f}"
    return f"{v:.{places}f}"


def _aggregate(args) -> None:
    """Collect every *_turing_summary.json in a dir into one leaderboard.

    The tau-USI analogue is agents/tau_usi/aggregate_usi.py. Sorted by mean
    rating: higher = the judge more often mistook the model's turn for the real
    human's. gt_accuracy is the mirror image (lower = better for the model), and
    0.50 would mean perfectly indistinguishable.
    """
    out_dir = Path(args.dir)
    rows = []
    for p in sorted(out_dir.glob("*_turing_summary.json")):
        try:
            s = json.loads(p.read_text())
        except Exception as e:
            print(f"[turing_eval] skipping {p.name}: {type(e).__name__}: {e}")
            continue
        s["_label"] = p.name.replace("_turing_summary.json", "")
        rows.append(s)
    if not rows:
        print(f"[turing_eval] no *_turing_summary.json under {out_dir}")
        return

    # Collapse -seed<N> runs of the same model into one row with mean+/-std, the
    # way agents/tau_usi/aggregate_usi.py does. Without this each seed lands in
    # the leaderboard as if it were a separate model, three times over, and the
    # ranking silently compares single draws against each other.
    import re as _re
    import statistics as _st
    _SEED = _re.compile(r"-seed\d+$")
    NUMERIC = ["turing_rating_mean", "turing_reward_mean", "gt_accuracy", "tie_rate",
               "assistant_like_penalty_mean", "source_copy_penalty_mean",
               "wrong_target_or_role_penalty_mean"]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(_SEED.sub("", r["_label"]), []).append(r)
    merged = []
    for key, rs in groups.items():
        if len(rs) == 1:
            rs[0]["_n_seeds"] = 1
            # Use the seed-stripped key so a model with only seed0 scored so far
            # displays under the same name it will have once its other seeds land,
            # instead of appearing as a distinct "<model>-seed0" row.
            rs[0]["_label"] = key
            merged.append(rs[0])
            continue
        m = {"_label": key, "_n_seeds": len(rs),
             "judge_model": rs[0].get("judge_model")}
        # n_scored is the number of ROWS evaluated, which is the same heldout set
        # each seed -- so report the per-seed count, not the sum. Summing made a
        # 3-seed row read as n=2640 and tripped the "odd n" footnote below, which
        # then declared it incomparable with the single-seed rows it matches.
        ns = [r.get("n_scored") or 0 for r in rs]
        m["n_scored"] = round(sum(ns) / len(ns))
        m["n_parse_error"] = sum(r.get("n_parse_error") or 0 for r in rs)
        for f in NUMERIC:
            vals = [r[f] for r in rs if isinstance(r.get(f), (int, float))]
            if vals:
                m[f] = sum(vals) / len(vals)
                # Standard ERROR of the mean, not SD: the table is used to compare
                # models, which is a question about how precisely each mean is
                # pinned. SEM = SD/sqrt(n); at n=3 that is SD/1.73, and the SEM is
                # itself estimated from three draws.
                m[f + "_std"] = (
                    _st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
                )
        merged.append(m)
    rows = merged
    rows.sort(key=lambda r: -(r.get("turing_rating_mean") or 0))

    # Runs scored over different row counts are not peers: a 20-row smoke run's
    # rating carries ~6x the standard error of an 880-row one, yet it sorts into
    # the same ranking looking equally authoritative. Mark anything materially
    # off the modal n instead of letting it pass silently. The 5% tolerance keeps
    # the odd judge-skipped record (879 vs 880) out of the footnote.
    modal_n = Counter(r.get("n_scored") for r in rows).most_common(1)[0][0]
    odd = [r for r in rows if abs((r.get("n_scored") or 0) - modal_n) > 0.05 * modal_n]
    for r in odd:
        r["_label"] += " *"

    hdr = ["Model", "Seeds", "n", "Rating", "Reward", "GT acc", "Tie", "AsstLike", "SrcCopy", "Parse err"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        lines.append(
            "| " + " | ".join([
                r["_label"],
                str(r.get("_n_seeds", 1)),
                str(r.get("n_scored", "")),
                _pm(r, "turing_rating_mean", 2),
                _pm(r, "turing_reward_mean", 3),
                _pm(r, "gt_accuracy", 3),
                f"{r.get('tie_rate', float('nan')):.3f}",
                f"{r.get('assistant_like_penalty_mean', float('nan')):.3f}",
                f"{r.get('source_copy_penalty_mean', float('nan')):.3f}",
                str(r.get("n_parse_error", "")),
            ]) + " |"
        )
    table = "\n".join(lines)
    judges = {r.get("judge_model") for r in rows}
    note = (
        f"\nJudge: {', '.join(sorted(j for j in judges if j))}. "
        "Rating 1-7, higher = model's turn looked more human than the real one "
        "(4 = judge cannot tell). GT acc = fraction where the judge correctly "
        "picked the real human, ties excluded; 0.50 = indistinguishable.\n"
    )
    if odd:
        note += (
            f"* scored over a different number of rows than the modal n={modal_n}, "
            "so NOT comparable with the rest of the table: "
            + ", ".join(f"{r['_label'][:-2]} (n={r.get('n_scored')})" for r in odd)
            + "\n"
        )
    print("\n" + table + note)
    dest = out_dir / "turing_summary.md"
    dest.write_text(table + "\n" + note)
    print(f"[turing_eval] wrote {dest}")


# ──────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="roll the served model over the heldout set")
    g.add_argument("--test-parquet", required=True)
    g.add_argument("--model", default=None, help="served model name (vLLM --served-model-name)")
    g.add_argument("--api-model", default=None,
                   help="evaluate an API model instead: the hosted endpoint registry name "
                        "(gpt-5.5 / claude-opus-5 / gemini-3.6-flash). No vLLM, no GPU.")
    g.add_argument("--reasoning-effort", default=os.getenv("OPENAI_AGENT_REASONING_EFFORT", "low"),
                   help="API mode only; ignored by /v1/messages models")
    g.add_argument("--base-url", default="http://localhost:8100/v1")
    g.add_argument("--api-key", default=os.getenv("OPENAI_AGENT_API_KEY", "EMPTY"))
    g.add_argument("--out", required=True)
    g.add_argument("--workers", type=int, default=32)
    g.add_argument("--max-tokens", type=int, default=16384,
                   help="16k, not 2k: a thinking model spends most of its budget on the "
                        "reasoning block (see run_eval.sh)")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=None,
                   help="sampling seed for the LOCAL vLLM path, forwarded per request. "
                        "Different seeds give independent draws at temperature 1.0; the "
                        "same seed reproduces a run. Ignored in --api-model mode (neither "
                        "the Responses API nor Anthropic /v1/messages accepts a seed).")
    g.add_argument("--limit", type=int, default=None, help="smoke-test on the first N targets")

    s = sub.add_parser("score", help="run the Turing judge over generations")
    s.add_argument("--gen", required=True, help="JSON written by `generate`")
    s.add_argument("--out-dir", default=None)
    s.add_argument("--judge-model", default=os.getenv("TURING_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    s.add_argument("--workers", type=int, default=4,
                   help="concurrent judge calls; the global pacer "
                        "(PERSONA_OPENAI_MIN_INTERVAL_SECONDS, default 2s) is the real "
                        "rate cap, so raising this alone will not go faster")
    s.add_argument("--limit", type=int, default=None)

    a = sub.add_parser("aggregate", help="build a leaderboard from *_turing_summary.json")
    a.add_argument("--dir", default="outputs/turing")

    args = ap.parse_args(argv)
    if args.cmd == "generate":
        if bool(args.model) == bool(args.api_model):
            ap.error("pass exactly one of --model (local vLLM) or --api-model (the hosted endpoint)")
        asyncio.run(_generate(args))
    elif args.cmd == "score":
        _score(args)
    else:
        _aggregate(args)


if __name__ == "__main__":
    main()
