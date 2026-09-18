"""Standalone eval for userLLM agent on cmu-lti/osim-post-training (userllm_test).

Evaluates meta-llama/Llama-3.1-8B-Instruct as user simulator.
Uses local model for generation, Llama API (gpt-5-5-genai-responses) as judge.

Usage:
  cd MIMESIS
  with-proxy PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=4 python -m agents.userllm.run_eval
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from agents.userllm.agent import (
    _as_test_case,
    _intent_1gram_overlap_compatible,
    _resolve_related_metrics,
    compute_userllm_aggregates,
)
from agents.userllm.helpers import (
    _extract_choice_texts,
    _extract_intent,
    _extract_intent_adherence_fields,
    _format_first_turn_prompt,
    _format_sequential_turn_prompt,
    _normalize_for_choice_match,
    _to_optional_bool,
    INTENT_ADHERENCE_JUDGE_PROMPT,
)
from agents.utils import remove_think
from openai import AsyncOpenAI

# Judge (from api.py)
JUDGE_API_KEY = os.getenv(
    "OPENAI_API_KEY", os.environ.get("LLAMA_API_KEY", "")
)
JUDGE_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
)
JUDGE_MODEL = os.getenv("DEFAULT_JUDGE_MODEL", "gpt-5-5")

_judge_client: AsyncOpenAI | None = None


def log(msg: str):
    print(msg, flush=True)


def _get_judge() -> AsyncOpenAI:
    global _judge_client
    if _judge_client is None:
        _judge_client = AsyncOpenAI(
            api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL, timeout=60
        )
    return _judge_client


async def judge_call(prompt: str, retries: int = 5) -> str:
    client = _get_judge()
    for i in range(retries):
        try:
            r = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=20,
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            if i == retries - 1:
                return ""
            wait = min(2 ** (i + 1), 30)
            if "429" in str(e) or "rate" in str(e).lower():
                wait = max(wait, 15)
            await asyncio.sleep(wait)
    return ""


class LocalModel:
    def __init__(self, model_name: str, gpu: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = f"cuda:{gpu}"
        log(f"Loading {model_name} → {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, token=os.getenv("HF_TOKEN")
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map=self.device,
                token=os.getenv("HF_TOKEN"),
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map=self.device,
                local_files_only=True,
            )
        self.model.eval()
        log("Model loaded.")

    def generate_sync(
        self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 512
    ) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens, pad_token_id=self.tokenizer.eos_token_id
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )


async def run_eval(args):
    import numpy as np
    import pandas as pd

    if args.data_path:
        log(f"Loading: {args.data_path}")
        df = pd.read_parquet(args.data_path)
        rows = []
        for _, r in df.iterrows():
            extra = r["extra_info"] if isinstance(r["extra_info"], dict) else {}
            for k, v in extra.items():
                if isinstance(v, np.ndarray):
                    extra[k] = v.tolist()
            rows.append(extra)
    else:
        from datasets import load_dataset

        log(f"Loading HF: {args.dataset}/{args.config}")
        ds = load_dataset(
            args.dataset, args.config, split="train", token=os.getenv("HF_TOKEN")
        )
        rows = [
            r.get("extra_info", r) if isinstance(r.get("extra_info"), dict) else r
            for r in ds
        ]

    if args.n:
        rows = rows[: args.n]
    log(f"Evaluating {len(rows)} rows | model={args.model} | judge={JUDGE_MODEL}")

    model = LocalModel(args.model, gpu=args.gpu)

    # Phase 1: Generate all outputs
    log("Phase 1: Generating outputs...")
    t0 = time.monotonic()
    gen_results = []
    for i, row in enumerate(rows):
        tc = _as_test_case(row)
        intent = _extract_intent(tc)
        conv = row.get("conversation_history")
        is_first = conv == "" or conv is None
        prompt = (
            _format_first_turn_prompt(intent)
            if is_first
            else _format_sequential_turn_prompt(intent, str(conv))
        )

        msgs = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
        raw = model.generate_sync(
            msgs, temperature=args.temperature, max_tokens=args.max_tokens
        )
        output = remove_think(raw) if raw else ""

        pred_end = "<|endconversation|>" in output
        if pred_end:
            output = (
                output.split("<|endconversation|>")[0].strip() + "<|endconversation|>"
            )

        gen_results.append(
            {
                "output": output,
                "pred_end": pred_end,
                "is_first": is_first,
                "intent": intent,
            }
        )
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            log(f"  [{i+1}/{len(rows)}] {time.monotonic()-t0:.0f}s")

    # Phase 2: Compute metrics (judge concurrent)
    log("Phase 2: Scoring...")
    sem = asyncio.Semaphore(args.judge_workers)

    async def score_one(i: int) -> Dict[str, Any]:
        row = rows[i]
        g = gen_results[i]
        output, pred_end, is_first, intent = (
            g["output"],
            g["pred_end"],
            g["is_first"],
            g["intent"],
        )
        has_intent = bool(intent.strip())
        related = _resolve_related_metrics(row)
        true_end = _to_optional_bool(row.get("is_last_turn"))

        intent_decomp = None
        ai_score = None
        role_adh = None
        intent_adh = None

        if "intent_decomposition" in related and has_intent:
            intent_decomp = _intent_1gram_overlap_compatible(intent, output)

        if "role_adherence" in related:
            choices = _extract_choice_texts(row)
            if choices:
                out_norm = _normalize_for_choice_match(output)
                mentioned = sum(
                    1
                    for c in choices
                    if c and _normalize_for_choice_match(c) in out_norm
                )
                if mentioned != len(choices):
                    role_adh = 0.0 if mentioned in (1, 2) else 1.0
                else:
                    role_adh = 0.0
            else:
                role_adh = 0.0

        if args.use_judge:
            async with sem:
                if "ai_detector_score" in related:
                    text = output.replace("<|endconversation|>", "").strip()
                    if text:
                        p = f"You are an AI detection judge. Decide if this text was written by AI or human.\n\nText:\n{text}\n\nReply one word: AI or HUMAN."
                        r = await judge_call(p)
                        lbl = r.strip().upper().split()[0] if r.strip() else ""
                        ai_score = (
                            0.0 if lbl == "AI" else (1.0 if lbl == "HUMAN" else None)
                        )

                if "intent_adherence" in related:
                    tc2 = _as_test_case(row)
                    fields = _extract_intent_adherence_fields(tc2)
                    if fields and output:
                        q, s = fields
                        p = INTENT_ADHERENCE_JUDGE_PROMPT.format(
                            question=q, refusal=s, output=output
                        )
                        r = await judge_call(p)
                        lbl = r.strip().upper().split()[0] if r.strip() else ""
                        intent_adh = (
                            1.0
                            if lbl == "REFUSED"
                            else (0.0 if lbl == "ACCEPTED" else None)
                        )

        return {
            "index": i,
            "source": str(row.get("source") or ""),
            "generated_output": output,
            "related_metrics": sorted(list(related)),
            "is_first_turn": is_first,
            "has_intent": has_intent,
            "pred_endconversation": pred_end if "termination_f1" in related else None,
            "true_endconversation": true_end if "termination_f1" in related else None,
            "intent_decomposition": intent_decomp,
            "ai_detector_score": ai_score,
            "role_adherence": role_adh,
            "intent_adherence": intent_adh,
        }

    results = await asyncio.gather(*[score_one(i) for i in range(len(rows))])
    results = sorted(results, key=lambda r: r["index"])
    log(f"  Scoring done. {time.monotonic()-t0:.0f}s total")

    agg = compute_userllm_aggregates(results)
    elapsed = time.monotonic() - t0

    log("\n" + "=" * 60)
    log(f"  userLLM Eval — {args.model}")
    log("=" * 60)
    for m, v in agg.items():
        log(f"  {m:<25s}: {v:.4f} ({v*100:.2f}%)")
    log("=" * 60)
    log(f"  Time: {elapsed:.0f}s | Rows: {len(results)}\n")

    out_path = args.out or f"outputs/{args.model.replace('/','_')}_userllm_eval.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "n": len(results),
                "aggregates": agg,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    log(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument(
        "--data-path", default=None, help="Local parquet (skips HF download)"
    )
    ap.add_argument("--dataset", default="cmu-lti/osim-post-training")
    ap.add_argument("--config", default="userllm_test")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-judge", dest="use_judge", action="store_false", default=True)
    ap.add_argument("--judge-workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    asyncio.run(run_eval(ap.parse_args()))


if __name__ == "__main__":
    main()
