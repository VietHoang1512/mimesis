#!/usr/bin/env python3
"""Make an SFT checkpoint's tokenizer loadable by the RL environment.

    python sft/fix_ckpt_tokenizer.py $AGENT_MODEL_DIR/sft/qwen3-8b

The two stages deliberately run in different conda envs (LLaMA-Factory needs a
recent transformers; sglang 0.4.7 pins transformers==4.52.3). That is fine for
code, but the *artifact* crosses the boundary, and the tokenizer config does not
round-trip:

    transformers 5.x  writes  "extra_special_tokens": []      (list)
    transformers 4.52.3 does  kwargs.pop("extra_special_tokens", {})
                              ... list(special_tokens.keys())  (expects dict)

so RL dies at startup with

    AttributeError: 'list' object has no attribute 'keys'
      tokenization_utils_base.py:1193 in _set_model_specific_special_tokens

Full fine-tuning does not modify the tokenizer, only re-serialises it, so
normalising the field is lossless. --from-base copies the base model's tokenizer
files wholesale instead, which is the bigger hammer if more fields drift.

Pure JSON: no transformers import, so it runs in either env. If transformers
happens to be importable it also verifies the result by actually loading it.
"""

import argparse
import json
import pathlib
import shutil
import sys

TOKENIZER_FILES = (
    "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
)


def _key(tok):
    """Comparable identity for a token entry, which may be a str or a
    serialised AddedToken dict."""
    return tok["content"] if isinstance(tok, dict) and "content" in tok else tok


def normalize(cfg_path: pathlib.Path) -> bool:
    cfg = json.loads(cfg_path.read_text())
    est = cfg.get("extra_special_tokens")
    if not isinstance(est, list):
        print(f"  extra_special_tokens is {type(est).__name__}, nothing to do")
        return False

    # transformers 5.x writes the list that 4.x calls `additional_special_tokens`
    # under the name `extra_special_tokens`. In 4.52.3 those are two different
    # things: `additional_special_tokens` is a list in SPECIAL_TOKENS_ATTRIBUTES,
    # while `extra_special_tokens` is a {attribute_name: token} dict. Moving the
    # list to its 4.x home and emptying the dict is a pure rename -- the tokens
    # are Qwen3's own (<|im_start|>, <|vision_pad|>, ...), already in the vocab,
    # and full fine-tuning does not add any.
    existing = cfg.get("additional_special_tokens") or []
    seen = {_key(t) for t in existing}
    added = [t for t in est if _key(t) not in seen]

    shutil.copy2(cfg_path, cfg_path.with_suffix(".json.bak"))
    cfg["additional_special_tokens"] = list(existing) + added
    cfg["extra_special_tokens"] = {}
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    print(f"  extra_special_tokens: list[{len(est)}] -> {{}}")
    print(f"  additional_special_tokens: {len(existing)} -> {len(cfg['additional_special_tokens'])}"
          f" ({len(added)} migrated)")
    if not added:
        print("    (all were already listed there; the move was a no-op for content)")
    print(f"  backup: {cfg_path.name}.bak")
    return True


def copy_from_base(base: pathlib.Path, ckpt: pathlib.Path) -> None:
    for name in TOKENIZER_FILES:
        src = base / name
        if src.exists():
            shutil.copy2(src, ckpt / name)
            print(f"  copied {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=pathlib.Path, help="SFT output dir")
    ap.add_argument("--from-base", type=pathlib.Path, default=None,
                    help="copy tokenizer files from this base model instead")
    args = ap.parse_args()

    ckpt = args.ckpt
    if not (ckpt / "config.json").exists():
        print(f"!! no config.json under {ckpt} -- is this a finished checkpoint?", file=sys.stderr)
        sys.exit(1)

    print(f"checkpoint: {ckpt}")
    if args.from_base:
        copy_from_base(args.from_base, ckpt)
    else:
        normalize(ckpt / "tokenizer_config.json")

    # Best-effort verification: this only means something in an env whose
    # transformers matches the one RL will use.
    try:
        from transformers import AutoTokenizer
        import transformers
        AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
        print(f"  verified: loads under transformers {transformers.__version__}")
    except ImportError:
        print("  (transformers not importable here -- skipping load check)")
    except Exception as e:
        print(f"  !! still fails to load: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
