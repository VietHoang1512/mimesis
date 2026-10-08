#!/usr/bin/env python3
"""Make an SFT checkpoint's tokenizer loadable by the RL environment."""

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
    return tok["content"] if isinstance(tok, dict) and "content" in tok else tok


def normalize(cfg_path: pathlib.Path) -> bool:
    cfg = json.loads(cfg_path.read_text())
    est = cfg.get("extra_special_tokens")
    if not isinstance(est, list):
        print(f"  extra_special_tokens is {type(est).__name__}, nothing to do")
        return False

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
