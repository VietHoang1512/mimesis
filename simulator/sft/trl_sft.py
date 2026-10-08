"""SFT a thinking user-simulator on the ThoughtTrace conversations."""

import logging
import os
import shutil
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer, TrlParser

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.environ.get("MIMESIS_MODEL_DIR", os.path.join(REPO_DIR, "models"))
DATA_DIR = os.environ.get("MIMESIS_DATA_DIR", os.path.join(REPO_DIR, "data"))

SURVEY_FIELDS = (
    ("age", "Age"),
    ("gender", "Gender"),
    ("education", "Education"),
    ("occupation", "Occupation"),
    ("frequency", "AI-usage frequency"),
    ("purposes", "Typical uses"),
)

SYSTEM_INTRO = (
    "You are simulating a real human user chatting with an AI assistant. Stay in "
    "character and reply only with the user's next message — natural, first-person, "
    "one turn at a time."
)

SYSTEM_OUTRO = (
    "Before each message, think privately about your motivation, context, and "
    "constraints, then send the message."
)

GREETING = "Hi! How can I help you today?"


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(
        default=os.path.join(MODEL_DIR, "mimesis-9b-midtrain"),
        metadata={"help": "Base model. Lives here rather than on SFTConfig, which has no such field."},
    )
    dataset_path: str = field(
        default=os.path.join(DATA_DIR, "thoughttrace", "ThoughtTrace.jsonl"),
        metadata={"help": "JSONL dump of annotated human<->AI conversations."},
    )
    include_reactions: bool = field(
        default=False,
        metadata={
            "help": "Also fold the human's reaction to the previous AI reply into the <think> "
            "block. Off by default: ThoughtTrace defines the pre-message thought as `reasons`, "
            "and the published runs used reasons only. Turn on to additionally use the ~5.5k "
            "reaction annotations."
        },
    )
    greeting: str = field(
        default=GREETING,
        metadata={
            "help": "Opening AI turn prepended to every conversation, so the model's first "
            "generation replies to a greeting exactly as it does under agents/tau_usi. "
            "Pass an empty string to start cold from the system prompt."
        },
    )
    drop_trailing_context: bool = field(
        default=True,
        metadata={
            "help": "Drop trailing AI turns that carry no loss. Every conversation ends on an AI "
            "reply, so this frees context budget for turns that actually train. Never drops the "
            "last remaining AI turn -- the chat template requires one `user` message."
        },
    )
    eval_ratio: float = field(default=0.0, metadata={"help": "Fraction held out for eval. 0 disables eval."})
    inspect_only: bool = field(
        default=False,
        metadata={"help": "Render examples, verify the loss mask and report token lengths, then exit."},
    )
    attn_implementation: str = field(default="flash_attention_2")
    model_dtype: str = field(default="bfloat16")


PROCESSOR_FILES = (
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "image_preprocessor_config.json",
)


class CopyProcessorFiles(TrainerCallback):
    """Copy the base model's processor configs into every checkpoint we write."""

    def __init__(self, source: str):
        self.source = source

    def copy_into(self, dest: str) -> None:
        for name in PROCESSOR_FILES:
            src = os.path.join(self.source, name)
            dst = os.path.join(dest, name)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
                logger.info("Copied %s into %s", name, dest)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.copy_into(os.path.join(args.output_dir, f"checkpoint-{state.global_step}"))


def _annotations(items) -> list[str]:
    """Pull the non-empty `content` out of a reasons/reactions list."""
    out = []
    for item in items or []:
        text = str((item or {}).get("content") or "").strip()
        if text:
            out.append(text)
    return out


def build_system_prompt(record: dict) -> str:
    """Persona + task the human was pursuing, i.e. the simulator's conditioning."""
    parts = [SYSTEM_INTRO]

    task_summary = str(record.get("task_summary") or "").strip()
    if task_summary:
        parts.append(f"\nYour goal:\n{task_summary}")

    task_expectation = str(record.get("task_expectation") or "").strip()
    if task_expectation:
        parts.append(f"\nWhat you expect from the assistant:\n{task_expectation}")

    survey = record.get("survey_answers") or []
    profile = survey[0] if isinstance(survey, list) and survey else {}
    bits = []
    for key, label in SURVEY_FIELDS:
        value = profile.get(key) if isinstance(profile, dict) else None
        if value is None:
            continue
        value = str(value).strip()
        if value:
            bits.append(f"{label}: {value}")
    if bits:
        parts.append("\nAbout you:\n- " + "\n- ".join(bits))

    parts.append(f"\n{SYSTEM_OUTRO}")
    return "\n".join(parts)


def _alternates(messages: list) -> bool:
    """Strictly alternating and starting with the human, as the renderer requires."""
    if not messages:
        return False
    return all(m.get("type") == ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(messages))


def build_messages(record: dict, greeting: str, include_reactions: bool, drop_trailing_context: bool) -> dict:
    """Rewrite one annotated conversation into role-swapped chat messages."""
    source = record.get("messages") or []
    if not _alternates(source):
        return {"messages": []}

    out = [{"role": "system", "content": build_system_prompt(record)}]
    if greeting:
        out.append({"role": "user", "content": greeting})

    pending_reaction: list[str] = []

    for msg in source:
        content = str(msg.get("content") or "").strip()
        if not content:
            continue

        if msg.get("type") == "assistant":
            out.append({"role": "user", "content": content})
            pending_reaction = _annotations(msg.get("reactions")) if include_reactions else []
        else:
            thought = "\n\n".join(pending_reaction + _annotations(msg.get("reasons")))
            pending_reaction = []
            out.append({"role": "assistant", "content": f"<think>\n{thought}\n</think>\n\n{content}"})

    if drop_trailing_context:
        while len(out) > 2 and out[-1]["role"] == "user":
            out.pop()

    if not any(m["role"] == "assistant" for m in out) or not any(m["role"] == "user" for m in out):
        return {"messages": []}
    return {"messages": out}


def load_thought_trace(script_args: ScriptArguments, num_proc: int | None = None):
    raw = load_dataset("json", data_files=script_args.dataset_path, split="train")
    dataset = raw.map(
        build_messages,
        fn_kwargs={
            "greeting": script_args.greeting,
            "include_reactions": script_args.include_reactions,
            "drop_trailing_context": script_args.drop_trailing_context,
        },
        remove_columns=raw.column_names,
        num_proc=num_proc,
        desc="Building role-swapped messages",
    )
    kept = dataset.filter(lambda ex: len(ex["messages"]) > 0, num_proc=num_proc, desc="Dropping unrenderable")
    dropped = len(dataset) - len(kept)
    if dropped:
        logger.warning("Dropped %d/%d conversations with no renderable turn pair.", dropped, len(dataset))
    return kept


def inspect_dataset(dataset, model_path: str, chat_template_path: str, max_length: int) -> None:
    """Verify the template, the assistant-only loss mask and the token budget."""
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with open(chat_template_path) as fh:
        tokenizer.chat_template = fh.read()

    example = dataset[0]
    print("=" * 80, "RENDERED EXAMPLE 0", "=" * 80, sep="\n")
    print(tokenizer.apply_chat_template(example["messages"], tokenize=False)[:6000])

    out = tokenizer.apply_chat_template(example["messages"], return_dict=True, return_assistant_tokens_mask=True)
    mask = np.asarray(out["assistant_masks"])
    ids = np.asarray(out["input_ids"])
    print("=" * 80, "TOKENS THAT CARRY LOSS (must be the human's turns, incl. <think>)", "=" * 80, sep="\n")
    print(tokenizer.decode(ids[mask == 1])[:4000])
    if mask.sum() == 0:
        raise RuntimeError(
            f"assistant_masks is all zeros -- the {{% generation %}} block in {chat_template_path} "
            "is not being picked up, so training would be a no-op."
        )
    print(f"\nloss tokens: {int(mask.sum())}/{len(ids)} ({mask.mean():.1%})")

    lengths, loss_frac = [], []
    for row in dataset:
        enc = tokenizer.apply_chat_template(row["messages"], return_dict=True, return_assistant_tokens_mask=True)
        lengths.append(len(enc["input_ids"]))
        loss_frac.append(float(np.mean(enc["assistant_masks"])))
    lengths = np.asarray(lengths)
    print("=" * 80, "TOKEN LENGTHS", "=" * 80, sep="\n")
    for q in (50, 90, 95, 99, 100):
        print(f"  p{q}: {int(np.percentile(lengths, q))}")
    over = int((lengths > max_length).sum())
    print(f"  mean {lengths.mean():.0f} | total {lengths.sum() / 1e6:.2f}M tokens")
    print(f"  > max_length={max_length}: {over}/{len(lengths)} ({over / len(lengths):.1%})")
    print(f"  mean loss fraction: {np.mean(loss_frac):.1%}")


def main() -> None:
    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_and_config()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if training_args.assistant_only_loss and not training_args.chat_template_path:
        raise ValueError(
            "assistant_only_loss=True needs a chat template containing {% generation %}; "
            "pass --chat_template_path qwen3_5_think_training.jinja."
        )

    dataset = load_thought_trace(script_args, num_proc=training_args.dataset_num_proc)

    if script_args.inspect_only:
        inspect_dataset(
            dataset,
            script_args.model_name_or_path,
            training_args.chat_template_path,
            training_args.max_length,
        )
        return

    training_args.model_init_kwargs = {
        **(training_args.model_init_kwargs or {}),
        "dtype": script_args.model_dtype,
        "attn_implementation": script_args.attn_implementation,
    }

    eval_dataset = None
    if script_args.eval_ratio > 0:
        split = dataset.train_test_split(test_size=script_args.eval_ratio, seed=training_args.seed)
        dataset, eval_dataset = split["train"], split["test"]

    logger.info("train=%d eval=%d", len(dataset), len(eval_dataset) if eval_dataset else 0)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    trainer = SFTTrainer(
        model=script_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    copy_processor = CopyProcessorFiles(script_args.model_name_or_path)
    trainer.add_callback(copy_processor)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        copy_processor.copy_into(training_args.output_dir)


if __name__ == "__main__":
    main()
