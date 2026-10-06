# Stage 1: supervised warm start

RL does not start from the stock base model. The actor in every reported run is
Qwen3-8B after a short supervised pass over interaction traces, and the training
curves all begin from that checkpoint. Starting RL from the stock model is a
different experiment and gives different numbers.

This stage runs in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
rather than in verl, so it is kept as a config plus a thin driver rather than
vendored.

## Run it

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
pip install -e LLaMA-Factory

source scripts/env.sh
bash scripts/run_sft.sh /path/to/Qwen3-8B
```

`qwen3_customized.yaml` holds the hyper-parameters as used: full fine-tuning,
DeepSpeed ZeRO-3, lr 1e-5 cosine with 0.1 warmup, 3 epochs, per-device batch 2
with 4-step accumulation, bf16, 16384 cutoff.

## The dataset

`dataset: merged_gym_sft` is the supervised corpus, built from successful
interaction traces over the five training gyms. It is not shipped here -- see
the repository root README for how the corpus is assembled from the upstream
UserRL task data.

Register it with LLaMA-Factory by adding an entry to its `data/dataset_info.json`
pointing at your built file, as that framework requires.

## After training

```bash
python sft/fix_ckpt_tokenizer.py $AGENT_MODEL_DIR/sft/qwen3-8b
```

LLaMA-Factory writes a checkpoint whose tokenizer config does not round-trip the
chat template that the rollout engine then relies on. This repairs it in place.
Skipping it produces a checkpoint that loads fine and tokenizes multi-turn
conversations wrongly, which is not obvious until reward curves come out flat.
