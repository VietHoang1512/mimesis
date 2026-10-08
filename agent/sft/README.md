# Stage 1: supervised warm start

RL does not start from the stock base model. The actor in every reported run is
Qwen3-8B after a short supervised pass over interaction traces, and all training
curves begin from that checkpoint. Starting RL from the stock model gives
different results.

This stage runs in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory),
not in verl. The release provides a config and a driver script; LLaMA-Factory
itself is not vendored.

## Run it

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
pip install -e LLaMA-Factory

source scripts/env.sh
bash scripts/run_sft.sh /path/to/Qwen3-8B
```

`qwen3_customized.yaml` holds the hyper-parameters used for the warm start: full
fine-tuning, DeepSpeed ZeRO-3, lr 1e-5 cosine with 0.1 warmup, 3 epochs,
per-device batch 2 with 4-step accumulation, bf16, 16384 cutoff.

## The dataset

`dataset: merged_gym_sft` is the supervised corpus of successful interaction
traces over the five training gyms, assembled from the upstream UserRL task
data. It is not shipped here.

Register it with LLaMA-Factory by adding an entry for your built file to its
`data/dataset_info.json`.

## After training

```bash
python sft/fix_ckpt_tokenizer.py $AGENT_MODEL_DIR/sft/qwen3-8b
```

LLaMA-Factory writes a tokenizer config that the RL environment cannot load, so
RL fails at startup without this step. The script repairs the config in place.
