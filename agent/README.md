# Agent training

Reinforcement learning for the user-facing agent, with **SDPO** — an auxiliary
self-distillation term that supplies dense per-turn signal where multi-turn GRPO
gives almost none.

The agent is trained and evaluated against a simulated user. The simulator half
of the paper lives in [`../simulator`](../simulator).

```
L = L_GRPO + alpha * L_SDPO
```

At each hintable turn a *hint writer* reads the conversation so far plus two
privileged channels the agent cannot see — the simulated user's private
reasoning, and what they say next — and writes a short forward-looking briefing.
That briefing is spliced in as a user turn, the resulting teacher log-probs are
compared against the policy's own, and the difference is gated through a
sigmoid so a token the teacher dislikes is attenuated toward zero rather than
pushed down. See `verl/workers/rollout/sdpo_hint.py`.

---

## Install

The vendored `verl` here declares the package name `verl`, exactly as
`../simulator` does, so **the two cannot share one environment**. Use a separate
one for each.

```bash
conda create -n mimesis-agent python=3.12 && conda activate mimesis-agent
pip install -r requirements.txt
pip install -e .

bash scripts/install_gyms.sh      # clones upstream UserRL, applies our patch
bash scripts/prepare_data.sh      # builds data/rl_split and the eval parquets
```

Then edit `scripts/env.sh`, which is the only file in this repo that holds
paths. Everything else derives from it.

## Train

```bash
source scripts/env.sh

bash scripts/run_sft.sh /path/to/Qwen3-8B     # stage 1, see sft/README.md
bash scripts/run_sdpo.sh                      # stage 2, our method
bash scripts/run_grpo.sh                      # stage 2 control
```

Stage 2 needs a simulator already serving — see below. The two stage-2 scripts
both `source scripts/_train_common.sh` and differ only in `SDPO_COEF`
(`0.01` vs `0`), so the control is the same binary with the auxiliary term
switched off rather than a separately written baseline.

A two-step smoke run, which is the fastest way to confirm the whole loop is
wired up:

```bash
bash scripts/run_sdpo.sh trainer.total_training_steps=2
```

Check the log reports a non-zero `sdar/gate_mean`. An arm whose hint path is
silently broken still trains, still logs healthy-looking reward, and is just
GRPO — this metric is how you tell the difference.

## Serve the simulated user

```bash
bash scripts/serve_simulators.sh /path/to/mimesis-9b mimesis-9b 8000 8710
```

Two processes: vLLM on `:8000`, and `llm/shim.py` on `:8710`. **Point
`SIM_BASE_URL` at the shim, never at vLLM.** The shim does two things that are
load-bearing rather than cosmetic:

* strips `<think>` blocks before the gym sees the reply. A gym uses
  `message.content` verbatim as the user's utterance, so an unstripped reasoning
  trace becomes something the user "said" — the agent reads it, the judge grades
  against it, and the episode is corrupted with no error raised anywhere.
* keeps that stripped reasoning, keyed by episode, as SDPO's privileged channel.
  `verl/tools/interact_tool.py` reads it back over `GET /ep/<id>/privileged`.
  This is why `verl/tools/env_manager.py` gives each gym instance a
  `/ep/<request_id>/v1` base URL: an unscoped call cannot be attributed to a
  trajectory, and guessing would pair one episode's reasoning with another's
  turn — which yields plausible hints and no error.

Any OpenAI-compatible endpoint works as the upstream, including a hosted
provider; see `llm/README.md`.

## Evaluate

```bash
bash scripts/run_eval.sh /path/to/checkpoint my-agent
```

One agent against one simulator across all 15 evaluation sets. An FSDP training
checkpoint is merged to HF format first; pass an HF directory and the merge is
skipped. Repeat per simulator, then build the table:

```bash
python eval/digest.py -o results/main_results.json
python eval/make_main_table.py --digest results/main_results.json
```

`results/main_results.json` already ships, so the second command alone
reproduces the main table. The raw reward caches behind it are several GB and
about 40 GPU-hours to regenerate; the digest is a few hundred KB and gives
identical numbers.

The eight `travel*` variants differ in how many preference slots the simulated
user withholds; `eval/table3.py` collapses them into one TravelGym column and
weights `Avg.` by uid count.

## What we changed in verl

Vendored fork of [verl](https://github.com/volcengine/verl). Dead backends are
pruned — megatron throughout, the vLLM rollout and `third_party/vllm`, the
reward-model worker, the SFT/eval/generation trainers, and every
`reward_score` module except `interact` (all task rows are `interact_*`). An
import crawl from `verl.trainer.main_ppo` confirms none is reachable for this
configuration: 243 Python files before pruning, 110 after.

| File | Change | What it does |
|---|---|---|
| `workers/rollout/sdpo_hint.py` | new, ~710 L | hint writer: prompts, privileged-channel plumbing, the ablation switch |
| `workers/actor/dp_actor.py` | +750 | the SDPO loss, including the SDAR sigmoid-gated estimator |
| `workers/rollout/sglang_rollout/sglang_rollout_customized.py` | +449 | teacher rows for hinted turns, alongside the ordinary rollout |
| `tools/interact_tool.py` | +130 | reads the privileged channel back from the shim |
| `tools/env_manager.py` | +108 | per-episode base URLs, so reasoning is attributable |
| `workers/fsdp_workers.py` | +35 | tensor-arrival probe for the SDPO path |
| `utils/activation_offload.py` | +34 | offload interop with the extra teacher forward |
| `workers/rollout/schemas.py` | +20 | `sdpo_*` fields on the rollout schema |
| `trainer/config/ppo_trainer.yaml` | +18 | `sdpo_coef`, `sdpo_estimator` |
| `workers/reward_manager/naive.py` | +10 | `interact_*` scoring branch (turn sum, max, normalised) |

`trainer/ppo/core_algos.py` and `trainer/ppo/ray_trainer.py` also carry the
`grpo_multiturn` advantage estimator, which is a fork addition rather than one
of ours.

## What we changed in the gyms

`patches/gyms.patch` — 13 files, 407 insertions, applied over upstream
[UserRL](https://github.com/SalesforceAIResearch/UserRL) by
`scripts/install_gyms.sh`. These are correctness fixes the reported numbers
depend on:

* **TravelGym** — the async elicitation branch never assigned the simulator's
  reply before using it, so it raised `UnboundLocalError` and fell through to a
  canned response worth 0 reward. Present in upstream `HEAD`; the synchronous
  twin has the assignment. This alone was 96% of all malformed turns, 6.7% →
  0.1%.
* **Persuade / Turtle / Telepathy / Search** — `_coerce_model_json`, a four-tier
  parser for simulator output that is meant to be JSON and is not (curly quotes,
  apostrophes inside single-quoted dicts). Every parse failure previously became
  the same zero-reward canned reply.
* **All gyms** — `USERRL_SIM_TIMEOUT` / `USERRL_GYM_SEED`. The stock 10 s timeout
  turns a merely slow simulator turn into that canned reply as well.

`AlfworldGym` and `TemplateGym` are not reachable from `tools/env_manager.py`
and are not installed.

## Layout

```
scripts/      env.sh (the only place paths live) + one driver per stage
llm/          api.py (OpenAI-compatible client), shim.py (think-stripping proxy)
verl/         vendored RL framework, SDPO lives here
eval/         eval.py, table3.py, make_main_table.py, digest.py, merge.py
examples/     hydra configs and the data-split script
patches/      gyms.patch
sft/          stage-1 config and checkpoint repair
results/      score digest, so the main table regenerates without a rerun
```
