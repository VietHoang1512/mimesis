# userLLM agent

`agents/userllm/agent.py` implements the userLLM user-simulation evaluation.

## What It Does

- Builds a user-simulation prompt from `intent` and `conversation_history`.
- Generates one user turn (or reuses `row.output` if provided).
- Computes per-case fields for the 6 main userLLM metrics:
  - `first_turn_diversity` (aggregated only)
  - `intent_decomposition`
  - `termination_f1` (aggregated only; per-case uses pred/true flags)
  - `ai_detector_score`
  - `role_adherence`
  - `intent_adherence`
- Returns a result dict with `chat`, `reward`, metric fields, and debug flags.

## Input Schema (`data["row"]`)

Common fields:

- `id` or `case_id`
- `intent` (or equivalent fields accepted by the suite helper)
- `conversation_history` (empty string means first turn)
- `turn`, `is_last_turn`
- `source`
- Optional: `output` (skip generation)

Metric-specific fields:

- `choices` for `role_adherence`
- `question` + `assistant_suggestion_turn` for `intent_adherence`

Metric routing:

- Preferred: `related_metrics` (list/string/set of metric names)
- Backward-compatible alias: `related_metric`
- If missing, inferred from `source`:
  - `commonsense_qa` -> `role_adherence`
  - `natural_questions` -> `intent_adherence`
  - otherwise -> PRISM-like first 4 metrics

## Context Fields

Expected in `context`:

- `client` (`AsyncOpenAI`)
- `model`
- optional `judge_model` (defaults to `model`)
- optional `temperature`
- optional `max_tokens` (default `256`)
- optional `max_retries` (default `5`)

## Output Fields (per case)

Key fields returned by `agent_loop`:

- `reward` (first available metric, in this order):
  - `ai_detector_score` if available, else
  - `role_adherence` if available, else
  - `intent_adherence` if available, else `0.0`
- `chat`: `[{role: "system", ...}, {role: "assistant", ...}]`
- `generated_output`
- `related_metrics`
- `is_first_turn`, `has_intent`
- `pred_endconversation`, `true_endconversation`
- metric values:
  - `intent_decomposition`, `ai_detector_score`, `role_adherence`, `intent_adherence`
  - `first_turn_diversity` and `termination_f1` are left as `None` per case and computed in aggregate

## Aggregation

Use `compute_userllm_aggregates(results)` to compute the 6 main metrics.

- Outputs are raw scores in `[0, 1]` where applicable.
- `run_eval.py` prints both raw and `x100` views.

## Example

Run from the `simulator` directory:

```bash
python agents/userllm/run_eval.py --model /path/to/mimesis-9b --n 20
```

By default this loads the `userllm_test` config of `cmu-lti/osim-post-training` from Hugging Face (`--dataset`, `--config`); `--data-path` reads a local parquet instead.
