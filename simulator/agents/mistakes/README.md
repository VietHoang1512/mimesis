# Mistakes agent

`mistakes` is a single-turn benchmark. Given a multiple-choice math question and a student misconception, the model should simulate which option a student with that misconception would choose.

## Input

`agent_loop(data, context)` uses:

- `data["row"]`: one row of the Mistakes evaluation set
  - common fields: `QuestionText`, `AnswerAText..AnswerDText`, `MisconceptionName`, `TargetOption`
- `context["client"]`: `AsyncOpenAI` client
- `context["model"]`: model name
- `context["temperature"]`: optional sampling temperature

## Prompt Strategy

- `system`: zero-shot instruction that asks the model to simulate a student with the misconception and follow this format:
  - `Reasoning: ...`
  - `Incorrect Student Answer: <A/B/C/D>`
- `user`: built from `Question + Answer Choices + Student Misconception`.

## Output

`agent_loop` returns:

- `reward`: `1.0` / `0.0` (whether predicted option matches `TargetOption`)
- `chat`: conversation history (`system` / `user` / `assistant`)
- `predicted`: extracted option letter
- `correct`: target option letter
- additional fields: `final_answer`, `final_answer_mode`, `target_text`, `misconception_id`, `misconception_name`, `case_id`

## Run

The agent runs as part of the SOUL evaluation. From the `simulator` directory:

```bash
bash scripts/run_eval.sh
```
