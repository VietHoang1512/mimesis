# Judge and partner models

Every benchmark here needs a second model besides the one under test: a
conversational partner (the assistant the simulated user talks to) and a judge
(which scores the resulting transcript). `api.py` is the client for both.

## Configuration

```bash
export OPENAI_AGENT_BASE_URL=http://localhost:8000/v1   # any OpenAI-compatible endpoint
export OPENAI_AGENT_API_KEY=...                         # "EMPTY" for a local server
export OPENAI_MODEL_NAME=gpt-5-5                        # judge / partner model id
```

Any server that implements the OpenAI chat-completions API works: vLLM, SGLang,
a hosted provider, or your own gateway. The simulator under evaluation is served
separately (see `scripts/env.sh`); this module handles only the partner and the
judge.

The published results used a judge and partner equivalent to `gpt-5.5`, with
`OPENAI_AGENT_REASONING_EFFORT=low`. Providers interpret this setting
differently (some ignore it, some map it to a token budget), so a judge on a
different provider is not necessarily comparable even at the same nominal
setting.

## If you adapt this client

The retry budget must outlast the provider's rate-limit window. Providers meter
usage over windows measured in minutes, so a retry loop that gives up after
~45 s cannot wait one out. The call then fails silently: it returns nothing, the
turn is dropped, and the conversation ends early with a score of 0. In the
output, this is indistinguishable from a simulator that stopped responding.
Backoff is jittered for the same reason: without jitter, concurrent rollouts
that hit one limit retry in lockstep and collide again.

An empty completion is treated as an error. A reasoning model that uses up its
token budget before emitting visible text returns a well-formed response with
no content. `chat_async` raises an error in that case instead of returning
`""`, because an empty string would enter the transcript as a real (empty) turn
and be scored as one.

## Surface

| function | use |
|---|---|
| `chat_async` | one completion, retried; everything else wraps it |
| `chat_parse` | same, constrained to JSON |
| `chat_messages_async` | for call sites that pass the system prompt separately |
| `merge_alternating` | collapse consecutive same-role turns, which some chat templates reject |
| `normalize_model`, `is_messages_model`, `is_gemini_model` | compatibility shims from when call sites dispatched per provider family; one endpoint serves everything now |
