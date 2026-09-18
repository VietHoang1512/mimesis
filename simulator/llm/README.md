# Judge and partner models

Every benchmark here needs a second model besides the one under test: a
conversational **partner** (the assistant the simulated user talks to) and a
**judge** (which scores the resulting transcript). `api.py` is the client for
both.

## Configuration

```bash
export OPENAI_AGENT_BASE_URL=http://localhost:8000/v1   # any OpenAI-compatible endpoint
export OPENAI_AGENT_API_KEY=...                         # "EMPTY" for a local server
export OPENAI_MODEL_NAME=gpt-5-5                        # judge / partner model id
```

Anything speaking the OpenAI chat-completions API works: vLLM, SGLang, a hosted
provider, or a gateway of your own. The simulator under evaluation is served
separately (see `scripts/env.sh`); this module is only the other side of the
conversation.

The published results used a judge and partner equivalent to `gpt-5.5`, with
`OPENAI_AGENT_REASONING_EFFORT=low`. Note that providers interpret that
differently — some ignore it, some map it to a token budget — so a judge on a
different provider is not automatically comparable even at the same nominal
setting.

## Two behaviours worth preserving if you adapt this

**The retry budget must outlast the provider's rate-limit window.** Providers
meter over windows measured in minutes. A retry loop that gives up in ~45 s
cannot ride one out, and the failure mode is silent rather than loud: the call
returns nothing, the turn is dropped, the conversation ends early and scores 0 —
indistinguishable in the output from a simulator that simply stopped talking.
An earlier version of this code lost 27% of its rollouts that way before the
cause was found. Backoff is jittered for the same reason: without jitter,
concurrent rollouts that hit one limit retry in lockstep and collide again.

**An empty completion is an error, not a result.** A reasoning model whose token
budget is consumed before it emits visible text returns a perfectly well-formed
response with no content. `chat_async` raises on that instead of returning `""`,
because an empty string propagates into the transcript as a real (empty) turn
and quietly corrupts the score.

## Surface

| function | use |
|---|---|
| `chat_async` | one completion, retried; everything else wraps it |
| `chat_parse` | same, constrained to JSON |
| `chat_messages_async` | for call sites that pass the system prompt separately |
| `merge_alternating` | collapse consecutive same-role turns, which some chat templates reject |
| `normalize_model`, `is_messages_model`, `is_gemini_model` | compatibility shims from when call sites dispatched per provider family; one endpoint serves everything now |
