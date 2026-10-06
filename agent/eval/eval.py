import os
import json
import re
import time
import argparse
import yaml
import copy
from openai import AsyncOpenAI
# Gemini SDK: imported lazily because it is only reachable via the
# `"gemini" in model_name` branches below, and installing google-genai into this
# env would drag pydantic>=2.12.5 / httpx>=0.28.1 / anyio>=4.8.0 into a stack
# pinned around sglang 0.4.7 and transformers==4.52.3. Unpinned transitive deps
# are what broke compressed-tensors and uvloop here already. Evaluating a Gemini
# model raises a clear error instead of failing at import for everyone else.
try:
    from google.genai import Client, types
except ImportError as _genai_err:      # noqa: N816
    Client = None

    class _MissingGenai:
        def __getattr__(self, name):
            raise ImportError(
                f"google.genai is required for gemini models (used types.{name}); "
                f"pip install google-genai -- but check it does not move pydantic/httpx "
                f"in this env first. Original: {_genai_err}"
            )

    types = _MissingGenai()
import pandas as pd
import asyncio
import hashlib

MAX_WORKER_NUM = int(os.environ.get("MAX_WORKER_NUM", 25))
semaphore = asyncio.Semaphore(MAX_WORKER_NUM)

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "./") # Please change the path to the `eval` folder
# Where RESULTS go. PROJECT_ROOT stays pointed at the eval/ source tree, because
# it also resolves ../data/*.parquet and schema/interact_tool.yaml -- those must
# not move. Only the output side is redirectable, so a sweep can write GBs of
# caches and transcripts to scratch storage instead of into the repo working tree.
# Defaults to PROJECT_ROOT, i.e. unchanged behaviour when unset.
EVAL_OUTPUT_DIR = os.getenv("EVAL_OUTPUT_DIR") or PROJECT_ROOT
    
async def load_data(env_name, one_choice=True, split="test"):
    if one_choice and "travel" in env_name:
        path = f"{PROJECT_ROOT}/../data/{env_name}_multiturn_onechoice/{split}.parquet"
    else:
        path = f"{PROJECT_ROOT}/../data/{env_name}_multiturn/{split}.parquet"
    df = pd.read_parquet(path)
     # turn into a list of data, using only the prompt
    data = []
    for i in range(len(df)):
        data.append({
            "env_name": env_name,
            "gold": str(df.iloc[i]["reward_model"]["id"]) if (env_name == "intention" or env_name == "persuasion" or env_name == "bamboogle" or env_name == "alfworld" or env_name == "tau" or env_name == "function" or "travel" in env_name) else str(df.iloc[i]["reward_model"]["title"]),
            "messages": list(df.iloc[i]["prompt"]),
        })
    print(f"Loaded {len(data)} data from {path}")
    return data


class _Transcripts:
    """Agent-side and fully-stripped transcripts, one JSON file per episode.

    The reward cache keeps only post-strip text, so without this neither side's
    raw output survives anywhere -- and a misread of the agent's own output is
    then undetectable after the fact. The concrete hazard: vLLM exposes the
    agent's reasoning as `reasoning`, while other serving stacks use
    `reasoning_content`. Reading only one of those names yields empty reasoning
    for an entire run with nothing to contradict it, so turn() reads BOTH names
    deliberately and records the raw message -- a wrong read then shows up as
    empty transcript fields instead of as nothing at all.

    Completions are stored verbatim -- they are the point, and they are small.
    The prompt is stored ONCE per episode: prompts outweigh completions 20:1
    (31.8M vs 1.56M tokens on a measured cell) and re-serialising the prefix
    every turn turns ~25 MB/cell into ~200 MB.

    Inert unless TRANSCRIPT_DIR is set. Never raises: a transcript problem must
    not take down an evaluation.
    """

    def __init__(self):
        self.dir = os.environ.get("TRANSCRIPT_DIR", "")
        self.failed = False
        self._eps = {}
        # hash_id is a hash of the TASK only, so it is identical for the same
        # task across every (agent, simulator, judge) cell. Prefixing the run
        # keeps files from colliding, and keeps a judge shim shared by several
        # concurrent cells unambiguous about which cell a record came from.
        self.run = ""

    def episode_id(self, hash_id):
        # 16 hex chars is ample for ~1k tasks per cell and keeps the id usable as
        # a URL path segment and a CLI argument. The FULL hash_id stays in the
        # file body, which is what joins a transcript to the reward cache.
        short = str(hash_id)[:16]
        if not self.run:
            return short
        run = re.sub(r"[^A-Za-z0-9._-]", "_", self.run)
        return f"{run}--{short}"

    def _path(self, hash_id, env_name):
        d = os.path.join(self.dir, "agent", env_name)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self.episode_id(hash_id)}.json")

    def begin(self, hash_id, data, model_name):
        if not self.dir or self.failed:
            return
        try:
            self._eps[hash_id] = {
                "episode": {
                    "hash_id": hash_id,
                    "episode_id": self.episode_id(hash_id),
                    "run": self.run,
                    "env": data["env_name"],
                    "gold": data.get("gold"),
                    "agent": model_name,
                    "simulator": os.environ.get("USER_MODEL_NAME"),
                    "judge": os.environ.get("JUDGE_MODEL_NAME") or None,
                },
                "initial_messages": copy.deepcopy(data["messages"]),
                "turns": [],
            }
        except Exception as e:                              # noqa: BLE001
            self.failed = True
            print(f"[transcript] disabled after error: {type(e).__name__}: {e}")

    def turn(self, hash_id, turn, response, choice, content, feedback, reward):
        ep = self._eps.get(hash_id)
        if ep is None or self.failed:
            return
        try:
            raw = {}
            try:
                msg = response.choices[0].message
                # BOTH names on purpose. vLLM 0.24 + --reasoning-parser qwen3
                # emits `reasoning`; older/other stacks use `reasoning_content`.
                # Reading only the latter is exactly the bug this exists to catch.
                raw = {
                    "content": getattr(msg, "content", None),
                    "reasoning": (getattr(msg, "reasoning", None)
                                  or getattr(msg, "reasoning_content", None)),
                    "finish_reason": getattr(response.choices[0], "finish_reason", None),
                    "tool_call": None,
                }
                tc = getattr(msg, "tool_calls", None)
                if tc:
                    raw["tool_call"] = {"name": tc[0].function.name,
                                        "arguments": tc[0].function.arguments}
            except Exception:
                raw = {"unparsed": str(response)[:4000]}
            ep["turns"].append({
                "turn": turn, "agent_raw": raw,
                "action": {"choice": choice, "content": content},
                "feedback": feedback, "reward": reward,
            })
            # Written every turn, not at the end: eval.py dumps the reward cache
            # only after asyncio.gather, so a crashed cell currently loses the
            # whole env. Transcripts should survive that.
            with open(self._path(hash_id, ep["episode"]["env"]), "w") as fh:
                json.dump(ep, fh, ensure_ascii=False)
        except Exception as e:                              # noqa: BLE001
            self.failed = True
            print(f"[transcript] disabled after error: {type(e).__name__}: {e}")


_tx = _Transcripts()


def _episode_urls(episode):
    """(simulator_url, judge_url) scoped to one episode, or (None, None).

    llm/shim.py serves /ep/<episode>/v1/... as an alias of /v1/..., so a gym's
    simulator and judge calls carry the episode id with them and stay
    attributable to one trajectory when a single shim fronts many episodes at
    once. The shipped shim keys its privileged channel on exactly this id.
    Returning None leaves each gym's config untouched, so with TRANSCRIPT_DIR
    unset this whole path is inert.

    TauGym is deliberately excluded by the caller: its gym config *writes*
    OPENAI_BASE_URL into os.environ for litellm, which is process-global and
    would race across the 10 concurrent episodes in this process.
    """
    if not episode or not os.environ.get("TRANSCRIPT_DIR"):
        return None, None

    def scope(url):
        if not url:
            return None
        base = url[:-3] if url.endswith("/v1") else url.rstrip("/")
        return f"{base}/ep/{episode}/v1"

    return scope(os.environ.get("OPENAI_BASE_URL")), scope(os.environ.get("JUDGE_BASE_URL"))


async def build_env(data, max_turns, episode=None):
    gold = data["gold"]
    env_name = data["env_name"]
    model_name = os.environ.get("USER_MODEL_NAME", "gpt-4o")
    # Scope this episode's gym->LLM calls to a per-episode URL so llm/shim.py can
    # attribute them. Every gym reads its endpoint out of its own config and
    # calls it unmodified, so this costs zero changes at the eight gym call
    # sites -- which is why the scoping lives in the URL rather than in the gyms.
    # Without it a shared judge shim sees ~110 concurrent episodes interleaved
    # with no way to tell them apart.
    sim_url, judge_url = _episode_urls(episode)

    print("Building environment...", env_name)
    if "travel" in env_name:
        import travelgym
        config = travelgym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        config.model_name = model_name
        if sim_url:
            config.base_url = sim_url
        if judge_url and hasattr(config, "judge_base_url"):
            config.judge_base_url = judge_url
        # configure the one choice option
        config.one_choice_per_aspect = True
        config.search_correct_reward = 0.2
        config.preference_correct_reward = 0.6
        env = travelgym.TravelEnv(config=config)
        env.reset()
        return env
    elif env_name == "turtle":
        import turtlegym
        config = turtlegym.get_default_config()
        config.max_steps = max_turns
        config.success_threshold = 1.0
        config.data_mode = "single"
        config.data_source = gold
        config.model_name = model_name
        if sim_url:
            config.base_url = sim_url
        if judge_url and hasattr(config, "judge_base_url"):
            config.judge_base_url = judge_url
        env = turtlegym.StoryEnv(config=config)
        env.reset()
        return env
    elif env_name == "telepathy":
        import telepathygym
        config = telepathygym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        config.model_name = model_name
        if sim_url:
            config.base_url = sim_url
        if judge_url and hasattr(config, "judge_base_url"):
            config.judge_base_url = judge_url
        env = telepathygym.TelepathyEnv(config=config)
        env.reset()
        return env
    elif env_name == "intention":
        import intentiongym
        config = intentiongym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        config.model_name = model_name
        if sim_url:
            config.base_url = sim_url
        if judge_url and hasattr(config, "judge_base_url"):
            config.judge_base_url = judge_url
        env = intentiongym.IntentionEnv(config=config)
        env.reset()
        return env
    elif env_name == "persuasion":
        import persuadegym
        config = persuadegym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        config.model_name = model_name
        if sim_url:
            config.base_url = sim_url
        if judge_url and hasattr(config, "judge_base_url"):
            config.judge_base_url = judge_url
        env = persuadegym.PersuadeEnv(config=config)
        env.reset()
        return env
    elif env_name == "bamboogle":
        import searchgym
        config = searchgym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        config.eval_method = "llm"
        config.model_name = model_name
        if sim_url:
            config.base_url = sim_url
        if judge_url and hasattr(config, "judge_base_url"):
            config.judge_base_url = judge_url
        # 5 is default value
        # config.max_search_results = 5
        # config.max_search_steps = 5
        env = searchgym.SearchEnv(config=config)
        env.reset()
        return env
    elif env_name == "alfworld":
        import alfworldgym
        config = alfworldgym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = int(gold)
        env = alfworldgym.AlfworldEnv(config=config)
        env.reset()
        return env
    elif env_name == "tau":
        import taugym
        config = taugym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        config.user_model = model_name
        # special handling for task_category and task_split
        if "retail" in gold:
            config.task_category = "retail"
        else:
            config.task_category = "airline"
        if "train" in gold:
            config.task_split = "train"
        else:
            config.task_split = "test"
        env = taugym.TauEnv(config=config)
        env.reset()
        return env
    elif env_name == "function":
        import functiongym
        config = functiongym.get_default_config()
        config.max_steps = max_turns
        config.data_mode = "single"
        config.data_source = gold
        env = functiongym.FunctionEnv(config=config)
        env.reset()
        return env
    else:
        raise ValueError(f"Environment {env_name} not supported")


async def gen_response(client, data, schema, temperature, model_name):
    for _ in range(10):
        try:
            if "gemini" in model_name:
                interact_tool = copy.deepcopy(schema["function"])
                tools = types.Tool(function_declarations=[interact_tool])
                config = types.GenerateContentConfig(
                    tools=[tools],
                    temperature=temperature,
                    max_output_tokens=2048,
                    tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.ANY))
                )
                contents = [
                    types.Content(role="user", parts=[
                        types.Part(text=data["messages"][0]["content"]),
                        types.Part(text=data["messages"][1]["content"])
                    ])
                ]
                for message in data["messages"][2:]:
                    contents.append(message["content"])
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                part_exists = response.candidates[0].content.parts[0]
                return response
            
            else:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=data["messages"],
                    tools=[schema],
                    tool_choice="required" if str(os.environ.get("TOOL_CHOICE", "required")) == "required" else "auto",
                    temperature=temperature,
                    max_tokens=2048,
                    n=1,
                    # reasoning_effort="high"
                )
                return response
        
        except Exception as e:
            print(f"[local tool call] failed: {e}")
            time.sleep(2)
    
    print(f"[local tool call] failed three times, please check the model name and API key.")
    raise RuntimeError("!!! Local function_call failed three times !!!")


def hash(data):
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


async def rollout(data, client, function, temperature, max_turns, model_name):
    turn = 0
    rewards = []
    hash_id = hash(data)
    history = []
    json_data = copy.deepcopy(data)

    try:
        env = await build_env(data, max_turns, episode=_tx.episode_id(hash_id))
        _tx.begin(hash_id, data, model_name)
        while turn < max_turns:
            if "gemini" in model_name:
                response = await gen_response(client, data, function, temperature, model_name)
                # print(f"Model response: {response}")
                try:
                    json_args = response.candidates[0].content.parts[0].function_call.args
                    if isinstance(json_args, str):
                        json_args = json.loads(json_args)
                except Exception as e:
                    assert False, f"Invalid response (response without tool call detected): {response}"
            else:
                response = await gen_response(client, data, function, temperature, model_name)
                print(f"Model response: {response}")
                try:
                    args = response.choices[0].message.tool_calls[0].function.arguments
                    if isinstance(args, str):
                        json_args = json.loads(args)
                    else:
                        json_args = args
                except Exception as e:
                    assert False, f"Invalid response (response without tool call detected): {response}"

            assert "choice" in json_args and "content" in json_args, f"Invalid response: {args}"

            choice = json_args["choice"]
            content = json_args["content"]
            if choice == "action" and not content.startswith("[action]"):
                formatted_action = "[action] " + content
            elif choice == "answer" and not content.startswith("[answer]"):
                formatted_action = "[answer] " + content
            elif choice == "search" and not content.startswith("[search]"):
                formatted_action = "[search] " + content
            else:
                formatted_action = content
            
            observation, reward, terminated, truncated, info = await asyncio.wait_for(
                env.step_async(formatted_action),
                timeout=120.0
            )
            feedback = observation["feedback"]

            if len(feedback) > 512 and choice == "search":
                output_feedback = feedback[:256] + "  ... ... " + feedback[-256:]
            else:
                output_feedback = feedback
            
            print(f"In {data['env_name']}, turn {turn}, action: {formatted_action}, feedback: {output_feedback}, reward: {reward}")
            rewards.append(reward)

            history.append({
                "turn": turn,
                "choice": choice,
                "content": content,
                "feedback": feedback,
                "reward": reward,
            })
            _tx.turn(hash_id, turn, response, choice, content, feedback, reward)

            json_data["messages"].append({
                "role": "assistant",
                "content": {"name": "interact_with_env", "arguments": json_args}
            })
            json_data["messages"].append({
                "role": "tool",
                "content": feedback
            })

            if terminated or truncated:
                break

            if "gpt" in model_name:
                tool_call = response.choices[0].message.tool_calls[0]
                content = response.choices[0].message.content
                data["messages"].append({"role": "assistant", "tool_calls": [tool_call], "content": content})
                data["messages"].append({"role": "tool", "tool_call_id": tool_call.id, "content": feedback})
            elif "gemini" in model_name:
                tool_call = response.candidates[0].content.parts[0].function_call
                function_response_part = types.Part.from_function_response(name=tool_call.name, response={"result": feedback})
                response_content = types.Content(role="user", parts=[function_response_part])
                data["messages"].append({"role": "assistant", "content": response.candidates[0].content})
                data["messages"].append({"role": "tool", "content": response_content})
            else:
                tool_call = response.choices[0].message.tool_calls[0]
                try:
                    content = response.choices[0].message.content
                    content = "" if not content else content
                except:
                    content = ""
                # Add this for reasoning models like qwen3
                try:
                    reasoning_content = response.choices[0].message.reasoning_content
                    reasoning_content = "" if not reasoning_content else "<think>" + reasoning_content + "</think>"
                except:
                    reasoning_content = ""
                final_content = reasoning_content + content
                data["messages"].append({"role": "assistant", "tool_calls": [tool_call], "content": final_content})
                data["messages"].append({"role": "tool", "content": feedback})
            turn += 1
            
        total_reward = rewards
        return {"hash_id": hash_id, "reward": total_reward, "history": history, "data": json_data}
    
    except asyncio.TimeoutError:
        print(f"==================== [local] rollout timeout !!! ====================")
        total_reward = rewards if len(rewards) > 0 else [0]
        return {"hash_id": hash_id, "reward": total_reward, "history": history, "data": json_data}
    
    except Exception as e:
        print(f"==================== [local] rollout failed: {e} ====================")
        total_reward = rewards if len(rewards) > 0 else [0]
        return {"hash_id": hash_id, "reward": total_reward, "history": history, "data": json_data}


def _label_offline_search(results):
    """Rename `bamboogle` when SearchGym ran with the search tool stubbed out.

    SEARCH_OFFLINE=1 (searchgym/utils.py) replaces every search result with a
    fixed "search is unavailable" notice, so the episode measures whether the
    agent can answer Bamboogle's two-hop questions from parametric memory. The
    score is still correct -- Bamboogle ships gold answers and scoring never
    consults search -- but it is NOT a SearchGym score and must never be joined
    to one.

    Renaming the key is the guard. eval/table3.py maps only
    ("bamboogle", "SearchGym"), so an unrecognised key is silently *skipped*
    rather than mis-attributed: the column comes out ABSENT instead of WRONG,
    which is the safe direction to fail. Putting it in the table again takes a
    deliberate ("bamboogle_closedbook", "SearchGym (closed-book)") entry.

    Returns a copy; the live `results` dict is left alone so a resumed run still
    finds its own keys.
    """
    if os.getenv("SEARCH_OFFLINE", "0") != "1" or "bamboogle" not in results:
        return results
    out = {k: v for k, v in results.items() if k != "bamboogle"}
    out["bamboogle_closedbook"] = results["bamboogle"]
    out["_meta"] = {
        "search_mode": "offline",
        "note": ("SearchGym ran with the search backend stubbed out; this is "
                 "closed-book two-hop QA, not a search score. Not comparable "
                 "to any SearchGym number obtained with live search."),
        "stub": os.getenv("SEARCH_OFFLINE_NOTICE", "<default notice>"),
    }
    return out


async def post_process_results(results, reward_cache, env, pass_k):
    if "travel" in env:
        results[env][str(pass_k)] = {}
        number_of_1 = []
        number_of_08 = []
        micro_avg = []
        micro_max = []
        for hash_id in reward_cache[env]:
            turn_scores = reward_cache[env][hash_id]["reward"][-1]
            number_of_1.append(turn_scores.count(1.0)) # best choice
            number_of_08.append(turn_scores.count(0.8)) # correct choice
            micro_avg.append(sum(turn_scores) / len(turn_scores) if len(turn_scores) > 0 else 0)
            micro_max.append(max(turn_scores) if len(turn_scores) > 0 else 0)
        results[env][str(pass_k)]["micro_avg"] = sum(micro_avg) / len(micro_avg)
        results[env][str(pass_k)]["micro_max"] = sum(micro_max) / len(micro_max)
        results[env][str(pass_k)]["avg_number_of_08"] = sum(number_of_08) / len(number_of_08)
        results[env][str(pass_k)]["avg_number_of_1"] = sum(number_of_1) / len(number_of_1)
        print(f"\n ######### Pass {pass_k} reward: {results[env][str(pass_k)]["micro_max"]} ######### \n")
    else: # env in ["turtle", "telepathy", "intention", "persuasion", etc.]
        results[env][str(pass_k)] = {}
        micro_avg = []
        for hash_id in reward_cache[env]:
            all_rewards = reward_cache[env][hash_id]["reward"]
            all_rewards = [sum(r) if isinstance(r, list) else r for r in all_rewards]
            micro_avg.append(sum(all_rewards) / len(all_rewards) if len(all_rewards) > 0 else 0)
        results[env][str(pass_k)]["micro_avg"] = sum(micro_avg) / len(micro_avg)
        micro_max = []
        for hash_id in reward_cache[env]:
            all_rewards = reward_cache[env][hash_id]["reward"]
            all_rewards = [sum(r) if isinstance(r, list) else r for r in all_rewards]
            micro_max.append(max(all_rewards) if len(all_rewards) > 0 else 0)
        results[env][str(pass_k)]["micro_max"] = sum(micro_max) / len(micro_max)
        print(f"\n ######### Pass {pass_k} reward: {results[env][str(pass_k)]["micro_max"]} ######### \n")
    return results


async def limited_rollout(*args, **kwargs):
    async with semaphore:
        return await rollout(*args, **kwargs)
    

async def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct")
    arg_parser.add_argument("--port", type=int, default=8000)
    arg_parser.add_argument("--max_turns", type=int, default=8)
    arg_parser.add_argument("--pass_k", type=int, nargs="+", default=[1])
    arg_parser.add_argument("--temperature", type=float, default=1.0)
    arg_parser.add_argument("--envs", type=str, nargs="+", default=["travel22", "travel33", "travel44"])
    arg_parser.add_argument("--save_name", type=str, default="results")

    args = arg_parser.parse_args()

    print(args)

    # AGENT_BASE_URL wins over the name heuristics below. Without it any agent
    # whose name contains "gpt" is sent to https://api.openai.com/v1, which is
    # wrong whenever the agent is served elsewhere -- a locally served
    # checkpoint, or a model behind your own gateway. scripts/run_eval.sh sets
    # this to the vLLM server it just started.
    if os.environ.get("AGENT_BASE_URL"):
        client = AsyncOpenAI(api_key=os.environ.get("AGENT_API_KEY", "local-shim"),
                             base_url=os.environ["AGENT_BASE_URL"])
    elif "gpt" in args.model_name and "gpt-oss" not in args.model_name:
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1")
    elif "gemini" in args.model_name:
        if Client is None:
            raise ImportError("google.genai not installed; cannot evaluate a gemini model")
        client = Client(api_key=os.environ["GENAI_API_KEY"])
    else:
        base_url = f"http://localhost:{args.port}/v1"
        client = AsyncOpenAI(api_key="dummy", base_url=base_url)

    function = yaml.safe_load(open(f"{PROJECT_ROOT}/schema/interact_tool.yaml", "r"))["tool_schema"]

    # e.g. "outputs/results_qwen3-8b-...__Ditto-8B__J-<judge>" -> the cell name.
    # Transcript filenames key on this, because hash_id alone is the same task in
    # every cell of the matrix.
    _tx.run = args.save_name.rsplit("/", 1)[-1].removeprefix("results_")
    if _tx.dir:
        print(f"transcripts: {_tx.dir} (run={_tx.run})")

    if os.path.exists(f"{EVAL_OUTPUT_DIR}/{args.save_name}_results.json"):
        results = json.load(open(f"{EVAL_OUTPUT_DIR}/{args.save_name}_results.json", "r"))
    else:
        results = {}
    if os.path.exists(f"{EVAL_OUTPUT_DIR}/{args.save_name}_reward_cache.json"):
        reward_cache = json.load(open(f"{EVAL_OUTPUT_DIR}/{args.save_name}_reward_cache.json", "r"))
    else:
        reward_cache = {}
    
    for env in args.envs:
        print(f"Evaluating {env}...")
        data = await load_data(env)
        if env not in results:
            results[env] = {}
        if env not in reward_cache:
            reward_cache[env] = {}
        
        for pass_k in args.pass_k:
            if str(pass_k) in results[env]:
                print(f"Pass {pass_k} already evaluated, skipping...")
                continue

            reqs = []
            # use limited rollout to avoid getting stuck
            for d in data:
                data_id = hash(d)
                existing_number = len(reward_cache[env][data_id]["reward"]) if data_id in reward_cache[env] else 0
                needed_number = max(0, pass_k - existing_number)
                if needed_number == 0:
                    print(f"Data {data_id} already has enough rollouts, skipping...")
                    continue
                for _ in range(needed_number):
                    reqs.append(
                        limited_rollout(copy.deepcopy(d), client, function, args.temperature, args.max_turns, args.model_name)
                    )

            # Run all rollout requests in parallel
            rewards = await asyncio.gather(*reqs)

            # Process the rewards
            for r in rewards:
                hash_id = r["hash_id"]
                reward = r["reward"]
                history = r["history"]
                data = r["data"]
                post_processed_reward = reward # if "travel" in env else sum(reward)
                if hash_id not in reward_cache[env]:
                    reward_cache[env][hash_id] = {"history": [], "reward": [], "data": []}
                reward_cache[env][hash_id]["history"].append(history)
                reward_cache[env][hash_id]["data"].append(data)
                reward_cache[env][hash_id]["reward"].append(post_processed_reward)
            
            # Post-process the results
            results = await post_process_results(results, reward_cache, env, pass_k)

            save_dir = args.save_name.rsplit("/", 1)[0]
            if not os.path.exists(f"{EVAL_OUTPUT_DIR}/{save_dir}"):
                os.makedirs(f"{EVAL_OUTPUT_DIR}/{save_dir}")

            with open(f"{EVAL_OUTPUT_DIR}/{args.save_name}_results.json", "w") as f:
                json.dump(_label_offline_search(results), f, indent=4)
            with open(f"{EVAL_OUTPUT_DIR}/{args.save_name}_reward_cache.json", "w") as f:
                json.dump(reward_cache, f, indent=4)
        

if __name__ == "__main__":
    asyncio.run(main())


    
    
    