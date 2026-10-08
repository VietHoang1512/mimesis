from typing import Any, Optional, Tuple
import asyncio
import os

import httpx

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
from .env_manager import get_environment_manager

# ---- privileged channel (SDPO) ----
_SDPO = os.environ.get("USERRL_SDPO") == "1"
_privileged_client: "httpx.AsyncClient | None" = None
_privileged_stats = {"ok": 0, "unscoped": 0, "failed": 0, "empty": 0}


def _privileged_url(env) -> "str | None":
    """Derive the shim's privileged endpoint from the gym's own base_url."""
    url = getattr(getattr(env, "config", None), "base_url", "") or ""
    if "/ep/" not in url or not url.endswith("/v1"):
        return None
    return url[:-3] + "/privileged"


async def _fetch_think(env) -> "str | None":
    """The simulator's reasoning behind the reply this turn just received."""
    global _privileged_client
    if not _SDPO:
        return None
    url = _privileged_url(env)
    if url is None:
        _privileged_stats["unscoped"] += 1
        _report_privileged()
        return None
    try:
        if _privileged_client is None:
            _privileged_client = httpx.AsyncClient(timeout=5.0)
        r = await _privileged_client.get(url)
        r.raise_for_status()
        think = (r.json().get("think") or [])
    except Exception as e:                                   # noqa: BLE001
        _privileged_stats["failed"] += 1
        if _privileged_stats["failed"] % 100 == 1:
            print(f"[SDPO] privileged fetch failed ({_privileged_stats['failed']}x): "
                  f"{type(e).__name__}: {str(e)[:160]}")
        return None
    if not think:
        _privileged_stats["empty"] += 1
        _report_privileged()
        return None
    _privileged_stats["ok"] += 1
    _report_privileged()
    return think[-1]


def _report_privileged(every: int = 50):
    """Periodically print the fetch breakdown."""
    total = sum(_privileged_stats.values())
    if total and total % every == 0:
        print(f"[SDPO] privileged fetches: {_privileged_stats}")

class InteractTool(BaseTool):
    """A tool for interacting with environments across multi-turn conversations."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._conversation_data = {}  # request_id -> conversation state
        self._env_manager = get_environment_manager()

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str, env_name: Optional[str] = None, max_turns: int = 15, **kwargs) -> str:
        """Create environment and initialize conversation state."""
        if instance_id in self._conversation_data:
            print(f"!!!!!!!! Conversation {instance_id} already exists !!!!!!!!")
            return instance_id
        
        # Create environment through environment manager
        if env_name:
            kwargs["max_turns"] = max_turns
            self._env_manager.create_environment(instance_id, env_name, **kwargs)
        
        # Initialize conversation state (separate from environment)
        self._conversation_data[instance_id] = {
            "history": [],
            "reward": 0.0,
            "ground_truth": kwargs.get("ground_truth"),
            "env_name": env_name,
        }
        
        print(f"Created conversation {instance_id} with {env_name} environment")
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], current_turns, **kwargs) -> Tuple[str, float, dict]:
        """Execute action in the persistent environment."""
        
        if instance_id not in self._conversation_data:
            raise ValueError(f"Conversation {instance_id} not found. Call create() first.")
        
        # Get persistent environment
        env = self._env_manager.get_environment(instance_id)
        if env is None:
            raise ValueError(f"Environment for conversation {instance_id} not found")
        
        # Parse action parameters
        choice = str(parameters.get("choice", ""))
        content = str(parameters.get("content", ""))
        
        # Format action for environment
        if choice == "action" and not content.startswith("[action]"):
            formatted_action = "[action] " + content
        elif choice == "answer" and not content.startswith("[answer]"):
            formatted_action = "[answer] " + content
        elif choice == "search" and not content.startswith("[search]"):
            formatted_action = "[search] " + content
        elif choice == "finish":
            formatted_action = "[finish]"
        else:
            formatted_action = content
        
        env_failed = False
        try:
            observation, reward, terminated, truncated, info = await asyncio.wait_for(
                env.step_async(formatted_action),
                timeout=float(os.environ.get("USERRL_STEP_TIMEOUT", "300"))
            )
        except asyncio.TimeoutError:
            print(f"Environment step timed out for {instance_id} after "
                  f"{os.environ.get('USERRL_STEP_TIMEOUT', '300')}s")
            try:
                print(f"Attempting fallback process isolation for {instance_id}")
                result = await asyncio.to_thread(
                    self._run_env_in_process, env, formatted_action
                )
                observation, reward, terminated, truncated, info = result
            except Exception as e:
                print(f"Process isolation fallback failed: {e}")
                observation = {"feedback": "Environment operation failed completely"}
                reward, terminated, truncated, info = 0.0, True, False, {}
                env_failed = True
        except Exception as e:
            print(f"Environment step failed for {instance_id}: {e}")
            # Return safe fallback values
            observation = {"feedback": f"Error: {str(e)}"}
            reward, terminated, truncated, info = 0.0, True, False, {}
            env_failed = True

        # Update conversation state
        conversation_state = self._conversation_data[instance_id]
        current_env_name = conversation_state["env_name"]
        conversation_state["reward"] = reward
        conversation_state["history"].append({
            "choice": choice,
            "content": content,
            "observation": observation,
            "reward": reward,
            "info": info
        })
        
        # Format response
        feedback = observation.get("feedback", "") if isinstance(observation, dict) else str(observation)
        response_text = f"{feedback}\nReward: {reward}"
        
        is_done = terminated or truncated
        print(f"Turn {current_turns}: Executed {choice} in conversation {instance_id} (Env: {current_env_name}), action: {formatted_action}, feedback: {feedback}, reward: {reward}, done: {is_done}")

        metrics = {}
        if _SDPO and not env_failed:
            metrics["privileged"] = {
                "reply": feedback,
                "think": await _fetch_think(env),
            }

        return response_text, reward, is_done, choice, content, metrics

    def _run_env_in_process(self, env, formatted_action):
        """Run environment step in separate process to isolate from NCCL context."""
        try:
            if hasattr(env, 'step'):
                return env.step(formatted_action)
            else:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(env.step_async(formatted_action))
        except Exception as e:
            return {"feedback": f"Process error: {e}"}, 0.0, True, False, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate final reward for the conversation."""
        if instance_id not in self._conversation_data:
            print(f"!!!!!!!! Conversation {instance_id} not found for reward calculation !!!!!!!!")
            return 0.0
        
        conversation_state = self._conversation_data[instance_id]
        
        # Return the highest reward achieved during the conversation
        if conversation_state["history"]:
            max_reward = max(step["reward"] for step in conversation_state["history"])
            return max_reward
        
        return conversation_state["reward"]
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up conversation and environment."""
        # Clean up conversation state
        if instance_id in self._conversation_data:
            del self._conversation_data[instance_id]
        
        # Clean up environment through manager
        self._env_manager.release_environment(instance_id)
        
