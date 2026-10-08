"""
Environment Manager for Multi-turn Interactions
"""

import logging
import os
from typing import Any, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


def _tune_llm_client(env_config) -> None:
    """Apply training-time overrides to a gym's LLM client settings."""
    timeout = float(os.environ.get("USERRL_GYM_TIMEOUT", "180"))
    if hasattr(env_config, "timeout"):
        env_config.timeout = timeout

    max_tokens = int(os.environ.get("USERRL_GYM_MAX_TOKENS", "0"))
    if max_tokens and hasattr(env_config, "max_tokens"):
        env_config.max_tokens = max_tokens


# Gyms whose simulator endpoint cannot be episode-scoped.
_NO_EPISODE_SCOPE = {"TauGym", "FunctionGym"}

_sdpo_preflight_done = False
_scope_logged = False


def _scope_to_episode(env, env_name: str, request_id: str) -> None:
    """Point this gym instance's simulator calls at /ep/<request_id>/v1."""
    global _sdpo_preflight_done
    if os.environ.get("USERRL_SDPO") != "1":
        return

    if not _sdpo_preflight_done:
        _sdpo_preflight_done = True
        judge = (os.environ.get("SDPO_HINT_BASE_URL")
                 or os.environ.get("JUDGE_BASE_URL", ""))
        sim = os.environ.get("OPENAI_BASE_URL", "")
        if not judge:
            raise RuntimeError(
                "USERRL_SDPO=1 requires SDPO_HINT_BASE_URL: without a separate coach "
                "endpoint, coach calls land in the simulator's privileged buffer "
                "and every hint is attributed to the wrong turn (see agent/README.md).")
        if sim and judge.rstrip("/") == sim.rstrip("/"):
            raise RuntimeError(
                f"USERRL_SDPO=1 requires the coach on a different shim from the "
                f"simulator, but both are {judge}. Point SDPO_HINT_BASE_URL at a "
                f"second shim (see agent/README.md).")

    if env_name in _NO_EPISODE_SCOPE:
        return
    cfg = getattr(env, "config", None)
    if cfg is None or not hasattr(cfg, "base_url"):
        logger.warning(f"[SDPO] {env_name} has no config.base_url; not episode-scoped")
        return

    url = cfg.base_url or os.environ.get("OPENAI_BASE_URL", "")
    if not url:
        logger.warning(f"[SDPO] {env_name} has an empty base_url and OPENAI_BASE_URL "
                       f"is unset; not episode-scoped")
        return
    base = url[:-3] if url.endswith("/v1") else url.rstrip("/")
    cfg.base_url = f"{base}/ep/{request_id}/v1"

    global _scope_logged
    if not _scope_logged:
        _scope_logged = True
        print(f"[SDPO] episode scoping ACTIVE: {env_name} -> {cfg.base_url}")


class EnvironmentManager:
    """Manages environment instances for multi-turn conversations."""
    
    def __init__(self):
        self._environments: Dict[str, Any] = {}  # request_id -> environment
        self._env_configs: Dict[str, Dict] = {}   # request_id -> config
    
    def create_environment(self, request_id: str, env_name: str, **kwargs) -> str:
        """Create and store environment for a conversation."""
        if request_id in self._environments:
            logger.warning(f"Environment for request_id {request_id} already exists")
            return request_id
            
        env = None
        
        if env_name == "TurtleGym":
            env = self._create_turtlegym_environment(**kwargs)
        elif env_name == "TelepathyGym":
            env = self._create_telepathygym_environment(**kwargs)
        elif env_name == "PersuadeGym":
            env = self._create_persuadegym_environment(**kwargs)
        elif env_name == "IntentionGym":
            env = self._create_intentiongym_environment(**kwargs)
        elif env_name == "TravelGym":
            env = self._create_travelgym_environment(**kwargs)
        elif env_name == "SearchGym":
            env = self._create_searchgym_environment(**kwargs)
        elif env_name == "TauGym":
            env = self._create_taugym_environment(**kwargs)
        elif env_name == "FunctionGym":
            env = self._create_functiongym_environment(**kwargs)
        else:
            raise ValueError(f"Unknown environment type: {env_name}")

        _scope_to_episode(env, env_name, request_id)

        self._environments[request_id] = env
        self._env_configs[request_id] = {
            "env_name": env_name,
            "kwargs": kwargs
        }
        logger.info(f"Created {env_name} environment for request {request_id}")
        return request_id
    
    def get_environment(self, request_id: str) -> Optional[Any]:
        """Get environment for a conversation."""
        return self._environments.get(request_id)
    
    def release_environment(self, request_id: str) -> None:
        """Clean up environment for a conversation."""
        if request_id in self._environments:
            # Cleanup environment if it has cleanup methods
            env = self._environments[request_id]
            if hasattr(env, 'close'):
                env.close()
            
            del self._environments[request_id]
            del self._env_configs[request_id]
            logger.info(f"Released environment for request {request_id}")
    
    def _create_turtlegym_environment(self, **kwargs):
        """Create TurtleGym environment."""
        import turtlegym
        
        env_config = turtlegym.get_default_config()
        
        # Configure from kwargs
        title = kwargs.get("title")
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        
        env_config.max_steps = max_turns
        env_config.success_threshold = 1.0
        env_config.data_mode = "single"
        env_config.data_source = title
        env_config.model_name = model_name
        
        _tune_llm_client(env_config)
        
        env = turtlegym.StoryEnv(config=env_config)
        env.reset()
        
        return env
    
    def _create_telepathygym_environment(self, **kwargs):
        """Create TelepathyGym environment."""
        import telepathygym
        
        env_config = telepathygym.get_default_config()
        
        # Configure from kwargs
        title = kwargs.get("title")
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        
        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = title
        env_config.model_name = model_name

        _tune_llm_client(env_config)

        env = telepathygym.TelepathyEnv(config=env_config)
        env.reset()
        
        return env

    def _create_persuadegym_environment(self, **kwargs):
        """Create PersuadeGym environment."""
        import persuadegym
        
        env_config = persuadegym.get_default_config()
        
        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        id = kwargs.get("id")
        
        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.model_name = model_name
        
        _tune_llm_client(env_config)
        
        env = persuadegym.PersuadeEnv(config=env_config)
        env.reset()
        
        return env

    def _create_intentiongym_environment(self, **kwargs):
        """Create IntentionGym environment."""
        import intentiongym
        
        env_config = intentiongym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.model_name = model_name

        _tune_llm_client(env_config)

        env = intentiongym.IntentionEnv(config=env_config)
        env.reset()

        return env

    def _create_travelgym_environment(self, **kwargs):
        """Create TravelGym environment."""
        import travelgym
        
        env_config = travelgym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        model_name = kwargs.get("model_name", "gpt-4o")
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.model_name = model_name
        env_config.one_choice_per_aspect = True

        env_config.search_correct_reward = 0.0
        env_config.preference_correct_reward = 0.8

        _tune_llm_client(env_config)

        env = travelgym.TravelEnv(config=env_config)
        env.reset()

        return env

    def _create_searchgym_environment(self, **kwargs):
        """Create SearchGym environment."""
        import searchgym
        
        env_config = searchgym.get_default_config()
        model_name = kwargs.get("model_name", "gpt-4o")
        
        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.eval_method = "llm"
        env_config.model_name = model_name

        _tune_llm_client(env_config)

        env = searchgym.SearchEnv(config=env_config)
        env.reset()

        return env
    
    def _create_taugym_environment(self, **kwargs):
        """Create TauGym environment."""
        import taugym
        
        env_config = taugym.get_default_config()
        
        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        model_name = kwargs.get("model_name", "gpt-4o")
        id = kwargs.get("id")
        
        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.user_model = model_name

        if "retail" in id:
            env_config.task_category = "retail"
        else:
            env_config.task_category = "airline"
        if "train" in id:
            env_config.task_split = "train"
        else:
            env_config.task_split = "test"

        _tune_llm_client(env_config)

        env = taugym.TauEnv(config=env_config)
        env.reset()
        
        return env
    
    def _create_functiongym_environment(self, **kwargs):
        """Create FunctionGym environment."""
        import functiongym
        
        env_config = functiongym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        id = kwargs.get("id")
        
        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        
        _tune_llm_client(env_config)
        
        env = functiongym.FunctionEnv(config=env_config)
        env.reset()
        
        return env

# Global environment manager instance
_env_manager = EnvironmentManager()


def get_environment_manager() -> EnvironmentManager:
    """Get the global environment manager instance."""
    return _env_manager 