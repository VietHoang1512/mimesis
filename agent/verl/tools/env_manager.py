"""
Environment Manager for Multi-turn Interactions
Handles environment lifecycle and persistence across conversation turns.
"""

import logging
import os
from typing import Any, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


def _tune_llm_client(env_config) -> None:
    """Apply training-time overrides to a gym's LLM client settings.

    Only `timeout` today, and it is cheap insurance. Every gym dataclass defaults
    it to 10s (TravelGym 15s), sized for gpt-4o-mini. Measured against the
    serving endpoint at SHIM_REASONING_EFFORT=none, the simulator model answers
    in ~2s (p90 2.4s at 32-way concurrency), so 10s does fit -- but with no
    headroom for a degraded window at the provider, a longer prompt, or a bump to
    effort=low, where separate measurements put the median at 25-30s.

    Overrunning it is not a loud failure: the gyms catch the timeout and
    substitute canned fallback text ("That's an interesting question.") rather
    than raising, so the run would look healthy while training the policy against
    a random-phrase user. That asymmetry is why the ceiling is raised rather than
    left at a value that merely usually works.

    Set USERRL_GYM_TIMEOUT to override. Guarded by hasattr because FunctionGym
    (pure math) and TauGym (litellm, which reads its own env) have no such field.
    """
    timeout = float(os.environ.get("USERRL_GYM_TIMEOUT", "180"))
    if hasattr(env_config, "timeout"):
        env_config.timeout = timeout

    # A THINKING simulator needs a much larger output budget than the gyms
    # default to. The failure is silent and total: the chat template prefills
    # <think>, so a truncated completion carries NO tag at all -- not even the
    # closing one, which arrives as "</think" without the ">" -- and
    # llm/shim.py's remove_think cannot match it (see that file's note on the
    # pre-filled-opener case being indistinguishable from a plain answer). The
    # gym then receives the raw chain-of-thought as the simulator's utterance
    # AND as its judge verdict, parses nothing useful out of it, and scores
    # 0.000 forever. No error is logged anywhere: not in the shim, not in the
    # gym, not in the trainer. One run behaved exactly that way against a
    # thinking simulator checkpoint, while an otherwise identical run against a
    # non-thinking simulator scored normally.
    max_tokens = int(os.environ.get("USERRL_GYM_MAX_TOKENS", "0"))
    if max_tokens and hasattr(env_config, "max_tokens"):
        env_config.max_tokens = max_tokens


# Gyms whose simulator endpoint cannot be episode-scoped.
#   TauGym    -- taugym/config.py *writes* OPENAI_BASE_URL into os.environ for
#                litellm. That is process-global and would race across the
#                concurrent episodes in this worker. eval/eval.py carves it out
#                for the same reason.
#   FunctionGym -- pure math, no LLM, so no reasoning to capture.
_NO_EPISODE_SCOPE = {"TauGym", "FunctionGym"}

_sdpo_preflight_done = False
_scope_logged = False


def _scope_to_episode(env, env_name: str, request_id: str) -> None:
    """Point this gym instance's simulator calls at /ep/<request_id>/v1.

    llm/shim.py serves that path as an alias of /v1 and records the episode id
    with every call, which is what lets the shim hold the simulator's reasoning
    against the trajectory that produced it. Without the id the shim sees the
    concurrent episodes in this worker interleaved with no way to tell them
    apart, and a hint built from that buffer would be attached to some other
    trajectory's turn -- plausible text, wrong response, nothing downstream can
    detect it.

    Mutating `env.config` after construction is sound: every gym rebuilds its
    `model_config` dict from `self.config` on each step (e.g.
    travel_env.py:429) and constructs the OpenAI client per call from that dict,
    so none of them cache a base_url. Verified across all six scopeable gyms --
    if a future gym caches a client at construction, this silently stops taking
    effect, so pair any such change with a check that the rewritten base_url is
    the one actually used at call time.

    Inert unless USERRL_SDPO=1.
    """
    global _sdpo_preflight_done
    if os.environ.get("USERRL_SDPO") != "1":
        return

    if not _sdpo_preflight_done:
        _sdpo_preflight_done = True
        # A judge on its own shim means the simulator shim's buffer sees only
        # simulator traffic -- exactly one call per turn, since every gym's
        # evaluate_action reaches one `return` per branch. Two ways to lose that:
        # leave JUDGE_BASE_URL unset, so _as_judge() hands the judge the same
        # config; or point judge and simulator at the SAME shim, which
        # scripts/run_sdpo.sh does by default -- SDPO_HINT_BASE_URL falls back
        # to the simulator's own endpoint. Either way both sides land in one
        # buffer and thought-to-turn alignment slips by however many judge calls
        # occurred -- silently, and every hint then describes a different turn
        # than the one it is attached to. Refuse rather than train on that.
        judge = (os.environ.get("SDPO_HINT_BASE_URL")
                 or os.environ.get("JUDGE_BASE_URL", ""))
        sim = os.environ.get("OPENAI_BASE_URL", "")
        if not judge:
            raise RuntimeError(
                "USERRL_SDPO=1 requires JUDGE_BASE_URL: without a separate judge "
                "endpoint, judge calls land in the simulator's privileged buffer "
                "and every hint is attributed to the wrong turn.")
        if sim and judge.rstrip("/") == sim.rstrip("/"):
            raise RuntimeError(
                f"USERRL_SDPO=1 requires the judge on a DIFFERENT shim from the "
                f"simulator, but both are {judge}. scripts/run_sdpo.sh defaults "
                f"SDPO_HINT_BASE_URL to the simulator's own endpoint; point it "
                f"at a second shim serving the judge model.")

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

    # If this function silently fails to take effect, every simulator call lands
    # in the shim's unscoped bucket while the post-pass still runs normally, so
    # the only symptom is hints with no reasoning attached. Say so once, loudly,
    # rather than leaving "did scoping happen" to be inferred downstream.
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
        """Create and store environment for a conversation.
        
        Args:
            request_id: Unique identifier for the conversation
            env_name: Type of environment (TurtleGym, TelepathyGym, etc.)
            **kwargs: Environment-specific configuration
            
        Returns:
            request_id for the created environment
        """
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